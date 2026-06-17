"""E36: 2D-radial spectral surgery on the diffusion LATENT, applied DURING generation.

The token-axis toolkit (`text_spectral_ops.py`, E24/E30/E32/E35) edits a prompt's
sequence embedding ONCE before generation. This module is its latent-space analogue:
it FFTs the 2D **spatial** axes of a diffusion latent `(B, C, H, W)` and edits the
*radial* frequency spectrum -- and, crucially, it is applied *inside the denoising
loop* via diffusers' `callback_on_step_end`. That makes **when** an op fires (init /
every step / early / late / last-step) a first-class control the token sweep could
not have.

Frequency convention (matches spectral_ops.py): everything works UNSHIFTED (DC at
[0,0], torch.fft convention). A real-valued mask/gain that is symmetric under
f -> -f keeps the spectrum Hermitian, so `ifft2(.).real` loses only ~1e-6 residue;
the radial frequency `rr` (spectral_ops.radial_bins) is exactly such a symmetric
field. `cut`/`lo`/`hi` are the **normalised radial frequency** in `[0, 1]`
(0 = DC, 1 = the corner / highest radial frequency), so a `cut` means the same thing
for any latent size -- the direct 2D analogue of the token axis's normalised `[0,1]`.

Three op groups:
  - SINGLE-LATENT FILTERS / PHASE-MAG: pure spectral masks on the current latent
    (low/high-pass, band gain, notch, phase-only/mag-only, phase band, quantize phase).
  - SBN toward a TARGET spectrum: per-(channel, radial band) magnitude clamp toward a
    cfg=1 proxy or real-image power reference (reuses spectral_ops.psd_match /
    band_power, the E8/E9/E16/E23 operator), plus scalar band modulation / global power.
  - TWO-LATENT (offline recombination of two finished latents): full-spectrum hybrid
    split and phase swap (E18) -- exposed in the demo, where the user supplies two
    prompts.

`LatentOpCallback` is the loop hook that applies any of the per-step ops on the
schedule; `unpack`/`pack` default to identity (SD1.5, latents already unpacked) and
take Flux's `_unpack_latents`/`_pack_latents` for the packed-latent models.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_ops import (radial_bins, band_index_map, band_power, psd_match,
                          phase_only, magnitude_only, quantize_phase,
                          _restore_self_conj)
from style_ops import restyle_latent, color_noise, blend_references  # noqa: F401 (re-export)

SCHEDULES = ["init", "every", "early", "late", "last", "interval"]


# ---------------------------------------------------------------------------
# radial frequency bookkeeping (normalised [0,1])
# ---------------------------------------------------------------------------

def radial_norm(H, W, device="cuda"):
    """(H, W) normalised radial frequency in [0, 1] (DC at [0,0] -> 0, corner -> 1).

    rr / rr.max() of spectral_ops.radial_bins; symmetric under f -> -f, so any
    mask/gain built from it is Hermitian-preserving (ifft2 stays real)."""
    rr = radial_bins(H, W, device)
    return rr / rr.max().clamp(min=1e-12)


def _band_sel(H, W, lo, hi, device, keep_dc=False, drop_dc=False):
    """{0,1}-ish bool mask of the normalised radial band [lo, hi]. keep_dc adds the
    DC bin; drop_dc removes it (mutually exclusive uses)."""
    r = radial_norm(H, W, device)
    sel = (r >= lo) & (r <= hi)
    if keep_dc:
        sel = sel | (r == 0)
    if drop_dc:
        sel = sel & (r != 0)
    return sel


def band_to_norm(cut, band):
    """('low'|'high', cut) -> (lo, hi) normalised-radius band edges."""
    return (0.0, cut) if band == "low" else (cut, 1.0)


# ---------------------------------------------------------------------------
# single-latent radial filters (batched over (B, C, H, W))
# ---------------------------------------------------------------------------

def band_filter_2d(x, lo, hi, keep_dc=True):
    """Keep radial frequencies in normalised [lo, hi]; zero the rest. DC kept when
    keep_dc (so a high-pass latent isn't degenerate). 2D analogue of band_filter_1d."""
    B, C, H, W = x.shape
    F = torch.fft.fft2(x.float())
    mask = _band_sel(H, W, lo, hi, x.device, keep_dc=keep_dc).to(F.dtype)
    return torch.fft.ifft2(F * mask[None, None]).real.to(x.dtype)


def band_gain_2d(x, lo, hi, gain, keep_dc=True):
    """Multiply radial frequencies in normalised [lo, hi] by `gain`; DC left at unity
    when keep_dc so the latent's global level is preserved. 2D analogue of band_gain_1d."""
    B, C, H, W = x.shape
    F = torch.fft.fft2(x.float())
    sel = _band_sel(H, W, lo, hi, x.device, drop_dc=keep_dc)
    g = torch.where(sel, torch.as_tensor(float(gain), device=x.device),
                    torch.ones((), device=x.device)).to(F.dtype)
    return torch.fft.ifft2(F * g[None, None]).real.to(x.dtype)


def band_notch_2d(x, lo, hi):
    """Zero exactly the radial band [lo, hi] (knockout), keep the complement incl. DC.
    notch + band_filter(lo,hi,keep_dc=False) == x (linearity)."""
    return x - band_filter_2d(x, lo, hi, keep_dc=False)


def band_phase_filter_2d(x, lo, hi, keep_dc=True):
    """PHASE band-pass: keep the latent's spatial phase only inside the normalised band
    [lo, hi] (flatten phase to 0 elsewhere) while KEEPING the magnitude everywhere. The
    band-limited 2D phase-only -- which radial bands' phase carries the layout. DC + the
    real Nyquist bins are restored from source so the result stays real."""
    B, C, H, W = x.shape
    F = torch.fft.fft2(x.float())
    keep = _band_sel(H, W, lo, hi, x.device, keep_dc=keep_dc)[None, None]
    new_ph = torch.where(keep, F.angle(), torch.zeros_like(F.angle()))
    Fp = torch.polar(F.abs(), new_ph)
    _restore_self_conj(Fp, F, H, W)
    return torch.fft.ifft2(Fp).real.to(x.dtype)


def band_phase_gain_2d(x, lo, hi, gain, keep_dc=True):
    """PHASE gain: scale the spatial phase angle by `gain` inside the radial band
    [lo, hi] (magnitude kept). gain=0 removes phase there (-> mag-only in the band),
    1 = identity, >1 amplifies. DC/Nyquist left unscaled (stay real)."""
    B, C, H, W = x.shape
    F = torch.fft.fft2(x.float())
    sel = _band_sel(H, W, lo, hi, x.device, drop_dc=keep_dc)[None, None]
    new_ph = torch.where(sel, F.angle() * float(gain), F.angle())
    Fp = torch.polar(F.abs(), new_ph)
    _restore_self_conj(Fp, F, H, W)
    return torch.fft.ifft2(Fp).real.to(x.dtype)


def _batched(fn, x):
    """Apply a (1,C,H,W)->(1,C,H,W) op to every batch element (small B)."""
    return torch.cat([fn(x[b:b + 1]) for b in range(x.shape[0])], dim=0)


def phase_only_2d(x):
    """Oppenheim-Lim phase-only latent (spectral_ops.phase_only): keep phase, flatten
    every non-DC magnitude to a per-channel constant preserving total power."""
    return phase_only(x.float()).to(x.dtype)


def mag_only_2d(x, generator=None):
    """Keep the magnitude, replace phase with a fresh Hermitian-uniform field
    (spectral_ops.magnitude_only). Looped per batch element so seeds get distinct phase."""
    return _batched(lambda s: magnitude_only(s.float(), generator=generator), x).to(x.dtype)


def quantize_phase_2d(x, k):
    """Quantize the 2D FFT phase to k uniform levels (spectral_ops.quantize_phase),
    magnitude + DC kept. k=None is identity."""
    return quantize_phase(x.float(), k).to(x.dtype)


# ---------------------------------------------------------------------------
# SBN toward a target spectrum (reuses spectral_ops.psd_match / band_power)
# ---------------------------------------------------------------------------

def sbn_clamp(x, ref_band, idx_map, n_bins):
    """Clamp each batch element's per-(channel, radial band) power toward ref_band
    (C, n_bins); phase untouched. The E8/E9/E16/E23 operator, batched."""
    return _batched(lambda s: psd_match(s, ref_band, idx_map, n_bins), x)


def latent_band_power_batched(x, idx_map, n_bins):
    """Mean (C, n_bins) band power across the batch -- the descriptor used to build a
    style target / measure the realised spectrum."""
    F2 = (torch.fft.fft2(x.float()).abs() ** 2).mean(0)   # (C,H,W) batch-mean
    return band_power(F2, idx_map, n_bins)


def band_modulate(x, idx_map, n_bins, cut_band, lo_gain, hi_gain):
    """Scalar hi/lo-band magnitude modulation (E9 bandnorm.modulate_reference): multiply
    band power below `cut_band` by lo_gain**2 and at/above it by hi_gain**2 (so the
    realised magnitude gain is lo_gain / hi_gain), via psd_match toward the modulated
    own-power. cut_band is a band index in [0, n_bins)."""
    out = []
    for b in range(x.shape[0]):
        cur = band_power((torch.fft.fft2(x[b:b + 1].float()).abs() ** 2)[0], idx_map, n_bins)
        g = torch.ones(n_bins, device=cur.device)
        g[:cut_band] = lo_gain ** 2
        g[cut_band:] = hi_gain ** 2
        out.append(psd_match(x[b:b + 1], cur * g[None], idx_map, n_bins))
    return torch.cat(out, dim=0)


def global_power(x, scale):
    """Scale the whole latent by `scale` (Parseval: a flat power gain, no FFT). The
    'global-norm' control -- changes total power without touching spectral shape."""
    return (x.float() * float(scale)).to(x.dtype)


# ---------------------------------------------------------------------------
# two-latent recombination (offline; both finished latents, E18) -- normalised cut
# ---------------------------------------------------------------------------

def hybrid_split_2d(lat_low, lat_high, cut):
    """Hybrid latent: full complex spectrum (magnitude AND phase) from lat_low inside
    the normalised radial band [0, cut], from lat_high outside. The strong offline
    hybrid (coarse structure of lat_low + fine detail of lat_high). Like
    style_ops.band_spectrum_split but with normalised-radius `cut` (not area-quantile)."""
    assert lat_low.shape == lat_high.shape
    B, C, H, W = lat_low.shape
    Fl = torch.fft.fft2(lat_low.float())
    Fh = torch.fft.fft2(lat_high.float())
    low = _band_sel(H, W, 0.0, cut, lat_low.device, keep_dc=True).to(Fl.dtype)[None, None]
    return torch.fft.ifft2(Fl * low + Fh * (1.0 - low)).real.to(lat_low.dtype)


def phase_swap_2d(lat_a, lat_b, cut, mag_from="A"):
    """Phase from A inside the normalised band [0, cut], phase from B outside; magnitude
    everywhere from `mag_from`. The E18 'phaseA_magB' content/style split, normalised-cut."""
    assert lat_a.shape == lat_b.shape
    B, C, H, W = lat_a.shape
    Fa = torch.fft.fft2(lat_a.float())
    Fb = torch.fft.fft2(lat_b.float())
    low = _band_sel(H, W, 0.0, cut, lat_a.device, keep_dc=True).to(torch.float32)[None, None]
    phi = Fa.angle() * low + Fb.angle() * (1.0 - low)
    M = Fa.abs() if mag_from == "A" else Fb.abs()
    Fmix = torch.polar(M, phi)
    src = Fa if mag_from == "A" else Fb
    for (i, j) in ((0, 0), (H // 2, 0), (0, W // 2), (H // 2, W // 2)):
        Fmix[..., i, j] = src[..., i, j]
    return torch.fft.ifft2(Fmix).real.to(lat_a.dtype)


# ---------------------------------------------------------------------------
# schedule + the loop callback
# ---------------------------------------------------------------------------

def fires_at(schedule, i, total, interval=None):
    """Whether a per-step op on `schedule` applies at step index i of `total`.
    'init' fires at no in-loop step (it pre-sets the initial latent instead).
    'interval' fires on the inclusive step window `interval=(i_lo, i_hi)` -- the
    free-form analogue of the early/late presets and the latent twin of the velocity
    tab's "timesteps to intervene" gate (velocity_spectral_ops.make_velocity_override)."""
    if schedule == "every":
        return True
    if schedule == "last":
        return i == total - 1
    if schedule == "early":
        return i < max(1, total // 3)
    if schedule == "late":
        return i >= (2 * total) // 3
    if schedule == "interval":
        i_lo, i_hi = interval if interval is not None else (0, total - 1)
        return i_lo <= i <= i_hi
    return False  # 'init' handled outside the loop


class LatentOpCallback:
    """diffusers callback_on_step_end that applies `op_fn` to the latent on `schedule`.

    op_fn(lat (B,C,H,W) tensor, step_i) -> lat. unpack/pack convert to/from the
    pipeline's latent layout: leave None for SD-style pipelines (latents already
    (B,C,H,W)); pass Flux's _unpack_latents / _pack_latents (already closed over
    size / channels) for packed-latent models. `interval=(i_lo,i_hi)` is the inclusive
    step window used only by schedule='interval'. Records the post-edit packed latent in
    `.last` (like e8's ClampPSD) so callers can grab exactly what the VAE decoded."""

    def __init__(self, op_fn, schedule, total_steps, unpack=None, pack=None, interval=None):
        self.op_fn = op_fn
        self.schedule = schedule
        self.total = total_steps
        self.unpack = unpack
        self.pack = pack
        self.interval = interval
        self.last = None

    def __call__(self, pipe, i, t, kw):
        packed = kw["latents"]
        if not fires_at(self.schedule, i, self.total, self.interval):
            self.last = packed
            return {}
        lat = self.unpack(packed) if self.unpack is not None else packed
        lat = self.op_fn(lat, i).to(lat.dtype)
        new_packed = self.pack(lat) if self.pack is not None else lat
        self.last = new_packed
        return {"latents": new_packed.to(packed.dtype)}
