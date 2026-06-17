"""E37: spectral surgery on the CFG-combined VELOCITY, applied DURING generation.

Where `latent_spectral_ops.py` (E36) edits the diffusion *latent* on a step-end
callback, this module edits the *velocity* (the flow-matching model output) BEFORE
the Euler step. The motivating idea: classifier-free guidance is the extrapolation

    v_w = v_uncond + w * (v_cond - v_uncond)                    (the CFG velocity)

which buys prompt adherence but systematically over-amplifies certain frequency
*magnitudes* (the oversaturated / over-contrasty CFG look). The unconditional field
v_uncond (== the w=1 / cfg=1 flow field) sits on the model's natural manifold. So we
force v_w to carry v_uncond's FFT *amplitude* while keeping v_w's *phase* (phase is
where layout/composition live). This is the one-pass, scale-correct cousin of the
E8/E16/E23 SBN latent clamp: because v_uncond is the SAME-STEP flow field, its
amplitude is already at the right scale for step i -- unlike a fixed clean-image
target (the every-step `SBN→real` bug, E23/E36) which only matches at the last step.

Cost is two `fft2`/`ifft2` traversals per step; both v_uncond and v_cond are already
computed by CFG, so there is NO extra transformer forward.

Frequency convention matches `spectral_ops`/`latent_spectral_ops`: everything works
UNSHIFTED (DC at [0,0]); `lo`/`hi` are the normalised radial frequency in [0,1]
(0 = DC, 1 = the corner). Real-valued radially-symmetric masks/gains keep the spectrum
Hermitian, so `ifft2(.).real` loses only ~1e-6; the four self-conjugate bins (DC +
Nyquist axes) are restored from the source so realness (and v_w's global level) is exact.

Two NORMALIZE modes (both exposed in the demo):
  - "mag"        per-BIN magnitude transplant   |V_w| <- |V_uncond| inside [lo,hi].
  - "band power" per-BAND mean-power match       psd_match(v_w -> band_power(v_uncond)).
plus a band amplify/reduce gain (independent of v_uncond), and an interval gate so the
op fires only on a contiguous window of denoising steps.

The interception point is `make_velocity_override`, an SD3-style scheduler.step
override (see `e17_sd35.gen_sd3`): it receives the batched [uncond, cond] transformer
output and the pipeline's already-CFG-combined `model_output` (== v_w), and returns the
edited velocity. `callback_on_step_end` fires AFTER the Euler step and cannot see the
velocity, so the step override is the only correct hook.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_ops import band_index_map, band_power, psd_match, _restore_self_conj
from latent_spectral_ops import radial_norm, _band_sel, _batched  # noqa: F401 (shared helpers)

VEL_MODES = ["mag", "band power"]


# ---------------------------------------------------------------------------
# the CFG velocity (for reference / tests; the override uses model_output directly,
# which the pipeline already computed as exactly this with w = guidance_scale)
# ---------------------------------------------------------------------------

def cfg_velocity(v_uncond, v_cond, w):
    """v_w = v_uncond + w*(v_cond - v_uncond) -- the standard CFG-combined velocity."""
    return v_uncond + float(w) * (v_cond - v_uncond)


# ---------------------------------------------------------------------------
# normalize modes: pull v_w's amplitude toward v_uncond's inside a radial band
# ---------------------------------------------------------------------------

def mag_transplant_band(v_w, v_src, lo, hi, strength=1.0):
    """Per-BIN magnitude transplant. Inside the normalised radial band [lo, hi],
    replace |V_w| with |V_src| (blended by `strength`: 1=full, 0=identity) while
    KEEPING v_w's phase; outside the band v_w is untouched. DC + Nyquist self-conj
    bins are restored from v_w so the global level stays and ifft2 is real.

        mag = (1-s)|V_w| + s|V_src|  on [lo,hi];  |V_w| elsewhere
        V'  = mag * exp(i * angle(V_w))

    v_w, v_src: (B, C, H, W) real (same shape). Math in fp32, cast back to v_w.dtype."""
    dt = v_w.dtype
    B, C, H, W = v_w.shape
    Fw = torch.fft.fft2(v_w.float())
    Fs = torch.fft.fft2(v_src.float())
    sel = _band_sel(H, W, lo, hi, v_w.device, drop_dc=True)[None, None]   # exclude DC
    s = float(strength)
    new_mag = torch.where(sel, (1.0 - s) * Fw.abs() + s * Fs.abs(), Fw.abs())
    Fp = torch.polar(new_mag, Fw.angle())
    _restore_self_conj(Fp, Fw, H, W)
    return torch.fft.ifft2(Fp).real.to(dt)


def bandpower_match_band(v_w, v_src, lo, hi, strength=1.0, n_bins=24):
    """Per-BAND mean-power match (the SBN/psd_match operator, E8/E16/E23). Drive each
    radial band's mean power in v_w toward v_src's, but ONLY for bands whose centre
    falls in [lo, hi]; other bands are left at v_w's own power (gain 1 = identity).
    `strength` blends the target between v_w's own power (0) and v_src's (1).

    Phase untouched. Batched over B (psd_match takes a single (1,C,H,W))."""
    B, C, H, W = v_w.shape
    idx = band_index_map(H, W, n_bins, v_w.device)
    # normalised radial centre of each band -> which bands lie inside [lo, hi]
    centres = (torch.arange(n_bins, device=v_w.device) + 0.5) / n_bins
    in_band = (centres >= lo) & (centres <= hi)                    # (n_bins,) bool
    s = float(strength)

    def one(xw, xs):
        cur = band_power((torch.fft.fft2(xw.float()).abs() ** 2)[0], idx, n_bins)   # (C,nb)
        ref = band_power((torch.fft.fft2(xs.float()).abs() ** 2)[0], idx, n_bins)   # (C,nb)
        tgt = torch.where(in_band[None], (1.0 - s) * cur + s * ref, cur)            # (C,nb)
        return psd_match(xw, tgt, idx, n_bins)

    return torch.cat([one(v_w[b:b + 1], v_src[b:b + 1]) for b in range(B)], dim=0)


def band_gain_velocity(v_w, lo, hi, gain):
    """Amplify/reduce v_w's magnitude inside the normalised band [lo, hi] by `gain`
    (DC kept at unity); independent of v_uncond. The E9 band-gain lever on the velocity."""
    dt = v_w.dtype
    B, C, H, W = v_w.shape
    Fw = torch.fft.fft2(v_w.float())
    sel = _band_sel(H, W, lo, hi, v_w.device, drop_dc=True)
    g = torch.where(sel, torch.as_tensor(float(gain), device=v_w.device),
                    torch.ones((), device=v_w.device)).to(Fw.dtype)
    return torch.fft.ifft2(Fw * g[None, None]).real.to(dt)


# ---------------------------------------------------------------------------
# the scheduler.step override (SD3 batched [uncond, cond]; see e17_sd35.gen_sd3)
# ---------------------------------------------------------------------------

def make_velocity_override(op, lo, hi, strength, gain, i_lo, i_hi, n_bins=24):
    """Build a `step_override(records, model_output, sample) -> new_model_output`
    closure for `gen_sd3`-style interception.

      op       : "mag" | "band power" | "gain"  (operator on v_w)
      lo, hi   : normalised radial band edges in [0, 1]
      strength : blend toward v_uncond's amplitude (mag / band power modes)
      gain     : magnitude gain inside the band ("gain" mode)
      i_lo,i_hi: inclusive step-index window; the op fires only when i_lo <= i <= i_hi
                 (outside it, the pipeline's plain CFG velocity passes through unchanged)

    `model_output` IS the pipeline's CFG velocity v_w (== v_uncond + w*(v_cond-v_uncond)
    for the run's guidance_scale), so we use it directly and only need v_uncond from the
    batched transformer output `records[-1]`. Passthrough when there is no CFG batch
    (guidance<=1) so the demo's no-CFG sanity case can't crash."""
    state = {"i": 0}

    def override(records, model_output, sample):
        i = state["i"]
        state["i"] += 1
        if not (i_lo <= i <= i_hi):
            return model_output
        out = records[-1]                                  # batched [uncond, cond]
        if out.shape[0] != 2 * sample.shape[0]:
            return model_output                            # no cfg (guidance<=1)
        v_uncond, _ = out.chunk(2)
        v_w = model_output
        if op == "gain":
            mod = band_gain_velocity(v_w, lo, hi, gain)
        elif op == "band power":
            mod = bandpower_match_band(v_w, v_uncond.to(v_w.dtype), lo, hi, strength, n_bins)
        else:  # "mag"
            mod = mag_transplant_band(v_w, v_uncond.to(v_w.dtype), lo, hi, strength)
        return mod.to(model_output.dtype)

    return override
