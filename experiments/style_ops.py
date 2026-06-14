"""Spectral style transfer / blending operators -- the "two-image spectrum"
direction (E18-E22).

The premise inherited from E7-E14: in the diffusion latent's 2D Fourier domain,
PHASE (esp. low-band) carries content/layout/identity, and per-(channel, radial
band) MAGNITUDE/POWER carries "style" -- the radial energy envelope (texture
slope) plus palette/contrast (the DC + low bands). That is the Gatys/AdaIN split
moved into frequency space: re-leveling per-band power == AdaIN on the radial
power spectrum ("AdaIN-in-Fourier"), and psd_match is exactly that operator.

Two regimes, sharing one notion of a per-band gain g[c,k]:
  - OFFLINE (E18): restyle a concrete latent -- keep its phase + within-band
    texture, drive its per-band power toward a style latent's. -> restyle_latent.
  - GENERATION-TIME (E19-E22): bend the per-step SBN reference that ClampPSD3
    clamps to, so the generated image's spectrum tracks a style image/prompt
    while its layout follows the content prompt. -> build_style_reference and
    friends return a `ref` dict consumable by e17_sd35.ClampPSD3 / bandnorm.

This generalizes bandnorm.modulate_reference (a scalar hi/lo-band gain) to an
arbitrary per-(channel, band) gain vector derived from a style target.

ISOTROPY CAVEAT: every operator here works on *radial* bands, so it transfers
texture-energy + palette, NOT oriented/stroke style (Gram matrices would). This
is the stated scope limit the experiments probe; an anisotropic-band variant is
a later extension.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_ops import band_index_map, band_power, psd_match, radial_bins

EPS = 1e-8


# ---------------------------------------------------------------------------
# Latent-space hybrid image (Oliva et al. 2006), offline
# ---------------------------------------------------------------------------

def band_spectrum_split(lat_low, lat_high, c):
    """Hybrid image in latent Fourier space: the FULL complex spectrum (magnitude
    AND phase) comes from lat_low inside the lowest-c radial fraction and from
    lat_high outside it. c in [0,1] (0 -> all high, 1 -> all low). The
    radial-quantile mask is exactly Hermitian-symmetric (same construction as
    spectral_ops.band_phase_swap), so conjugate partners never straddle the
    cutoff and the ifft stays real. This is the strong offline hybrid (coarse
    structure of lat_low + fine detail of lat_high); the generation-time
    build_hybrid_reference only splits the energy envelope, not the phase."""
    assert lat_low.shape == lat_high.shape
    B, C, H, W = lat_low.shape
    Fl = torch.fft.fft2(lat_low.float())
    Fh = torch.fft.fft2(lat_high.float())
    if c > 0:
        rr = radial_bins(H, W, lat_low.device)
        low = (rr <= torch.quantile(rr.flatten(), c)).float()[None, None]
    else:
        low = torch.zeros(1, 1, H, W, device=lat_low.device)
    mix = Fl * low + Fh * (1.0 - low)
    return torch.fft.ifft2(mix).real


# ---------------------------------------------------------------------------
# Per-band gains: the shared currency of style transfer
# ---------------------------------------------------------------------------

def band_shape(band, eps=EPS):
    """(C, n_bins) band power -> per-channel *shape* (sums to 1 across bands).

    Divides out absolute level so only the radial *distribution* of power -- the
    envelope/slope that reads as texture + palette -- remains."""
    band = band.float()
    return band / band.sum(dim=-1, keepdim=True).clamp(min=eps)


def style_gain(content_band, style_band, strength, eps=EPS, gmax=None):
    """Per-(channel, band) magnitude gain bending content's spectral envelope
    toward style's: gain = (style_shape / content_shape) ** strength.

    strength=0 -> all-ones (no change); strength=1 -> full style envelope. Shapes
    are level-normalized (band_shape) so only the across-band *distribution*
    transfers, not absolute level -- which would destabilize early, noisy steps
    when applied as a per-step clamp target. gmax (if set) clamps each gain to
    [1/gmax, gmax] to bound extreme ratios in bands where content power ~ 0."""
    cs = band_shape(content_band, eps)
    ss = band_shape(style_band, eps)
    g = (ss / cs.clamp(min=eps)).clamp(min=eps) ** strength
    if gmax is not None:
        g = g.clamp(1.0 / gmax, gmax)
    return g                                        # (C, n_bins)


def split_gain(content_band, band_low, band_high, cut_band, strength,
               eps=EPS, gmax=None):
    """Hybrid-image gain (E20): style envelope from `band_low` on radial bands
    below `cut_band`, from `band_high` at/above it. Bands index 0..n_bins-1 by
    increasing radial frequency (band 0 = DC/structure, high = fine detail)."""
    g_lo = style_gain(content_band, band_low, strength, eps, gmax)
    g_hi = style_gain(content_band, band_high, strength, eps, gmax)
    n_bins = content_band.shape[-1]
    sel_low = (torch.arange(n_bins, device=g_lo.device) < cut_band)[None]
    return torch.where(sel_low, g_lo, g_hi)


def morph_gain(content_band, band_a, band_b, alpha, strength, eps=EPS, gmax=None):
    """Spectral morph gain (E21): geometric interpolation between style A and
    style B envelopes at alpha in [0,1] (0 -> A, 1 -> B). Geometric (log-space)
    interpolation keeps the path on the positive-power manifold."""
    g_a = style_gain(content_band, band_a, strength, eps, gmax)
    g_b = style_gain(content_band, band_b, strength, eps, gmax)
    return g_a ** (1.0 - alpha) * g_b ** alpha


# ---------------------------------------------------------------------------
# Generation-time references (consumed by e17_sd35.ClampPSD3)
# ---------------------------------------------------------------------------

def apply_gain(content_ref, gain):
    """Return a copy of an SBN reference with its per-step band targets scaled by
    a per-(channel, band) `gain` (broadcast over steps). The single primitive the
    build_* helpers below funnel through."""
    out = {k: (v.clone() if torch.is_tensor(v) else v)
           for k, v in content_ref.items()}
    out["band"] = content_ref["band"].clone() * gain[None]   # (T,C,B)*(1,C,B)
    return out


def build_style_reference(content_ref, style_band, strength, gmax=None):
    """Generation-time spectral style transfer reference (E19). content_ref is a
    cfg=1 per-step SBN reference (record_reference_sd3); style_band is a style
    image's (C, n_bins) band power (band_power of sd3_vae_encode'd image). The
    gain is computed from content's FINAL-step shape (the finished-image endpoint
    that matches the style image) and applied at every step."""
    g = style_gain(content_ref["band"][-1], style_band, strength, gmax=gmax)
    return apply_gain(content_ref, g)


def build_hybrid_reference(content_ref, band_low, band_high, cut_band, strength,
                           gmax=None):
    """Generation-time hybrid-image reference (E20): low-band envelope from
    band_low, high-band from band_high. NOTE this controls the energy SPLIT only;
    phase (hence coarse structure) still comes from the prompt -- a softer effect
    than the offline phase+magnitude hybrid (band_phase_swap)."""
    g = split_gain(content_ref["band"][-1], band_low, band_high, cut_band,
                   strength, gmax=gmax)
    return apply_gain(content_ref, g)


def build_morph_reference(content_ref, band_a, band_b, alpha, strength, gmax=None):
    """Generation-time spectral morph reference (E21) at alpha in [0,1]."""
    g = morph_gain(content_ref["band"][-1], band_a, band_b, alpha, strength,
                   gmax=gmax)
    return apply_gain(content_ref, g)


def blend_references(ref_a, ref_b, w):
    """Two-prompt SBN mixing (E22): blend two cfg=1 per-step references in log
    space (per-band geometric mean), w in [0,1] (0 -> A, 1 -> B). Geometric so a
    band that is bright in one prompt and dark in the other interpolates
    multiplicatively rather than being dominated by the brighter one."""
    a, b = ref_a["band"].clamp(min=EPS), ref_b["band"].clamp(min=EPS)
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in ref_a.items()}
    out["band"] = a ** (1.0 - w) * b ** w
    if "total" in ref_a and "total" in ref_b:
        out["total"] = ref_a["total"] ** (1.0 - w) * ref_b["total"] ** w
    return out


# ---------------------------------------------------------------------------
# Offline restyle (E18): operate directly on a concrete latent
# ---------------------------------------------------------------------------

def restyle_latent(lat, style_band, idx_map, n_bins, strength=1.0, eps=EPS):
    """Offline spectral style transfer: keep lat's PHASE + within-band texture,
    drive its per-band power toward style_band. strength interpolates the target
    in log space between lat's own band power (0) and style_band (1). At
    strength=1 this is psd_match(lat, style_band) -- content layout, style
    envelope + palette. Returns a real latent (phase untouched)."""
    cur = band_power((torch.fft.fft2(lat.float()).abs() ** 2)[0], idx_map, n_bins)
    sb = style_band.to(cur.device).float()
    if strength >= 1.0:
        target = sb
    else:
        target = cur.clamp(min=eps) ** (1.0 - strength) * sb.clamp(min=eps) ** strength
    return psd_match(lat, target, idx_map, n_bins)


def latent_band_power(lat, idx_map, n_bins):
    """(C, n_bins) band power of a single (1,C,H,W) latent -- the style descriptor
    extracted from an encoded style image."""
    return band_power((torch.fft.fft2(lat.float()).abs() ** 2)[0], idx_map, n_bins)
