"""E39: spectral band-AdaIN -- a soft-band frequency knob on the sampler's output.

Unlike the network's internal AdaLN (the `vec` -> shift/scale/gate path, a *semantic*
/timestep knob), this is a *frequency* knob that lives OUTSIDE the network, in the
integration loop, on the velocity `v_theta` or on a latent. It partitions the 2D
spatial-frequency spectrum into SOFT radial bands `m_k(omega)` that form a partition of
unity (sum_k m_k = 1), matches MAGNITUDE statistics per band (mean AND std -- full
first+second moment), and REUSES the content phase. Because every mask is a function of
the radial frequency |omega| (via latent_spectral_ops.radial_norm) it is symmetric under
omega -> -omega, so with content-phase reuse the output stays real (ifft2(.).real loses
only ~1e-6 residue); the 4 self-conjugate bins are restored from the content to keep
realness + the DC mean exactly.

Contrast with the repo's existing SBN (spectral_ops.psd_match / latent_spectral_ops.
sbn_clamp): that uses HARD radial bands (band_index_map) and matches MEAN POWER only.
Here the bands are soft (overlapping Gaussian rings, so moments are mask-WEIGHTED) and we
match the full mean+std of the magnitude.

Everything works on (B, C, H, W) for BOTH diffusion latents (C=16) and RGB images (C=3),
so the pixel sanity demo reuses the exact same operator as the latent/velocity demos.
"""
import os
import sys

import torch
import torch.fft as fft
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latent_spectral_ops import radial_norm, hybrid_split_2d
from spectral_ops import _restore_self_conj


# ---------------------------------------------------------------------------
# soft radial bands + mask-weighted moments
# ---------------------------------------------------------------------------

def soft_band_masks(H, W, centers, widths, device="cuda"):
    """(K, H, W) soft radial-band masks forming a partition of unity (sum_k = 1).

    Gaussian rings exp(-0.5*((r-c)/w)**2) over the normalised radius r in [0,1]
    (0 = DC, 1 = corner), then normalised so sum_k m_k(omega) = 1 everywhere. Each
    mask is a function of |omega| only -> symmetric under omega -> -omega, hence
    Hermitian-preserving. Wider widths = softer overlap = less ringing.
    """
    r = radial_norm(H, W, device)
    M = torch.stack([torch.exp(-0.5 * ((r - c) / w) ** 2)
                     for c, w in zip(centers, widths)], 0)
    return M / M.sum(0, keepdim=True).clamp_min(1e-8)


def band_moments(mag, M):
    """Mask-weighted per-band magnitude (mean, std).

    mag: (B, C, H, W) FFT magnitude. M: (K, H, W). Returns mu, sig each (K, B, C).
    Weighted (not hard-masked) because soft bands overlap and share frequencies.
    """
    w = M[:, None, None]                       # (K,1,1,H,W)
    x = mag[None]                              # (1,B,C,H,W)
    wsum = w.sum((-1, -2)).clamp_min(1e-8)     # (K,1,1)
    mu = (w * x).sum((-1, -2)) / wsum          # (K,B,C)
    var = (w * (x - mu[..., None, None]) ** 2).sum((-1, -2)) / wsum
    return mu, var.clamp_min(1e-12).sqrt()


def _reassemble(per_band_mag, M, phase, V):
    """Convex blend sum_k M[k]*|.|_k (clamped >=0), recombine with the content phase,
    restore the self-conjugate bins from V, and return the real iFFT."""
    out = sum(M[k] * per_band_mag[k].clamp_min(0.0) for k in range(M.shape[0]))
    Fmix = torch.polar(out, phase)
    _restore_self_conj(Fmix, V, V.shape[-2], V.shape[-1])
    return fft.ifft2(Fmix, norm="ortho").real


# ---------------------------------------------------------------------------
# the operator
# ---------------------------------------------------------------------------

def spectral_adain(v_content, sources, M, eps=1e-6):
    """Per-band magnitude AdaIN: rewrite |V|'s mean+std in each band to sources[k]'s,
    keep v_content's phase. `sources` is a length-K list (one per band):
      - source == v_content  -> identity in that band,
      - source == reference  -> moment-matching (the "matching" formulation),
      - different per band    -> a mix (low anchored, high free, etc.).
    """
    dt = v_content.dtype
    V = fft.fft2(v_content.float(), norm="ortho")
    magV, phase = V.abs(), V.angle()
    muc, sigc = band_moments(magV, M)          # (K,B,C)
    per_band = []
    for k in range(M.shape[0]):
        Sk = fft.fft2(sources[k].float(), norm="ortho").abs()
        mus, sigs = band_moments(Sk, M[k:k + 1])
        mus, sigs = mus[0], sigs[0]            # (B,C)
        norm = (magV - muc[k][..., None, None]) / (sigc[k][..., None, None] + eps)
        per_band.append(sigs[..., None, None] * norm + mus[..., None, None])
    return _reassemble(per_band, M, phase, V).to(dt)


def band_mag_mix(content, A, B, M, alphas):
    """Two-latent mixing primitive: |V~_k| = a_k|A_k| + (1-a_k)|B_k|, phase from
    `content` (= A). Low-band a->0 imports B's color/global layout; high-band a->1
    keeps A's texture. Phase is NOT interpolated (it is circular)."""
    dt = content.dtype
    Vc = fft.fft2(content.float(), norm="ortho")
    Amag = fft.fft2(A.float(), norm="ortho").abs()
    Bmag = fft.fft2(B.float(), norm="ortho").abs()
    per_band = [alphas[k] * Amag + (1.0 - alphas[k]) * Bmag for k in range(M.shape[0])]
    return _reassemble(per_band, M, Vc.angle(), Vc).to(dt)


def freq_mixed_init(x0, eps, cut):
    """One-shot SDEdit init F^-1(low(F x0) + high(F eps)): coarse structure from the
    real latent x0, fine detail from noise. Thin wrapper over hybrid_split_2d."""
    return hybrid_split_2d(x0, eps, cut)


# ---------------------------------------------------------------------------
# learned per-band/per-time schedule (Demo 3)
# ---------------------------------------------------------------------------

class BandSchedule(nn.Module):
    """A tiny learnable table {g_k(t), b_k(t)} of size K x T_bins x 2 -- a learned
    frequency-shaping schedule over the trajectory:

        |V~_k| = g_k(t) * (|V| - mu_k)/sig_k + b_k(t)      (phase from content)

    Fit it (Demo 3) to reproduce a reference velocity (true-CFG or distilled), or
    condition it on a concept and read off which (k, t) cells it drives toward
    attenuation -- a candidate frequency-domain erasure operator.
    """

    def __init__(self, n_bands, t_bins):
        super().__init__()
        self.t_bins = t_bins
        self.g = nn.Parameter(torch.ones(n_bands, t_bins))
        self.b = nn.Parameter(torch.zeros(n_bands, t_bins))

    def t_bin(self, sigma):
        """sigma in [0,1] (1 = pure noise, decreasing) -> integer time bin."""
        i = int((1.0 - float(sigma)) * self.t_bins)
        return max(0, min(self.t_bins - 1, i))

    def forward(self, v, M, sigma, eps=1e-6):
        tb = self.t_bin(sigma)
        V = fft.fft2(v.float(), norm="ortho")
        magV, phase = V.abs(), V.angle()
        muc, sigc = band_moments(magV, M)
        per_band = []
        for k in range(M.shape[0]):
            norm = (magV - muc[k][..., None, None]) / (sigc[k][..., None, None] + eps)
            per_band.append(self.g[k, tb] * norm + self.b[k, tb])
        return _reassemble(per_band, M, phase, V).to(v.dtype)
