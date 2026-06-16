"""E29 analysis helpers: phase-inheritance metrics (seed z_T vs output z_0).

The question: with the deterministic DDIM map z_T -> z_0, how much of the OUTPUT
latent's FFT phase is determined by the SEED's FFT phase? Prior repo work showed
the FFT *phase* of a latent carries image structure; here we quantify, per radial
frequency band, the statistical link between seed phase and output phase.

All FFTs are unshifted (DC at [0,0], torch.fft convention), matching
spectral_ops.band_index_map / _SELF_CONJ. We pool over the 4 channels and over
all (H, W) bins that fall in a radial band, and we DROP the 4 self-conjugate bins
(DC + Nyquist axes) from circular statistics -- their phase is 0/pi (real), so a
circular correlation there is degenerate.

Primary metric (`inheritance_spectrum`): per FFT bin, the Jammalamadaka-SenGupta
CIRCULAR CORRELATION between seed phase and output phase across N seeds, then
radially averaged into bands. ~0 = no link (independent uniform phases),
~1 = output phase fully predicted by seed phase. A permutation null gives the
chance level.

This is NOT what spectral_ops.phase_coherence computes -- that is the
cross-sample resultant within one phase stack, not the paired seed<->output link.
"""
import math

import torch

from spectral_ops import _SELF_CONJ, band_index_map  # noqa: F401

EPS = 1e-8


# ---------------------------------------------------------------------------
# small primitives
# ---------------------------------------------------------------------------

def fft_phase(lat):
    """(B,C,H,W) real latent -> (B,C,H,W) unshifted FFT phase in (-pi, pi]."""
    return torch.fft.fft2(lat.float()).angle()


def fft_logmag(lat):
    """(B,C,H,W) real latent -> (B,C,H,W) log FFT magnitude."""
    return torch.log(torch.fft.fft2(lat.float()).abs() + EPS)


def non_selfconj_mask(H, W, device):
    """(H, W) {0,1} weight that zeros the 4 self-conjugate bins (DC + Nyquist
    axes). Used to exclude degenerate (real-phase) bins from circular stats."""
    m = torch.ones(H, W, device=device)
    for (i, j) in _SELF_CONJ(H, W):
        m[i, j] = 0.0
    return m


def band_average(field, idx_map, n_bins, mask=None):
    """Channel-pooled, weighted radial-band average.

    field: (K, H, W) -- K channel-like maps (e.g. the 4 latent channels, or
           N*C stacked). idx_map: (H, W) band ids from band_index_map.
    mask:  optional (H, W) per-bin weight (default all-ones).
    Returns (n_bins,): the weighted mean of `field` over all (k, h, w) whose
    bin falls in each band. Empty bands -> 0.
    """
    K, H, W = field.shape
    dev = field.device
    flat_idx = idx_map.reshape(-1).to(dev)
    w = (torch.ones(H * W, device=dev) if mask is None
         else mask.reshape(-1).to(dev))
    fvals = field.reshape(K, -1)
    num = torch.zeros(n_bins, device=dev)
    for k in range(K):
        num.scatter_add_(0, flat_idx, fvals[k] * w)
    den_bin = torch.zeros(n_bins, device=dev).scatter_add_(0, flat_idx, w)
    den = den_bin * K
    return (num / den.clamp(min=EPS)).cpu()


# ---------------------------------------------------------------------------
# per-bin statistics across N samples
# ---------------------------------------------------------------------------

def circ_corr_bins(phiA, phiB):
    """Jammalamadaka-SenGupta circular correlation, per bin, over N samples.

    phiA, phiB: (N, C, H, W) FFT phases. Returns (C, H, W) in [-1, 1]:
      abar = atan2(sum sin a, sum cos a)          # circular mean over N
      r = sum_n sin(a-abar) sin(b-bbar)
          / sqrt( sum sin(a-abar)^2 * sum sin(b-bbar)^2 )
    Bins where either marginal is degenerate (den ~ 0) -> 0.
    """
    abar = torch.atan2(torch.sin(phiA).sum(0), torch.cos(phiA).sum(0))
    bbar = torch.atan2(torch.sin(phiB).sum(0), torch.cos(phiB).sum(0))
    sa = torch.sin(phiA - abar)
    sb = torch.sin(phiB - bbar)
    num = (sa * sb).sum(0)
    den = torch.sqrt((sa ** 2).sum(0) * (sb ** 2).sum(0))
    r = num / den.clamp(min=EPS)
    return torch.where(den > EPS, r, torch.zeros_like(r))


def pearson_bins(xA, xB):
    """Per-bin Pearson correlation over N samples. xA, xB: (N, C, H, W) ->
    (C, H, W) in [-1, 1]."""
    xa = xA - xA.mean(0)
    xb = xB - xB.mean(0)
    num = (xa * xb).sum(0)
    den = torch.sqrt((xa ** 2).sum(0) * (xb ** 2).sum(0))
    r = num / den.clamp(min=EPS)
    return torch.where(den > EPS, r, torch.zeros_like(r))


# ---------------------------------------------------------------------------
# spectra (the headline outputs)
# ---------------------------------------------------------------------------

def inheritance_spectrum(seed_lats, out_lats, n_bins=24, n_perm=20,
                         generator=None):
    """Phase-inheritance spectrum + permutation null.

    seed_lats, out_lats: (N, C, H, W) real latents (z_T and z_0), paired by seed.
    Returns dict (cpu tensors):
      r_band   : (n_bins,) circular corr of seed-phase vs output-phase per band
      r2d      : (H, W) per-bin circular corr (channel-mean), fftshifted, for a heatmap
      null_mean, null_std : (n_bins,) permutation-null band curves
      centers  : (n_bins,) band radial-frequency centers
    """
    N, C, H, W = seed_lats.shape
    dev = seed_lats.device
    idx = band_index_map(H, W, n_bins, device=dev)
    mask = non_selfconj_mask(H, W, dev)

    phiS = fft_phase(seed_lats)          # (N,C,H,W)
    phiO = fft_phase(out_lats)

    r = circ_corr_bins(phiS, phiO)       # (C,H,W)
    r_band = band_average(r, idx, n_bins, mask=mask)

    # 2D heatmap: channel-mean per-bin corr, fftshifted, self-conj zeroed
    r2d = (r.mean(0) * mask)
    r2d = torch.fft.fftshift(r2d).cpu()

    # permutation null: break the seed<->output pairing
    null = torch.zeros(n_perm, n_bins)
    for p in range(n_perm):
        perm = torch.randperm(N, device=dev, generator=generator)
        rp = circ_corr_bins(phiS, phiO[perm])
        null[p] = band_average(rp, idx, n_bins, mask=mask)

    _, edges = _band_centers(H, W, n_bins, dev)
    return {
        "r_band": r_band,
        "r2d": r2d,
        "null_mean": null.mean(0),
        "null_std": null.std(0),
        "centers": edges,
    }


def magnitude_spectrum_corr(seed_lats, out_lats, n_bins=24):
    """Control: per-band Pearson of log|FFT| seed vs output (expected weak --
    the model drives the PSD toward natural-image statistics). Returns (n_bins,)."""
    N, C, H, W = seed_lats.shape
    dev = seed_lats.device
    idx = band_index_map(H, W, n_bins, device=dev)
    mask = non_selfconj_mask(H, W, dev)
    r = pearson_bins(fft_logmag(seed_lats), fft_logmag(out_lats))
    return band_average(r, idx, n_bins, mask=mask)


def dphi_resultant(seed_lats, out_lats, n_bins=24):
    """Secondary diagnostic: per (seed,output) pair, band-averaged resultant of
    the phase difference dphi = phi_out - phi_seed. High R within a band means
    the output phase is a *consistent transform* of the seed phase across that
    band (e.g. a constant offset = spatial shift); it does NOT by itself prove
    inheritance, hence secondary to the circular correlation. Returns (n_bins,)
    averaged over pairs and channels."""
    N, C, H, W = seed_lats.shape
    dev = seed_lats.device
    idx = band_index_map(H, W, n_bins, device=dev)
    flat_idx = idx.reshape(-1)
    counts = torch.zeros(n_bins, device=dev).scatter_add_(
        0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))
    dphi = fft_phase(out_lats) - fft_phase(seed_lats)        # (N,C,H,W)
    z = torch.exp(1j * dphi)
    acc = torch.zeros(n_bins, device=dev)
    for n in range(N):
        for c in range(C):
            re = torch.zeros(n_bins, device=dev).scatter_add_(
                0, flat_idx, z[n, c].real.reshape(-1))
            im = torch.zeros(n_bins, device=dev).scatter_add_(
                0, flat_idx, z[n, c].imag.reshape(-1))
            acc += torch.sqrt(re ** 2 + im ** 2) / counts.clamp(min=1)
    return (acc / (N * C)).cpu()


def spatial_pearson(seed_lats, out_lats):
    """Scalar sanity check: per-channel Pearson of z_T vs z_0 in PIXEL space,
    averaged over channels and samples. Expected low -- the phase relationship
    lives in frequency, not in raw spatial values."""
    N, C, H, W = seed_lats.shape
    a = seed_lats.reshape(N, C, -1).float()
    b = out_lats.reshape(N, C, -1).float()
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    num = (a * b).sum(-1)
    den = torch.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1)).clamp(min=EPS)
    return float((num / den).mean())


# ---------------------------------------------------------------------------
# causal transplant: follow score
# ---------------------------------------------------------------------------

def follow_score(out_A, out_Aprime, out_B, n_bins=24):
    """How far the transplanted output's phase moved from base A toward donor B,
    per band. out_*: (C, H, W) real OUTPUT latents (z_0) from the base seed A,
    the transplanted seed A', and the donor seed B.

      d_circ(x, y) = mean_band (1 - cos(phi_x - phi_y))     # circular distance
      follow = d_circ(A', A) / ( d_circ(A', A) + d_circ(A', B) )

    follow -> 1 : A' output phase followed the donor B (causal inheritance in
                  that band); ~0.5 : no net movement. Returns (n_bins,).
    """
    H, W = out_A.shape[-2:]
    dev = out_A.device
    idx = band_index_map(H, W, n_bins, device=dev)
    mask = non_selfconj_mask(H, W, dev)
    phiA = fft_phase(out_A[None])[0]
    phiP = fft_phase(out_Aprime[None])[0]
    phiB = fft_phase(out_B[None])[0]
    dA = band_average(1 - torch.cos(phiP - phiA), idx, n_bins, mask=mask)
    dB = band_average(1 - torch.cos(phiP - phiB), idx, n_bins, mask=mask)
    return dA / (dA + dB).clamp(min=EPS)


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------

def _band_centers(H, W, n_bins, device):
    """Radial-frequency band-center grid matching band_index_map's binning."""
    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.fftfreq(W, device=device)
    rr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    edges = torch.linspace(0, rr.max().item() + 1e-6, n_bins + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return rr, centers.cpu()
