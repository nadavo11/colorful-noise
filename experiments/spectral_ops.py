"""Spectral operations for exploring noise-PSD manipulation in diffusion latents.

Extends the Colorful-Noise (Cohen et al., SIGGRAPH 2026) frequency swap with:
  - colored noise generation (PSD ~ f^beta)
  - a generalized low-band conditioning that decouples phase / magnitude / DC
  - radial PSD measurement + the paper's band-wise whiteness metric
  - FFT phase surgery: Hermitian uniform-phase sampling, phase quantization
    with optional level omission (E6)
"""
import math

import torch


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def radial_bins(H, W, device):
    """Normalized radial frequency grid (DC at [0,0], unshifted)."""
    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.fftfreq(W, device=device)
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)  # max ~ sqrt(0.5)


def radial_psd(x, n_bins=24):
    """Radially averaged power spectral density.

    x: (B, C, H, W) real tensor.
    Returns (centers, psd) where psd is (C, n_bins), averaged over batch.
    """
    B, C, H, W = x.shape
    power = torch.fft.fft2(x.float()).abs() ** 2 / (H * W)
    rr = radial_bins(H, W, x.device)
    edges = torch.linspace(0, rr.max() + 1e-6, n_bins + 1, device=x.device)
    idx = torch.bucketize(rr.flatten(), edges) - 1
    idx = idx.clamp(0, n_bins - 1)
    psd = torch.zeros(C, n_bins, device=x.device)
    counts = torch.zeros(n_bins, device=x.device).scatter_add_(
        0, idx, torch.ones_like(idx, dtype=torch.float))
    flat = power.mean(0).reshape(C, -1)  # batch mean
    for c in range(C):
        psd[c] = torch.zeros(n_bins, device=x.device).scatter_add_(0, idx, flat[c])
    psd = psd / counts.clamp(min=1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.cpu(), psd.cpu()


def whiteness(x, n_bins=24):
    """Paper-style whiteness: std of band-wise power across bands, per channel,
    normalized by mean band power (0 = perfectly flat/white)."""
    _, psd = radial_psd(x, n_bins)
    return (psd.std(dim=1) / psd.mean(dim=1)).tolist()


# ---------------------------------------------------------------------------
# Colored noise (E1)
# ---------------------------------------------------------------------------

def colored_noise(shape, beta, device="cuda", normalize="global", generator=None):
    """Gaussian noise with PSD ~ f^beta.

    beta: -2 red/brown, -1 pink, 0 white, +1 blue, +2 violet.
    normalize: 'global' rescales each sample to zero mean / unit variance,
               'none' leaves raw filtered noise.
    """
    B, C, H, W = shape
    white = torch.randn(shape, device=device, generator=generator)
    if beta == 0 and normalize == "global":
        return white
    fft = torch.fft.fft2(white)
    rr = radial_bins(H, W, device)
    r_min = rr[rr > 0].min()
    filt = rr.clamp(min=r_min) ** (beta / 2)  # amplitude ~ f^(beta/2) => PSD ~ f^beta
    x = torch.fft.ifft2(fft * filt[None, None]).real
    if normalize == "global":
        x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / x.std(dim=(1, 2, 3), keepdim=True)
    return x


# ---------------------------------------------------------------------------
# Generalized low-band conditioning (E2)
# ---------------------------------------------------------------------------

def paper_low_mask(H, W, p, device):
    """Exact reproduction of the paper's radial-percentile mask (fftshifted
    coordinates, lowest p fraction of bins by area)."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    cutoff = torch.quantile(rr.flatten(), p)
    return (rr <= cutoff).float()


def condition_latent(noise, img_latent, p=0.015, gamma=0.05,
                     phase="image", mag="image", dc="image", mag_scale=1.0):
    """Generalized version of fft_radial_frequency_swap.

    Inside the lowest-p radial band, builds coefficients M * exp(i*phi) from:
      phase: 'image' | 'noise'   -- where the spatial arrangement comes from
      mag:   'image' (gamma-scaled, the paper's peak) | 'noise' (white PSD)
      dc:    'image' (gamma-scaled) | 'noise' | 'zero' -- per-channel mean

    (phase='image', mag='image', dc='image')  == the paper's method.
    (phase='image', mag='noise', dc='noise')  == PSD-whitened conditioning.
    (phase='noise', mag='noise', dc='image')  == DC-only (channel means).
    """
    assert noise.shape == img_latent.shape
    B, C, H, W = noise.shape
    fft_n = torch.fft.fftshift(torch.fft.fft2(noise.float()), dim=(-2, -1))
    fft_i = torch.fft.fftshift(torch.fft.fft2(img_latent.float()), dim=(-2, -1))

    low = paper_low_mask(H, W, p, noise.device)[None, None]

    M = mag_scale * (gamma * fft_i.abs() if mag == "image" else fft_n.abs())
    phi = fft_i.angle() if phase == "image" else fft_n.angle()
    low_coeffs = M * torch.exp(1j * phi)

    mix = low_coeffs * low + fft_n * (1.0 - low)

    # DC bin sits at the fftshift center
    cy, cx = H // 2, W // 2
    if dc == "image":
        mix[..., cy, cx] = gamma * fft_i[..., cy, cx]
    elif dc == "noise":
        mix[..., cy, cx] = fft_n[..., cy, cx]
    elif dc == "zero":
        mix[..., cy, cx] = 0.0

    mix = torch.fft.ifftshift(mix, dim=(-2, -1))
    return torch.fft.ifft2(mix).real


# Named E2 conditions: (phase, mag, dc)
E2_CONDITIONS = {
    "white":      ("noise", "noise", "noise"),  # control = plain white noise
    "paper":      ("image", "image", "image"),  # Cohen et al.
    "paper_nodc": ("image", "image", "noise"),
    "phase_only": ("image", "noise", "noise"),  # USER: whitened PSD, image phase
    "phase_dc":   ("image", "noise", "image"),
    "mag_only":   ("noise", "image", "noise"),  # spectral shape, no layout
    "mag_dc":     ("noise", "image", "image"),
    "dc_only":    ("noise", "noise", "image"),  # channel means alone
}


# ---------------------------------------------------------------------------
# FFT phase surgery (E6)
# ---------------------------------------------------------------------------

def random_hermitian_phase(C, H, W, device="cuda", generator=None):
    """Uniform iid phase field with exact Hermitian (conjugate) symmetry.

    The FFT phase of a real white-Gaussian field is exactly uniform on
    (-pi, pi] with the required phi(-f) = -phi(f) antisymmetry, so the
    simplest correct construction is to take the angle of one.
    Returns a (1, C, H, W) tensor (unshifted, DC at [0, 0]).
    """
    g = torch.randn(1, C, H, W, device=device, generator=generator)
    return torch.fft.fft2(g).angle()


def quantize_phase(x, k, omit=None, mode="zero", preserve_dc=True,
                   return_stats=False):
    """Quantize the FFT phase of x to k uniform levels, keeping magnitudes.

    Works unshifted (fft2 convention). torch.round of the antisymmetric
    phase stays antisymmetric (half-to-even is an odd function), so the
    quantized spectrum remains Hermitian and ifft2(.).real discards only
    ~1e-7 residue.

    omit: int level index -- zero every bin whose quantized level falls in
          {omit % k, (-omit) % k}; conjugate +/- pairs are zeroed together so
          the drop mask is Hermitian-symmetric and the ifft stays real. For
          k=8 the distinct pairs are levels {0, 1, 2, 3, 4(=pi)}; 0 and 4 are
          self-conjugate singletons.
    mode: 'zero' leaves the power hole (std drops ~ sqrt(kept_fraction));
          'renorm' rescales the kept non-DC magnitudes per channel by
          sqrt(P_total / P_remaining) (Parseval) so unit variance is
          restored and only the phase hole differs from the control.
    preserve_dc: restore the original DC bin and the 3 real Nyquist bins
          after quantization/omission -- E0 showed the model is acutely
          sensitive to the DC level, so it is held fixed by default.
    k=None: early-return x unchanged (the k=inf control).
    """
    if k is None:
        if return_stats:
            return x, {"kept_fraction": 1.0, "power_ratio": 1.0,
                       "imag_residue": 0.0}
        return x
    B, C, H, W = x.shape
    F = torch.fft.fft2(x.float())
    mag, phi = F.abs(), F.angle()
    delta = 2 * math.pi / k
    lvl = torch.round(phi / delta)
    Fq = mag * torch.exp(1j * lvl * delta)

    kept_fraction = 1.0
    if omit is not None:
        lv = lvl.long() % k
        drop = (lv == omit % k) | (lv == (-omit) % k)
        Fq = torch.where(drop, torch.zeros_like(Fq), Fq)
        kept_fraction = float(1.0 - drop.float().mean())

    if mode == "renorm":
        pw = Fq.abs() ** 2
        p_total = (mag ** 2).sum(dim=(-2, -1)) - mag[..., 0, 0] ** 2
        p_rem = pw.sum(dim=(-2, -1)) - pw[..., 0, 0]
        Fq = Fq * torch.sqrt(p_total / p_rem.clamp(min=1e-12))[..., None, None]

    if preserve_dc:
        for (i, j) in ((0, 0), (H // 2, 0), (0, W // 2), (H // 2, W // 2)):
            Fq[..., i, j] = F[..., i, j]

    out = torch.fft.ifft2(Fq)
    if return_stats:
        stats = {
            "kept_fraction": kept_fraction,
            "power_ratio": float((Fq.abs() ** 2).sum() / (F.abs() ** 2).sum()),
            "imag_residue": float(out.imag.abs().max()),
        }
        return out.real, stats
    return out.real
