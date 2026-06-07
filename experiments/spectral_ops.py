"""Spectral operations for exploring noise-PSD manipulation in diffusion latents.

Extends the Colorful-Noise (Cohen et al., SIGGRAPH 2026) frequency swap with:
  - colored noise generation (PSD ~ f^beta)
  - a generalized low-band conditioning that decouples phase / magnitude / DC
  - radial PSD measurement + the paper's band-wise whiteness metric
"""
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
                     phase="image", mag="image", dc="image"):
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

    M = gamma * fft_i.abs() if mag == "image" else fft_n.abs()
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
