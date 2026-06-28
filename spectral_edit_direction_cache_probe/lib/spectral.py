"""Pure-numpy spectral helpers for E51. Operate on spatial latent tensors [C, H, W]
(channel-first), as produced by unpacking FLUX velocity predictions.

Used for (a) frequency-band energy of v_edit / delta_edit over timesteps and
(b) the SEA-style low-pass projection that drives the spectral cache decision.
"""
from __future__ import annotations
import numpy as np


def _radial(h, w):
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    return r / (r.max() + 1e-12)


def _fft2c(x):
    return np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1))


def _ifft2c(f):
    return np.fft.ifft2(np.fft.ifftshift(f, axes=(-2, -1)), axes=(-2, -1)).real


def band_energies(x, low, high):
    """Fractional power in radial low/mid/high bands, averaged over channels."""
    f = _fft2c(x)
    p = np.abs(f) ** 2                       # [C,H,W]
    r = _radial(*x.shape[-2:])
    lo, hi = r < low, r > high
    md = (~lo) & (~hi)
    tot = p.sum(axis=(-2, -1)) + 1e-12       # [C]
    e_lo = p[:, lo].sum(-1) / tot
    e_md = p[:, md].sum(-1) / tot
    e_hi = p[:, hi].sum(-1) / tot
    return float(e_lo.mean()), float(e_md.mean()), float(e_hi.mean())


def lowpass(x, frac):
    """Keep only radial frequencies below frac*Nyquist (SEA-style projection)."""
    f = _fft2c(x)
    r = _radial(*x.shape[-2:])
    return _ifft2c(f * (r < frac)[None])


def radial_power(x, nb=24):
    """Mean radial power profile (over channels), nb bins from DC to Nyquist."""
    f = _fft2c(x)
    p = (np.abs(f) ** 2).mean(0)
    r = _radial(*x.shape[-2:])
    edges = np.linspace(0, 1, nb + 1)
    out = np.zeros(nb)
    for i in range(nb):
        m = (r >= edges[i]) & (r < edges[i + 1])
        out[i] = p[m].mean() if m.any() else 0.0
    return out


def amp_image(x):
    """Log-amplitude FFT image (mean over channels), normalized to [0,1] for display."""
    f = _fft2c(x)
    a = np.log1p(np.abs(f).mean(0))
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def rel_l2(a, b):
    """Relative L2 distance ||a-b|| / ||b|| on flattened arrays."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
