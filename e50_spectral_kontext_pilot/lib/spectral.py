"""Frequency-domain operators for E50.

Per-channel 2-D FFT operators on RGB images, plus Fourier visualisation helpers. Pure numpy/PIL
(no torch) so this runs identically in both the uv and anaconda envs. All operators map
PIL.Image -> PIL.Image at a fixed size, so any output can be fed straight to FluxKontext.

Conventions:
  amplitude A = |F|,  phase P = angle(F),  reconstruction = Re( ifft2( A * exp(iP) ) ).
A radial mask on the fftshifted spectrum defines low/mid/high bands (fraction of Nyquist radius).
"""
from __future__ import annotations
import numpy as np
from PIL import Image

RNG = np.random.default_rng(0)   # fixed → reproducible random-phase / random-amplitude ops


def to_arr(img, size=512):
    im = img if isinstance(img, Image.Image) else Image.open(img)
    return np.asarray(im.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32)


def to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _fft(ch):
    return np.fft.fftshift(np.fft.fft2(ch))


def _ifft(F):
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))


def amp_phase(arr):
    """Per-channel amplitude & phase of an HxWx3 array."""
    A, P = [], []
    for c in range(3):
        F = _fft(arr[..., c])
        A.append(np.abs(F)); P.append(np.angle(F))
    return np.stack(A, -1), np.stack(P, -1)


def recon(amp, phase):
    out = np.zeros(amp.shape, np.float32)
    for c in range(3):
        F = amp[..., c] * np.exp(1j * phase[..., c])
        out[..., c] = _ifft(F)
    return out


def _renorm(out, ref):
    """Match per-channel mean/std of a reconstruction to a reference array (keeps it viewable)."""
    o = out.copy()
    for c in range(3):
        oc, rc = o[..., c], ref[..., c]
        s = oc.std() + 1e-6
        o[..., c] = (oc - oc.mean()) / s * (rc.std() + 1e-6) + rc.mean()
    return o


def _radial_r(size):
    cy = cx = size // 2
    y, x = np.indices((size, size))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return r / r.max()        # normalised radius in [0, ~1]


# ---------------------------------------------------------------- single-image ops (Exp C)
def op_raw(img, size=512):
    return to_img(to_arr(img, size))


def op_phase_only(img, size=512):
    """Keep phase, flatten amplitude to a constant → structure/edges, no spectral texture."""
    arr = to_arr(img, size)
    amp, phase = amp_phase(arr)
    flat = np.full_like(amp, amp.mean())
    return to_img(_renorm(recon(flat, phase), arr))


def op_amplitude_only(img, size=512):
    """Keep amplitude, randomise phase → texture statistics, structure destroyed."""
    arr = to_arr(img, size)
    amp, _ = amp_phase(arr)
    rp = RNG.uniform(-np.pi, np.pi, amp.shape).astype(np.float32)
    return to_img(_renorm(recon(amp, rp), arr))


def _band(img, lo, hi, size=512):
    arr = to_arr(img, size)
    r = _radial_r(size)
    mask = ((r >= lo) & (r < hi)).astype(np.float32)
    out = np.zeros_like(arr)
    for c in range(3):
        out[..., c] = _ifft(_fft(arr[..., c]) * mask)
    return arr, out


def op_low_band(img, size=512, cut=0.15):
    arr, out = _band(img, 0.0, cut, size)
    return to_img(out + arr.mean(axis=(0, 1)) * (0 if cut else 0))  # low band already carries DC


def op_mid_band(img, size=512):
    arr, out = _band(img, 0.15, 0.45, size)
    return to_img(_renorm(out, arr))


def op_high_band(img, size=512, cut=0.45):
    arr, out = _band(img, cut, 2.0, size)
    return to_img(_renorm(out, arr))


# ---------------------------------------------------------------- cross ops (Exp A: content x style)
def op_content_phase_style_amp(content, style, size=512):
    """Content structure (phase) + style texture statistics (amplitude)."""
    c, s = to_arr(content, size), to_arr(style, size)
    ac, pc = amp_phase(c); as_, ps = amp_phase(s)
    return to_img(_renorm(recon(as_, pc), c))


def op_style_phase_content_amp(content, style, size=512):
    """Style structure (phase) + content texture statistics (amplitude) — leakage probe."""
    c, s = to_arr(content, size), to_arr(style, size)
    ac, pc = amp_phase(c); as_, ps = amp_phase(s)
    return to_img(_renorm(recon(ac, ps), s))


def op_style_high_on_content(content, style, size=512, cut=0.45):
    """Graft only the high-frequency band of the style onto the raw content (texture-only graft)."""
    c = to_arr(content, size)
    r = _radial_r(size)
    hi = (r >= cut).astype(np.float32)
    lo = 1.0 - hi
    out = np.zeros_like(c)
    s = to_arr(style, size)
    for ch in range(3):
        Fc, Fs = _fft(c[..., ch]), _fft(s[..., ch])
        out[..., ch] = _ifft(Fc * lo + Fs * hi)
    return to_img(_renorm(out, c))


SOURCE_FN = {
    "raw": op_raw,
    "phase_only": op_phase_only,
    "amplitude_only": op_amplitude_only,
    "low_band": op_low_band,
    "high_band": op_high_band,
}
REF_FN = {
    "content_raw": lambda content, style, size=512: op_raw(content, size),
    "content_phase_style_amp": op_content_phase_style_amp,
    "style_phase_content_amp": op_style_phase_content_amp,
    "style_high_on_content": op_style_high_on_content,
}


# ---------------------------------------------------------------- Fourier visualisation
def amp_spectrum_img(img, size=256):
    """Log radial amplitude spectrum as a grayscale image (for figures)."""
    g = np.asarray(Image.open(img).convert("L").resize((size, size)) if not isinstance(img, Image.Image)
                   else img.convert("L").resize((size, size)), np.float32)
    A = np.log1p(np.abs(_fft(g)))
    A = (A - A.min()) / (A.ptp() + 1e-8) * 255
    return Image.fromarray(A.astype(np.uint8))


def phase_img(img, size=256):
    g = np.asarray(Image.open(img).convert("L").resize((size, size)) if not isinstance(img, Image.Image)
                   else img.convert("L").resize((size, size)), np.float32)
    P = np.angle(_fft(g))
    P = (P + np.pi) / (2 * np.pi) * 255
    return Image.fromarray(P.astype(np.uint8))


def radial_power(img, size=256):
    """Returns (radii, mean log-amplitude per radius) for radial power-spectrum plots."""
    g = np.asarray(Image.open(img).convert("L").resize((size, size)) if not isinstance(img, Image.Image)
                   else img.convert("L").resize((size, size)), np.float32)
    A = np.log1p(np.abs(_fft(g)))
    cy = cx = size // 2
    y, x = np.indices((size, size))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), A.ravel())
    nr = np.bincount(r.ravel())
    prof = tbin / np.maximum(nr, 1)
    n = size // 2
    return np.arange(n), prof[:n]
