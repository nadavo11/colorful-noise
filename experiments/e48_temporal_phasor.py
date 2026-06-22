"""E48 Probe 0: temporal-axis Fourier phasor sanity on LTX latents (follow-on to E45).

Fourier shift theorem on the TEMPORAL (F) axis of a video latent z[1,C,F,H,W]:
    fft_F(z)[k] * exp(-2pi j k Delta / F)   <=>   circular frame shift by Delta.
Integer Delta -> circular shift; fractional Delta -> band-limited interpolation.

This is the diagnostic for E48 (reframe of the handoff): a linear phasor is a CIRCULAR
shift, it cannot extrapolate new frames -- so we only ever test re-timing/interpolation.
The real question this probe answers: is the temporal phase of an LTX latent a faithful,
smoothly-manipulable carrier of motion? If decode(temporal_shift(z)) is coherent, the
deliverable (temporal-only phase preservation for edit consistency) is worth building.

Three checks, escalating from pure-math to model judgement:
  1. OPERATOR CORRECTNESS (latent space, no VAE): integer phasor == torch.roll exactly;
     +Delta then -Delta recovers z; two half-shifts == one integer shift. Code sanity.
  2. VAE SHIFT-EQUIVARIANCE (integer Delta): PSNR(decode(shift(z,1)), pixel_roll(decode(z),tcr)).
     Make-or-break: if low, the latent F-axis is not a clean time axis and the story is shaky.
     NB: LTX VAE encodes frame0 as 1 pixel-frame, the rest as `tcr` each -> exact equivariance
     is not expected; we read the INTERIOR PSNR (away from both wrap boundaries).
  3. FRACTIONAL COHERENCE (Delta=0.5): decode the phasor-interpolated latent; save mp4 next to
     a latent-lerp baseline. Eyeball + report PSNR(phasor, lerp) to quantify the difference.

Run (cluster): python e48_temporal_phasor.py --width 704 --height 480 --frames 49
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.5.1", "diffusers==0.36.0", "transformers==4.57.6", "accelerate",
#     "sentencepiece", "protobuf", "imageio", "imageio-ffmpeg", "numpy", "pillow",
# ]
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# ///
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e45_ltx_flowalign import load_ltx, ltx_encode, ltx_decode, ltx_conform

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e48")


def temporal_shift(z, delta):
    """Circular shift z by `delta` frames along the F axis (dim=2) via a temporal-FFT phasor.
    Integer delta == torch.roll(z, delta, dim=2); fractional delta == band-limited interp."""
    Fl = z.shape[2]
    freqs = torch.fft.fftfreq(Fl, device=z.device)               # k/F per bin
    phasor = torch.exp(-2j * np.pi * freqs * delta).view(1, 1, Fl, 1, 1)
    Z = torch.fft.fft(z.to(torch.complex64), dim=2)
    return torch.fft.ifft(Z * phasor, dim=2).real.float()


def psnr(a, b):
    a, b = a.astype(np.float64) / 255.0, b.astype(np.float64) / 255.0
    n = min(len(a), len(b))
    mse = float(np.mean((a[:n] - b[:n]) ** 2))
    return 99.0 if mse < 1e-12 else float(-10 * np.log10(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=704)            # LTX-native landscape
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=49)            # -> F_lat=7 (the native target)
    ap.add_argument("--video", default="imageio:cockatoo.mp4")   # real footage
    ap.add_argument("--src_caption", default="a white cockatoo bird perched on a branch, moving its head")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import imageio.v3 as iio

    pipe = load_ltx()
    tcr = pipe.vae_temporal_compression_ratio
    frames = ltx_conform(args.video, args.width, args.frames, args.height)
    iio.imwrite(os.path.join(OUT, "source.mp4"), frames, fps=8)
    z = ltx_encode(pipe, frames)                                 # (1,128,Fl,Hl,Wl)
    Fl = z.shape[2]
    print(f"[e48] clip {frames.shape} -> latent {tuple(z.shape)}  F_lat={Fl}  tcr={tcr}", flush=True)

    # --- check 1: operator correctness (pure latent-space FFT identities) ---
    int_err = (temporal_shift(z, 1.0) - torch.roll(z, 1, dims=2)).abs().max().item()
    rt_err = (temporal_shift(temporal_shift(z, 0.5), -0.5) - z).abs().max().item()
    half2_err = (temporal_shift(temporal_shift(z, 0.5), 0.5) - torch.roll(z, 1, dims=2)).abs().max().item()
    print(f"[e48] OP integer-shift==roll max|err|={int_err:.2e}  "
          f"roundtrip(+.5,-.5) max|err|={rt_err:.2e}  half+half==roll max|err|={half2_err:.2e}", flush=True)

    # --- check 2: VAE temporal shift-equivariance (integer Delta=1) ---
    recon = ltx_decode(pipe, z)
    iio.imwrite(os.path.join(OUT, "recon.mp4"), recon, fps=8)
    dec_int = ltx_decode(pipe, temporal_shift(z, 1.0))
    iio.imwrite(os.path.join(OUT, "shift_int1.mp4"), dec_int, fps=8)
    pix_roll = np.roll(recon, tcr, axis=0)                       # 1 latent frame ~ tcr pixel frames
    p_all = psnr(dec_int, pix_roll)
    lo, hi = 2 * tcr, len(recon) - 2 * tcr                       # interior: away from both wrap edges
    p_int = psnr(dec_int[lo:hi], pix_roll[lo:hi]) if hi > lo else float("nan")
    print(f"[e48] EQUIVARIANCE decode(shift1) vs roll(decode,{tcr}): "
          f"PSNR all={p_all:.2f} dB  interior[{lo}:{hi}]={p_int:.2f} dB", flush=True)

    # --- check 3: fractional coherence (Delta=0.5) vs latent-lerp baseline ---
    dec_frac = ltx_decode(pipe, temporal_shift(z, 0.5))
    iio.imwrite(os.path.join(OUT, "shift_frac0.5.mp4"), dec_frac, fps=8)
    z_lerp = 0.5 * z + 0.5 * torch.roll(z, -1, dims=2)           # naive each-frame-with-next average
    dec_lerp = ltx_decode(pipe, z_lerp)
    iio.imwrite(os.path.join(OUT, "lerp0.5.mp4"), dec_lerp, fps=8)
    print(f"[e48] FRACTIONAL Delta=0.5: PSNR(phasor, latent-lerp)={psnr(dec_frac, dec_lerp):.2f} dB "
          f"(diagnostic; lower => phasor differs more from naive lerp)", flush=True)

    recon_l1 = float(np.abs(frames[:len(recon)].astype(float) - recon[:len(recon)].astype(float)).mean() / 255.0)
    print(f"[e48] VAE round-trip L1={recon_l1:.4f}  (context for the PSNRs above)", flush=True)
    print(f"[e48] mp4s in {OUT}: source recon shift_int1 shift_frac0.5 lerp0.5", flush=True)
    print("[e48] DONE", flush=True)


if __name__ == "__main__":
    main()
