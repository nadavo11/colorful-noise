"""E44: FlowAlign on LTX-Video + our spectral phase op, for temporally-coherent video editing.

Motivation. FlowAlign (arXiv:2505.23145) edits VIDEO frame-by-frame on an IMAGE model (SD3):
the paper itself admits "temporal consistency for the edited object is limited, as no explicit
constraint is imposed." Our bet: run FlowAlign on a real video model (LTX-Video) and add the
E41/E43 low-band PHASE-keep op in the SPATIOTEMPORAL frequency domain (3D FFT over frame x H x W)
to constrain exactly that flicker. The 2D per-frame variant ~= the paper's approach (a control);
the 3D variant is the bet.

This is a one-clip FEASIBILITY probe (KEEP/KILL). Goal = a phase setting beats plain
FlowAlign-on-LTX (sbn off) on edited-object warp-error + structure-dist, holding CLIP-directional
editability within tol, on ONE LTX demo clip.

FlowAlign math is model-agnostic; this file ports the velocity/pack/VAE to LTX and reuses the
FlowAlign loop body. Paper hyperparams: zeta=0.01, CFG w in [5,7.5,10,13.5] (default 10), 33 steps.

Parts (--part): smoke (pipeline + VAE round-trip shape/const dump) ; gen ; analyze.
Run:  uv run experiments/e44_ltx_flowalign.py --part smoke --steps 8
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.5.1",
#     "diffusers==0.36.0",
#     "transformers==4.57.6",
#     "accelerate",
#     "sentencepiece",
#     "protobuf",
#     "imageio",
#     "imageio-ffmpeg",
#     "numpy",
#     "pillow",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# ///
import argparse
import os
import sys

import numpy as np
import torch

MODEL = "Lightricks/LTX-Video"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e44")


def load_ltx():
    from diffusers import LTXPipeline
    pipe = LTXPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def vae_roundtrip(pipe, frames_np):
    """frames_np: (F, H, W, 3) uint8. Encode -> normalize -> denormalize -> decode.
    Returns (recon_frames_np, latent) and prints every shape/const so the manual
    FlowAlign plumbing can be written against real numbers, not guesses."""
    vae = pipe.vae
    x = torch.from_numpy(frames_np).float() / 255.0
    x = (x.permute(3, 0, 1, 2)[None] * 2 - 1)          # (1, 3, F, H, W)
    x = x.to(vae.device, vae.dtype)
    print(f"[smoke] video tensor in: {tuple(x.shape)} dtype={x.dtype}", flush=True)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
    print(f"[smoke] raw latent: {tuple(z.shape)}  mean={z.float().mean():.3f} std={z.float().std():.3f}", flush=True)

    lm = pipe.__class__._normalize_latents
    mean = vae.latents_mean if hasattr(vae, "latents_mean") else vae.config.latents_mean
    std = vae.latents_std if hasattr(vae, "latents_std") else vae.config.latents_std
    mean = torch.tensor(mean) if not torch.is_tensor(mean) else mean
    std = torch.tensor(std) if not torch.is_tensor(std) else std
    sf = vae.config.scaling_factor
    print(f"[smoke] latents_mean shape {tuple(mean.shape)}  std shape {tuple(std.shape)}  scaling_factor={sf}", flush=True)
    zn = lm(z, mean, std, sf)
    print(f"[smoke] normalized latent: mean={zn.float().mean():.3f} std={zn.float().std():.3f}", flush=True)

    zd = pipe.__class__._denormalize_latents(zn, mean, std, sf).to(vae.dtype)
    # LTX VAE decode is timestep-conditioned; try the simple path, fall back if it needs temb.
    import inspect
    dparams = list(inspect.signature(vae.decode).parameters)
    print(f"[smoke] vae.decode params: {dparams}", flush=True)
    with torch.no_grad():
        try:
            ts = torch.zeros(zd.shape[0], device=zd.device, dtype=zd.dtype)
            out = vae.decode(zd, temb=ts, return_dict=False)[0]
        except TypeError:
            out = vae.decode(zd, return_dict=False)[0]
    print(f"[smoke] decoded tensor: {tuple(out.shape)}", flush=True)
    out = ((out.float() + 1) / 2).clamp(0, 1)[0].permute(1, 2, 3, 0).cpu().numpy()
    recon = (out * 255).round().astype(np.uint8)
    return recon, zn


def run_smoke(args):
    os.makedirs(OUT, exist_ok=True)
    pipe = load_ltx()
    print(f"[smoke] vae spatial/temporal compression: "
          f"{pipe.vae_spatial_compression_ratio}/{pipe.vae_temporal_compression_ratio}  "
          f"patch s/t: {pipe.transformer_spatial_patch_size}/{pipe.transformer_temporal_patch_size}", flush=True)

    H = W = args.size
    F = args.frames                                    # must be 8k+1
    gen = pipe(prompt="a calm sea with gentle waves at sunset",
               num_frames=F, height=H, width=W,
               num_inference_steps=args.steps,
               guidance_scale=3.0,
               generator=torch.Generator("cuda").manual_seed(0),
               output_type="np")
    frames = gen.frames[0]                              # (F, H, W, 3) float [0,1]
    frames = (np.asarray(frames) * 255).round().astype(np.uint8)
    print(f"[smoke] generated frames: {frames.shape} dtype={frames.dtype}", flush=True)
    import imageio.v3 as iio
    iio.imwrite(os.path.join(OUT, "smoke_gen.mp4"), frames, fps=8)

    recon, lat = vae_roundtrip(pipe, frames)
    n = min(len(frames), len(recon))
    err = np.abs(frames[:n].astype(float) - recon[:n].astype(float)).mean() / 255.0
    print(f"[smoke] VAE round-trip L1 (0..1): {err:.4f}  (frames out={recon.shape})", flush=True)
    iio.imwrite(os.path.join(OUT, "smoke_recon.mp4"), recon, fps=8)
    print(f"[smoke] latent packed-able shape (B,C,F,H,W)={tuple(lat.shape)}", flush=True)
    print("[smoke] DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="smoke")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    for part in args.part.split(","):
        if part == "smoke":
            run_smoke(args)
        else:
            print(f"[e44] part '{part}' not implemented yet", flush=True)


if __name__ == "__main__":
    main()
