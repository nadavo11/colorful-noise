"""E22: spectral image editing via DDIM inversion + frequency-band control (SDXL).

RF inversion on SD3.5 (E21) failed to reconstruct (naive + fixed-point both drift).
SDXL is an eps-prediction model where DDIM inversion is reliable, so we pivot here:
invert a real image to noise with DDIMInverseScheduler, then regenerate with a NEW
prompt while LOCKING chosen source frequencies. Phase (esp. low-band) carries
layout, so locking source low-band phase preserves composition while the prompt
edits appearance -- a frequency-decomposed editing control.

SDXL latent is 4x128x128 at 1024px -- same (H,W) grid as SD3.5, so spectral_ops /
style_ops apply unchanged (channel-agnostic). Runs locally (SDXL is cached).

Parts: preflight / invert (validate reconstruction -- the gate) / edit / analyze.
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid, SDXL_ID
from spectral_ops import band_phase_swap, band_index_map
from style_ops import restyle_latent, latent_band_power
from clip_sim import load_clip, clip_image_features, clip_text_features, cosine

OUT = os.path.join(RESULTS, "e22")
SIZE, H, W, C, N_BINS = 1024, 128, 128, 4, 24
EDITS = [
    ("photo_000.jpg", "a photo", "an oil painting"),
    ("photo_001.jpg", "a photo", "a pencil sketch"),
    ("photo_002.jpg", "a photo", "a watercolor painting"),
]


def load_sdxl():
    from diffusers import (StableDiffusionXLPipeline, DDIMScheduler,
                           DDIMInverseScheduler)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_ID, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inv = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe, inv


def encode(pipe, pil):
    """PIL -> SDXL latent (1,4,128,128) fp16 (VAE in fp32 for stability)."""
    pipe.vae.to(dtype=torch.float32)
    x = pipe.image_processor.preprocess(pil).to("cuda", torch.float32)
    with torch.no_grad():
        lat = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    pipe.vae.to(dtype=torch.float16)
    return lat.half()


@torch.no_grad()
def ddim_invert(pipe, inv_sched, latent, prompt, steps, guidance=1.0):
    """Clean latent -> inverted noise: run the pipeline with the INVERSE scheduler
    (timesteps 0->T), output the final latent."""
    ddim = pipe.scheduler
    pipe.scheduler = inv_sched
    try:
        out = pipe(prompt=prompt, latents=latent.half(), num_inference_steps=steps,
                   guidance_scale=guidance, output_type="latent").images
    finally:
        pipe.scheduler = ddim
    return out


class BandLock:
    """Step-end callback: lock source frequency content into the generating latent.
    mode='phase' keeps source low-band phase (layout) + generation magnitude/high;
    mode='power' re-levels per-band power to the source (palette). Active for the
    first `until` fraction of steps."""

    def __init__(self, x0, idx, n_bins, c=0.25, until=1.0, steps=30, mode="phase"):
        self.x0 = x0.cuda().float()
        self.idx, self.n_bins, self.c = idx, n_bins, c
        self.until, self.steps, self.mode = until, steps, mode
        self.srcband = latent_band_power(self.x0, idx, n_bins)

    def __call__(self, pipe, i, t, kw):
        if i >= self.until * self.steps:
            return {}
        lat = kw["latents"].float()
        if self.mode == "phase":
            out = band_phase_swap(self.x0, lat, c=self.c, mag_from="B").real
        else:
            out = restyle_latent(lat, self.srcband, self.idx, self.n_bins, 1.0)
        return {"latents": out.to(kw["latents"].dtype)}


@torch.no_grad()
def generate(pipe, prompt, latent, steps, guidance, cb=None):
    return pipe(prompt=prompt, latents=latent.half(), num_inference_steps=steps,
                guidance_scale=guidance, callback_on_step_end=cb,
                callback_on_step_end_tensor_inputs=["latents"]).images[0]


# ---------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------

def preflight(args):
    print("[e22] pre-flight (model-free) ...", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    src = torch.randn(1, C, H, W, device=dev)
    gen = torch.randn(1, C, H, W, device=dev) * 1.6
    recon = band_phase_swap(src, gen, c=1.0, mag_from="A").real
    assert float((recon - src).abs().max()) < 1e-2, "band recon"
    locked = band_phase_swap(src, gen, c=0.25, mag_from="B").real
    assert locked.shape == src.shape and locked.is_floating_point()
    print("[e22] pre-flight OK", flush=True)


def run_invert(args):
    pipe, inv = load_sdxl()
    model, proc = load_clip()
    os.makedirs(f"{OUT}/invert", exist_ok=True)
    rows, labels, report = [], [], {}
    for fn, src_p, _ in EDITS[: args.num]:
        pil = Image.open(os.path.join(args.imgs, fn)).convert("RGB").resize((SIZE, SIZE))
        lat = encode(pipe, pil)
        noise = ddim_invert(pipe, inv, lat, src_p, args.steps, args.inv_guidance)
        rec = generate(pipe, src_p, noise, args.steps, args.inv_guidance)
        f = clip_image_features(model, proc, [pil, rec])
        report[fn] = {"recon_clip_i": cosine(f[0], f[1]), "noise_std": float(noise.std())}
        rec.save(f"{OUT}/invert/{fn}_recon.png")
        rows.append([pil, rec]); labels.append(fn[:10])
        print(f"[e22] invert {fn}: recon CLIP-I={report[fn]['recon_clip_i']:.3f} "
              f"noise_std={report[fn]['noise_std']:.2f}", flush=True)
    save_grid(rows, labels, ["source", "DDIM reconstruction"], f"{OUT}/invert/grid.png")
    json.dump(report, open(f"{OUT}/invert.json", "w"), indent=2)
    print(f"[e22] -> {OUT}/invert.json", flush=True)


def run_edit(args):
    pipe, inv = load_sdxl()
    model, proc = load_clip()
    idx = band_index_map(H, W, N_BINS, "cuda")
    os.makedirs(f"{OUT}/edit", exist_ok=True)
    report = {}
    for fn, src_p, tgt_p in EDITS[: args.num]:
        pil = Image.open(os.path.join(args.imgs, fn)).convert("RGB").resize((SIZE, SIZE))
        x0 = encode(pipe, pil)
        noise = ddim_invert(pipe, inv, x0, src_p, args.steps, args.inv_guidance)
        variants = {"invert_only": None}
        for c in args.cuts:
            for u in args.untils:
                variants[f"lockphase_c{c:g}_u{u:g}"] = BandLock(
                    x0, idx, N_BINS, c=c, until=u, steps=args.steps, mode="phase")
        variants["lockpower"] = BandLock(x0, idx, N_BINS, steps=args.steps, mode="power")

        ftgt = clip_text_features(model, proc, [tgt_p])[0]
        fsrc = clip_image_features(model, proc, [pil])[0]
        report[fn] = {"target": tgt_p, "cells": {}}
        grid = [pil]
        cols = ["source"]
        for name, cb in variants.items():
            img = generate(pipe, tgt_p, noise, args.steps, args.cfg, cb)
            f = clip_image_features(model, proc, [img])[0]
            report[fn]["cells"][name] = {"struct_clip": cosine(f, fsrc),
                                         "edit_clip_t": cosine(f, ftgt)}
            grid.append(img); cols.append(name)
            print(f"[e22] {fn} {name}: struct={report[fn]['cells'][name]['struct_clip']:.3f} "
                  f"edit_t={report[fn]['cells'][name]['edit_clip_t']:.3f}", flush=True)
        save_grid([grid], [fn[:10]], cols, f"{OUT}/edit/grid_{fn}.png")
    json.dump(report, open(f"{OUT}/edit.json", "w"), indent=2)
    print(f"[e22] -> {OUT}/edit.json", flush=True)


def run_analyze(args):
    for jf in ("invert.json", "edit.json"):
        p = f"{OUT}/{jf}"
        if os.path.exists(p):
            print(f"\n=== {jf} ===\n" + json.dumps(json.load(open(p)), indent=1)[:2500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="preflight")
    ap.add_argument("--imgs", default=os.path.join(RESULTS, "e10", "real_photos"))
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--inv_guidance", type=float, default=1.0)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--cuts", default="0.1,0.25",
                    type=lambda s: [float(x) for x in s.split(",")])
    ap.add_argument("--untils", default="0.6,1.0",
                    type=lambda s: [float(x) for x in s.split(",")])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"preflight": preflight, "invert": run_invert, "edit": run_edit,
     "analyze": run_analyze}[args.part.split(",")[0]]  # validate first part name
    for part in filter(None, (p.strip() for p in args.part.split(","))):
        {"preflight": preflight, "invert": run_invert, "edit": run_edit,
         "analyze": run_analyze}[part](args)


if __name__ == "__main__":
    main()
