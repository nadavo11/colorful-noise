"""E21: spectral image editing via RF-inversion + frequency-band control on SD3.5.

Invert a real image to noise (reverse flow ODE), then regenerate with a NEW prompt
while LOCKING chosen frequency content to the source. The band ops decide WHICH
structure survives the edit: phase (esp. low-band) carries layout (E12-E14), so
locking the source's low-band phase preserves composition while the new prompt
edits appearance -- a frequency-decomposed editing control.

SD3.5 is rectified flow, so inversion = integrate the velocity field from sigma=0
(clean) up to sigma=1 (noise); NOT literal DDIM (cf. RF-Inversion / FlowEdit). The
generation Euler step is  x += (sigma_next - sigma_cur) * v(x, sigma); inversion
reverses it over the same sigma grid.

Parts (--part):
  preflight -- reverse-Euler exactness on a toy field + band-lock math (no model)
  invert    -- invert real images, reconstruct (same prompt), report fidelity
               (gates everything: if reconstruction is poor, editing is moot)
  edit      -- invert, then regenerate with a target prompt under band-lock variants
  analyze   -- grids + edit-success vs structure-preservation table
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid
from spectral_ops import band_phase_swap, band_index_map
from style_ops import restyle_latent, latent_band_power
from e17_sd35 import (load_sd35, sd3_vae_encode, sd3_vae_decode, gen_sd3,
                      SIZE, H, W, N_CH)

OUT = os.path.join(RESULTS, "e21")
N_BINS = 24
# (source image, source prompt, target/edit prompt)
EDITS = [
    ("photo_000.jpg", "a photograph", "an oil painting"),
    ("photo_001.jpg", "a photograph", "a pencil sketch"),
    ("photo_002.jpg", "a photograph", "a watercolor painting, pastel colors"),
]


# ---------------------------------------------------------------------------
# RF inversion (clean -> noise) and the band-lock callback
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_prompt(pipe, prompt):
    pe, _, pp, _ = pipe.encode_prompt(prompt=prompt, prompt_2=prompt, prompt_3=prompt,
                                      device="cuda", do_classifier_free_guidance=False)
    return pe, pp


@torch.no_grad()
def velocity(pipe, x, sigma, pe, pp):
    """Transformer velocity v(x, sigma) (flow-matching prediction), fp32."""
    nt = pipe.scheduler.config.num_train_timesteps
    t = torch.full((x.shape[0],), sigma * nt, device=x.device, dtype=pipe.dtype)
    return pipe.transformer(hidden_states=x.to(pipe.dtype), timestep=t,
                            encoder_hidden_states=pe, pooled_projections=pp,
                            return_dict=False)[0].float()


@torch.no_grad()
def invert_sd3(pipe, x0, prompt, steps=28, fp_iters=4):
    """Clean latent x0 (1,16,128,128) -> inverted noise over the flow ODE (guidance
    =1). fp_iters>0 uses FIXED-POINT (implicit) Euler -- solve
    x_hi = x_lo + (s_hi-s_lo)*v(x_hi, s_hi) by iteration -- which inverts the
    sampler far more accurately than naive forward Euler (fp_iters=1)."""
    pipe.scheduler.set_timesteps(steps, device="cuda")
    sig = pipe.scheduler.sigmas                        # (steps+1,) decreasing, [-1]=0
    pe, pp = encode_prompt(pipe, prompt)
    x = x0.cuda().float()
    for i in range(steps - 1, -1, -1):                 # walk sigma 0 -> 1
        s_lo, s_hi = float(sig[i + 1]), float(sig[i])
        x_hi = x.clone()
        for _ in range(max(1, fp_iters)):              # implicit: v at the NEXT sigma
            x_hi = x + (s_hi - s_lo) * velocity(pipe, x_hi, s_hi, pe, pp)
        x = x_hi
    return x.cpu()


class BandLock:
    """Step-end callback: lock the source's chosen frequency content into the
    generating latent. mode='phase' keeps the source low-band PHASE (layout) with
    the generation's magnitude + high-band phase; mode='power' re-levels per-band
    power to the source (palette). Applied for the first `until` fraction of steps,
    then released so the target prompt drives the rest."""

    def __init__(self, x0_src, idx, n_bins, c=0.25, until=1.0, steps=28, mode="phase"):
        self.x0 = x0_src.cuda().float()
        self.idx, self.n_bins, self.c = idx, n_bins, c
        self.until, self.steps, self.mode = until, steps, mode
        self.srcband = latent_band_power(self.x0, idx, n_bins)

    def __call__(self, p, i, t, kw):
        if i >= self.until * self.steps:
            return {}
        lat = kw["latents"].float()
        if self.mode == "phase":                       # source low-band phase + gen rest
            out = band_phase_swap(self.x0, lat, c=self.c, mag_from="B").real
        else:                                          # source per-band power
            out = restyle_latent(lat, self.srcband, self.idx, self.n_bins, 1.0)
        return {"latents": out.to(kw["latents"].dtype)}


# ---------------------------------------------------------------------------
# preflight (no model)
# ---------------------------------------------------------------------------

def preflight(args):
    print("[e21] pre-flight (model-free) ...", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    # 1. reverse-Euler exactness on a STATE-INDEPENDENT field v(sigma): generation
    #    then inversion must recover x0 (Euler is exact for v independent of x).
    steps = 28
    sig = torch.linspace(1, 0, steps + 1)
    a = torch.randn(1, N_CH, H, W, device=dev)         # constant velocity field
    x0 = torch.randn(1, N_CH, H, W, device=dev)
    xn = x0.clone()                                    # invert 0->1, then regen 1->0
    for i in range(steps - 1, -1, -1):                 # invert 0->1
        xn = xn + (float(sig[i]) - float(sig[i + 1])) * a
    xr = xn.clone()
    for i in range(steps):                             # regen 1->0
        xr = xr + (float(sig[i + 1]) - float(sig[i])) * a
    err = float((xr - x0).abs().max())
    assert err < 1e-3, f"reverse-Euler not exact on constant field: {err}"

    # 2. band-lock invariants: c=1 mag_from=A reconstructs the source exactly;
    #    mag_from=B keeps the generation's magnitude (phase taken from source).
    src = torch.randn(1, N_CH, H, W, device=dev)
    gen = torch.randn(1, N_CH, H, W, device=dev) * 1.7
    recon = band_phase_swap(src, gen, c=1.0, mag_from="A").real
    assert float((recon - src).abs().max()) < 1e-2, "c=1 mag=A must reconstruct source"
    locked = band_phase_swap(src, gen, c=1.0, mag_from="B").real
    mg = torch.fft.fft2(gen.float()).abs()
    rel = ((torch.fft.fft2(locked.float()).abs() - mg).abs() / mg.clamp(min=1e-3))
    assert float(rel.median()) < 0.05, "lock did not keep generation magnitude"
    print(f"[e21] pre-flight OK (euler_err={err:.1e}, recon + mag-lock verified)",
          flush=True)


# ---------------------------------------------------------------------------
# invert: validate reconstruction
# ---------------------------------------------------------------------------

def _clip():
    from e9_clipt import load_clip
    return load_clip("openai/clip-vit-large-patch14")


def run_invert(args):
    from clip_sim import clip_image_features, cosine
    pipe = load_sd35("gpu_resident")
    vae = pipe.vae
    model, proc = _clip()
    os.makedirs(f"{OUT}/invert", exist_ok=True)
    rows, labels, report = [], [], {}
    for fn, src_p, _ in EDITS[: args.num]:
        src_pil = Image.open(os.path.join(args.imgs, fn)).convert("RGB").resize((SIZE, SIZE))
        x0 = sd3_vae_encode(vae, src_pil)
        noise = invert_sd3(pipe, x0, src_p, args.steps)
        # reconstruct: generate from the inverted noise with the SAME prompt, g=1
        rec, _ = gen_sd3(pipe, src_p, 0, 1.0, args.steps, init_latents=noise)
        f = clip_image_features(model, proc, [src_pil, rec])
        report[fn] = {"recon_clip_i": cosine(f[0], f[1]),
                      "noise_std": float(noise.std())}
        rec.save(f"{OUT}/invert/{fn}_recon.png")
        rows.append([src_pil, rec]); labels.append(fn[:10])
        print(f"[e21] invert {fn}: recon CLIP-I={report[fn]['recon_clip_i']:.3f} "
              f"noise_std={report[fn]['noise_std']:.2f}", flush=True)
    save_grid(rows, labels, ["source", "reconstruction"], f"{OUT}/invert/grid.png")
    json.dump(report, open(f"{OUT}/invert.json", "w"), indent=2)
    print(f"[e21] reconstruction fidelity (CLIP-I to source) -> {OUT}/invert.json", flush=True)


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------

def run_edit(args):
    from clip_sim import clip_image_features, cosine
    from e9_clipt import clip_scores
    pipe = load_sd35("gpu_resident")
    vae = pipe.vae
    model, proc = _clip()
    idx = band_index_map(H, W, N_BINS, "cuda")
    os.makedirs(f"{OUT}/edit", exist_ok=True)
    report = {}
    for fn, src_p, tgt_p in EDITS[: args.num]:
        src_pil = Image.open(os.path.join(args.imgs, fn)).convert("RGB").resize((SIZE, SIZE))
        x0 = sd3_vae_encode(vae, src_pil)
        noise = invert_sd3(pipe, x0, src_p, args.steps)
        variants = {"invert_only": None}              # baseline: no band lock
        for c in args.cuts:
            for u in args.untils:
                variants[f"lockphase_c{c:g}_u{u:g}"] = BandLock(
                    x0, idx, N_BINS, c=c, until=u, steps=args.steps, mode="phase")
        variants["lockpower"] = BandLock(x0, idx, N_BINS, mode="power", steps=args.steps)

        fsrc = clip_image_features(model, proc, [src_pil])[0]
        rows, labels = [[src_pil]], ["source"]
        report[fn] = {"target": tgt_p, "cells": {}}
        imgs_for_grid = []
        for name, cb in variants.items():
            img, _ = gen_sd3(pipe, tgt_p, 0, args.cfg, args.steps,
                             init_latents=noise, cb_obj=cb)
            f = clip_image_features(model, proc, [img])[0]
            report[fn]["cells"][name] = {
                "struct_clip": cosine(f, fsrc),                       # preserve source
                "edit_clip_t": clip_scores(model, proc, tgt_p, [img])[0]}  # follow target
            imgs_for_grid.append((name, img))
            print(f"[e21] {fn} {name}: struct={report[fn]['cells'][name]['struct_clip']:.3f} "
                  f"edit={report[fn]['cells'][name]['edit_clip_t']:.3f}", flush=True)
        rows.append([im for _, im in imgs_for_grid])
        labels.append("edits")
        save_grid([[src_pil] + [im for _, im in imgs_for_grid]],
                  [fn[:10]], ["source"] + [n for n, _ in imgs_for_grid],
                  f"{OUT}/edit/grid_{fn}.png")
    json.dump(report, open(f"{OUT}/edit.json", "w"), indent=2)
    print(f"[e21] edit results -> {OUT}/edit.json", flush=True)


def run_analyze(args):
    for jf in ("invert.json", "edit.json"):
        p = f"{OUT}/{jf}"
        if os.path.exists(p):
            print(f"\n=== {jf} ===")
            print(json.dumps(json.load(open(p)), indent=1)[:2000])


def run_site(args):
    """Retired: per-experiment HTML is superseded by the roadmap site
    (docs/roadmap/, generated from roadmap_registry.py)."""
    print("[e21] --part site retired; see docs/roadmap/ "
          "(regen: python experiments/make_roadmap.py)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="preflight")
    ap.add_argument("--imgs", default=os.path.join(RESULTS, "e10", "real_photos"))
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--cuts", default="0.1,0.25",
                    type=lambda s: [float(x) for x in s.split(",")])
    ap.add_argument("--untils", default="0.6,1.0",
                    type=lambda s: [float(x) for x in s.split(",")])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    runners = {"preflight": preflight, "invert": run_invert, "edit": run_edit,
               "analyze": run_analyze, "site": run_site}
    for part in filter(None, (p.strip() for p in args.part.split(","))):
        runners[part](args)


if __name__ == "__main__":
    main()
