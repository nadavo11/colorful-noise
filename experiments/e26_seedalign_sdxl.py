"""E26 - Seed alignment on SDXL with long prompts, sweeping inner-step count.

Direction (continues E25 on SD1.5): instead of a random init noise z~N(0,I), iterate
z DIRECTLY IN LATENT SPACE with a few gradient steps so a decoded view of it aligns
with the prompt, while keeping standard-Gaussian first moments (zero mean / unit var,
so ||z||==sqrt(d) exactly -- a move ON the Gaussian sphere). This matches the "seed
leaves traces" idea: bias the seed, the trace survives into the generation.

CRUCIAL: the objective does NOT predict x0 and NEVER runs the UNet. It is purely
    loss = -cosine( CLIP-image(decode(z)), CLIP-text(prompt) ).
The UNet is only used inside the normal pipe(prompt, latents=z*) generation call.

This is E25's MODE="latent" (the gentlest, best-behaved one there) ported to SDXL.

Three extensions over E25:
  1. SDXL (1024px) instead of SD1.5.
  2. Long, dense DPG-Bench prompts (dpg_bench.load_dpg_prompts). NB: SDXL's two CLIP
     text encoders -- and the CLIP scorer -- truncate at 77 tokens, while DPG prompts
     run ~70-90 words. So the objective/metric use a LONG-AWARE text feature: split the
     prompt into clauses, CLIP-encode each (<=77 tok), mean-pool. Seed-alignment can thus
     carry prompt info the truncated text conditioning drops.
  3. Sweep N = number of inner gradient (latent-space) steps: the cheap one-step linear
     solution N=1, a "strengthened" single step (larger lr), then a few more N=2,3,5.

Run:  python experiments/e26_seedalign_sdxl.py [quick]
Out:  experiments/results/e26/{grid.png, deltaclip_vs_N.png, report.json, pairs/}
"""
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

from clip_sim import load_clip, clip_image_features, cosine
from common import RESULTS, load_pipe, save_grid, generate
from dpg_bench import load_dpg_prompts

OUT = os.path.join(RESULTS, "e26")

# ---- knobs -------------------------------------------------------------------
N_PROMPTS = int(os.environ.get("E26_PROMPTS", 10))
SEEDS = [int(s) for s in os.environ.get("E26_SEEDS", "0").split(",")]
SWEEP_N = [1, 2, 3, 5]            # inner latent-space gradient-step counts to snapshot
N_MAX = max(SWEEP_N)
LR = float(os.environ.get("E26_LR", 0.05))
STRONG_LR = float(os.environ.get("E26_STRONG_LR", 0.20))   # the "strengthened" 1-step
GEN_STEPS = int(os.environ.get("E26_GEN_STEPS", 40))
GUIDANCE = float(os.environ.get("E26_GUIDANCE", 7.0))
SIZE = int(os.environ.get("E26_SIZE", 1024))               # drop to 768 if OOM
DTYPE = torch.float16

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
VAE_FP16_FIX = "madebyollin/sdxl-vae-fp16-fix"


# ---- model loading -----------------------------------------------------------
def load_sdxl():
    pipe = load_pipe()                       # SDXL base, fp16, on cuda
    # SDXL's stock fp16 VAE NaNs on decode; swap in the fp16-fix VAE if available.
    try:
        from diffusers import AutoencoderKL
        pipe.vae = AutoencoderKL.from_pretrained(
            VAE_FP16_FIX, torch_dtype=DTYPE).to("cuda")
        print("[e26] using sdxl-vae-fp16-fix", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[e26] fp16-fix VAE unavailable ({e}); using fp32 VAE for decode",
              flush=True)
        pipe.vae = pipe.vae.to(torch.float32)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    return pipe


# ---- long-prompt-aware CLIP text feature -------------------------------------
@torch.no_grad()
def clip_text_features_long(model, proc, prompt):
    """Whole-prompt joint-space vector: split into clauses (each <=77 tok), encode,
    mean-pool, renormalize. Beats plain CLIP-T which truncates long prompts at 77."""
    chunks = [c.strip() for c in re.split(r"[.,;]", prompt) if c.strip()] or [prompt]
    tin = proc(text=chunks, return_tensors="pt", padding=True, truncation=True,
               max_length=77).to("cuda")
    f = model.get_text_features(**tin).to(DTYPE)
    f = f / f.norm(dim=-1, keepdim=True)
    m = f.mean(0)
    return (m / m.norm()).float().cpu()


# ---- differentiable latent-space objective: decode(z) -> CLIP -> cosine -------
def clip_pixel_values(img_m1p1):
    img = (img_m1p1 / 2 + 0.5).clamp(0, 1)
    img = F.interpolate(img, size=224, mode="bicubic", align_corners=False,
                        antialias=True)
    mean = torch.tensor(CLIP_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=img.device).view(1, 3, 1, 1)
    return (img - mean) / std


def _align_cosine(pipe, clip_model, z, tfeat, sf):
    """cosine( CLIP-image(decode(z)), tfeat ) -- differentiable wrt z. No UNet."""
    img = pipe.vae.decode((z / sf).to(pipe.vae.dtype)).sample
    feat = clip_model.get_image_features(
        pixel_values=clip_pixel_values(img).to(DTYPE)).float()
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return (feat * tfeat).sum()


def optimize_seed(pipe, clip_model, z0, tfeat, lr, n_max, snap_at):
    """Iterate z in latent space; return {n: z_snapshot} for n in snap_at, plus the
    per-step alignment history. Re-standardize to zero-mean/unit-var each step."""
    sf = pipe.vae.config.scaling_factor
    tfeat = tfeat.to("cuda")
    z = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    snaps, hist = {}, []
    for step in range(1, n_max + 1):
        opt.zero_grad()
        align = _align_cosine(pipe, clip_model, z, tfeat, sf)
        (-align).backward()
        opt.step()
        with torch.no_grad():                 # hard moment constraint -> ||z||=sqrt(d)
            z.data = (z.data - z.data.mean()) / (z.data.std() + 1e-8)
        hist.append(float(align.detach()))
        if step in snap_at:
            snaps[step] = z.detach().clone()
    return snaps, hist


def moments(z):
    return {"mean": float(z.mean()), "std": float(z.std()), "norm": float(z.norm())}


# ---- main --------------------------------------------------------------------
def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    n_prompts = 2 if quick else N_PROMPTS
    seeds = SEEDS[:1]
    sweep = [1, 2] if quick else SWEEP_N
    n_max = max(sweep)
    os.makedirs(os.path.join(OUT, "pairs"), exist_ok=True)

    pipe = load_sdxl()
    clip_model, clip_proc = load_clip()
    clip_model = clip_model.to(DTYPE)
    clip_model.requires_grad_(False)

    prompts = load_dpg_prompts(n=n_prompts)
    shape = (1, pipe.unet.config.in_channels, SIZE // 8, SIZE // 8)
    d = shape[1] * shape[2] * shape[3]
    sqrt_d = d ** 0.5
    cols = ["baseline"] + [f"N={n}" for n in sweep] + ["N=1*strong"]
    print(f"SDXL {SIZE}px  d={d} sqrt(d)={sqrt_d:.1f}  prompts={len(prompts)} "
          f"seeds={seeds}  sweep={sweep}  cols={cols}", flush=True)

    rows, row_labels, records = [], [], []
    delta_by_n = {n: [] for n in sweep + ["1*"]}
    for pid, prompt in prompts:
        tfeat = clip_text_features_long(clip_model, clip_proc, prompt)
        for seed in seeds:
            g = torch.Generator("cuda").manual_seed(seed)
            z0 = torch.randn(shape, generator=g, device="cuda", dtype=torch.float32)

            snaps, hist = optimize_seed(pipe, clip_model, z0, tfeat, LR, n_max, set(sweep))
            strong, _ = optimize_seed(pipe, clip_model, z0, tfeat, STRONG_LR, 1, {1})
            seeds_by_col = [("baseline", z0)] + \
                [(f"N={n}", snaps[n]) for n in sweep] + [("N=1*strong", strong[1])]

            imgs, clip_by_col = [], {}
            for col, z in seeds_by_col:
                img = generate(pipe, prompt, z, steps=GEN_STEPS, guidance=GUIDANCE)
                imgs.append(img)
                tag = f"{pid}_s{seed}_{col.replace('=', '').replace('*', 's')}"
                img.save(os.path.join(OUT, "pairs", f"{tag}.png"))

            feats = clip_image_features(clip_model, clip_proc, imgs)  # (C, D)
            clip_scores = [cosine(feats[i], tfeat) for i in range(len(imgs))]
            base = clip_scores[0]
            for j, (col, z) in enumerate(seeds_by_col):
                clip_by_col[col] = clip_scores[j]
            for n in sweep:
                delta_by_n[n].append(clip_by_col[f"N={n}"] - base)
            delta_by_n["1*"].append(clip_by_col["N=1*strong"] - base)

            rows.append(imgs)
            row_labels.append(f"{pid} s{seed}")
            records.append({
                "pid": pid, "prompt": prompt, "seed": seed, "n_words": len(prompt.split()),
                "clip_long": clip_by_col, "obj_hist": hist,
                "z_moments": {c: moments(z) for c, z in seeds_by_col},
            })
            torch.cuda.empty_cache()
            print(f"[{pid} s{seed}] ({len(prompt.split())}w) "
                  f"obj {hist[0]:.3f}->{hist[-1]:.3f}  "
                  f"clipL base={base:.3f} " +
                  " ".join(f"{c.split('=')[1] if '=' in c else c}={clip_by_col[c]:+.3f}"
                           for c, _ in seeds_by_col[1:]) +
                  f"  ||z(N={n_max})||={moments(snaps[n_max])['norm']:.1f}", flush=True)

    save_grid(rows, row_labels, cols, os.path.join(OUT, "grid.png"))
    plot_delta(delta_by_n, sweep, os.path.join(OUT, "deltaclip_vs_N.png"))

    mean_delta = {str(n): sum(v) / len(v) for n, v in delta_by_n.items() if v}
    summary = {
        "config": {"model": "sdxl", "size": SIZE, "lr": LR, "strong_lr": STRONG_LR,
                   "sweep_n": sweep, "gen_steps": GEN_STEPS, "guidance": GUIDANCE,
                   "latent_dim": d, "sqrt_d": sqrt_d, "metric": "long-aware CLIP-T"},
        "mean_clip_delta_by_n": mean_delta,
        "records": records,
    }
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nmean long-CLIP delta vs baseline by N:",
          {k: round(v, 4) for k, v in mean_delta.items()})
    print(f"wrote {OUT}/{{grid.png, deltaclip_vs_N.png, report.json}}")


def plot_delta(delta_by_n, sweep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = sweep
    means = [sum(delta_by_n[n]) / len(delta_by_n[n]) for n in xs]
    plt.figure(figsize=(6.5, 4.2))
    for n in xs:                                   # per-prompt faint lines
        plt.scatter([n] * len(delta_by_n[n]), delta_by_n[n], s=8, alpha=0.3, color="C0")
    plt.plot(xs, means, "o-", color="C0", label="mean Δ (N steps)")
    s = delta_by_n.get("1*", [])
    if s:
        plt.scatter([1] * len(s), s, s=8, alpha=0.3, color="C3")
        plt.scatter([1], [sum(s) / len(s)], marker="*", s=200, color="C3",
                    label="mean Δ (1 strengthened step)", zorder=5)
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("N = inner latent-space gradient steps")
    plt.ylabel("Δ long-CLIP-T  (aligned − baseline)")
    plt.title("E26: SDXL seed alignment vs #steps (long DPG prompts)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
