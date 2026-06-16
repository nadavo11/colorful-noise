"""E25 - Text-aligned seed optimization (SD1.5).

Hypothesis (from "the seed leaves traces" / golden-noise observations): the
initial noise `z ~ N(0, I)` is highly correlated with the final generated latent,
so biasing `z` toward the prompt *before* denoising should bias the output.

We optimize `z` for a few gradient steps to:
  1. keep standard-Gaussian first moments (zero mean, unit variance) -- enforced
     by re-standardizing z after every step;
  2. raise a text-alignment objective.

The space-mismatch fix: a VAE-space noise latent can't be cosine'd with a CLIP
text embedding directly. The bridge is the model's ONE-STEP predicted clean
latent x0_hat: decode it, CLIP-image-encode it, cosine that against the CLIP-text
embedding. So we push the seed's *trace* (its one-step denoised guess) toward the
prompt, while constraining z to look like white Gaussian noise.

Then we run the SAME standard SD1.5 generation from the baseline seed and from the
aligned seed and compare side by side.

Run:  python experiments/e25_seedalign.py [quick]
Output: experiments/results/e25/{grid.png, objective_curves.png, report.json, pairs/}
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from clip_sim import load_clip, clip_image_features, clip_text_features, cosine
from common import RESULTS, save_grid

# ---- config (knobs) ----------------------------------------------------------
SD15_IDS = ["sd-legacy/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"]
OUT = os.path.join(RESULTS, "e25")

PROMPTS = [
    "a red cube on top of a blue sphere",
    "a photo of a cat wearing a tiny hat",
    "a green bench next to a yellow umbrella",
    "an astronaut riding a horse",
]
SEEDS = [0, 1]

INNER_STEPS = int(os.environ.get("E25_STEPS", 40))   # MAX seed-opt steps (early-stop below)
LR = float(os.environ.get("E25_LR", 0.05))           # Adam lr on z
OBJ_T = int(os.environ.get("E25_T", 600))            # denoise timestep (/1000) for x0_hat
KEEP_LAMBDA = float(os.environ.get("E25_LAMBDA", 0.0))  # lambda*||z-z0||^2 anchor (0=off)
# Early-stop: stop as soon as the inner objective reaches TARGET, so it's a small
# nudge (a natural image's CLIP-T is ~0.25-0.30) rather than an over-cooked seed.
TARGET = float(os.environ.get("E25_TARGET", 0.28))
# Objective MODE: "x0" = optimize one-step predicted clean latent x0_hat (uses UNet);
# "latent" = optimize the RAW seed's own decode (no UNet) -- the literal "traces" idea.
MODE = os.environ.get("E25_MODE", "x0")
GEN_STEPS = 50        # final generation steps
GUIDANCE = 7.5        # final generation CFG
SIZE = 512            # SD1.5 native

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# fp16 for the frozen models keeps the footprint small on a shared GPU; z stays
# fp32 for stable optimization. Set E25_DTYPE=fp32 if you have the whole card.
DTYPE = torch.float16 if os.environ.get("E25_DTYPE", "fp16") == "fp16" else torch.float32


# ---- model loading -----------------------------------------------------------
def load_sd15():
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    last_err = None
    for mid in SD15_IDS:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                mid, torch_dtype=DTYPE, safety_checker=None,
                requires_safety_checker=False)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            pipe = None
    if pipe is None:
        raise RuntimeError(f"could not load SD1.5: {last_err}")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    return pipe


# ---- objective: one-step x0_hat -> decode -> CLIP-image vs CLIP-text ---------
def clip_pixel_values(img_m1p1):
    """(-1,1) image tensor -> CLIP-normalized 224px pixel_values (differentiable)."""
    img = (img_m1p1 / 2 + 0.5).clamp(0, 1)
    img = F.interpolate(img, size=224, mode="bicubic", align_corners=False,
                        antialias=True)
    mean = torch.tensor(CLIP_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=img.device).view(1, 3, 1, 1)
    return (img - mean) / std


def align_seed(pipe, clip_model, z0, cond_embeds, text_feat):
    """Optimize a copy of z0 so a decoded view of it aligns with text_feat.

    MODE="x0":  decode the one-step predicted clean latent x0_hat (uses UNet).
    MODE="latent": decode the RAW seed z directly (no UNet) -- the literal "traces"
                   idea, biasing the seed's own appearance.
    Early-stops as soon as the objective reaches TARGET (small nudge). z is
    re-standardized to zero-mean/unit-var after every step.

    Returns (z_star, history, stop_step).
    """
    ac = pipe.scheduler.alphas_cumprod.to("cuda")
    sqrt_ab = ac[OBJ_T].sqrt()
    sqrt_1mab = (1 - ac[OBJ_T]).sqrt()
    t = torch.tensor(OBJ_T, device="cuda")
    sf = pipe.vae.config.scaling_factor
    tfeat = text_feat.to("cuda")

    z = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=LR)
    hist = []
    stop_step = INNER_STEPS
    for step in range(INNER_STEPS + 1):     # +1 for a final no-step eval
        opt.zero_grad()
        if MODE == "x0":
            zin = pipe.scheduler.scale_model_input(z, t).to(DTYPE)
            eps = pipe.unet(zin, t, encoder_hidden_states=cond_embeds).sample
            dec_lat = (z - sqrt_1mab * eps.float()) / sqrt_ab     # fp32 x0_hat
        else:                                # "latent": decode the raw seed
            dec_lat = z
        img = pipe.vae.decode((dec_lat / sf).to(DTYPE)).sample
        feat = clip_model.get_image_features(
            pixel_values=clip_pixel_values(img).to(DTYPE)).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        align = (feat * tfeat).sum()
        hist.append(float(align.detach()))
        if float(align) >= TARGET or step == INNER_STEPS:   # early-stop / cap
            stop_step = step
            break
        loss = -align
        if KEEP_LAMBDA:
            loss = loss + KEEP_LAMBDA * ((z - z0) ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():           # hard moment constraint
            z.data = (z.data - z.data.mean()) / (z.data.std() + 1e-8)
    return z.detach(), hist, stop_step


# ---- generation --------------------------------------------------------------
@torch.no_grad()
def generate(pipe, prompt, z):
    return pipe(prompt=prompt, latents=z.to("cuda").to(pipe.dtype),
                num_inference_steps=GEN_STEPS, guidance_scale=GUIDANCE,
                height=SIZE, width=SIZE).images[0]


def moments(z):
    return {"mean": float(z.mean()), "std": float(z.std()),
            "norm": float(z.norm())}


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    prompts = PROMPTS[:2] if quick else PROMPTS
    seeds = SEEDS[:1] if quick else SEEDS
    os.makedirs(os.path.join(OUT, "pairs"), exist_ok=True)

    pipe = load_sd15()
    clip_model, clip_proc = load_clip()
    clip_model = clip_model.to(DTYPE)
    clip_model.requires_grad_(False)
    shape = (1, pipe.unet.config.in_channels, SIZE // 8, SIZE // 8)
    d = shape[1] * shape[2] * shape[3]
    sqrt_d = d ** 0.5
    print(f"mode={MODE}  target={TARGET}  lr={LR}  max_steps={INNER_STEPS}  "
          f"latent dim d={d}  sqrt(d)={sqrt_d:.2f}", flush=True)

    rows, row_labels, records, curves = [], [], [], []
    for prompt in prompts:
        cond_embeds, _ = pipe.encode_prompt(prompt, "cuda", 1, False)
        cond_embeds = cond_embeds.detach()
        tfeat = clip_text_features(clip_model, clip_proc, [prompt])[0]  # (D,)
        for seed in seeds:
            g = torch.Generator("cuda").manual_seed(seed)
            z0 = torch.randn(shape, generator=g, device="cuda", dtype=torch.float32)

            z_star, hist, stop_step = align_seed(
                pipe, clip_model, z0, cond_embeds, tfeat)

            base_img = generate(pipe, prompt, z0)
            aligned_img = generate(pipe, prompt, z_star)

            # final-image CLIP-T (text-image cosine) baseline vs aligned
            with torch.autocast("cuda", dtype=DTYPE, enabled=DTYPE == torch.float16):
                feats = clip_image_features(clip_model, clip_proc,
                                            [base_img, aligned_img])
            clip_base = cosine(feats[0], tfeat)
            clip_aligned = cosine(feats[1], tfeat)

            tag = f"{MODE}_{prompt[:22].strip().replace(' ', '_')}_s{seed}"
            base_img.save(os.path.join(OUT, "pairs", f"{tag}_baseline.png"))
            aligned_img.save(os.path.join(OUT, "pairs", f"{tag}_aligned.png"))

            mom = moments(z_star)
            rows.append([base_img, aligned_img])
            row_labels.append(f"{prompt[:18]} s{seed}")
            curves.append((tag, hist))
            records.append({
                "prompt": prompt, "seed": seed,
                "clip_baseline": clip_base, "clip_aligned": clip_aligned,
                "clip_delta": clip_aligned - clip_base,
                "obj_start": hist[0], "obj_end": hist[-1], "stop_step": stop_step,
                "z_star_moments": mom,
            })
            print(f"[{tag}] obj {hist[0]:.3f}->{hist[-1]:.3f} @step{stop_step}  "
                  f"clipT {clip_base:.3f}->{clip_aligned:.3f} "
                  f"(d={clip_aligned - clip_base:+.3f})  "
                  f"z* mean={mom['mean']:+.3f} std={mom['std']:.3f} "
                  f"norm={mom['norm']:.2f} (sqrt(d)={sqrt_d:.2f})", flush=True)

    save_grid(rows, row_labels, ["baseline", f"aligned ({MODE})"],
              os.path.join(OUT, f"grid_{MODE}.png"))
    plot_curves(curves, os.path.join(OUT, f"objective_curves_{MODE}.png"))

    summary = {
        "config": {"mode": MODE, "target": TARGET, "max_steps": INNER_STEPS,
                   "lr": LR, "obj_t": OBJ_T, "keep_lambda": KEEP_LAMBDA,
                   "gen_steps": GEN_STEPS, "guidance": GUIDANCE,
                   "latent_dim": d, "sqrt_d": sqrt_d},
        "mean_clip_delta": sum(r["clip_delta"] for r in records) / len(records),
        "mean_z_norm": sum(r["z_star_moments"]["norm"] for r in records) / len(records),
        "records": records,
    }
    with open(os.path.join(OUT, f"report_{MODE}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nmode={MODE}  mean clipT delta (aligned - baseline): "
          f"{summary['mean_clip_delta']:+.4f}  "
          f"mean ||z*||={summary['mean_z_norm']:.2f} (sqrt(d)={sqrt_d:.2f})")
    print(f"wrote {OUT}/{{grid_{MODE}.png, objective_curves_{MODE}.png, "
          f"report_{MODE}.json}}")


def plot_curves(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    for tag, hist in curves:
        plt.plot(range(len(hist)), hist, label=tag, alpha=0.8)
    plt.xlabel("seed-optimization step")
    plt.ylabel(f"CLIP(decode {MODE}) . text  (alignment)")
    plt.axhline(TARGET, color="k", ls="--", lw=0.8, alpha=0.5)
    plt.title(f"E25: inner-loop seed/text alignment (mode={MODE})")
    plt.legend(fontsize=6, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
