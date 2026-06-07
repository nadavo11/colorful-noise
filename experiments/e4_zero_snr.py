"""E4: Does low-frequency noise conditioning survive zero terminal SNR?

Reruns the informative E2 conditions + colored-noise betas on Playground v2.5
(SDXL architecture, EDM schedule, sigma_max=80 => terminal SNR ~ 0).
Hypothesis: SDXL's non-zero terminal SNR is *why* Colorful-Noise works; on a
zero-SNR model the conditioning effect should weaken or vanish.

Note: the pipeline scales input latents by init_noise_sigma internally, so we
pass our unit-variance conditioned latents exactly as for SDXL.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import encode_image, generate, save_grid, INPUTS, RESULTS
from spectral_ops import condition_latent, colored_noise, E2_CONDITIONS

PG_ID = "playgroundai/playground-v2.5-1024px-aesthetic"
E2_SUBSET = ["white", "paper", "phase_only", "dc_only", "mag_only"]
BETAS = [-2, 0, 2]
CASE = ("cat_orange.png", "A photo of cat in the park")


def load_playground():
    from diffusers import StableDiffusionXLPipeline, EDMDPMSolverMultistepScheduler
    pipe = StableDiffusionXLPipeline.from_pretrained(
        PG_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    assert isinstance(pipe.scheduler, EDMDPMSolverMultistepScheduler), type(pipe.scheduler)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main(args):
    out = os.path.join(RESULTS, "e4")
    os.makedirs(out, exist_ok=True)
    pipe = load_playground()
    print(f"[e4] init_noise_sigma = {pipe.scheduler.init_noise_sigma}", flush=True)

    img_name, prompt = CASE
    z = encode_image(pipe, os.path.join(INPUTS, img_name)).float()

    # Part A: conditioning matrix subset
    rows, row_labels = [], []
    for cond in E2_SUBSET:
        phase, mag, dc = E2_CONDITIONS[cond]
        row = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            noise = torch.randn(1, 4, 128, 128, device="cuda")
            lat = condition_latent(noise, z, p=args.alpha, gamma=args.gamma,
                                   phase=phase, mag=mag, dc=dc)
            img = generate(pipe, prompt, lat, steps=args.steps, guidance=3.0)
            img.save(f"{out}/cond_{cond}_s{seed}.png")
            row.append(img)
            print(f"[e4] cond {cond} seed{seed} done", flush=True)
        rows.append(row)
        row_labels.append(cond)
    save_grid(rows, row_labels, [f"seed {s}" for s in range(args.seeds)],
              f"{out}/grid_conditioning.png")
    print("[e4] grid conditioning saved", flush=True)

    # Part B: colored noise subset
    rows, row_labels = [], []
    for beta in BETAS:
        row = []
        for seed in range(args.seeds):
            gen = torch.Generator("cuda").manual_seed(seed)
            lat = colored_noise((1, 4, 128, 128), beta, generator=gen)
            img = generate(pipe, prompt, lat, steps=args.steps, guidance=3.0)
            img.save(f"{out}/beta{beta}_s{seed}.png")
            row.append(img)
            print(f"[e4] beta {beta} seed{seed} done", flush=True)
        rows.append(row)
        row_labels.append(f"beta={beta}")
    save_grid(rows, row_labels, [f"seed {s}" for s in range(args.seeds)],
              f"{out}/grid_colored.png")
    print("[e4] grid colored saved", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.015)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    main(ap.parse_args())
