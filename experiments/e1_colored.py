"""E1: Text-to-image generation from pure colored noise (no source image).

Sweeps PSD ~ f^beta for beta in {-2,-1,0,+1,+2} (red, pink, white, blue,
violet), generates per prompt x seed, and measures the radial PSD of the
*generated images* to see whether the model restores a natural 1/f spectrum
regardless of the input spectrum.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_pipe, generate, save_grid, RESULTS
from spectral_ops import colored_noise, radial_psd, whiteness

BETAS = [-2, -1, 0, 1, 2]
NAMES = {-2: "red(-2)", -1: "pink(-1)", 0: "white(0)", 1: "blue(+1)", 2: "violet(+2)"}
PROMPTS = [
    "A photo of cat in the park",
    "A vibrant city street at sunset, photorealistic",
    "A bowl of fruit on a wooden table, studio lighting",
]


def image_psd(pil_img):
    x = torch.from_numpy(np.array(pil_img.convert("L"))).float()[None, None] / 255.0
    return radial_psd(x)


def main(args):
    out = os.path.join(RESULTS, "e1")
    os.makedirs(out, exist_ok=True)
    pipe = load_pipe()
    report, out_psds = {}, {}

    for prompt_i, prompt in enumerate(PROMPTS[: args.num_prompts]):
        rows, row_labels = [], []
        for beta in BETAS:
            row = []
            for seed in range(args.seeds):
                gen = torch.Generator("cuda").manual_seed(seed)
                lat = colored_noise((1, 4, 128, 128), beta, generator=gen,
                                    normalize=args.normalize)
                if seed == 0 and prompt_i == 0:
                    report[NAMES[beta]] = {"whiteness": whiteness(lat),
                                           "std": float(lat.std())}
                img = generate(pipe, prompt, lat, steps=args.steps)
                img.save(f"{out}/p{prompt_i}_beta{beta}_s{seed}.png")
                row.append(img)
                c, psd = image_psd(img)
                out_psds.setdefault(beta, []).append(psd.mean(0))
                print(f"[e1] p{prompt_i} {NAMES[beta]} seed{seed} done", flush=True)
            rows.append(row)
            row_labels.append(NAMES[beta])
        save_grid(rows, row_labels, [f"seed {s}" for s in range(args.seeds)],
                  f"{out}/grid_prompt{prompt_i}.png")
        print(f"[e1] grid for prompt {prompt_i} saved", flush=True)

    # PSD of generated images, grouped by input beta
    fig, ax = plt.subplots(figsize=(7, 5))
    for beta in BETAS:
        mean_psd = torch.stack(out_psds[beta]).mean(0)
        ax.loglog(c[1:], mean_psd[1:], label=f"input {NAMES[beta]}")
    ax.set(title="PSD of GENERATED images by input-noise color",
           xlabel="radial frequency", ylabel="power")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/generated_image_psd.png", dpi=130)

    with open(f"{out}/latent_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--num_prompts", type=int, default=3)
    ap.add_argument("--normalize", type=str, default="global", choices=["global", "none"])
    main(ap.parse_args())
