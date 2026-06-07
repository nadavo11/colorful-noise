"""E5: Conditioning-strength sweep for whitened (image-phase) conditioning.

E2 showed: phase carries layout, but at full white amplitude (mag_scale=1) the
low-band injection is so strong the output flips to flat-illustration style,
while the paper's gamma-crushed magnitudes (~0.15x white) stay photoreal.

Here: phase='image', mag='noise' (flat profile), dc='image', sweeping
mag_scale s. s=1 is E2's phase_dc; small s approaches the paper's strength but
with a *uniform* low-band level instead of the image's magnitude profile.
A 'paper' row is included for reference.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_pipe, encode_image, generate, save_grid, INPUTS, RESULTS
from spectral_ops import condition_latent

SCALES = [0.15, 0.3, 0.5, 1.0]
CASES = [
    ("cat_orange.png", "A photo of cat in the park"),
    ("savana.png", "A photo of giraffe and an elephant in the savanna"),
]


def main(args):
    out = os.path.join(RESULTS, "e5")
    os.makedirs(out, exist_ok=True)
    pipe = load_pipe()

    for img_name, prompt in CASES[: args.num_cases]:
        z = encode_image(pipe, os.path.join(INPUTS, img_name)).float()
        rows, row_labels = [], []
        for s in SCALES:
            row = []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                noise = torch.randn(1, 4, 128, 128, device="cuda")
                lat = condition_latent(noise, z, p=args.alpha, gamma=args.gamma,
                                       phase="image", mag="noise", dc="image",
                                       mag_scale=s)
                img = generate(pipe, prompt, lat, steps=args.steps)
                img.save(f"{out}/{img_name[:-4]}_s{s}_seed{seed}.png")
                row.append(img)
                print(f"[e5] {img_name} scale={s} seed{seed} done", flush=True)
            rows.append(row)
            row_labels.append(f"s={s}")
        # paper reference row
        row = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            noise = torch.randn(1, 4, 128, 128, device="cuda")
            lat = condition_latent(noise, z, p=args.alpha, gamma=args.gamma)
            img = generate(pipe, prompt, lat, steps=args.steps)
            img.save(f"{out}/{img_name[:-4]}_paper_seed{seed}.png")
            row.append(img)
            print(f"[e5] {img_name} paper seed{seed} done", flush=True)
        rows.append(row)
        row_labels.append("paper")
        save_grid(rows, row_labels, [f"seed {s}" for s in range(args.seeds)],
                  f"{out}/grid_{img_name[:-4]}.png")
        print(f"[e5] grid {img_name} saved", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.015)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--num_cases", type=int, default=2)
    main(ap.parse_args())
