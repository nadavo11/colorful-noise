"""E2: Where does the conditioning signal live -- phase, magnitude, or DC?

For each input image x prompt, generates one image per E2 condition per seed,
plus a PSD/whiteness report of every initial latent used.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_pipe, encode_image, generate, save_grid, INPUTS, RESULTS
from spectral_ops import condition_latent, whiteness, E2_CONDITIONS

CASES = [  # (input image, prompt) -- the paper's own pairs
    ("cat_orange.png", "A photo of cat in the park"),
    ("savana.png", "A photo of giraffe and an elephant in the savanna"),
]


def main(args):
    out = os.path.join(RESULTS, "e2")
    os.makedirs(out, exist_ok=True)
    pipe = load_pipe()
    report = {}

    for img_name, prompt in CASES[: args.num_cases]:
        z = encode_image(pipe, os.path.join(INPUTS, img_name)).float()
        rows, row_labels = [], []
        for cond, (phase, mag, dc) in E2_CONDITIONS.items():
            row = []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                noise = torch.randn(1, 4, 128, 128, device="cuda")
                lat = condition_latent(noise, z, p=args.alpha, gamma=args.gamma,
                                       phase=phase, mag=mag, dc=dc)
                if seed == 0:
                    report[f"{img_name}/{cond}"] = {
                        "whiteness": whiteness(lat),
                        "std": float(lat.std()),
                        "channel_means": lat.mean(dim=(2, 3)).flatten().tolist(),
                    }
                img = generate(pipe, prompt, lat, steps=args.steps)
                img.save(f"{out}/{img_name[:-4]}_{cond}_s{seed}.png")
                row.append(img)
                print(f"[e2] {img_name} {cond} seed{seed} done", flush=True)
            rows.append(row)
            row_labels.append(cond)
        grid = save_grid(rows, row_labels,
                         [f"seed {s}" for s in range(args.seeds)],
                         f"{out}/grid_{img_name[:-4]}.png")
        print(f"[e2] grid -> {grid}", flush=True)

    with open(f"{out}/latent_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.015)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--num_cases", type=int, default=2)
    main(ap.parse_args())
