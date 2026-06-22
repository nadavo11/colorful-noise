"""E46 Probe 0 -- does low-band SEED phase control generated layout? (Recipe B mechanism check)

Reconstruction setting (prompt = source prompt, no x0 carried at inference):
  - arm "white" : plain N(0,I) seed -> full SDXL generation.
  - arm "phaseB": seed magnitude = white, low-band [0,cut] phase = source latent's phase.
Hypothesis: if seed phase controls layout, phaseB reproduces the source layout
(DINO structure distance to source LOWER than white). KILL if no gap.

Run: python experiments/e46_seedphase.py   (writes results/e46/)
"""
import json
import os

import torch
from PIL import Image

import common
from latent_spectral_ops import phase_swap_2d
from struct_metrics import load_dino, structure_distance

CUT = 0.2          # low-band fraction (the "0.2" knob, interpretation (a))
SEEDS = [0, 1, 2, 3]
STEPS = 30
GUID = 5.0

# (file under colorful_noise/inputs, source prompt for reconstruction)
ITEMS = [
    ("cat_orange.png", "a photograph of an orange cat"),
    ("black_panther.png", "a black panther in the jungle"),
    ("savana.png", "a savanna landscape with trees and grass"),
]


def main():
    out = os.path.join(common.RESULTS, "e46")
    os.makedirs(out, exist_ok=True)
    pipe = common.load_pipe()
    dino = load_dino()

    rows, row_labels, recs = [], [], []
    for fname, prompt in ITEMS:
        path = os.path.join(common.INPUTS, fname)
        src = Image.open(path).convert("RGB").resize((1024, 1024))
        x0 = common.encode_image(pipe, path)  # (1,4,128,128) fp16 on cuda

        for arm in ("white", "phaseB"):
            row = [src]
            for s in SEEDS:
                g = torch.Generator(device="cuda").manual_seed(s)
                z = torch.randn(x0.shape, generator=g, device="cuda", dtype=x0.dtype)
                seed = z if arm == "white" else phase_swap_2d(x0, z, CUT, mag_from="B")
                img = common.generate(pipe, prompt, seed, steps=STEPS, guidance=GUID)
                d = structure_distance(dino, img, src)
                recs.append({"item": fname, "arm": arm, "seed": s, "struct_dist": d})
                print(f"{fname:18s} {arm:7s} seed{s}  struct_dist={d:.4f}")
                row.append(img)
            rows.append(row)
            row_labels.append(f"{fname.split('.')[0]}/{arm}")

    common.save_grid(rows, row_labels, ["source"] + [f"seed{s}" for s in SEEDS],
                     os.path.join(out, "grid.png"))
    json.dump(recs, open(os.path.join(out, "scores.json"), "w"), indent=2)

    # verdict summary: mean struct_dist per arm per item, and the gap (white - phaseB)
    print("\n=== per-item mean struct_dist (lower=closer to source layout) ===")
    for fname, _ in ITEMS:
        m = {a: sum(r["struct_dist"] for r in recs if r["item"] == fname and r["arm"] == a)
             / len(SEEDS) for a in ("white", "phaseB")}
        gap = m["white"] - m["phaseB"]
        print(f"{fname:18s} white={m['white']:.4f}  phaseB={m['phaseB']:.4f}  "
              f"gap={gap:+.4f}  {'<-- phaseB closer' if gap > 0 else ''}")
    print(f"\nartifacts: {out}/grid.png, {out}/scores.json")


if __name__ == "__main__":
    main()
