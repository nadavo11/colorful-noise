"""E46 Probe 2 -- FULL-band phase transplant seed (user's recipe).

Seed = ifft( |FFT(randn)| * exp(i * angle(FFT(x0_latent))) ): take the source latent's
phase at EVERY frequency, keep the random seed's amplitude, generate with target prompt.
== phase_swap_2d(x0, z, cut=1.0, mag_from="B"). Contrast arms: vanilla SDEdit, and the old
low-band (cut=0.2) seed.

Run: python experiments/e46_probe2.py   (writes results/e46_p2/)
"""
import json
import os

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

import common
from latent_spectral_ops import phase_swap_2d
from struct_metrics import load_dino, structure_distance, clip_directional, load_clip
from e46_probe1 import make_sources, sdedit, STEPS, GUID, STRENGTH

SEEDS = [0, 1]


def main():
    out = os.path.join(common.RESULTS, "e46_p2")
    os.makedirs(out, exist_ok=True)
    pipe = common.load_pipe()
    img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
    img2img.set_progress_bar_config(disable=True)
    dino, clip = load_dino(), load_clip()
    items = make_sources(pipe)

    rows, row_labels, recs = [], [], []
    for it in items:
        src = it["img"]
        x0 = common.encode_img_sdxl(pipe, src)
        firsts = {}
        for s in SEEDS:
            g = torch.Generator(device="cuda").manual_seed(s)
            z = torch.randn(x0.shape, generator=g, device="cuda", dtype=x0.dtype)
            seed_full = phase_swap_2d(x0, z, 1.0, mag_from="B")  # ALL-freq image phase + random amp
            seed_low = phase_swap_2d(x0, z, 0.2, mag_from="B")   # low-band only (Probe 1 "B")
            outs = {
                "vanilla": sdedit(img2img, src, it["tgt"], STRENGTH, STEPS, GUID,
                                  torch.Generator("cuda").manual_seed(s)),
                "Cfull": common.generate(pipe, it["tgt"], seed_full, steps=STEPS, guidance=GUID),
                "Blow": common.generate(pipe, it["tgt"], seed_low, steps=STEPS, guidance=GUID),
            }
            for a, im in outs.items():
                recs.append({"key": it["key"], "arm": a, "seed": s,
                             "struct_dist": structure_distance(dino, im, src),
                             "clip_dir": clip_directional(clip, src, im, it["src"], it["tgt"])})
                if s == SEEDS[0]:
                    firsts[a] = im
        rows.append([src, firsts["vanilla"], firsts["Cfull"], firsts["Blow"]])
        row_labels.append(it["key"][:22])
        print(f"done {it['key']}", flush=True)

    common.save_grid(rows, row_labels, ["source", "vanilla", "Cfull", "Blow"],
                     os.path.join(out, "grid.png"))
    json.dump(recs, open(os.path.join(out, "scores.json"), "w"), indent=2)

    keys = sorted({r["key"] for r in recs})
    def m(a, k, key=None):
        v = [r[k] for r in recs if r["arm"] == a and (key is None or r["key"] == key)]
        return sum(v) / len(v)
    print("\n=== means (struct down / clip_dir up) ===")
    for a in ("vanilla", "Cfull", "Blow"):
        print(f"  {a:8s} struct={m(a,'struct_dist'):.4f}  clip_dir={m(a,'clip_dir'):.4f}")
    print("\n=== chair (change_material) ===")
    for a in ("vanilla", "Cfull", "Blow"):
        print(f"  {a:8s} struct={m(a,'struct_dist','change_material'):.4f}  "
              f"clip_dir={m(a,'clip_dir','change_material'):.4f}")
    print(f"\nartifacts: {out}/grid.png, {out}/scores.json")


if __name__ == "__main__":
    main()
