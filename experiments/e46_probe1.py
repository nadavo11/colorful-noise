"""E46 Probe 1 -- editing frontier: seed-phase recipes vs vanilla SDEdit (SDXL, PIE-Bench).

Three arms, target prompt, cut=0.2 low-band source phase:
  vanilla : SDEdit (SDXL img2img) @ strength 0.8, plain white noise.   [baseline]
  A       : same SDEdit, but the injected noise is phase-structured
            (white magnitude + source low-band phase). x0 term still carried.
  B       : full gen from a structured seed (no x0 carried). Recipe B from Probe 0.

Frontier: DINO structure distance to source (down) x CLIP-directional editability (up).
A and B each must Pareto-beat vanilla on >=3 items to survive.

Run: python experiments/e46_probe1.py   (writes results/e46_p1/)
"""
import json
import os

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
import diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img as _img2img_mod

import common
from latent_spectral_ops import phase_swap_2d
from struct_metrics import load_dino, structure_distance, clip_directional, load_clip

CUT = 0.2
SEEDS = [0, 1]
STEPS = 30
GUID = 5.0
STRENGTH = 0.8
N_PER_TYPE = 1


# Local proxy for PIE-Bench (datasets lib not installed locally; official cache lives on
# the cluster). Sources are SDXL-generated so phase is natural/on-manifold; edits cover the
# PIE edit-type families (object/color/content/background/material/style). Run the official
# PIE-Bench frontier on the cluster to confirm.
SOURCES = [
    ("change_object", "a photo of a golden retriever sitting on grass",
     "a photo of a grey cat sitting on grass"),
    ("change_color", "a red sports car parked on a city street",
     "a blue sports car parked on a city street"),
    ("change_content", "a bowl of red apples on a wooden table",
     "a bowl of oranges on a wooden table"),
    ("change_background", "a cottage beside a calm lake in summer",
     "a cottage beside a calm lake in heavy winter snow"),
    ("change_attribute", "a cup of black coffee on a white saucer",
     "a cup of green tea on a white saucer"),
    ("change_material", "a wooden chair in an empty bright room",
     "a shiny metal chair in an empty bright room"),
    ("change_style", "a mountain landscape at sunset, photograph",
     "a mountain landscape at sunset, oil painting"),
    ("change_object2", "a single red rose in a glass vase",
     "a single yellow sunflower in a glass vase"),
]
SRC_SEED = 100


def make_sources(pipe):
    items = []
    for key, src_prompt, tgt_prompt in SOURCES:
        g = torch.Generator(device="cuda").manual_seed(SRC_SEED)
        z = torch.randn((1, 4, 128, 128), generator=g, device="cuda", dtype=pipe.dtype)
        img = common.generate(pipe, src_prompt, z, steps=STEPS, guidance=GUID)
        items.append({"key": key, "img": img, "src": src_prompt, "tgt": tgt_prompt})
    print(f"generated {len(items)} source images", flush=True)
    return items


def sdedit(img2img, src_img, prompt, strength, steps, guidance, gen, structured_noise=None):
    """SDXL img2img SDEdit. If structured_noise given, inject it as the forward noise
    (Recipe A) by patching the pipeline's randn_tensor for this call."""
    if structured_noise is None:
        return img2img(prompt=prompt, image=src_img, strength=strength,
                       num_inference_steps=steps, guidance_scale=guidance,
                       generator=gen).images[0]
    orig = _img2img_mod.randn_tensor
    _img2img_mod.randn_tensor = lambda shape, **kw: structured_noise.to(kw.get("device", "cuda")).type(kw.get("dtype", structured_noise.dtype))
    try:
        return img2img(prompt=prompt, image=src_img, strength=strength,
                       num_inference_steps=steps, guidance_scale=guidance,
                       generator=gen).images[0]
    finally:
        _img2img_mod.randn_tensor = orig


def main():
    out = os.path.join(common.RESULTS, "e46_p1")
    os.makedirs(out, exist_ok=True)
    pipe = common.load_pipe()
    img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
    img2img.set_progress_bar_config(disable=True)
    dino, clip = load_dino(), load_clip()
    items = make_sources(pipe)

    rows, row_labels, recs = [], [], []
    for it in items:
        src = it["img"]
        x0 = common.encode_img_sdxl(pipe, src)  # (1,4,128,128) fp16 cuda
        row_imgs = {a: [] for a in ("vanilla", "A", "B")}
        for s in SEEDS:
            g = torch.Generator(device="cuda").manual_seed(s)
            z = torch.randn(x0.shape, generator=g, device="cuda", dtype=x0.dtype)
            phase_noise = phase_swap_2d(x0, z, CUT, mag_from="B")  # white mag + src low-band phase

            outs = {
                "vanilla": sdedit(img2img, src, it["tgt"], STRENGTH, STEPS, GUID,
                                  torch.Generator("cuda").manual_seed(s)),
                "A": sdedit(img2img, src, it["tgt"], STRENGTH, STEPS, GUID,
                            torch.Generator("cuda").manual_seed(s), structured_noise=phase_noise),
                "B": common.generate(pipe, it["tgt"], phase_noise, steps=STEPS, guidance=GUID),
            }
            for a, im in outs.items():
                sd = structure_distance(dino, im, src)
                cd = clip_directional(clip, src, im, it["src"], it["tgt"])
                recs.append({"key": it["key"], "arm": a, "seed": s,
                             "struct_dist": sd, "clip_dir": cd})
                row_imgs[a].append(im)
            print(f"{it['key'][:28]:28s} seed{s} " +
                  " ".join(f"{a}(sd={structure_distance(dino, outs[a], src):.3f})" for a in outs),
                  flush=True)
        rows.append([src] + [row_imgs[a][0] for a in ("vanilla", "A", "B")])
        row_labels.append(it["key"][:22])

    common.save_grid(rows, row_labels, ["source", "vanilla", "A", "B"],
                     os.path.join(out, "grid.png"))
    json.dump(recs, open(os.path.join(out, "scores.json"), "w"), indent=2)

    # frontier verdict: per-item seed-mean, count Pareto wins over vanilla
    print("\n=== per-arm means (struct down / clip_dir up) ===")
    keys = sorted({r["key"] for r in recs})
    def mean(arm, k, key=None):
        v = [r[k] for r in recs if r["arm"] == arm and (key is None or r["key"] == key)]
        return sum(v) / len(v)
    for a in ("vanilla", "A", "B"):
        print(f"  {a:7s}  struct={mean(a,'struct_dist'):.4f}  clip_dir={mean(a,'clip_dir'):.4f}")
    for a in ("A", "B"):
        wins = sum(1 for k in keys
                   if mean(a, "struct_dist", k) < mean("vanilla", "struct_dist", k)
                   and mean(a, "clip_dir", k) > mean("vanilla", "clip_dir", k))
        print(f"  {a} Pareto-beats vanilla (struct lower AND clip higher) on {wins}/{len(keys)} items")
    print(f"\nartifacts: {out}/grid.png, {out}/scores.json")


if __name__ == "__main__":
    main()
