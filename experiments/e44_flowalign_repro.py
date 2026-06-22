"""E44: reproduce FlowAlign on PIE-Bench (official SD3 code + official PnPInversion metrics),
then (later) score our spectral phase-clamp on the SAME loop.

Two parts (like e43):
  --part gen      GPU: load the OFFICIAL `SD3FlowAlign` sampler, edit each PIE-Bench image at a
                  given CFG (seed 123, NFE 33, 1024px), save edited+source PNGs (512) keyed by id,
                  plus a per-run meta.json (prompts, RLE mask, edit-type) so analyze is self-contained.
  --part analyze  Score with the OFFICIAL PnPInversion MetricsCalculator: structure_distance,
                  background (unedit-part) PSNR/LPIPS/MSE/SSIM, CLIP whole + edited-part. Aggregate
                  per edit-type + overall; emit the (edited-CLIP, bg-PSNR, struct-dist) point that
                  lands on FlowAlign's Fig-3a curve.

Data: cached HF++ PIE-Bench (`UB-CVML-Group/PIE_Bench_pp`); its `mask` field is the same RLE the
official `mask_decode` consumes (caveat: HF++ is a repackaging of the original 700).

Method = official, untouched: `diffusion.editing.sd3_edit.SD3FlowAlign` from the staged repo.
`--ours` flips on the ported spectral phase-clamp (added later; baseline run ignores it).

Run (cluster):  python e44_flowalign_repro.py --part gen,analyze --cfg 7.5 --n_per_type 0 --tag cfg75
                ( --n_per_type 0 = all 700; >0 = stratified subset for quick checks )
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

FA = os.environ.get("CN_FLOWALIGN", "/storage/malnick/flowalign_official")
PNP = os.environ.get("CN_PNP", "/storage/malnick/pnpinversion")
RESULTS = os.environ.get("CN_RESULTS",
                         "/storage/malnick/colorful-noise/experiments/results")
OUT = os.path.join(RESULTS, "e44")

# The 8 official PIE-Bench columns (PnPInversion defaults).
METRICS = ["structure_distance", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part",
           "ssim_unedit_part", "clip_similarity_source_image", "clip_similarity_target_image",
           "clip_similarity_target_image_edit_part"]


def _parse_rle(mask_field):
    """HF++ stores the RLE as a whitespace string like '0 262144'; original is a list of ints."""
    if isinstance(mask_field, str):
        return [int(x) for x in mask_field.split()]
    return [int(x) for x in mask_field]


def load_items(n_per_type):
    """PIE-Bench from the HF++ cache. n_per_type=0 -> all; >0 -> stride-sampled per category."""
    cache = os.environ.get("CN_PIEBENCH_CACHE", "/storage/malnick/datasets/pie_bench_hf")
    from datasets import get_dataset_config_names, load_dataset
    items = []
    for cfg in sorted(get_dataset_config_names("UB-CVML-Group/PIE_Bench_pp")):
        dd = load_dataset("UB-CVML-Group/PIE_Bench_pp", cfg, cache_dir=cache)
        ds = dd[next(iter(dd))]
        etype = cfg.split("_")[0]                     # leading index == editing_type_id
        idxs = range(len(ds))
        if n_per_type > 0:
            idxs = sorted(set(np.linspace(0, len(ds) - 1, n_per_type).astype(int).tolist()))
        for j in idxs:
            r = ds[j]
            items.append({
                "key": f"{cfg}_{r['id']}",
                "img": r["image"].convert("RGB"),
                "src_prompt": r["source_prompt"],
                "tgt_prompt": r["target_prompt"],
                "mask_rle": _parse_rle(r["mask"]),
                "edit_type": etype,
            })
    print(f"[e44] loaded {len(items)} PIE-Bench items "
          f"({'all' if n_per_type == 0 else f'{n_per_type}/type'})", flush=True)
    return items


# ---------------------------------------------------------------------------
# Part: gen  (GPU; official FlowAlign, untouched)
# ---------------------------------------------------------------------------

def run_gen(args):
    sys.path.insert(0, FA)
    from diffusion.editing.sd3_edit import get_editor
    from torchvision.utils import save_image

    d = os.path.join(OUT, args.tag)
    edited_dir, src_dir = os.path.join(d, "edited"), os.path.join(d, "source")
    os.makedirs(edited_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    items = load_items(args.n_per_type)
    meta = {"params": vars(args), "items": {}}

    sampler = get_editor("flowalign").to(device="cuda")

    for it in items:
        key = it["key"]
        meta["items"][key] = {"src_prompt": it["src_prompt"], "tgt_prompt": it["tgt_prompt"],
                              "mask_rle": it["mask_rle"], "edit_type": it["edit_type"]}
        outp = os.path.join(edited_dir, f"{key}.png")
        srcp = os.path.join(src_dir, f"{key}.png")
        if not os.path.exists(srcp):
            it["img"].resize((512, 512)).save(srcp)
        if os.path.exists(outp):
            continue

        # match official run_edit.py: load at img_shape, scale to [-1,1]
        from torchvision import transforms
        src = transforms.ToTensor()(it["img"].resize((args.img_shape, args.img_shape)))
        src = (src.unsqueeze(0) * 2.0 - 1.0)

        # brackets are PIE-Bench edit-word markers; the model prompt is the plain text
        src_prompt = it["src_prompt"].replace("[", "").replace("]", "")
        tgt_prompt = it["tgt_prompt"].replace("[", "").replace("]", "")

        with torch.no_grad():
            out = sampler.sample(src_img=src, src_prompt=src_prompt, tgt_prompt=tgt_prompt,
                                 null_prompt="", NFE=args.NFE, img_shape=(args.img_shape, args.img_shape),
                                 cfg_scale=args.cfg, n_start=args.NFE)
        # out is in [-1,1]; save_image(normalize=True) -> [0,1] PNG, then downsize to 512 for metrics
        tmp = os.path.join(edited_dir, f".{key}.tmp.png")
        save_image(out, tmp, normalize=True)
        Image.open(tmp).convert("RGB").resize((512, 512)).save(outp)
        os.remove(tmp)
        print(f"[e44] {args.tag}/{key} edited", flush=True)

    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[e44] gen done -> {d} ({len(items)} items)", flush=True)


# ---------------------------------------------------------------------------
# Part: analyze  (official PnPInversion metrics)
# ---------------------------------------------------------------------------

def run_analyze(args):
    sys.path.insert(0, PNP)
    from evaluation.matrics_calculator import MetricsCalculator
    from evaluation.evaluate import mask_decode, calculate_metric

    d = os.path.join(OUT, args.tag)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    mc = MetricsCalculator(args.device)

    per_type, overall = {}, {m: [] for m in METRICS}
    rows = {}
    for key, info in meta["items"].items():
        srcp = os.path.join(d, "source", f"{key}.png")
        tgtp = os.path.join(d, "edited", f"{key}.png")
        if not (os.path.exists(srcp) and os.path.exists(tgtp)):
            continue
        src_img, tgt_img = Image.open(srcp), Image.open(tgtp)
        mask = mask_decode(info["mask_rle"])[:, :, np.newaxis].repeat(3, axis=2)
        op = info["src_prompt"].replace("[", "").replace("]", "")
        ep = info["tgt_prompt"].replace("[", "").replace("]", "")
        res = {}
        for m in METRICS:
            v = calculate_metric(mc, m, src_img, tgt_img, mask, mask, op, ep)
            res[m] = None if (v == "nan" or v is None) else float(v)
        rows[key] = res
        et = info["edit_type"]
        per_type.setdefault(et, {m: [] for m in METRICS})
        for m in METRICS:
            if res[m] is not None:
                per_type[et][m].append(res[m])
                overall[m].append(res[m])

    def mean(xs):
        return None if not xs else sum(xs) / len(xs)

    summary = {
        "tag": args.tag, "n": len(rows), "params": meta["params"],
        "overall": {m: mean(overall[m]) for m in METRICS},
        "per_type": {et: {m: mean(v[m]) for m in METRICS} for et, v in per_type.items()},
    }
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    o = summary["overall"]
    print(f"\n[e44] === {args.tag} (n={summary['n']}, cfg={meta['params']['cfg']}) ===", flush=True)
    print(f"[e44]  structure_distance = {o['structure_distance']}", flush=True)
    print(f"[e44]  bg PSNR={o['psnr_unedit_part']}  LPIPS={o['lpips_unedit_part']}  "
          f"MSE={o['mse_unedit_part']}  SSIM={o['ssim_unedit_part']}", flush=True)
    print(f"[e44]  CLIP whole={o['clip_similarity_target_image']}  "
          f"edited={o['clip_similarity_target_image_edit_part']}", flush=True)
    print(f"[e44]  >> curve point: (edited-CLIP={o['clip_similarity_target_image_edit_part']}, "
          f"bg-PSNR={o['psnr_unedit_part']}, struct={o['structure_distance']})", flush=True)


# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(OUT, exist_ok=True)
    parts = [p.strip() for p in args.part.split(",") if p.strip()]
    if "gen" in parts:
        run_gen(args)
    if "analyze" in parts:
        run_analyze(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="gen,analyze")
    ap.add_argument("--cfg", type=float, default=7.5, help="CFG omega (sweep {5,7.5,10,13.5})")
    ap.add_argument("--NFE", type=int, default=33)
    ap.add_argument("--img_shape", type=int, default=1024)
    ap.add_argument("--n_per_type", type=int, default=0, help="0=all 700; >0=subset per category")
    ap.add_argument("--tag", default="cfg75")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ours", action="store_true", help="enable ported spectral clamp (added later)")
    main(ap.parse_args())
