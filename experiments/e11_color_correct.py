"""E11: cheap image-level color/contrast correction of band-norm (SBN) outputs.

E9 showed band-normalized generation clamps cfg=3.5 latent power back to the
cfg=1.0 per-step PSD reference. A measured side effect (results/e9/report.json):
SBN outputs are washed out vs plain cfg=3.5 -- ~-23% RMS contrast and slightly
lower colorfulness. They "lack color." E11 asks whether quick post-processing
on the *already-saved* SBN PNGs (no regeneration, no GPU) can restore color and
contrast toward the cfg=3.5 look WITHOUT discarding the detail gain that
motivated the method -- i.e. without re-baking. The detail-preservation criterion
is the contrast-invariant high-frequency fraction `hf_frac`: a good correction
lifts colorfulness/saturation/rms_contrast while keeping hf_frac near the SBN
baseline.

Methods (PIL + numpy only -- no cv2/skimage):
  sat        -- HSV-style saturation multiply (PIL ImageEnhance.Color), factor sweep
  contrast   -- global contrast (ImageEnhance.Contrast) + autocontrast variant
  lum_eq     -- equalize the Y channel only (YCbCr) -> no hue shift
  hist_match -- per-channel CDF match to the paired cfg=3.5 image (reference upper bound)

Outputs:
  results/e11/report.json          per-class/per-method mean metrics + deltas
  results/e11/{class}/{method}/    corrected PNGs
  results/e11/{class}/grids/*.png  SBN | corrected | cfg3.5 contact sheets
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid
from e9_bandnorm_classes import CLASSES, METRICS, image_metrics

OUT = os.path.join(RESULTS, "e11")
E9 = os.path.join(RESULTS, "e9")


# ---------------------------------------------------------------------------
# Correction methods. Each maps (sbn_img, ref_img) -> corrected PIL image.
# `ref_img` is the paired cfg=3.5 image (only hist_match uses it).
# ---------------------------------------------------------------------------

def correct_sat(img, ref, f):
    return ImageEnhance.Color(img).enhance(f)


def correct_contrast(img, ref, f):
    return ImageEnhance.Contrast(img).enhance(f)


def correct_autocontrast(img, ref):
    return ImageOps.autocontrast(img.convert("RGB"))


def correct_lum_eq(img, ref):
    """Histogram-equalize luminance only -> brighter/punchier without hue shift."""
    y, cb, cr = img.convert("YCbCr").split()
    return Image.merge("YCbCr", (ImageOps.equalize(y), cb, cr)).convert("RGB")


def correct_hist_match(img, ref):
    """Match each RGB channel's CDF of `img` to the paired cfg=3.5 `ref`."""
    s = np.asarray(img.convert("RGB"))
    r = np.asarray(ref.convert("RGB"))
    out = np.empty_like(s)
    for c in range(3):
        sv, sidx, scnt = np.unique(s[..., c].ravel(),
                                   return_inverse=True, return_counts=True)
        rv, rcnt = np.unique(r[..., c].ravel(), return_counts=True)
        sq = np.cumsum(scnt) / s[..., c].size
        rq = np.cumsum(rcnt) / r[..., c].size
        mapped = np.interp(sq, rq, rv)
        out[..., c] = mapped[sidx].reshape(s[..., c].shape)
    return Image.fromarray(out.astype(np.uint8))


def build_methods(args):
    """Return {method_name: fn(sbn_img, ref_img) -> corrected img}."""
    m = {}
    if "sat" in args.methods:
        for f in args.sat_factors:
            m[f"sat{f}"] = lambda img, ref, f=f: correct_sat(img, ref, f)
    if "contrast" in args.methods:
        for f in args.contrast_factors:
            m[f"contrast{f}"] = lambda img, ref, f=f: correct_contrast(img, ref, f)
        m["autocontrast"] = correct_autocontrast
    if "lum_eq" in args.methods:
        m["lum_eq"] = correct_lum_eq
    if "hist_match" in args.methods:
        m["hist_match"] = correct_hist_match
    return m


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def discover_seeds(cdir):
    imgs = os.path.join(cdir, "images")
    seeds = []
    for fn in os.listdir(imgs):
        if fn.startswith("bandnorm_s") and fn.endswith(".png"):
            seeds.append(int(fn[len("bandnorm_s"):-len(".png")]))
    return sorted(seeds)


def mean_metrics(rows):
    """rows: list of metric dicts -> dict of per-metric mean."""
    return {k: float(np.mean([r[k] for r in rows])) for k in METRICS}


def run(args):
    methods = build_methods(args)
    classes = [(k, p) for k, p in CLASSES if k in args.classes]
    report = {}
    print(f"[e11] methods: {list(methods)}", flush=True)

    for key, _prompt in classes:
        src = os.path.join(E9, key)
        seeds = discover_seeds(src)
        if args.seeds is not None:
            seeds = [s for s in seeds if s in args.seeds]
        if not seeds:
            print(f"[e11] {key}: no seeds, skip", flush=True)
            continue

        # per-method accumulators of per-image metric dicts
        acc = {m: [] for m in methods}
        sbn_rows, cfg_rows = [], []
        # cache corrected images of the first grid_n seeds for contact sheets
        grid_cache = {m: [] for m in methods}
        grid_sbn, grid_cfg = [], []

        for s in seeds:
            sbn = Image.open(f"{src}/images/bandnorm_s{s}.png").convert("RGB")
            ref = Image.open(f"{src}/images/cfg3.5_s{s}.png").convert("RGB")
            sbn_rows.append(image_metrics(sbn))
            cfg_rows.append(image_metrics(ref))
            for m, fn in methods.items():
                cor = fn(sbn, ref)
                acc[m].append(image_metrics(cor))
                outdir = f"{OUT}/{key}/{m}"
                os.makedirs(outdir, exist_ok=True)
                cor.save(f"{outdir}/{m}_s{s}.png")
                if len(grid_cache[m]) < args.grid_n:
                    grid_cache[m].append(cor)
            if len(grid_sbn) < args.grid_n:
                grid_sbn.append(sbn)
                grid_cfg.append(ref)

        sbn_mean = mean_metrics(sbn_rows)
        cfg_mean = mean_metrics(cfg_rows)
        cres = {
            "n_seeds": len(seeds),
            "baseline_sbn": sbn_mean,
            "target_cfg3.5": cfg_mean,
            "methods": {},
        }
        for m in methods:
            mm = mean_metrics(acc[m])
            cres["methods"][m] = {
                "metrics": mm,
                # how far each metric moved from SBN toward cfg=3.5
                "delta_vs_sbn": {k: mm[k] - sbn_mean[k] for k in METRICS},
                "delta_vs_cfg3.5": {k: mm[k] - cfg_mean[k] for k in METRICS},
            }
            # contact sheet: SBN | corrected | cfg3.5
            save_grid(
                [grid_sbn, grid_cache[m], grid_cfg],
                ["SBN", m, "cfg3.5"],
                [f"s{s}" for s in seeds[:args.grid_n]],
                f"{OUT}/{key}/grids/{m}.png",
            )
        report[key] = cres
        print(f"[e11] {key}: {len(seeds)} seeds, "
              f"sbn color={sbn_mean['colorfulness']:.4f} "
              f"contrast={sbn_mean['rms_contrast']:.4f} "
              f"hf={sbn_mean['hf_frac']:.4f}", flush=True)
        for m in methods:
            d = cres["methods"][m]["delta_vs_sbn"]
            print(f"        {m:14s} dcolor={d['colorfulness']:+.4f} "
                  f"dcontrast={d['rms_contrast']:+.4f} "
                  f"dhf={d['hf_frac']:+.4f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[e11] wrote {OUT}/report.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=[k for k, _ in CLASSES])
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="subset of seeds (default: all found on disk)")
    ap.add_argument("--methods", nargs="+",
                    default=["sat", "contrast", "lum_eq", "hist_match"])
    ap.add_argument("--sat-factors", nargs="+", type=float, default=[1.2, 1.4])
    ap.add_argument("--contrast-factors", nargs="+", type=float, default=[1.2])
    ap.add_argument("--grid-n", type=int, default=8)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
