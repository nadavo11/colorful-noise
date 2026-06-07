"""E9: band-normalized generation as a method, across prompt classes.

E8 found that clamping cfg=3.5 latents to the cfg=1.0 per-step PSD reference
keeps the guidance look (power is a correlate, not the cause) -- and on the
cat prompt the band-normalized row reads as MORE finely detailed / less baked
than plain cfg=3.5. E9 tests whether that generalizes: 6 prompt classes
(photo and non-photo), 25 paired seeds each, plain cfg=3.5 vs band-norm,
plus a TRANSFER condition (band-norm driven by E8's cat-prompt reference)
that asks whether one universal reference suffices -- i.e. whether the
method is plug-and-play or needs per-prompt calibration.

Parts (--part):
  gen     -- per class: ref_seeds cfg=1.0 runs -> per-step band-power
             reference (cached, images double as cfg=1 baselines); `pairs`
             plain cfg=3.5 + `pairs` band-norm at identical seeds; 2 transfer
             seeds with the E8 cat reference. Image+latent cached per file,
             so a killed run resumes for free.
  analyze -- paired image metrics (Laplacian-variance sharpness, image-FFT
             high-frequency fraction, RMS contrast, Hasler-Suesstrunk
             colorfulness, mean saturation) + latent std/slope, per class;
             grids (first 8 seeds); cross-class reference-curve comparison;
             plots + report.json.

Needs results/e8/ref_psd.pt (the cat reference) for the transfer condition.
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_grid, RESULTS
from spectral_ops import (radial_psd, spectral_slope, radial_bins,
                          band_index_map)
from e7_flux_phase import load_flux
from e8_psd_clamp import RecordPSD, gen_with_cb, preflight as e8_preflight
from bandnorm import record_reference, generate_bandnorm

CLASSES = [
    ("animal",     "A photo of a red fox standing in a misty forest"),
    ("portrait",   "A portrait photo of an elderly fisherman with a weathered face"),
    ("landscape",  "A landscape photo of jagged mountains above a turquoise lake at sunrise"),
    ("urban_night","A photo of a rainy city street at night with glowing neon signs"),
    ("abstract",   "An abstract painting of swirling colors and interlocking geometric shapes"),
    ("watercolor", "A watercolor illustration of a sailing ship in a storm"),
]
CAT_REF = os.path.join(RESULTS, "e8", "ref_psd.pt")
METRICS = ["sharpness", "hf_frac", "rms_contrast", "colorfulness", "saturation"]


# ---------------------------------------------------------------------------
# Image metrics
# ---------------------------------------------------------------------------

def image_metrics(img):
    """Detail/contrast/color statistics of a PIL image (all torch cpu)."""
    x = torch.from_numpy(np.asarray(img.convert("RGB"))).float() / 255.0
    gray = x @ torch.tensor([0.299, 0.587, 0.114])
    lap_k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    lap = TF.conv2d(gray[None, None], lap_k[None, None], padding=1)
    F2 = torch.fft.fft2(gray).abs() ** 2
    F2[0, 0] = 0.0  # drop DC
    rr = radial_bins(*gray.shape, "cpu")
    rg = x[..., 0] - x[..., 1]
    yb = 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    mx, mn = x.max(-1).values, x.min(-1).values
    return {
        "sharpness": float(lap.var()),
        "hf_frac": float(F2[rr > 0.25].sum() / F2.sum()),
        "rms_contrast": float(gray.std()),
        "colorfulness": float(torch.sqrt(rg.std() ** 2 + yb.std() ** 2) +
                              0.3 * torch.sqrt(rg.mean() ** 2 + yb.mean() ** 2)),
        "saturation": float(((mx - mn) / (mx + 1e-8)).mean()),
    }


def lat_metrics(lat):
    centers, psd = radial_psd(lat.cuda())
    slopes = spectral_slope(centers, psd)
    return {"lat_std": float(lat.std()),
            "lat_slope": sum(slopes) / len(slopes)}


# ---------------------------------------------------------------------------
# Generation (cached per file -> killed runs resume for free)
# ---------------------------------------------------------------------------

def cached_gen(cdir, tag, fn):
    """Run fn() -> (img, lat) unless both files exist; save both."""
    ipath = f"{cdir}/images/{tag}.png"
    lpath = f"{cdir}/latents/{tag}.pt"
    if os.path.exists(ipath) and os.path.exists(lpath):
        print(f"[e9] {tag} (cached)", flush=True)
        return False
    img, lat = fn()
    img.save(ipath)
    torch.save(lat, lpath)
    return True


def run_gen(args, report, out):
    pipe = load_flux(args.mem)
    cat_ref = torch.load(CAT_REF, weights_only=True)
    for key, prompt in CLASSES[: args.num_classes]:
        cdir = f"{out}/{key}"
        os.makedirs(f"{cdir}/images", exist_ok=True)
        os.makedirs(f"{cdir}/latents", exist_ok=True)

        # reference (cfg=1.0) -- cached as ref_psd.pt + its images/latents
        ref_path = f"{cdir}/ref_psd.pt"
        ref_tags = [f"cfg{args.ref_cfg}_s{s}" for s in range(args.ref_seeds)]
        if os.path.exists(ref_path) and all(
                os.path.exists(f"{cdir}/images/{t}.png") for t in ref_tags):
            ref = torch.load(ref_path, weights_only=True)
            print(f"[e9] {key} reference (cached)", flush=True)
        else:
            ref, outs = record_reference(pipe, prompt, args.ref_seeds,
                                         args.ref_cfg, args.steps, args.n_bins)
            for t, (img, lat) in zip(ref_tags, outs):
                img.save(f"{cdir}/images/{t}.png")
                torch.save(lat, f"{cdir}/latents/{t}.pt")
            torch.save(ref, ref_path)
            print(f"[e9] {key} reference recorded "
                  f"(std[-1]={ref['std'][-1]:.3f})", flush=True)

        idx_map = band_index_map(128, 128, args.n_bins, "cuda")
        for s in range(args.pairs):
            # plain cfg=3.5
            cached_gen(cdir, f"cfg{args.cfg}_s{s}", lambda: gen_with_cb(
                pipe, prompt, s, args.cfg, args.steps,
                RecordPSD(idx_map, args.n_bins, args.steps)))
            # band-norm with the class's own reference
            cached_gen(cdir, f"bandnorm_s{s}", lambda: generate_bandnorm(
                pipe, prompt, s, ref, args.cfg, args.steps, args.n_bins)[:2])
        # transfer: band-norm driven by E8's cat reference
        for s in range(args.xfer_seeds):
            cached_gen(cdir, f"xfer_s{s}", lambda: generate_bandnorm(
                pipe, prompt, s, cat_ref, args.cfg, args.steps,
                args.n_bins)[:2])
        print(f"[e9] {key} generation complete", flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_set(cdir, cond, n):
    imgs, tags = [], []
    for s in range(n):
        p = f"{cdir}/images/{cond}_s{s}.png"
        if os.path.exists(p):
            imgs.append(Image.open(p))
            tags.append(f"{cond}_s{s}")
    return imgs, tags


def agg(vals):
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return {"mean": m, "std": sd, "n": len(vals)}


def run_analyze(args, report, out):
    cat_ref = torch.load(CAT_REF, weights_only=True)
    deltas = {m: {} for m in METRICS}  # metric -> class -> list of paired diffs
    fig_r, ax_r = plt.subplots(figsize=(7, 4))
    ax_r.plot(cat_ref["std"], "k--", lw=2, label="e8 cat (transfer ref)")

    for key, prompt in CLASSES[: args.num_classes]:
        cdir = f"{out}/{key}"
        ref = torch.load(f"{cdir}/ref_psd.pt", weights_only=True)
        ax_r.plot(ref["std"], label=key)
        std_dev = float(((ref["std"] - cat_ref["std"]).abs()
                         / cat_ref["std"]).max())
        band_dev = float(((ref["band"] - cat_ref["band"]).abs()
                          / cat_ref["band"].clamp(min=1e-12)).median())

        conds = {f"cfg{args.ref_cfg}": args.ref_seeds,
                 f"cfg{args.cfg}": args.pairs,
                 "bandnorm": args.pairs,
                 "xfer": args.xfer_seeds}
        centry = {"prompt": prompt,
                  "ref_vs_cat_std_maxrel": std_dev,
                  "ref_vs_cat_band_medrel": band_dev}
        per_cond_metrics = {}
        rows, row_labels = [], []
        for cond, n in conds.items():
            imgs, tags = load_set(cdir, cond, n)
            if not imgs:
                continue
            ms = [image_metrics(im) for im in imgs]
            ls = [lat_metrics(torch.load(f"{cdir}/latents/{t}.pt",
                                         weights_only=True)) for t in tags]
            per_cond_metrics[cond] = ms
            centry[cond] = {m: agg([v[m] for v in ms]) for m in METRICS}
            centry[cond]["lat_std"] = agg([v["lat_std"] for v in ls])
            centry[cond]["lat_slope"] = agg([v["lat_slope"] for v in ls])
            rows.append([im for im in imgs[:8]])
            row_labels.append(cond)
        # paired deltas bandnorm - plain at identical seeds
        plain_key = f"cfg{args.cfg}"
        npair = min(len(per_cond_metrics.get(plain_key, [])),
                    len(per_cond_metrics.get("bandnorm", [])))
        for m in METRICS:
            deltas[m][key] = [per_cond_metrics["bandnorm"][i][m]
                              - per_cond_metrics[plain_key][i][m]
                              for i in range(npair)]
            centry[f"delta_{m}"] = agg(deltas[m][key])
        report[f"class/{key}"] = centry
        save_grid(rows, row_labels,
                  [f"seed {s}" for s in range(8)], f"{out}/grid_{key}.png")
        print(f"[e9] {key}: d_sharp={centry['delta_sharpness']['mean']:+.4f} "
              f"d_hf={centry['delta_hf_frac']['mean']:+.4f} "
              f"d_contrast={centry['delta_rms_contrast']['mean']:+.4f} "
              f"d_color={centry['delta_colorfulness']['mean']:+.4f} "
              f"refdev(std)={std_dev:.3f}", flush=True)

    os.makedirs(f"{out}/plots", exist_ok=True)
    ax_r.set(xlabel="step", ylabel="latent std",
             title="cfg=1.0 reference std curves per prompt class")
    ax_r.legend(fontsize=8)
    fig_r.savefig(f"{out}/plots/ref_std_curves.png", dpi=120,
                  bbox_inches="tight")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 4))
    names = [k for k, _ in CLASSES[: args.num_classes]]
    for ax, m in zip(axes, METRICS):
        means = [sum(deltas[m][k]) / len(deltas[m][k]) for k in names]
        sems = [(agg(deltas[m][k])["std"] / math.sqrt(len(deltas[m][k])))
                for k in names]
        ax.bar(range(len(names)), means, yerr=sems, capsize=3)
        ax.axhline(0, c="k", lw=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Δ {m} (bandnorm − cfg{args.cfg})", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/plots/metrics_delta.png", dpi=120,
                bbox_inches="tight")
    plt.close("all")
    print("[e9] plots saved", flush=True)


def main(args):
    out = os.path.join(RESULTS, "e9")
    os.makedirs(out, exist_ok=True)
    assert os.path.exists(CAT_REF), f"need E8 reference at {CAT_REF}"
    e8_preflight(args)

    report = {"params": vars(args)}
    runners = {"gen": run_gen, "analyze": run_analyze}
    for part in filter(None, (p.strip() for p in args.part.split(","))):
        runners[part](args, report, out)

    path = f"{out}/report.json"
    if os.path.exists(path):
        with open(path) as f:
            merged = json.load(f)
        merged.update(report)
        report = merged
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e9] report -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="gen,analyze")
    ap.add_argument("--pairs", type=int, default=25)
    ap.add_argument("--ref_seeds", type=int, default=3)
    ap.add_argument("--xfer_seeds", type=int, default=2)
    ap.add_argument("--num_classes", type=int, default=len(CLASSES))
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--ref_cfg", type=float, default=1.0)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--mem", default="bnb4", choices=["bnb4", "seq_offload"])
    main(ap.parse_args())
