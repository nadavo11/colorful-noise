"""Universal (prompt-agnostic) band reference for Spectral Band Normalization.

Per-prompt band-norm needs a short cfg=1.0 reference pass for *each* prompt. This
asks whether one GENERAL band-power profile -- the average across all prompt
classes -- works well enough that the per-prompt pass can be skipped entirely
(amortizing its cost to ~0).

Two parts:
  build   -- average the 6 per-class ref_psd.pt into universal_ref.pt; measure how
             far each class's own reference sits from the universal one (a single
             "generality" number + per-class), -> universal.json.
  gen     -- generate a demo set with the universal reference (uninorm_s*.png) so
             the page can compare universal- vs per-prompt-reference band-norm.

    python e9_universal_ref.py --part build,gen --seeds 5
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e9_bandnorm_classes import CLASSES, image_metrics
from bandnorm import generate_bandnorm

OUT = os.path.join(RESULTS, "e9")
UNI = os.path.join(OUT, "universal_ref.pt")


def agg(vals):
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return {"mean": m, "std": sd, "n": len(vals)}


def build():
    refs = {k: torch.load(f"{OUT}/{k}/ref_psd.pt", weights_only=True)
            for k, _ in CLASSES}
    keys = list(refs)
    uni = {
        "band": torch.stack([refs[k]["band"] for k in keys]).mean(0),
        "std": torch.stack([refs[k]["std"] for k in keys]).mean(0),
        "total": torch.stack([refs[k]["total"] for k in keys]).mean(0),
    }
    torch.save(uni, UNI)

    # How far each class's own reference sits from the universal profile.
    # Mean (not median) relative band-power deviation captures the spread; the
    # final latent std is the headline "power level" each reference encodes.
    per_class = {}
    for k in keys:
        rel = ((refs[k]["band"] - uni["band"]).abs()
               / uni["band"].clamp(min=1e-12))
        per_class[k] = {
            "own_std_final": float(refs[k]["std"][-1]),
            "mean_reldev": float(rel.mean()),
            "p90_reldev": float(rel.flatten().quantile(0.9)),
        }
    overall_mean = sum(v["mean_reldev"] for v in per_class.values()) / len(keys)
    report = {"per_class": per_class,
              "overall_mean_reldev": overall_mean,
              "uni_std_final": float(uni["std"][-1])}
    with open(f"{OUT}/universal.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[uni] built universal_ref.pt; uni std={float(uni['std'][-1]):.3f}; "
          f"overall mean reldev={overall_mean:.3f}", flush=True)
    for k in keys:
        print(f"[uni]   {k}: own_std={per_class[k]['own_std_final']:.3f} "
              f"reldev={per_class[k]['mean_reldev']:.3f}", flush=True)
    return report


def gen(args):
    from e7_flux_phase import load_flux
    uni = torch.load(UNI, weights_only=True)
    pipe = load_flux(args.mem)
    metrics = {}
    for key, prompt in CLASSES:
        cdir = f"{OUT}/{key}"
        os.makedirs(f"{cdir}/images", exist_ok=True)
        os.makedirs(f"{cdir}/latents", exist_ok=True)
        ms = []
        for s in range(args.seeds):
            ip = f"{cdir}/images/uninorm_s{s}.png"
            lp = f"{cdir}/latents/uninorm_s{s}.pt"
            if os.path.exists(ip) and os.path.exists(lp):
                ms.append(image_metrics(Image.open(ip)))
                continue
            img, lat, _ = generate_bandnorm(pipe, prompt, s, uni, args.cfg,
                                            args.steps, args.n_bins)
            img.save(ip)
            torch.save(lat, lp)
            ms.append(image_metrics(img))
            print(f"[uni] {key} uninorm_s{s} done", flush=True)
        metrics[key] = {m: agg([v[m] for v in ms])
                        for m in ["hf_frac", "rms_contrast", "colorfulness",
                                  "saturation", "sharpness"]}
    # merge metrics into universal.json
    path = f"{OUT}/universal.json"
    rep = json.load(open(path)) if os.path.exists(path) else {}
    rep["uninorm_metrics"] = metrics
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print("[uni] gen complete", flush=True)


def main(args):
    for part in filter(None, (p.strip() for p in args.part.split(","))):
        if part == "build":
            build()
        elif part == "gen":
            gen(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="build,gen")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--mem", default="bnb4", choices=["bnb4", "seq_offload"])
    main(ap.parse_args())
