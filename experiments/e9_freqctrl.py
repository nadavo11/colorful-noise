"""Selective high/low-frequency control via Spectral Band Normalization.

Plain band-norm clamps the latent's per-band power to a reference. Here we modify
that reference before clamping: scale only the HIGH bands (fine detail) or only the
LOW bands (large-scale structure) by a gain g, leaving the rest at the reference.
Because the clamp targets a level every step (it does not compound), this is a
stable knob:
  - high-band gain  -> more / less fine texture (moves hf_frac)
  - low-band gain   -> more / less structure / contrast

Focused sweep: portrait (broad texture) + landscape (strong structure), 2 seeds,
target in {low, high}, g in {0.7, 0.85, 1.0, 1.15, 1.3}. g=1.0 is plain band-norm
(generated once, shared as the baseline).

    python e9_freqctrl.py --seeds 2
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from spectral_ops import band_index_map, band_power
from e9_bandnorm_classes import CLASSES, image_metrics
from bandnorm import generate_bandnorm, modulate_reference, band_centers

OUT = os.path.join(RESULTS, "e9")
FC = os.path.join(OUT, "freqctrl")
PROMPTS = ["portrait", "landscape"]
GAINS = [0.55, 0.7, 0.85, 1.0, 1.15, 1.3]
TARGETS = ["low", "high"]


def agg(vals):
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return {"mean": m, "std": sd, "n": len(vals)}


def lowband_power(lat, idx_map, n_bins, cut_freq=0.25):
    """Fraction of latent spectral power in the low bands (structure proxy)."""
    F2 = torch.fft.fft2(lat.float()[0]).abs() ** 2
    bp = band_power(F2, idx_map, n_bins).mean(0)   # mean over channels
    centers = band_centers(n_bins)
    low = bp[centers < cut_freq].sum()
    return float(low / bp.sum().clamp(min=1e-12))


def main(args):
    from e7_flux_phase import load_flux
    prompts = dict(CLASSES)
    idx_map = band_index_map(128, 128, args.n_bins, "cuda")
    pipe = load_flux(args.mem)

    report = {"gains": GAINS, "targets": TARGETS, "seeds": args.seeds,
              "cut_freq": args.cut_freq, "prompts": {}}
    for pkey in PROMPTS:
        prompt = prompts[pkey]
        ref = torch.load(f"{OUT}/{pkey}/ref_psd.pt", weights_only=True)
        pdir = f"{FC}/{pkey}"
        os.makedirs(f"{pdir}/images", exist_ok=True)
        os.makedirs(f"{pdir}/latents", exist_ok=True)
        entry = {"prompt": prompt, "cells": {}}

        # (target, g) -> tag; g==1.0 collapses to a single shared baseline
        combos = [("baseline", 1.0, None)]
        for t in TARGETS:
            for g in GAINS:
                if g != 1.0:
                    combos.append((f"{t}_g{g}", g, t))

        for tag, g, target in combos:
            ms, lows = [], []
            for s in range(args.seeds):
                ip = f"{pdir}/images/{tag}_s{s}.png"
                lp = f"{pdir}/latents/{tag}_s{s}.pt"
                if os.path.exists(ip) and os.path.exists(lp):
                    img = Image.open(ip)
                    lat = torch.load(lp, weights_only=True)
                else:
                    use_ref = (ref if target is None
                               else modulate_reference(ref, target, g,
                                                       args.cut_freq, args.n_bins))
                    img, lat, _ = generate_bandnorm(pipe, prompt, s, use_ref,
                                                    args.cfg, args.steps,
                                                    args.n_bins)
                    img.save(ip)
                    torch.save(lat, lp)
                    print(f"[fc] {pkey} {tag} s{s} done", flush=True)
                ms.append(image_metrics(img))
                lows.append(lowband_power(lat.cuda(), idx_map, args.n_bins,
                                          args.cut_freq))
            entry["cells"][tag] = {
                "target": target, "g": g,
                "hf_frac": agg([v["hf_frac"] for v in ms]),
                "rms_contrast": agg([v["rms_contrast"] for v in ms]),
                "colorfulness": agg([v["colorfulness"] for v in ms]),
                "lowband_power": agg(lows),
            }
        # cfg3.5 full-guidance baseline for this prompt, computed from the
        # images + latents the main e9 run already saved (no generation needed).
        # Lets the site read prompts[k].cfg35 and show each freqctrl cell against it.
        cms, clows = [], []
        for s in range(args.seeds):
            cip = f"{OUT}/{pkey}/images/cfg3.5_s{s}.png"
            clp = f"{OUT}/{pkey}/latents/cfg3.5_s{s}.pt"
            if not (os.path.exists(cip) and os.path.exists(clp)):
                print(f"[fc] {pkey} cfg3.5 s{s} missing, skipping", flush=True)
                continue
            cimg = Image.open(cip)
            clat = torch.load(clp, weights_only=True)
            cms.append(image_metrics(cimg))
            clows.append(lowband_power(clat.cuda(), idx_map, args.n_bins,
                                       args.cut_freq))
        if cms:
            entry["cfg35"] = {
                "hf_frac": agg([v["hf_frac"] for v in cms]),
                "rms_contrast": agg([v["rms_contrast"] for v in cms]),
                "colorfulness": agg([v["colorfulness"] for v in cms]),
                "lowband_power": agg(clows),
            }
        report["prompts"][pkey] = entry
        print(f"[fc] {pkey} complete", flush=True)

    with open(f"{OUT}/freqctrl.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[fc] wrote {OUT}/freqctrl.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--cut_freq", type=float, default=0.25)
    ap.add_argument("--mem", default="bnb4", choices=["bnb4", "seq_offload"])
    main(ap.parse_args())
