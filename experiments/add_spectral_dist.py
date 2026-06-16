"""Merge spectral-distance-to-real into an existing scores.json without re-running
the slow scorers (ImageReward etc.). Computes spectral_dist from the cached latents
against the SD3.5 real-image PSD reference (results/e10/sd35_real_latents.pt) and
writes it into each cond's entry. Usage:

    CN_RESULTS=/storage/.../results python add_spectral_dist.py e17
    CN_RESULTS=/storage/.../results python add_spectral_dist.py e17cb
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from fidelity_metrics import load_real_psd, spectral_dist_to_real, SD35_REAL_LATENTS


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return {"mean": m, "std": sd, "n": len(vals)}


def main(exp, n_bins=24):
    OUT = os.path.join(RESULTS, exp)
    scores = json.load(open(f"{OUT}/scores.json"))
    ref = load_real_psd(n_bins, path=SD35_REAL_LATENTS)
    assert ref is not None, "SD3.5 real ref missing"
    n = 0
    for pid, pd in scores.items():
        cdir = f"{OUT}/{pid}"
        for cond, entry in pd["conds"].items():
            sd = []
            for s in entry.get("seeds", []):
                lp = f"{cdir}/latents/{cond}_s{s}.pt"
                lat = torch.load(lp, weights_only=True) if os.path.exists(lp) else None
                sd.append(spectral_dist_to_real(lat, ref, n_bins) if lat is not None else None)
            entry.setdefault("per_seed", {})["spectral_dist"] = sd
            entry["spectral_dist"] = agg(sd)
            n += 1
    with open(f"{OUT}/scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(f"[add-spectral] {exp}: spectral_dist added to {n} cond entries "
          f"over {len(scores)} prompts", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "e17")
