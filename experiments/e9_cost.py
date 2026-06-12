"""Measure SBN's per-step wall-clock overhead vs an unmodified sampling loop.

Times the interval between step callbacks for (a) a plain full-guidance loop with a
no-op hook and (b) the same loop with the SBN band-correction hook. The difference
is the per-step cost of SBN (one FFT + IFFT + per-band reduction on the latent).

    python e9_cost.py --seeds 2
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from spectral_ops import band_index_map
from e7_flux_phase import load_flux
from e8_psd_clamp import ClampPSD, gen_with_cb
from e9_bandnorm_classes import CLASSES

OUT = os.path.join(RESULTS, "e9")


class Timed:
    """Wrap a step hook; record perf_counter at each call."""
    def __init__(self, inner):
        self.inner, self.t, self.last = inner, [], None

    def __call__(self, p, i, t, kw):
        self.t.append(time.perf_counter())
        out = self.inner(p, i, t, kw) if self.inner else {}
        self.last = out.get("latents", kw["latents"])
        return out

    def per_step_ms(self):
        d = [(b - a) * 1000 for a, b in zip(self.t[:-1], self.t[1:])]
        return sum(d) / len(d) if d else float("nan")


def mean_per_step(pipe, prompt, seed, cfg, steps, inner):
    cb = Timed(inner)
    gen_with_cb(pipe, prompt, seed, cfg, steps, cb)
    return cb.per_step_ms()


def main(args):
    prompt = dict(CLASSES)["animal"]
    ref = torch.load(f"{OUT}/animal/ref_psd.pt", weights_only=True)
    idx_map = band_index_map(128, 128, args.n_bins, "cuda")
    pipe = load_flux(args.mem)

    plain, clamp = [], []
    for s in range(args.seeds):
        plain.append(mean_per_step(pipe, prompt, s, args.cfg, args.steps, None))
        clamp.append(mean_per_step(pipe, prompt, s, args.cfg, args.steps,
                                   ClampPSD("band", ref, idx_map, args.n_bins)))
        print(f"[cost] seed {s}: plain={plain[-1]:.1f}ms clamp={clamp[-1]:.1f}ms",
              flush=True)
    p = sum(plain) / len(plain)
    c = sum(clamp) / len(clamp)
    rep = {"plain_ms_per_step": p, "clamp_ms_per_step": c,
           "overhead_pct": (c - p) / p * 100, "steps": args.steps,
           "seeds": args.seeds}
    with open(f"{OUT}/cost.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[cost] plain={p:.1f}ms clamp={c:.1f}ms overhead=+{rep['overhead_pct']:.1f}%"
          f" -> {OUT}/cost.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--mem", default="bnb4", choices=["bnb4", "seq_offload"])
    main(ap.parse_args())
