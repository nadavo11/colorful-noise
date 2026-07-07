"""Analysis core for HorizonCache-PC (E57).

Two questions, honestly:
  1. Does the cached-endpoint correction beat PLAIN HorizonCache at matched speed
     (esp. in the 2.8-3.2x overshoot band)?  -> pc_minus_plain
  2. Is the curvature score a real signal for a bad jump?  -> curvature_signal

Reuses the E56 matched-achieved-speedup protocol (per-image paired, interpolate the
reference frontier at each image's own speedup, percentile bootstrap).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd


def boot_ci(vals, n_boot=5000, seed=0):
    a = np.asarray(vals, float)
    if len(a) < 2:
        return (None, None)
    rng = np.random.RandomState(seed)
    m = a[rng.randint(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def _frontier_points_by_key(df, templates, metric):
    """{key: [(speedup, metric), ...]} over the reference method templates."""
    out = collections.defaultdict(list)
    for t in templates:
        sub = df[df.method == t]
        for _, r in sub.iterrows():
            out[r["key"]].append((r["compute_speedup"], r[metric]))
    return out


def matched_delta_vs(df, method, ref_templates, tau_grid, metric="psnr", higher=True):
    """Per-image delta of `method` vs the reference frontier interpolated at each image's
    own achieved speedup. delta>0 => method better (LPIPS sign-flipped)."""
    ref = _frontier_points_by_key(df, ref_templates, metric)
    hor = df[df.method == method]
    deltas, wins = [], []
    for _, r in hor.iterrows():
        pts = sorted(ref.get(r["key"], []))
        if len(pts) < 2:
            continue
        at = float(np.interp(r["compute_speedup"], [p[0] for p in pts], [p[1] for p in pts]))
        d = (r[metric] - at) if higher else (at - r[metric])
        deltas.append(d); wins.append(1.0 if d > 0 else 0.0)
    return deltas, wins


def pc_minus_plain(df, family, tau, metric="psnr", higher=True):
    """Paired per-image (PC - plain) for the SAME variant family + tau (pure PC effect).
    Returns dict with mean delta, CI, win, and the speedup gap (PC pays the endpoint cost)."""
    pc_m = f"horizon_pc0.5_{family}_t{tau:g}"
    pl_m = f"horizon_{family}_t{tau:g}"
    pc = {r["key"]: r for _, r in df[df.method == pc_m].iterrows()}
    pl = {r["key"]: r for _, r in df[df.method == pl_m].iterrows()}
    keys = set(pc) & set(pl)
    if not keys:
        return None
    d = [( (pc[k][metric]-pl[k][metric]) if higher else (pl[k][metric]-pc[k][metric]) ) for k in keys]
    lo, hi = boot_ci(d)
    return dict(
        family=family, tau=tau, n=len(keys),
        pc_speedup=float(np.mean([pc[k]["compute_speedup"] for k in keys])),
        plain_speedup=float(np.mean([pl[k]["compute_speedup"] for k in keys])),
        pc_metric=float(np.mean([pc[k][metric] for k in keys])),
        plain_metric=float(np.mean([pl[k][metric] for k in keys])),
        mean_delta=float(np.mean(d)), ci=(lo, hi),
        excl0=(lo is not None and (lo > 0 or hi < 0)),
        win=float(np.mean([1.0 if x > 0 else 0.0 for x in d])),
    )


def variants_present(df):
    return sorted({m[len("horizon_"):m.rfind("_t")] for m in df.method.unique() if m.startswith("horizon_")})


def pc_families(df):
    """Variant families that have BOTH a pc0.5_ and a plain version."""
    vs = set(variants_present(df))
    fams = []
    for v in vs:
        if v.startswith("pc0.5_"):
            base = v[len("pc0.5_"):]
            if base in vs:
                fams.append(base)
    return sorted(fams)


def curvature_signal(gen: Path):
    """Correlate the per-jump curvature score with the end-to-end damage of that trajectory.
    Damage proxy: (matched SeaCache PSNR - this method PSNR) is trajectory-level, so we use a
    per-jump surrogate: mean curvature of a trajectory vs its final PSNR drop vs full.
    Also returns curvature percentiles (to check scaling for the kappa sweep)."""
    df = pd.read_csv(gen / "metrics.csv")
    tdir = gen / "traces"
    curv_rows = []  # (mean_curv, jf, sigma, psnr, method)
    all_curv = []
    for _, r in df[df.method.str.contains("pc0.5_", na=False)].iterrows():
        tp = tdir / f"{r['key']}__{r['method']}.json"
        if not tp.exists():
            continue
        jj = json.loads(tp.read_text()).get("jumps", [])
        cs = [j.get("curvature_l1") for j in jj if j.get("curvature_l1") is not None]
        all_curv += cs
        for j in jj:
            if j.get("curvature_l1") is not None:
                curv_rows.append((j["curvature_l1"], j.get("jump_factor"), j.get("old_sigma"), r["psnr"]))
    pct = {}
    if all_curv:
        a = np.asarray(all_curv, float)
        pct = {p: float(np.percentile(a, p)) for p in (50, 65, 80, 90, 95)}
    return {"n_jumps": len(all_curv), "percentiles": pct, "rows": curv_rows}


def summarize(gen: Path, tau_grid):
    df = pd.read_csv(gen / "metrics.csv")
    fams = pc_families(df)
    out = {"families": fams, "pc_minus_plain": {}, "curvature": curvature_signal(gen)}
    for fam in fams:
        out["pc_minus_plain"][fam] = []
        for tau in tau_grid:
            r = pc_minus_plain(df, fam, tau, "psnr", True)
            if r:
                out["pc_minus_plain"][fam].append(r)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--tau-grid", type=float, nargs="+", default=[0.4, 0.5, 0.65])
    a = ap.parse_args()
    s = summarize(Path(a.gen_dir), a.tau_grid)
    print("families:", s["families"])
    print("curvature n_jumps:", s["curvature"]["n_jumps"], "percentiles:", s["curvature"]["percentiles"])
    for fam, rows in s["pc_minus_plain"].items():
        print(f"--- PC minus plain: {fam}")
        for r in rows:
            print(f"  t{r['tau']:g}: PC {r['pc_speedup']:.2f}x/{r['pc_metric']:.2f}dB  "
                  f"plain {r['plain_speedup']:.2f}x/{r['plain_metric']:.2f}dB  "
                  f"Δ={r['mean_delta']:+.2f} CI[{r['ci'][0]},{r['ci'][1]}] win={r['win']:.2f} excl0={r['excl0']}")
