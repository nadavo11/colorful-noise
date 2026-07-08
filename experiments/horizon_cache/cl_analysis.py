"""Analysis core for E60 — Closed-Loop Residual Motion (online innovation-fit β̂ + midpoint-λ).

The four pre-registered questions (docs/methods/closed_loop_residual_motion.md):
  1. Does adaptive β̂ match or beat fixed β=0.5?        -> cl_minus_fixed (paired, per τ)
  2. Does it automatically turn RM off above ~4×?      -> beta_by_tau + cl/fixed minus plain
     (E59 measured fixed-β RM−plain at −0.26* @4.47×, −0.97* @τ1.4 — β̂→0 must remove that)
  3. Does innovation magnitude predict when RM helps?  -> innovation_vs_gain (per-image corr)
  4. Final rule "RM iff β̂>0"?                          -> beta_zero_fraction by τ
Plus the standing E56 protocol: matched-achieved-speedup vs the SeaCache frontier, per band.

Reads metrics.json (NOT metrics.csv — the E60 per-run aggregates cl_beta_hat_final /
cl_beta_hat_mean / cl_innov_rel_mean / rm_beta_used_mean live only in the JSON rows).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import rm_analysis as RA
from .so_analysis import BANDS, band_of

boot_ci = RA.boot_ci

_CL_RE = re.compile(r"^rmcl([0-9.]+)(?:w([0-9.]+))?(?:m([0-9.]+))?(?:x([0-9.]+))?"
                    r"(?:g([0-9.]+))?(?:k([0-9.]+))?(mid)?_(.+)$")
_FO_MID_RE = re.compile(r"^rm(raw|lowpass|topk|sea)([0-9.]+)mid_(.+)$")


def load_df(gen: Path) -> pd.DataFrame:
    rows = json.loads((gen / "metrics.json").read_text())
    for r in rows:
        r.pop("actions", None)
    return pd.DataFrame(rows)


def classify(variant: str) -> dict:
    """-> {family: plain|fo|fo_mid|cl|cl_mid, base, prior, forget, mu, beta_max, pairs}.
    pairs = dict of named paired ablations this variant anchors (variant name -> label)."""
    m = _CL_RE.match(variant)
    if m:
        prior, w, mu, bx, gate, scale, mid, base = m.groups()
        fam = "cl_mid" if mid else "cl"
        d = dict(family=fam, base=base, prior=float(prior),
                 forget=float(w) if w else 0.85, mu=float(mu) if mu else 1.0,
                 beta_max=float(bx) if bx else 1.0,
                 gate=float(gate) if gate else 0.0, scale=float(scale) if scale else 1.0)
        core = f"rmcl{prior}" + (f"w{w}" if w else "") + (f"m{mu}" if mu else "") + \
               (f"x{bx}" if bx else "") + (f"g{gate}" if gate else "") + \
               (f"k{scale}" if scale else "")
        # the fixed-β twin: gate mode applies rm_beta=0.5; scale mode's effective initial
        # gain is prior·κ; plain cl uses the prior itself
        eff = 0.5 if gate else (d["prior"] * d["scale"])
        d["pairs"] = {f"rmraw{eff:g}{'mid' if mid else ''}_{base}": "cl_minus_fixed",
                      base: "cl_minus_plain"}
        if mid:
            d["pairs"][f"{core}_{base}"] = "mid_minus_point"
        return d
    m = _FO_MID_RE.match(variant)
    if m:
        proj, beta, base = m.groups()
        return dict(family="fo_mid", base=base, prior=float(beta), forget=None, mu=None,
                    beta_max=None,
                    pairs={f"rm{proj}{beta}_{base}": "mid_minus_point", base: "cl_minus_plain"})
    b = RA.rm_base(variant)
    if b:
        proj, beta = RA.rm_proj_beta(variant)
        return dict(family="fo", base=b, prior=beta, forget=None, mu=None, beta_max=None,
                    pairs={b: "fixed_minus_plain"})
    return dict(family="plain", base=variant, prior=None, forget=None, mu=None, beta_max=None,
                pairs={})


# ------------------------------------------------------------------ β̂ behavior
def beta_by_tau(df: pd.DataFrame, tau_grid) -> list[dict]:
    """β̂ distribution per τ (and achieved speedup) for every closed-loop variant.
    Uses the per-run aggregates written by run.py; zero_frac = trajectories whose mean
    USED β (clamped) fell below 0.05 — the 'RM effectively off' fraction."""
    out = []
    cl_vars = [v for v in RA.variants_present(df) if classify(v)["family"].startswith("cl")]
    for v in cl_vars:
        for tau in tau_grid:
            sub = df[df.method == f"horizon_{v}_t{tau:g}"]
            if sub.empty or "cl_beta_hat_final" not in sub.columns:
                continue
            bh = sub["cl_beta_hat_final"].dropna()
            bu = sub["rm_beta_used_mean"].dropna() if "rm_beta_used_mean" in sub else bh
            if not len(bh):
                continue
            out.append({
                "variant": v, "tau": tau,
                "speedup": float(sub.compute_speedup.mean()),
                "band": band_of(float(sub.compute_speedup.mean())),
                "n": int(len(bh)),
                "beta_hat_final_p25": float(np.percentile(bh, 25)),
                "beta_hat_final_p50": float(np.percentile(bh, 50)),
                "beta_hat_final_p75": float(np.percentile(bh, 75)),
                "beta_used_mean_p50": float(np.percentile(bu, 50)) if len(bu) else None,
                "zero_frac": float(np.mean(bu < 0.05)) if len(bu) else None,
                "innov_rel_p50": (float(sub["cl_innov_rel_mean"].dropna().median())
                                  if "cl_innov_rel_mean" in sub else None),
                "n_obs_p50": (float(sub["cl_n_obs"].dropna().median())
                              if "cl_n_obs" in sub else None),
            })
    return out


def beta_trajectories(gen: Path, variant: str, tau_grid, max_traj=200) -> list[dict]:
    """Per-trajectory (σ-ordered) β̂ paths from traces, for the report figure."""
    tdir = gen / "traces"
    out = []
    for tau in tau_grid:
        for tp in sorted(tdir.glob(f"*__horizon_{variant}_t{tau:g}.json"))[:max_traj]:
            tr = json.loads(tp.read_text())["traces"]
            path = [{"sigma": t["sigma"], "beta_hat": t["cl_beta_hat"]}
                    for t in tr if "cl_beta_hat" in t]
            if path:
                out.append({"tau": tau, "key": tp.name.split("__")[0], "path": path})
    return out


# ------------------------------------------------------------------ innovation vs gain
def innovation_vs_gain(df: pd.DataFrame, tau_grid) -> dict:
    """Pre-registered prediction (b): per-image innovation norm anti-correlates with the
    per-image RM−plain gain. Innovation from the CL variant's rows (measured on the same
    trajectory family); gain from the FIXED-β RM−plain pair (the thing β̂ modulates)."""
    xs, ys, taus = [], [], []
    vs = set(RA.variants_present(df))
    cl_v = next((v for v in sorted(vs) if classify(v)["family"] == "cl"), None)
    if cl_v is None:
        return {"n": 0, "corr": None}
    base = classify(cl_v)["base"]
    fixed = f"rmraw{classify(cl_v)['prior']:g}_{base}"
    if fixed not in vs:
        return {"n": 0, "corr": None}
    for tau in tau_grid:
        cl = {r["key"]: r for _, r in df[df.method == f"horizon_{cl_v}_t{tau:g}"].iterrows()}
        rm = {r["key"]: r for _, r in df[df.method == f"horizon_{fixed}_t{tau:g}"].iterrows()}
        pl = {r["key"]: r for _, r in df[df.method == f"horizon_{base}_t{tau:g}"].iterrows()}
        for k in set(cl) & set(rm) & set(pl):
            innov = cl[k].get("cl_innov_rel_mean")
            if innov is None or (isinstance(innov, float) and np.isnan(innov)):
                continue
            xs.append(float(innov))
            ys.append(float(rm[k]["psnr"] - pl[k]["psnr"]))
            taus.append(tau)
    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 3 else None
    return {"n": len(xs), "corr": corr, "cl_variant": cl_v, "fixed_variant": fixed,
            "scatter": {"innov_rel": xs, "rm_minus_plain_psnr": ys, "tau": taus}}


# ------------------------------------------------------------------ summary
def summarize(gen: Path, tau_grid) -> dict:
    df = load_df(gen)
    variants = RA.variants_present(df)
    fam = {v: classify(v) for v in variants}
    out = {"variants": variants, "families": {v: f["family"] for v, f in fam.items()},
           "vs_seacache": {}, "vs_seacache_lpips": {}, "paired": {}, "paired_lpips": {},
           "method_points": {}}

    for v in variants:
        out["vs_seacache"][v] = RA.matched_vs_seacache(df, v, tau_grid, "psnr", True)
        out["vs_seacache_lpips"][v] = RA.matched_vs_seacache(df, v, tau_grid, "lpips", False)
        pts = []
        for tau in tau_grid:
            sub = df[df.method == f"horizon_{v}_t{tau:g}"]
            if not sub.empty:
                pts.append({"tau": tau, "speedup": float(sub.compute_speedup.mean()),
                            "psnr": float(sub.psnr.mean()),
                            "lpips": float(sub.lpips.mean()) if "lpips" in df.columns else None})
        out["method_points"][v] = pts

    sea_pts = []
    for tau in tau_grid:
        sub = df[df.method == f"seacache_t{tau:g}"]
        if not sub.empty:
            sea_pts.append({"tau": tau, "speedup": float(sub.compute_speedup.mean()),
                            "psnr": float(sub.psnr.mean()),
                            "lpips": float(sub.lpips.mean()) if "lpips" in df.columns else None})
    out["seacache_points"] = sea_pts
    out["seacache_max_speedup"] = max((p["speedup"] for p in sea_pts), default=None)

    # named paired ablations: cl−fixed, cl−plain, fixed−plain, mid−point
    for v in variants:
        for other, label in fam[v]["pairs"].items():
            if other not in variants:
                continue
            rows, rows_lp = [], []
            for tau in tau_grid:
                r = RA.rm_minus_plain(df, v, other, tau)
                if r is not None:
                    rows.append(r)
                rl = RA.rm_minus_plain(df, v, other, tau, "lpips", False)
                if rl is not None:
                    rows_lp.append(rl)
            if rows:
                out["paired"][f"{label}::{v}::{other}"] = rows
            if rows_lp:
                out["paired_lpips"][f"{label}::{v}::{other}"] = rows_lp

    # band table (best delta-vs-SeaCache per family, as in E59)
    band_rows = []
    for bname, lo, hi in BANDS:
        row = {"band": bname}
        for family, label in [("plain", "plain"), ("fo", "fixed_rm"), ("fo_mid", "fixed_rm_mid"),
                              ("cl", "closed_loop"), ("cl_mid", "closed_loop_mid")]:
            best = None
            for v in variants:
                if fam[v]["family"] != family:
                    continue
                for r in out["vs_seacache"][v]:
                    if lo <= r["speedup"] < hi and (best is None or r["delta"] > best["delta"]):
                        best = {**r, "variant": v}
            if best:
                lp = next((x for x in out["vs_seacache_lpips"][best["variant"]]
                           if x["tau"] == best["tau"]), None)
                row[label] = {"variant": best["variant"], "tau": best["tau"],
                              "speedup": round(best["speedup"], 3),
                              "delta_psnr_vs_seacache": round(best["delta"], 3),
                              "ci": [round(best["ci"][0], 3), round(best["ci"][1], 3)],
                              "excl0": best["excl0"], "win": round(best["win"], 3),
                              "n": best["n"],
                              "delta_lpips_vs_seacache": round(lp["delta"], 4) if lp else None}
        if len(row) > 1:
            row["seacache_reachable"] = (out["seacache_max_speedup"] is not None
                                         and lo <= out["seacache_max_speedup"] + 0.05)
            band_rows.append(row)
    out["bands"] = band_rows

    highest = {}
    for label in ("plain", "fixed_rm", "fixed_rm_mid", "closed_loop", "closed_loop_mid"):
        hb = None
        for row in band_rows:
            d = row.get(label)
            if d and d["delta_psnr_vs_seacache"] > 0 and d["excl0"]:
                hb = row["band"]
        highest[label] = hb
    out["highest_positive_vs_seacache_band"] = highest

    out["beta_by_tau"] = beta_by_tau(df, tau_grid)
    out["innovation_vs_gain"] = innovation_vs_gain(df, tau_grid)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--tau-grid", type=float, nargs="+",
                    default=[0.3, 0.5, 0.575, 0.65, 0.8, 1.0, 1.2, 1.4])
    a = ap.parse_args()
    s = summarize(Path(a.gen_dir), a.tau_grid)
    print("variants:", s["variants"])
    print("highest positive band:", s["highest_positive_vs_seacache_band"])
    print("--- beta_hat by tau (closed-loop variants)")
    for r in s["beta_by_tau"]:
        print(f"  {r['variant']} t{r['tau']:g} {r['speedup']:.2f}x: "
              f"b^p50={r['beta_hat_final_p50']:+.3f} used_p50={r['beta_used_mean_p50']:.3f} "
              f"zero_frac={r['zero_frac']:.2f} innov_p50={r['innov_rel_p50']}")
    ig = s["innovation_vs_gain"]
    print(f"--- innovation vs gain: corr={ig['corr']} (n={ig['n']})")
    for k, rows in sorted(s["paired"].items()):
        print(f"--- {k}")
        for r in rows:
            print(f"  t{r['tau']:g}: {r['rm_speedup']:.2f}x  Δ={r['mean_delta']:+.3f} "
                  f"CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}] win={r['win']:.2f} excl0={r['excl0']}")
    for row in s["bands"]:
        parts = [row["band"] + ("" if row["seacache_reachable"] else "†")]
        for k in ("plain", "fixed_rm", "closed_loop", "closed_loop_mid"):
            if k in row:
                parts.append(f"{k}:{row[k]['delta_psnr_vs_seacache']:+.2f}"
                             f"{'*' if row[k]['excl0'] else ''}@{row[k]['speedup']:.2f}x")
        print("  ".join(parts))
