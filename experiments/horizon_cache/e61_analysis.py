"""Analysis core for E61 — Innovation-Gated Fixed Residual Motion.

Keeps fixed β=0.5 in-band; throttles it only via a causal gate g_a computed from the FIXED-β
forecast innovation at the last fresh anchor (independent of E60's LS β̂, which is only an
optional risk signal in `betahat_gate`). Central questions:
  1. Does the gate MATCH fixed RM in-band (2.0-3.5x)?             -> gate_minus_fixed
  2. Does it FIX the high-speed penalty vs plain HorizonCache?     -> gate_minus_plain
  3. Does I_a (or Ī_a) predict RM-plain gain, reproducing/improving E60's corr? -> innovation diag
  4. Mean β_i / frac(β_i<0.5) / frac(β_i=0) by speed band.

Reuses the E56/E58/E60 matched-achieved-speedup protocol unchanged.
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

_GATE_RE = re.compile(r"^rmg([hsf])([0-9.]+)(?:g([0-9.]+))?(?:a([0-9.]+))?(?:n([rdp]))?_(.+)$")
_BG_RE = re.compile(r"^rmbg([0-9.]*)_(.+)$")
_GATE_NAME = {"h": "hard", "s": "soft", "f": "floor"}


def load_df(gen: Path) -> pd.DataFrame:
    rows = json.loads((gen / "metrics.json").read_text())
    for r in rows:
        r.pop("actions", None)
    return pd.DataFrame(rows)


def classify(variant: str) -> dict:
    """-> {family: plain|fo|cl|gate_hard|gate_soft|gate_floor|betahat_gate, base, kappa, gmin,
    ema_alpha, norm}."""
    m = _GATE_RE.match(variant)
    if m:
        t, kappa, gmin, alpha, norm, base = m.groups()
        return dict(family=f"gate_{_GATE_NAME[t]}", base=base, kappa=float(kappa),
                    gmin=float(gmin) if gmin else 0.0, ema_alpha=float(alpha) if alpha else 1.0,
                    norm=norm or "r", fixed_twin=f"rmraw0.5_{base}")
    m = _BG_RE.match(variant)
    if m:
        gmin, base = m.groups()
        return dict(family="betahat_gate", base=base, kappa=None,
                    gmin=float(gmin) if gmin else 0.0, ema_alpha=None, norm=None,
                    fixed_twin=f"rmraw0.5_{base}")
    if variant.startswith("rmcl"):
        from .cl_analysis import classify as cl_classify
        d = cl_classify(variant)
        return dict(family="cl", base=d["base"], kappa=None, gmin=None, ema_alpha=None,
                   norm=None, fixed_twin=f"rmraw{d['prior']:g}_{d['base']}")
    b = RA.rm_base(variant)
    if b:
        return dict(family="fo", base=b, kappa=None, gmin=None, ema_alpha=None, norm=None,
                   fixed_twin=None)
    return dict(family="plain", base=variant, kappa=None, gmin=None, ema_alpha=None, norm=None,
               fixed_twin=None)


def beta_by_tau(df: pd.DataFrame, tau_grid, families=("gate_hard", "gate_soft", "gate_floor",
                                                       "betahat_gate")) -> list[dict]:
    """β_i distribution + gate-off fractions per τ for every gated variant present."""
    out = []
    variants = [v for v in RA.variants_present(df) if classify(v)["family"] in families]
    for v in variants:
        for tau in tau_grid:
            sub = df[df.method == f"horizon_{v}_t{tau:g}"]
            if sub.empty or "rm_beta_used_mean" not in sub.columns:
                continue
            bu = sub["rm_beta_used_mean"].dropna()
            if not len(bu):
                continue
            row = {"variant": v, "family": classify(v)["family"], "tau": tau,
                  "speedup": float(sub.compute_speedup.mean()),
                  "band": band_of(float(sub.compute_speedup.mean())), "n": int(len(bu)),
                  "beta_mean": float(bu.mean()), "beta_p10": float(np.percentile(bu, 10)),
                  "beta_p50": float(np.percentile(bu, 50)), "beta_p90": float(np.percentile(bu, 90)),
                  "frac_beta_lt_0p5": (float(sub["frac_beta_lt_0p5"].dropna().mean())
                                       if "frac_beta_lt_0p5" in sub else None),
                  "frac_beta_eq_0": (float(sub["frac_beta_eq_0"].dropna().mean())
                                    if "frac_beta_eq_0" in sub else None)}
            if "gate_Ibar_mean" in sub:
                row["Ibar_mean"] = float(sub["gate_Ibar_mean"].dropna().mean())
            out.append(row)
    return out


def innovation_vs_gain(df: pd.DataFrame, tau_grid, n_bins: int = 6) -> dict:
    """Per-image I_a (mean Ibar over refreshes of a representative gated run) vs the per-image
    fixed-RM − plain gain; plus binned E[gain | I_a]."""
    xs, ys, taus = [], [], []
    vs = set(RA.variants_present(df))
    gate_v = next((v for v in sorted(vs) if classify(v)["family"].startswith("gate_")), None)
    if gate_v is None:
        return {"n": 0, "corr": None, "bins": []}
    cls = classify(gate_v)
    base, fixed = cls["base"], cls["fixed_twin"]
    if fixed not in vs:
        return {"n": 0, "corr": None, "bins": []}
    for tau in tau_grid:
        ga = {r["key"]: r for _, r in df[df.method == f"horizon_{gate_v}_t{tau:g}"].iterrows()}
        rm = {r["key"]: r for _, r in df[df.method == f"horizon_{fixed}_t{tau:g}"].iterrows()}
        pl = {r["key"]: r for _, r in df[df.method == f"horizon_{base}_t{tau:g}"].iterrows()}
        for k in set(ga) & set(rm) & set(pl):
            ib = ga[k].get("gate_Ibar_mean")
            if ib is None or (isinstance(ib, float) and np.isnan(ib)):
                continue
            xs.append(float(ib))
            ys.append(float(rm[k]["psnr"] - pl[k]["psnr"]))
            taus.append(tau)
    if len(xs) < 3:
        return {"n": len(xs), "corr": None, "bins": []}
    xs_a, ys_a = np.asarray(xs), np.asarray(ys)
    corr = float(np.corrcoef(xs_a, ys_a)[0, 1])
    order = np.argsort(xs_a)
    edges = np.array_split(order, n_bins)
    bins = []
    for idx in edges:
        if len(idx) == 0:
            continue
        bins.append({"I_mean": float(xs_a[idx].mean()), "gain_mean": float(ys_a[idx].mean()),
                    "gain_std": float(ys_a[idx].std()), "n": int(len(idx))})
    return {"n": len(xs), "corr": corr, "gate_variant": gate_v, "fixed_variant": fixed,
           "bins": bins, "scatter": {"I": xs, "rm_minus_plain_psnr": ys, "tau": taus}}


def summarize(gen: Path, tau_grid) -> dict:
    df = load_df(gen)
    variants = RA.variants_present(df)
    fam = {v: classify(v) for v in variants}
    out = {"variants": variants, "families": {v: f["family"] for v, f in fam.items()},
          "vs_seacache": {}, "vs_seacache_lpips": {}, "gate_minus_fixed": {},
          "gate_minus_fixed_lpips": {}, "gate_minus_plain": {}, "method_points": {}}

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

    for v in variants:
        f = fam[v]
        if f["fixed_twin"] and f["fixed_twin"] in variants:
            rows = [r for tau in tau_grid if (r := RA.rm_minus_plain(df, v, f["fixed_twin"], tau)) is not None]
            if rows:
                out["gate_minus_fixed"][v] = rows
            rowslp = [r for tau in tau_grid
                     if (r := RA.rm_minus_plain(df, v, f["fixed_twin"], tau, "lpips", False)) is not None]
            if rowslp:
                out["gate_minus_fixed_lpips"][v] = rowslp
        if f["base"] in variants:
            rows = [r for tau in tau_grid if (r := RA.rm_minus_plain(df, v, f["base"], tau)) is not None]
            if rows:
                out["gate_minus_plain"][v] = rows

    band_rows = []
    gate_families = ["gate_hard", "gate_soft", "gate_floor", "betahat_gate"]
    for bname, lo, hi in BANDS:
        row = {"band": bname}
        for family, label in [("plain", "plain"), ("fo", "fixed_rm"), ("cl", "closed_loop")] + \
                             [(gf, gf) for gf in gate_families]:
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
                             "excl0": best["excl0"], "win": round(best["win"], 3), "n": best["n"],
                             "delta_lpips_vs_seacache": round(lp["delta"], 4) if lp else None}
        # gate-vs-fixed / gate-vs-plain best-in-band (across ALL gate families, for the headline)
        for src, key in [(out["gate_minus_fixed"], "best_gate_minus_fixed"),
                         (out["gate_minus_plain"], "best_gate_minus_plain")]:
            best = None
            for v, rows in src.items():
                for r in rows:
                    if lo <= r["rm_speedup"] < hi and (best is None or r["mean_delta"] > best["mean_delta"]):
                        best = {**r, "variant": v}
            if best:
                row[key] = {"variant": best["variant"], "tau": best["tau"],
                           "delta_psnr": round(best["mean_delta"], 3),
                           "ci": [round(best["ci"][0], 3), round(best["ci"][1], 3)],
                           "excl0": best["excl0"], "win": round(best["win"], 3)}
        if len(row) > 1:
            row["seacache_reachable"] = (out["seacache_max_speedup"] is not None
                                         and lo <= out["seacache_max_speedup"] + 0.05)
            band_rows.append(row)
    out["bands"] = band_rows

    highest = {}
    for label in ["plain", "fixed_rm", "closed_loop"] + gate_families:
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
    ig = s["innovation_vs_gain"]
    print(f"--- innovation vs gain: corr={ig['corr']} (n={ig['n']})")
    for b in ig.get("bins", []):
        print(f"  I~{b['I_mean']:.3f}: gain={b['gain_mean']:+.3f}±{b['gain_std']:.3f} (n={b['n']})")
    for k, rows in sorted(s["gate_minus_fixed"].items()):
        print(f"--- gate-fixed: {k}")
        for r in rows:
            print(f"  t{r['tau']:g}: {r['rm_speedup']:.2f}x  Δ={r['mean_delta']:+.3f} "
                 f"CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}] win={r['win']:.2f} excl0={r['excl0']}")
    for row in s["bands"]:
        parts = [row["band"]]
        for k in ("plain", "fixed_rm", "closed_loop"):
            if k in row:
                parts.append(f"{k}:{row[k]['delta_psnr_vs_seacache']:+.2f}{'*' if row[k]['excl0'] else ''}")
        print("  ".join(parts))
