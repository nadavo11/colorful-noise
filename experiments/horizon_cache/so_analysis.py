"""Analysis core for E59 — Second-Order Residual Hold + extreme-speed SeaCache comparison.

Questions (in priority order):
  1. Extreme speed: how far up the achieved-speedup axis does first-order Residual Motion
     stay above SeaCache — and where does every method break?      -> vs_seacache + band table
  2. Does a second-order residual hold (uniform Newton Δ² or nonuniform Lagrange quad) beat
     first-order RM at matched compute?                            -> so_minus_fo (paired)
  3. Does the second-order ASSUMPTION hold — is residual curvature smooth/stable enough to
     extrapolate (ρ2, cosΔ), and does it predict when SO helps?    -> so_diagnostics

Reuses the E56/E58 matched-achieved-speedup protocol (per-image paired, interpolate the
SeaCache frontier at each image's own achieved speedup, percentile bootstrap). UNCHANGED.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import rm_analysis as RA

boot_ci = RA.boot_ci

# E59 speed bands (extreme-speed comparison; extends the E58 bands)
BANDS = [("1.8-2.2x", 1.8, 2.2), ("2.3-2.7x", 2.3, 2.7), ("2.8-3.2x", 2.8, 3.2),
         ("3.3-3.7x", 3.3, 3.7), ("3.8-4.2x", 3.8, 4.2), ("4.3-4.7x", 4.3, 4.7),
         ("4.8-5.2x", 4.8, 5.2), (">5.2x", 5.2, 99.0)]

_SO_RE = re.compile(r"^rm2(raw|lowpass|topk|sea)([0-9.]+)b([0-9.]+)(?:g([0-9.]+))?(?:r([0-9.]+))?_(.+)$")
_Q_RE = re.compile(r"^rmq(raw|lowpass|topk|sea)([0-9.]+)_(.+)$")


def band_of(sp: float) -> str | None:
    for name, lo, hi in BANDS:
        if lo <= sp < hi:
            return name
    return None


def classify(variant: str) -> dict:
    """-> {family: plain|fo|so|quad, base, proj, beta1, beta2, gamma, rho_max, fo_equiv}.
    fo_equiv = the first-order RM variant a SO variant pairs against (same proj/β1/base)."""
    m = _SO_RE.match(variant)
    if m:
        return dict(family="so", proj=m.group(1), beta1=float(m.group(2)), beta2=float(m.group(3)),
                    gamma=float(m.group(4)) if m.group(4) else None,
                    rho_max=float(m.group(5)) if m.group(5) else None, base=m.group(6),
                    fo_equiv=f"rm{m.group(1)}{m.group(2)}_{m.group(6)}")
    m = _Q_RE.match(variant)
    if m:
        return dict(family="quad", proj=m.group(1), beta1=None, beta2=None, gamma=None,
                    rho_max=None, beta_quad=float(m.group(2)), base=m.group(3),
                    fo_equiv=f"rm{m.group(1)}0.5_{m.group(3)}")
    b = RA.rm_base(variant)
    if b:
        proj, beta = RA.rm_proj_beta(variant)
        return dict(family="fo", proj=proj, beta1=beta, beta2=None, gamma=None, rho_max=None,
                    base=b, fo_equiv=None)
    return dict(family="plain", proj=None, beta1=None, beta2=None, gamma=None, rho_max=None,
                base=variant, fo_equiv=None)


def paired_delta(df, method_a: str, method_b: str, tau: float, metric="psnr", higher=True):
    """Paired per-image (A − B) at the same τ. Positive = A better (LPIPS sign handled)."""
    return RA.rm_minus_plain(df, method_a, method_b, tau, metric, higher)


# --------------------------------------------------------------- diagnostics
def so_diagnostics(gen: Path, tau_grid):
    """Anchor-triple curvature stats from traces + per-image correlation with the SO−FO gain.
    Reads: fresh-step so_rho2/so_cos_delta (anchor triples) and cached-step rm_so_* fields."""
    df = pd.read_csv(gen / "metrics.csv")
    tdir = gen / "traces"
    rho2, cosd, rho_anchor, so_term, so_coeff = [], [], [], [], []
    per_image: dict[tuple, dict] = {}     # (key, method) -> {rho2:[], cos:[]}
    n_gated_off = 0
    n_so_apps = 0
    for _, r in df.iterrows():
        v = r["method"][len("horizon_"):r["method"].rfind("_t")] if r["method"].startswith("horizon_") else ""
        fam = classify(v)["family"] if v else "plain"
        if fam not in ("so", "quad"):
            continue
        tp = tdir / f"{r['key']}__{r['method']}.json"
        if not tp.exists():
            continue
        data = json.loads(tp.read_text())
        pi = per_image.setdefault((r["key"], r["method"]), {"rho2": [], "cos": []})
        for t in data.get("traces", []):
            if t.get("ran_full") and "so_rho2" in t:
                rho2.append(t["so_rho2"]); cosd.append(t["so_cos_delta"])
                rho_anchor.append(t.get("so_rho_anchor", np.nan))
                pi["rho2"].append(t["so_rho2"]); pi["cos"].append(t["so_cos_delta"])
            if t.get("rm_so_used"):
                n_so_apps += 1
                if t.get("rm_so_term_ratio") is not None:
                    so_term.append(t["rm_so_term_ratio"])
                if t.get("rm_so_coeff") is not None:
                    so_coeff.append(t["rm_so_coeff"])
            if t.get("rm_so_gated_off"):
                n_gated_off += 1

    def pct(a, ps=(5, 25, 50, 75, 95)):
        a = np.asarray([x for x in a if x == x], float)
        return {p: float(np.percentile(a, p)) for p in ps} if len(a) else {}

    # correlation: per-image mean rho2 / cosΔ vs the per-image SO−FO PSNR delta (same τ)
    xs_rho, xs_cos, ys = [], [], []
    for (key, method), d in per_image.items():
        if not d["rho2"]:
            continue
        v = method[len("horizon_"):method.rfind("_t")]
        tau = method[method.rfind("_t") + 2:]
        fo = f"horizon_{classify(v)['fo_equiv']}_t{tau}"
        a = df[(df.method == method) & (df.key == key)]
        b = df[(df.method == fo) & (df.key == key)]
        if a.empty or b.empty:
            continue
        xs_rho.append(float(np.mean(d["rho2"])))
        xs_cos.append(float(np.mean(d["cos"])))
        ys.append(float(a.iloc[0]["psnr"] - b.iloc[0]["psnr"]))

    def corr(x, y):
        if len(x) < 3:
            return None
        return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])

    return {"n_anchor_triples": len(rho2), "n_so_applications": n_so_apps,
            "n_so_gated_off": n_gated_off,
            "rho2_pct": pct(rho2), "cos_delta_pct": pct(cosd), "rho_anchor_pct": pct(rho_anchor),
            "so_term_ratio_pct": pct(so_term), "so_coeff_pct": pct(so_coeff),
            "rho2_corr_with_gain": corr(xs_rho, ys), "cos_delta_corr_with_gain": corr(xs_cos, ys),
            "scatter": {"rho2": xs_rho, "cos_delta": xs_cos, "so_minus_fo_psnr": ys}}


# --------------------------------------------------------------- summary
def summarize(gen: Path, tau_grid):
    df = pd.read_csv(gen / "metrics.csv")
    variants = RA.variants_present(df)
    fam = {v: classify(v) for v in variants}
    out = {"variants": variants, "families": {v: f["family"] for v, f in fam.items()},
           "vs_seacache": {}, "vs_seacache_lpips": {}, "so_minus_fo": {}, "so_minus_plain": {},
           "fo_minus_plain": {}, "method_points": {}, "bands": {}}

    # matched-achieved-speedup vs the SeaCache frontier (PSNR + LPIPS), every variant
    for v in variants:
        out["vs_seacache"][v] = RA.matched_vs_seacache(df, v, tau_grid, "psnr", True)
        out["vs_seacache_lpips"][v] = RA.matched_vs_seacache(df, v, tau_grid, "lpips", False)

    # SeaCache's own frontier points (per τ aggregates) for the band table
    sea_pts = []
    for tau in tau_grid:
        sub = df[df.method == f"seacache_t{tau:g}"]
        if not sub.empty:
            sea_pts.append({"tau": tau, "speedup": float(sub.compute_speedup.mean()),
                            "psnr": float(sub.psnr.mean()),
                            "lpips": float(sub.lpips.mean()) if "lpips" in df.columns else None})
    out["seacache_points"] = sea_pts

    # per-method per-τ aggregate points (for the frontier figure + band assignment)
    for v in variants:
        pts = []
        for tau in tau_grid:
            sub = df[df.method == f"horizon_{v}_t{tau:g}"]
            if not sub.empty:
                pts.append({"tau": tau, "speedup": float(sub.compute_speedup.mean()),
                            "psnr": float(sub.psnr.mean()),
                            "lpips": float(sub.lpips.mean()) if "lpips" in df.columns else None,
                            "wall_speedup": float(sub.wall_speedup.dropna().mean()) if "wall_speedup" in sub else None})
        out["method_points"][v] = pts

    # paired ablations at same τ
    for v in variants:
        f = fam[v]
        if f["family"] in ("so", "quad") and f["fo_equiv"] in variants:
            out["so_minus_fo"][v] = [r for tau in tau_grid
                                     if (r := paired_delta(df, v, f["fo_equiv"], tau)) is not None]
        if f["family"] in ("so", "quad", "fo") and f["base"] in variants:
            key = "fo_minus_plain" if f["family"] == "fo" else "so_minus_plain"
            out[key][v] = [r for tau in tau_grid
                           if (r := paired_delta(df, v, f["base"], tau)) is not None]
        # LPIPS side of the SO−FO ablation (regression check)
        if f["family"] in ("so", "quad") and f["fo_equiv"] in variants:
            out.setdefault("so_minus_fo_lpips", {})[v] = [
                r for tau in tau_grid
                if (r := paired_delta(df, v, f["fo_equiv"], tau, "lpips", False)) is not None]

    # band table: per band, best delta-vs-SeaCache per family + SO−FO
    band_rows = []
    for bname, lo, hi in BANDS:
        row = {"band": bname}
        for family, label in [("plain", "plain"), ("fo", "first_order_rm"),
                              ("so", "second_order_rm"), ("quad", "quad_rm")]:
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
        # SO−FO inside the band (best SO variant at a τ whose SO speedup falls in the band)
        best_sf = None
        for v, rows in out["so_minus_fo"].items():
            for r in rows:
                if lo <= r["rm_speedup"] < hi and (best_sf is None or r["mean_delta"] > best_sf["mean_delta"]):
                    best_sf = {**r, "variant": v}
        if best_sf:
            row["so_minus_fo"] = {"variant": best_sf["variant"], "tau": best_sf["tau"],
                                  "delta_psnr": round(best_sf["mean_delta"], 3),
                                  "ci": [round(best_sf["ci"][0], 3), round(best_sf["ci"][1], 3)],
                                  "excl0": best_sf["excl0"], "win": round(best_sf["win"], 3)}
        if len(row) > 1:
            band_rows.append(row)
    out["bands"] = band_rows

    # highest band where each family stays above SeaCache with CI excluding 0
    highest = {}
    for family, label in [("plain", "plain_horizon"), ("fo", "first_order_rm"),
                          ("so", "second_order_rm"), ("quad", "quad_rm")]:
        hb = None
        for row in band_rows:
            keymap = {"plain_horizon": "plain", "first_order_rm": "first_order_rm",
                      "second_order_rm": "second_order_rm", "quad_rm": "quad_rm"}
            d = row.get(keymap[label])
            if d and d["delta_psnr_vs_seacache"] > 0 and d["excl0"]:
                hb = row["band"]
        highest[label] = hb
    out["highest_positive_vs_seacache_band"] = highest

    out["diagnostics"] = so_diagnostics(gen, tau_grid)
    out["oracle"] = RA.oracle_residual(gen)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--tau-grid", type=float, nargs="+",
                    default=[0.3, 0.5, 0.575, 0.65, 0.8, 1.0, 1.2])
    a = ap.parse_args()
    s = summarize(Path(a.gen_dir), a.tau_grid)
    print("variants:", s["variants"])
    print("highest positive band:", s["highest_positive_vs_seacache_band"])
    d = s["diagnostics"]
    print(f"diag: {d['n_anchor_triples']} triples, rho2 p50={d['rho2_pct'].get(50)}, "
          f"cos p50={d['cos_delta_pct'].get(50)}, corr(rho2,gain)={d['rho2_corr_with_gain']}, "
          f"corr(cos,gain)={d['cos_delta_corr_with_gain']}")
    for v, rows in s["so_minus_fo"].items():
        print(f"--- SO minus FO: {v}")
        for r in rows:
            print(f"  t{r['tau']:g}: {r['rm_speedup']:.2f}x  Δ={r['mean_delta']:+.3f} "
                  f"CI[{r['ci'][0]},{r['ci'][1]}] win={r['win']:.2f} excl0={r['excl0']}")
    for row in s["bands"]:
        parts = [row["band"]]
        for k in ("plain", "first_order_rm", "second_order_rm", "quad_rm"):
            if k in row:
                parts.append(f"{k}:{row[k]['delta_psnr_vs_seacache']:+.2f}"
                             f"{'*' if row[k]['excl0'] else ''}@{row[k]['speedup']:.2f}x")
        print("  ".join(parts))
