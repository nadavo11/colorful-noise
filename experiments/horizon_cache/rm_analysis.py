"""Analysis core for HorizonCache Residual Motion Cache (E58).

Three honest questions:
  1. Does moving the cached block residual (r_pred = r_anchor + β·λ·P(Δr)) beat PLAIN
     HorizonCache at matched speed — especially past the ~2.7× safe band?  -> rm_minus_plain
  2. Does it shift the SeaCache frontier further than plain adaptive already does?  -> matched_delta_vs
  3. THE mechanism probe: does r_pred predict the TRUE block-residual motion better than the
     frozen r_anchor?  -> oracle_residual (from the --rm-oracle diagnostic traces).

Reuses the E56 matched-achieved-speedup protocol (per-image paired, interpolate the reference
frontier at each image's own achieved speedup, percentile bootstrap). Does NOT change it.
"""
from __future__ import annotations

import collections
import json
import re
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


_RM_RE = re.compile(r"^rm(raw|lowpass|topk|sea)([0-9.]+)_(.+)$")


def rm_base(variant: str) -> str | None:
    """rmlowpass0.25_adaptive_2.0 -> adaptive_2.0 ; non-RM -> None."""
    m = _RM_RE.match(variant)
    return m.group(3) if m else None


def rm_proj_beta(variant: str):
    m = _RM_RE.match(variant)
    return (m.group(1), float(m.group(2))) if m else (None, None)


def variants_present(df):
    return sorted({m[len("horizon_"):m.rfind("_t")] for m in df.method.unique() if m.startswith("horizon_")})


def rm_families(df):
    """List of (rm_variant, base) present with BOTH the rm and the plain base at some tau."""
    vs = set(variants_present(df))
    out = []
    for v in vs:
        b = rm_base(v)
        if b and b in vs:
            out.append((v, b))
    return sorted(out)


def _frontier_points_by_key(df, templates, metric):
    out = collections.defaultdict(list)
    for t in templates:
        sub = df[df.method == t]
        for _, r in sub.iterrows():
            out[r["key"]].append((r["compute_speedup"], r[metric]))
    return out


def matched_delta_vs(df, method, ref_templates, metric="psnr", higher=True):
    """Per-image delta of `method` vs the reference frontier interpolated at each image's own
    achieved speedup. delta>0 => method better (LPIPS sign-flipped)."""
    ref = _frontier_points_by_key(df, ref_templates, metric)
    hor = df[df.method == method]
    deltas = []
    for _, r in hor.iterrows():
        pts = sorted(ref.get(r["key"], []))
        if len(pts) < 2:
            continue
        at = float(np.interp(r["compute_speedup"], [p[0] for p in pts], [p[1] for p in pts]))
        deltas.append((r[metric] - at) if higher else (at - r[metric]))
    return deltas


def matched_vs_seacache(df, method, tau_grid, metric="psnr", higher=True):
    """Per-tau matched-achieved-speedup delta of `method` vs the SeaCache frontier (paired,
    each image vs SeaCache interpolated at that image's own speedup). Returns [{tau,speedup,
    delta,ci,win,n}]. This is the E56 fair-comparison protocol, unchanged."""
    sea = [f"seacache_t{t:g}" for t in tau_grid]
    out = []
    for tau in tau_grid:
        m = f"horizon_{method}_t{tau:g}"
        sub = df[df.method == m]
        if sub.empty:
            continue
        d = matched_delta_vs(df, m, sea, metric, higher)
        if not d:
            continue
        lo, hi = boot_ci(d)
        out.append({"tau": tau, "speedup": float(sub.compute_speedup.mean()),
                    "delta": float(np.mean(d)), "ci": (lo, hi),
                    "excl0": (lo is not None and (lo > 0 or hi < 0)),
                    "win": float(np.mean([1.0 if x > 0 else 0.0 for x in d])), "n": len(d)})
    return out


def rm_minus_plain(df, rm_variant, base, tau, metric="psnr", higher=True):
    """Paired per-image (RM - plain) for the SAME base family + tau (pure residual-motion effect)."""
    rm_m = f"horizon_{rm_variant}_t{tau:g}"
    pl_m = f"horizon_{base}_t{tau:g}"
    rm = {r["key"]: r for _, r in df[df.method == rm_m].iterrows()}
    pl = {r["key"]: r for _, r in df[df.method == pl_m].iterrows()}
    keys = set(rm) & set(pl)
    if not keys:
        return None
    d = [((rm[k][metric] - pl[k][metric]) if higher else (pl[k][metric] - rm[k][metric])) for k in keys]
    lo, hi = boot_ci(d)
    return dict(
        rm_variant=rm_variant, base=base, tau=tau, n=len(keys),
        rm_speedup=float(np.mean([rm[k]["compute_speedup"] for k in keys])),
        plain_speedup=float(np.mean([pl[k]["compute_speedup"] for k in keys])),
        rm_metric=float(np.mean([rm[k][metric] for k in keys])),
        plain_metric=float(np.mean([pl[k][metric] for k in keys])),
        rm_lpips=float(np.mean([rm[k]["lpips"] for k in keys])) if "lpips" in df.columns else None,
        plain_lpips=float(np.mean([pl[k]["lpips"] for k in keys])) if "lpips" in df.columns else None,
        mean_delta=float(np.mean(d)), ci=(lo, hi),
        excl0=(lo is not None and (lo > 0 or hi < 0)),
        win=float(np.mean([1.0 if x > 0 else 0.0 for x in d])),
    )


def motion_signal(gen: Path):
    """Residual-motion magnitude + λ distributions across all RM jumps (are we moving anything?)."""
    df = pd.read_csv(gen / "metrics.csv")
    tdir = gen / "traces"
    extrap, secant, lam, mph = [], [], [], []
    for _, r in df.iterrows():
        if rm_base(r["method"][len("horizon_"):r["method"].rfind("_t")] if r["method"].startswith("horizon_") else "") is None:
            continue
        tp = tdir / f"{r['key']}__{r['method']}.json"
        if not tp.exists():
            continue
        data = json.loads(tp.read_text())
        for j in data.get("jumps", []):
            if j.get("rm_used"):
                if j.get("rm_residual_extrapolation_ratio") is not None:
                    extrap.append(j["rm_residual_extrapolation_ratio"])
                if j.get("rm_residual_secant_norm") is not None:
                    secant.append(j["rm_residual_secant_norm"])
                if j.get("rm_lambda") is not None:
                    lam.append(j["rm_lambda"])
                if j.get("rm_motion_per_headroom") is not None:
                    mph.append(j["rm_motion_per_headroom"])
        # cached (non-jump) steps carry rm diag in traces too
        for t in data.get("traces", []):
            if t.get("rm_used") and t.get("action") == "cache":
                if t.get("rm_residual_extrapolation_ratio") is not None:
                    extrap.append(t["rm_residual_extrapolation_ratio"])
                if t.get("rm_residual_secant_norm") is not None:
                    secant.append(t["rm_residual_secant_norm"])
                if t.get("rm_lambda") is not None:
                    lam.append(t["rm_lambda"])

    def pct(a):
        a = np.asarray(a, float)
        return {p: float(np.percentile(a, p)) for p in (50, 65, 80, 90, 95)} if len(a) else {}
    return {"n": len(extrap), "extrapolation_ratio_pct": pct(extrap),
            "secant_norm_pct": pct(secant), "lambda_pct": pct(lam),
            "motion_per_headroom_pct": pct(mph)}


def oracle_residual(gen: Path):
    """THE mechanism figure: per cached step, frozen residual error ‖r_anchor-r_true‖ vs
    residual-motion error ‖r_pred-r_true‖ (both relative to ‖r_true‖). Reduction>0 => the
    secant genuinely predicts block-residual motion. Reads --rm-oracle traces."""
    tdir = gen / "traces"
    if not tdir.exists():
        return {"available": False}
    frozen, motion, by_age = [], [], collections.defaultdict(lambda: [[], []])
    for tp in sorted(tdir.glob("*.json")):
        data = json.loads(tp.read_text())
        for t in data.get("traces", []):
            fe = t.get("rm_oracle_frozen_err")
            me = t.get("rm_oracle_motion_err")
            if fe is not None and me is not None:
                frozen.append(fe)
                motion.append(me)
                age = int(t.get("refresh_distance", 0))
                by_age[age][0].append(fe)
                by_age[age][1].append(me)
    if not frozen:
        return {"available": False}
    fz = float(np.mean(frozen))
    mo = float(np.mean(motion))
    rows = []
    for age in sorted(by_age):
        f, m = by_age[age]
        rows.append({"step_distance": age, "n": len(f),
                     "frozen_err": float(np.mean(f)), "motion_err": float(np.mean(m)),
                     "reduction": float(np.mean(f)) - float(np.mean(m))})
    return {"available": True, "n": len(frozen),
            "frozen_residual_error": fz, "residual_motion_error": mo,
            "abs_error_reduction": fz - mo,
            "relative_error_reduction": (fz - mo) / (fz + 1e-9),
            "by_step_distance": rows}


# speed bands for the frontier tables
BANDS = [("1.7-2.0x", 1.7, 2.0), ("2.0-2.5x", 2.0, 2.5), ("2.5-2.8x", 2.5, 2.8),
         ("2.8-3.2x", 2.8, 3.2), ("3.2-3.5x", 3.2, 3.5)]


def summarize(gen: Path, tau_grid):
    df = pd.read_csv(gen / "metrics.csv")
    fams = rm_families(df)
    out = {"rm_families": fams, "rm_minus_plain": {}, "vs_seacache": {}, "motion": motion_signal(gen),
           "oracle": oracle_residual(gen)}
    bases = sorted({b for _, b in fams})
    for rm_variant, base in fams:
        out["rm_minus_plain"][rm_variant] = []
        for tau in tau_grid:
            r = rm_minus_plain(df, rm_variant, base, tau, "psnr", True)
            if r:
                out["rm_minus_plain"][rm_variant].append(r)
    # matched vs SeaCache for every RM variant AND its plain base (for the overshoot-flip verdict)
    for m in [v for v, _ in fams] + bases:
        out["vs_seacache"][m] = matched_vs_seacache(df, m, tau_grid, "psnr", True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--tau-grid", type=float, nargs="+", default=[0.4, 0.5, 0.65])
    a = ap.parse_args()
    s = summarize(Path(a.gen_dir), a.tau_grid)
    print("rm families:", s["rm_families"])
    print("motion extrap_ratio pct:", s["motion"]["extrapolation_ratio_pct"])
    print("motion lambda pct:", s["motion"]["lambda_pct"])
    orc = s["oracle"]
    if orc.get("available"):
        print(f"ORACLE frozen_err={orc['frozen_residual_error']:.4f} motion_err={orc['residual_motion_error']:.4f} "
              f"rel_reduction={orc['relative_error_reduction']:+.3f}")
    for rmv, rows in s["rm_minus_plain"].items():
        print(f"--- RM minus plain: {rmv}")
        for r in rows:
            print(f"  t{r['tau']:g}: RM {r['rm_speedup']:.2f}x/{r['rm_metric']:.2f}dB  "
                  f"plain {r['plain_speedup']:.2f}x/{r['plain_metric']:.2f}dB  "
                  f"Δ={r['mean_delta']:+.3f} CI[{r['ci'][0]},{r['ci'][1]}] win={r['win']:.2f} excl0={r['excl0']}")
