"""E51 aggregation (anaconda env). Turns diagnostics/results.jsonl + trajectory_*.json into:
  outputs/.../summary.json            (headline numbers, smoothness, Pareto, verdict)
  outputs/.../metrics.csv             (per-variant aggregate at the primary skip ratio)
  outputs/.../per_example_metrics.csv (every scored image, wide)

Compute-accounting (honest, two models, derived from recorded forward counts):
  * naive   baseline = 1 edit forward / step. full-cache speedup = n/fwd_edit;
            delta-cache ADDS the v_src base every step (speedup n/(n+fwd_edit) < 1).
  * cfg     baseline = src+edit every step (true-CFG editing, 2n). full-cache freezes the
            whole step; delta-cache freezes only the delta. speedup = 2n/variant_fwd.
The primary scientific axis is quality-vs-SKIP-RATIO; speedups are reported, not headlined.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

import config as C

PRIMARY = C.SKIP_PRIMARY


def _load_results():
    rows = [json.loads(l) for l in (C.DIAG / "results.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    n = df["n_steps"].replace(0, np.nan)
    fe, fs = df["fwd_edit"], df["fwd_src"]
    is_full = df["variant"].isin(["full_compute_reference", "raw_full_prediction_cache",
                                  "spectral_full_prediction_cache"])
    var_naive = np.where(is_full, fe, fe + fs)
    var_cfg = np.where(is_full, 2 * fe, fe + fs)
    df["speedup_naive"] = n / var_naive
    df["speedup_cfg"] = (2 * n) / var_cfg
    df["fwd_total_cfg"] = var_cfg
    return df


def _smoothness():
    """Is Δ_edit smoother (more cacheable) than v_edit? Uses ABSOLUTE adjacent L2 — the error a
    stale-reuse injects into the stepping velocity (same units for both, since v_src is exact for
    delta caching). Per-example ratios avoid one large-scale example dominating a naive pool."""
    files = sorted(C.DIAG.glob("trajectory_*.json"))
    abs_e_means, abs_d_means, spec_e_means, spec_d_means = [], [], [], []
    ratios_raw, ratios_spec, win_raw, win_spec = [], [], [], []
    for fp in files:
        d = json.loads(fp.read_text())
        ae, ad = np.array(d["abs_edit"][1:]), np.array(d["abs_delta"][1:])
        se, sd = np.array(d["spec_edit"][1:]), np.array(d["spec_delta"][1:])
        abs_e_means.append(ae.mean()); abs_d_means.append(ad.mean())
        spec_e_means.append(se.mean()); spec_d_means.append(sd.mean())
        ratios_raw.append(ae.mean() / (ad.mean() + 1e-12))
        ratios_spec.append(se.mean() / (sd.mean() + 1e-12))
        win_raw.append(float((ad < ae).mean()))
        win_spec.append(float((sd < se).mean()))
    return dict(
        abs_edit_mean=float(np.mean(abs_e_means)), abs_delta_mean=float(np.mean(abs_d_means)),
        spec_edit_mean=float(np.mean(spec_e_means)), spec_delta_mean=float(np.mean(spec_d_means)),
        raw_smoother_ratio=float(np.mean(ratios_raw)),       # mean per-example ||Δv_edit|| / ||ΔΔ_edit||
        spec_smoother_ratio=float(np.mean(ratios_spec)),
        raw_delta_smoother=bool(np.mean(ratios_raw) > 1.0),
        spec_delta_smoother=bool(np.mean(ratios_spec) > 1.0),
        delta_smoother_winrate_raw=float(np.mean(win_raw)),
        delta_smoother_winrate_spec=float(np.mean(win_spec)),
        n_examples=len(files),
    )


def _agg(df, group=("variant",)):
    qcols = ["dino_to_ref", "clipI_to_ref", "lpips_to_ref", "psnr_to_ref",
             "clipT_target", "clipT_gain", "clip_dir", "dino_to_src",
             "realized_skip", "speedup_naive", "speedup_cfg"]
    qcols = [c for c in qcols if c in df.columns]
    g = df.groupby(list(group))[qcols].mean().reset_index()
    return g


def _pareto(df):
    sub = df[(df["variant"] != "full_compute_reference")].copy()
    sub = sub[sub["id"].isin(_subset_ids())]
    out = {}
    for v in C.CACHE_VARIANTS:
        d = sub[sub["variant"] == v]
        pts = []
        for rho, dd in d.groupby("skip_ratio"):
            pts.append(dict(skip_ratio=float(rho), realized_skip=float(dd["realized_skip"].mean()),
                            dino_to_ref=float(dd["dino_to_ref"].mean()),
                            lpips_to_ref=float(dd["lpips_to_ref"].mean()),
                            psnr_to_ref=float(dd["psnr_to_ref"].mean()),
                            clipT_gain=float(dd["clipT_gain"].mean()),
                            speedup_cfg=float(dd["speedup_cfg"].mean())))
        out[v] = sorted(pts, key=lambda p: p["skip_ratio"])
    return out


def _subset_ids():
    man = json.loads((C.DIAG / "examples.json").read_text())
    return set(man["pareto_subset"])


def _verdict(smooth, prim, pareto):
    """Primary success: spectral_edit_delta beats spectral_full_prediction on fidelity-to-ref
    at comparable skip, on the primary point AND across the Pareto majority."""
    sd = prim[prim["variant"] == "spectral_edit_delta_cache"]
    sf = prim[prim["variant"] == "spectral_full_prediction_cache"]
    if sd.empty or sf.empty:
        return "NO-GO", {}
    sd, sf = sd.iloc[0], sf.iloc[0]
    prim_dino_win = sd["dino_to_ref"] > sf["dino_to_ref"]
    prim_lpips_win = sd["lpips_to_ref"] < sf["lpips_to_ref"]
    # Pareto: at matched skip_ratio, does spectral-delta dominate spectral-full on dino_to_ref?
    pf = {p["skip_ratio"]: p for p in pareto["spectral_full_prediction_cache"]}
    wins = tot = 0
    for p in pareto["spectral_edit_delta_cache"]:
        if p["skip_ratio"] in pf:
            tot += 1
            wins += int(p["dino_to_ref"] > pf[p["skip_ratio"]]["dino_to_ref"]
                        and p["lpips_to_ref"] <= pf[p["skip_ratio"]]["lpips_to_ref"] + 0.02)
    pareto_winrate = wins / tot if tot else 0.0
    quality_win = (prim_dino_win and prim_lpips_win)
    smoother = smooth["raw_delta_smoother"] or smooth["spec_delta_smoother"]
    ev = dict(prim_dino_win=bool(prim_dino_win), prim_lpips_win=bool(prim_lpips_win),
              pareto_winrate=pareto_winrate, delta_smoother=bool(smoother),
              raw_smoother_winrate=smooth["delta_smoother_winrate_raw"])
    if quality_win and pareto_winrate >= 0.6 and smoother:
        v = "STRONG GO"
    elif (quality_win or pareto_winrate >= 0.6) and smoother:
        v = "GO"
    elif smoother:
        v = "MIXED / INVESTIGATE FURTHER"
    else:
        v = "NO-GO"
    return v, ev


def main():
    df = _load_results()
    df.to_csv(C.OUT / "per_example_metrics.csv", index=False)

    prim = _agg(df[df["is_primary"] == True])  # noqa: E712
    prim.to_csv(C.OUT / "metrics.csv", index=False)

    smooth = _smoothness()
    pareto = _pareto(df)
    cat = _agg(df[df["is_primary"] == True], group=("category", "variant"))  # noqa: E712
    scope = _agg(df[df["is_primary"] == True], group=("scope", "variant"))   # noqa: E712
    verdict, ev = _verdict(smooth, prim, pareto)

    summary = dict(
        phase=C.PHASE_VERSION, n_examples=int(df["id"].nunique()),
        steps=C.STEPS, size=C.SIZE, guidance=C.GUIDANCE, strength=C.STRENGTH,
        primary_skip=PRIMARY,
        smoothness=smooth,
        primary_by_variant=prim.to_dict("records"),
        pareto=pareto,
        category_breakdown=cat.to_dict("records"),
        scope_breakdown=scope.to_dict("records"),
        verdict=verdict, verdict_evidence=ev,
    )
    C.write_json(C.OUT / "summary.json", summary)
    print(f"[analyze] verdict = {verdict}")
    print(f"[analyze] smoothness(abs): edit {smooth['abs_edit_mean']:.3f} vs delta "
          f"{smooth['abs_delta_mean']:.3f} -> {smooth['raw_smoother_ratio']:.2f}x smoother, "
          f"delta wins {smooth['delta_smoother_winrate_raw']*100:.0f}% of steps")
    print(prim[["variant", "dino_to_ref", "lpips_to_ref", "psnr_to_ref", "clipT_gain",
                "realized_skip", "speedup_cfg"]].to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
