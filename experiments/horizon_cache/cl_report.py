"""E60 report — Closed-Loop Residual Motion (online innovation-fit β̂ + midpoint-λ).

Builds reports/horizon_cache_closed_loop.{html,_summary.md,_summary.json}.
The four pre-registered questions:
  1. Does adaptive β̂ match or beat fixed β=0.5?
  2. Does it automatically turn RM off above ~4× (removing E59's fixed-β penalty)?
  3. Does innovation magnitude predict when RM helps?
  4. Does the final rule 'RM iff β̂>0' emerge — frontier ≥ max(plain, fixed-β RM)?

    python -m horizon_cache.cl_report --gen-dir <run> --tau-grid 0.3 ... 1.4 \
        --reports-dir ../reports [--samples-dir <run with --save-all-images>]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
REPO = HERE.parents[1]

from horizon_cache import figures as F
from horizon_cache import cl_analysis as CA
from horizon_cache.so_report import qual_grid, _fmt_ci, git_hash

BANDS = CA.BANDS
FAM_LABEL = {"plain": "plain HorizonCache", "fo": "fixed-β RM (E58)", "fo_mid": "fixed-β RM + midpoint-λ",
             "cl": "closed-loop RM (β̂)", "cl_mid": "closed-loop RM + midpoint-λ"}
FAM_COLOR = {"seacache": F.ACC, "plain": F.WARN, "fo": "#c084fc", "fo_mid": "#38bdf8",
             "cl": F.ACC2, "cl_mid": "#f472b6"}
BAND_KEY = {"plain": "plain", "fo": "fixed_rm", "fo_mid": "fixed_rm_mid",
            "cl": "closed_loop", "cl_mid": "closed_loop_mid"}


# ----------------------------------------------------------------- figures
def extreme_frontier(summ, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax in axes:
        F._style(ax)
        for x in (2, 3, 4, 5):
            ax.axvline(x, color=F.LINE, lw=0.8, ls=":")
    for ax, metric, ylab in [(axes[0], "psnr", "PSNR vs full (dB) ↑"),
                             (axes[1], "lpips", "LPIPS vs full ↓")]:
        sea = sorted((p["speedup"], p[metric]) for p in summ["seacache_points"]
                     if p[metric] is not None)
        if sea:
            ax.plot(*zip(*sea), marker="o", color=FAM_COLOR["seacache"], lw=2.6,
                    label="SeaCache", zorder=8)
        seen = set()
        for v, pts in summ["method_points"].items():
            fam = summ["families"][v]
            xy = sorted((p["speedup"], p[metric]) for p in pts if p[metric] is not None)
            if not xy:
                continue
            lab = FAM_LABEL[fam] if fam not in seen else None
            seen.add(fam)
            ax.plot(*zip(*xy), marker="D", color=FAM_COLOR[fam], lw=1.7,
                    ls="--" if fam.endswith("mid") else "-", alpha=0.85, label=lab, markersize=4)
        ax.set_xlabel("achieved speedup (block-stack-equivalent) →")
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=8, facecolor=F.PANEL, edgecolor=F.LINE, labelcolor=F.INK)
    fig.suptitle("Extreme-speed frontier — SeaCache vs plain vs fixed-β vs closed-loop "
                 "Residual Motion", color=F.INK, fontsize=11)
    fig.tight_layout()
    return F._save(fig, out)


def delta_vs_seacache_by_band(summ, out: Path):
    fams = ["plain", "fo", "cl", "cl_mid"]
    rows = summ["bands"]
    present = [f for f in fams if any(BAND_KEY[f] in r for r in rows)]
    if not rows or not present:
        return None
    fig, ax = plt.subplots(figsize=(max(9.0, 1.7 * len(rows)), 5.0))
    F._style(ax)
    w = 0.8 / len(present)
    for j, f in enumerate(present):
        xs, ys, lo, hi, hatch = [], [], [], [], []
        for i, r in enumerate(rows):
            d = r.get(BAND_KEY[f])
            if not d:
                continue
            xs.append(i + (j - (len(present) - 1) / 2) * w)
            ys.append(d["delta_psnr_vs_seacache"])
            lo.append(d["delta_psnr_vs_seacache"] - d["ci"][0])
            hi.append(d["ci"][1] - d["delta_psnr_vs_seacache"])
            hatch.append(not d["excl0"])
        bars = ax.bar(xs, ys, width=w * 0.92, color=FAM_COLOR[f], label=FAM_LABEL[f],
                      yerr=[lo, hi], capsize=3, ecolor=F.MUT, alpha=0.95)
        for b, h in zip(bars, hatch):
            if h:
                b.set_hatch("//"); b.set_alpha(0.55)
    ax.axhline(0, color=F.INK, lw=1.2)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["band"] for r in rows], fontsize=9)
    ax.set_ylabel("ΔPSNR vs SeaCache at matched achieved speedup (dB)")
    ax.set_title("Method − SeaCache per speed band (best variant per family, paired ±95% CI; "
                 "hatched = CI includes 0)", fontsize=10)
    ax.legend(fontsize=8.5, facecolor=F.PANEL, edgecolor=F.LINE, labelcolor=F.INK)
    fig.tight_layout()
    return F._save(fig, out)


def paired_fig(summ, label, title, ylabel, out: Path):
    """Grouped bar chart of one named paired ablation (cl_minus_fixed / mid_minus_point /
    cl vs fixed against plain) per τ, ±95% CI."""
    entries = [(k, rows) for k, rows in summ["paired"].items() if k.startswith(label + "::")]
    if not entries:
        return None
    n_groups = max(len(r) for _, r in entries)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.05 * n_groups * len(entries)), 4.6))
    F._style(ax)
    ax.axhline(0, color=F.INK, lw=1.2)
    xs_all, lbl_all = [], []
    idx = 0
    for k, rows in sorted(entries):
        v = k.split("::")[1]
        for r in sorted(rows, key=lambda r: r["tau"]):
            c = F.BAD if (r["mean_delta"] < 0 and r["excl0"]) else \
                (F.ACC2 if (r["mean_delta"] > 0 and r["excl0"]) else F.WARN)
            ax.bar(idx, r["mean_delta"], color=c,
                   yerr=[[r["mean_delta"] - r["ci"][0]], [r["ci"][1] - r["mean_delta"]]],
                   capsize=3, ecolor=F.MUT, alpha=0.95 if r["excl0"] else 0.55,
                   hatch=None if r["excl0"] else "//")
            xs_all.append(idx)
            lbl_all.append(f"{v.split('_adaptive')[0]}\nτ{r['tau']:g}\n{r['rm_speedup']:.2f}×")
            idx += 1
        idx += 1
    ax.set_xticks(xs_all)
    ax.set_xticklabels(lbl_all, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return F._save(fig, out)


def rm_penalty_fig(summ, out: Path):
    """Q2 figure: RM−plain for fixed-β vs closed-loop across achieved speedup. The E59
    fixed-β penalty at ≥4.3× must vanish for the closed loop (β̂→0 recovers plain)."""
    series = []
    for k, rows in summ["paired"].items():
        label, v, other = k.split("::")
        if label == "fixed_minus_plain":
            series.append((f"fixed-β: {v}", FAM_COLOR["fo"], "-", rows))
        elif label == "cl_minus_plain" and CA.classify(v)["family"].startswith("cl"):
            fam = CA.classify(v)["family"]
            series.append((f"closed-loop: {v}", FAM_COLOR[fam], "--" if fam.endswith("mid") else "-",
                           rows))
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    F._style(ax)
    ax.axhline(0, color=F.INK, lw=1.2)
    ax.axvspan(4.3, 5.6, color=F.BAD, alpha=0.06)
    for lab, col, ls, rows in sorted(series):
        rows = sorted(rows, key=lambda r: r["rm_speedup"])
        xs = [r["rm_speedup"] for r in rows]
        ys = [r["mean_delta"] for r in rows]
        lo = [r["mean_delta"] - r["ci"][0] for r in rows]
        hi = [r["ci"][1] - r["mean_delta"] for r in rows]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", color=col, ls=ls, lw=1.8,
                    capsize=3, markersize=4.5, label=lab)
    ax.set_xlabel("achieved speedup →")
    ax.set_ylabel("ΔPSNR: RM − plain HorizonCache (dB, paired)")
    ax.set_title("Does β̂ remove the fixed-β penalty at extreme speed? (shaded: the E59 "
                 "inversion region ≥4.3×)", fontsize=10)
    ax.legend(fontsize=8, facecolor=F.PANEL, edgecolor=F.LINE, labelcolor=F.INK)
    fig.tight_layout()
    return F._save(fig, out)


def beta_behavior_fig(summ, trajs, out: Path):
    """Q2/prediction (a): β̂ vs achieved speedup (p25/50/75 + zero fraction) and per-trajectory
    β̂ paths over σ."""
    bb = summ.get("beta_by_tau", [])
    if not bb:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    for ax in axes:
        F._style(ax)
    variants = sorted({r["variant"] for r in bb})
    for v in variants:
        rows = sorted([r for r in bb if r["variant"] == v], key=lambda r: r["speedup"])
        xs = [r["speedup"] for r in rows]
        med = [r["beta_hat_final_p50"] for r in rows]
        lo = [r["beta_hat_final_p50"] - r["beta_hat_final_p25"] for r in rows]
        hi = [r["beta_hat_final_p75"] - r["beta_hat_final_p50"] for r in rows]
        fam = CA.classify(v)["family"]
        axes[0].errorbar(xs, med, yerr=[lo, hi], marker="o", lw=1.8, capsize=3,
                         color=FAM_COLOR.get(fam, F.ACC2),
                         ls="--" if fam.endswith("mid") else "-", label=v, markersize=4)
        zf = [r["zero_frac"] for r in rows]
        axes[1].plot(xs, zf, marker="s", lw=1.8, color=FAM_COLOR.get(fam, F.ACC2),
                     ls="--" if fam.endswith("mid") else "-", label=v, markersize=4)
    axes[0].axhline(0.5, color=F.MUT, ls=":", lw=1.2)
    axes[0].axhline(0.0, color=F.INK, lw=1.0)
    axes[0].set_xlabel("achieved speedup →"); axes[0].set_ylabel("β̂ final (p50, bars p25–p75)")
    axes[0].set_title("prediction (a): β̂ ≈ 0.5 in-band, → 0 at speed", fontsize=9)
    axes[0].legend(fontsize=7, facecolor=F.PANEL, edgecolor=F.LINE, labelcolor=F.INK)
    axes[1].set_xlabel("achieved speedup →"); axes[1].set_ylabel("fraction of trajectories β̂≈0")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("'RM off' fraction (mean used β < 0.05)", fontsize=9)
    # spaghetti of β̂ over σ for a low and a high τ
    if trajs:
        taus = sorted({t["tau"] for t in trajs})
        show = [taus[0], taus[-1]] if len(taus) > 1 else taus
        cols = {show[0]: F.ACC2, show[-1]: F.BAD}
        for t in trajs:
            if t["tau"] not in show:
                continue
            xs = [p["sigma"] for p in t["path"]]
            ys = [p["beta_hat"] for p in t["path"]]
            axes[2].plot(xs, ys, color=cols[t["tau"]], alpha=0.35, lw=1.0)
        for tt in show:
            axes[2].plot([], [], color=cols[tt], label=f"τ={tt:g}")
        axes[2].axhline(0.5, color=F.MUT, ls=":", lw=1.2)
        axes[2].axhline(0.0, color=F.INK, lw=1.0)
        axes[2].invert_xaxis()
        axes[2].set_xlabel("σ (denoising →)"); axes[2].set_ylabel("β̂ (unclamped)")
        axes[2].set_title("per-trajectory β̂ paths", fontsize=9)
        axes[2].legend(fontsize=8, facecolor=F.PANEL, edgecolor=F.LINE, labelcolor=F.INK)
    fig.suptitle("Closed-loop gain behavior — the online β̂ across speed regimes",
                 color=F.INK, fontsize=10.5)
    fig.tight_layout()
    return F._save(fig, out)


def innovation_fig(summ, out: Path):
    """Q3/prediction (b): per-image innovation vs the per-image fixed-RM − plain gain."""
    ig = summ.get("innovation_vs_gain", {})
    sc = ig.get("scatter", {})
    xs = np.asarray(sc.get("innov_rel", []), float)
    ys = np.asarray(sc.get("rm_minus_plain_psnr", []), float)
    if len(xs) < 3:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    F._style(ax)
    taus = np.asarray(sc.get("tau", []), float)
    sct = ax.scatter(xs, ys, c=taus, cmap="viridis", s=16, alpha=0.75)
    cb = fig.colorbar(sct, ax=ax); cb.set_label("τ", color=F.INK)
    cb.ax.yaxis.set_tick_params(color=F.INK)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=F.INK)
    z = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xx, np.polyval(z, xx), color=F.INK, lw=1.4, ls="--")
    ax.axhline(0, color=F.MUT, lw=0.8)
    ax.set_xlabel("per-image innovation ‖e‖₁/‖r‖₁ (mean over refreshes, closed-loop run)")
    ax.set_ylabel("per-image fixed-β RM − plain ΔPSNR (dB)")
    ax.set_title(f"prediction (b): innovation anti-correlates with the RM gain · "
                 f"corr={ig.get('corr'):+.2f} (n={ig.get('n')})", fontsize=10)
    fig.tight_layout()
    return F._save(fig, out)


# ----------------------------------------------------------------- build
def build(gen: Path, tau_grid, reports_dir: Path, sha: str, samples_dir=None, run_dirs=None):
    df = CA.load_df(gen)
    summ = CA.summarize(gen, tau_grid)
    assets = reports_dir / "horizon_cache_closed_loop_assets"
    assets.mkdir(parents=True, exist_ok=True)

    cl_variants = [v for v, f in summ["families"].items() if f.startswith("cl")]
    trajs = []
    if cl_variants:
        trajs = CA.beta_trajectories(gen, cl_variants[0], tau_grid, max_traj=40)

    figs = {}
    figs["frontier"] = extreme_frontier(summ, assets / "extreme_frontier.png")
    for name, fn in [("delta_band", lambda: delta_vs_seacache_by_band(summ, assets / "delta_band.png")),
                     ("cl_minus_fixed", lambda: paired_fig(summ, "cl_minus_fixed",
                        "Closed-loop − fixed-β RM (paired, same base/τ — pure gain-adaptation effect)",
                        "ΔPSNR: β̂ − fixed β (dB)", assets / "cl_minus_fixed.png")),
                     ("mid_minus_point", lambda: paired_fig(summ, "mid_minus_point",
                        "Midpoint-λ − left-endpoint λ (paired — pure quadrature effect)",
                        "ΔPSNR: mid − point (dB)", assets / "mid_minus_point.png")),
                     ("rm_penalty", lambda: rm_penalty_fig(summ, assets / "rm_penalty.png")),
                     ("beta", lambda: beta_behavior_fig(summ, trajs, assets / "beta_behavior.png")),
                     ("innovation", lambda: innovation_fig(summ, assets / "innovation.png"))]:
        f = fn()
        if f:
            figs[name] = f

    # qualitative grids
    sdir = Path(samples_dir) if samples_dir else gen
    grids = []
    if (sdir / "samples").exists():
        try:
            qdf = pd.read_csv(sdir / "metrics.csv")
        except Exception:
            qdf = df
        keys = list(dict.fromkeys(qdf.key.tolist()))
        fam_best = {}
        for v in summ["variants"]:
            fam_best.setdefault(summ["families"][v], v)
        pl_v, fo_v = fam_best.get("plain"), fam_best.get("fo")
        cl_v = fam_best.get("cl") or fam_best.get("cl_mid")
        for tau, label in [(t, l) for t, l in
                           [(0.5, "~2.5×"), (0.65, "~3.4×"), (1.0, "~4.5×"), (1.4, "~5.3×")]
                           if t in tau_grid]:
            methods = [(f"seacache_t{tau:g}", "SeaCache")]
            for v, lab in [(pl_v, "plain"), (fo_v, "fixed-β RM"), (cl_v, "closed-loop RM")]:
                if v:
                    methods.append((f"horizon_{v}_t{tau:g}", lab))
            g = qual_grid(sdir / "samples", qdf, tau, methods, keys,
                          f"Generated samples at τ={tau:g} ({label} band)",
                          assets / f"qual_t{tau:g}.png")
            if g:
                grids.append((g, tau, label))

    bands = summ["bands"]
    highest = summ["highest_positive_vs_seacache_band"]
    bb = summ.get("beta_by_tau", [])
    ig = summ.get("innovation_vs_gain", {})

    # ---- pre-registered checks + verdicts ----
    cf_rows = [r for k, rows in summ["paired"].items() if k.startswith("cl_minus_fixed::")
               for r in rows]
    cf_lp = [r for k, rows in summ["paired_lpips"].items() if k.startswith("cl_minus_fixed::")
             for r in rows]
    mp_rows = [r for k, rows in summ["paired"].items() if k.startswith("mid_minus_point::")
               for r in rows]
    cl_plain = {k.split("::")[1]: rows for k, rows in summ["paired"].items()
                if k.startswith("cl_minus_plain::") and CA.classify(k.split("::")[1])["family"].startswith("cl")}
    fx_plain = [r for k, rows in summ["paired"].items() if k.startswith("fixed_minus_plain::")
                for r in rows]

    # prediction (a): β̂ in-band near the E58 value, falling toward 0 at speed
    inband = [r for r in bb if r["speedup"] < 3.0]
    fast = [r for r in bb if r["speedup"] >= 4.2]
    pred_a = None
    if inband and fast:
        in_p50 = float(np.median([r["beta_hat_final_p50"] for r in inband]))
        fast_p50 = float(np.median([r["beta_hat_final_p50"] for r in fast]))
        pred_a = {"inband_beta_p50": round(in_p50, 3), "fast_beta_p50": round(fast_p50, 3),
                  "fast_zero_frac": round(float(np.median([r["zero_frac"] for r in fast])), 3),
                  "pass": bool(0.25 <= in_p50 <= 0.9 and fast_p50 < in_p50)}
    # prediction (b): innovation anti-correlates with the RM−plain gain
    pred_b = {"corr": ig.get("corr"), "n": ig.get("n"),
              "pass": bool(ig.get("corr") is not None and ig["corr"] < -0.1)}

    # penalty removal: fixed-β CI-negative vs plain at high speed, closed loop not
    fx_bad = [r for r in fx_plain if r["rm_speedup"] >= 4.2 and r["mean_delta"] < 0 and r["excl0"]]
    cl_bad = [r for rows in cl_plain.values() for r in rows
              if r["rm_speedup"] >= 4.2 and r["mean_delta"] < 0 and r["excl0"]]
    penalty_removed = bool(fx_bad) and bool(cl_plain) and not cl_bad

    cl_wins = any(r["mean_delta"] > 0.05 and r["excl0"] for r in cf_rows)
    cl_loses_inband = any(r["mean_delta"] < -0.05 and r["excl0"] and r["rm_speedup"] < 3.8
                          for r in cf_rows)
    lpips_reg = any(r["mean_delta"] < -0.002 and r["excl0"] for r in cf_lp)
    cl_verdict = ("not_run" if not cf_rows else
                  ("STRONG_KEEP" if (penalty_removed and not cl_loses_inband and not lpips_reg) else
                   ("KEEP" if ((cl_wins or penalty_removed) and not cl_loses_inband and not lpips_reg) else
                    ("PARK" if not cl_loses_inband else "KILL"))))
    mid_wins = any(r["mean_delta"] > 0.05 and r["excl0"] for r in mp_rows)
    mid_loses = any(r["mean_delta"] < -0.05 and r["excl0"] for r in mp_rows)
    mid_verdict = ("not_run" if not mp_rows else
                   ("KEEP" if (mid_wins and not mid_loses) else
                    ("KILL" if (mid_loses and not mid_wins) else "PARK")))

    best_cf = max(cf_rows, key=lambda r: r["mean_delta"], default=None)
    cl_sea_best = None
    for v in cl_variants:
        for r in summ["vs_seacache"][v]:
            if cl_sea_best is None or r["delta"] > cl_sea_best["delta"]:
                cl_sea_best = {**r, "variant": v}

    headline = ""
    if cl_sea_best:
        headline += (f"Closed-loop RM ({cl_sea_best['variant']}) vs SeaCache at matched speed: best "
                     f"{cl_sea_best['delta']:+.2f} dB @ {cl_sea_best['speedup']:.2f}× "
                     f"(CI {_fmt_ci(cl_sea_best['ci'])}); highest CI-positive band "
                     f"{highest.get('closed_loop') or highest.get('closed_loop_mid') or 'none'}. ")
    if best_cf:
        headline += (f"β̂ − fixed β: best {best_cf['mean_delta']:+.2f} dB @ "
                     f"{best_cf['rm_speedup']:.2f}× (CI {_fmt_ci(best_cf['ci'])}); "
                     f"high-speed fixed-β penalty {'REMOVED' if penalty_removed else 'NOT removed'} "
                     f"by the closed loop → {cl_verdict}.")
    pa_txt = (f"β̂ p50 in-band {pred_a['inband_beta_p50']} vs fast {pred_a['fast_beta_p50']} "
              f"(zero-frac {pred_a['fast_zero_frac']}) — prediction (a) "
              f"{'PASS' if pred_a['pass'] else 'FAIL'}" if pred_a else "prediction (a): n/a")
    pb_txt = (f"corr(innovation, RM−plain gain) = {pred_b['corr']:+.2f} (n={pred_b['n']}) — "
              f"prediction (b) {'PASS' if pred_b['pass'] else 'FAIL'}"
              if pred_b.get("corr") is not None else "prediction (b): n/a")

    du = lambda p: F.data_uri(Path(p))

    def badge(v):
        c = {"STRONG_KEEP": "sk", "KEEP": "sk", "PARK": "park"}.get(v, "kill")
        return f"<span class='badge {c}'>{v.replace('_', ' ')}</span>"

    H = [f"<div class='wrap'><h1>Closed-Loop Residual Motion (E60)</h1>",
         f"<p class='sub'>Online innovation-fit secant gain β̂ + midpoint-λ · zero extra forwards · "
         f"FLUX 512px/28 · commit <code>{sha}</code> · cluster H100</p>"]
    H.append("<h2>1 · Executive summary</h2><div class='card'>"
             f"<p>Closed-loop β̂: {badge(cl_verdict)} · midpoint-λ: "
             f"{badge(mid_verdict) if mid_verdict != 'not_run' else 'not run'}</p>"
             f"<p class='hl'><b>Headline.</b> {headline}</p>"
             f"<p><b>Pre-registered predictions.</b> {pa_txt}; {pb_txt}.</p></div>")
    H.append("<h2>2 · Method</h2><div class='card'><ul>"
             "<li><b>Observation</b>: at every refresh the sampler computes the true residual r_k anyway; "
             "the previous anchor pair implies the forecast r̂_k = r_{k−1} + β·λ_k·Δr. The innovation "
             "e_k = r_k − r̂_k is a causal, per-trajectory, zero-cost measurement of the secant model.</li>"
             "<li><b>Estimator</b>: exponentially-forgetting regularized least squares over anchors, "
             "each observation normalized by ‖Δr‖²: β̂ = (μ·β_prior + Σ w^age λ⟨Δr,y⟩/‖Δr‖²)/(μ + Σ w^age λ²), "
             "clamped to [0, β_max]. β̂→0 recovers plain HorizonCache <i>by construction</i>.</li>"
             "<li><b>Midpoint-λ</b>: cached steps evaluate λ at (σ_i+σ_target)/2 — a midpoint-rule "
             "quadrature of the moving residual path (second-order accurate integration without the "
             "curvature term E59 killed).</li>"
             "<li>Defaults β_prior=0.5 (the E58 population value), w=0.85, μ=1.0, β_max=1.0. "
             "rm_beta_mode='fixed' is bit-identical to E58 (unit-tested); protocol/fixture/bootstrap "
             "unchanged from E56/E58/E59.</li></ul></div>")
    H.append("<h2>3 · Extreme-speed frontier</h2>")
    H.append(f"<div class='fig'><img src='{du(figs['frontier'])}'><div class='cap'>PSNR (left) and "
             "LPIPS (right) vs achieved speedup; dotted verticals at 2/3/4/5×.</div></div>")
    if "delta_band" in figs:
        H.append(f"<div class='fig'><img src='{du(figs['delta_band'])}'><div class='cap'>Best variant "
                 "per family minus SeaCache per speed band, paired ±95% CI (hatched = CI includes 0)."
                 "</div></div>")
    H.append("<h3>Δ vs SeaCache by speed band (best per family)</h3><table>"
             "<tr><th>band</th><th>plain</th><th>fixed-β RM</th><th>closed-loop</th>"
             "<th>closed-loop+mid</th></tr>")
    for r in bands:
        cells = []
        for k in ("plain", "fixed_rm", "closed_loop", "closed_loop_mid"):
            d = r.get(k)
            cells.append("—" if not d else
                         f"<span class='{'pos' if d['delta_psnr_vs_seacache'] > 0 else 'neg'}'>"
                         f"{d['delta_psnr_vs_seacache']:+.2f}</span>{'*' if d['excl0'] else ''} "
                         f"<span class='sub'>@{d['speedup']:.2f}× w{d['win']*100:.0f}%</span>")
        band_lab = r["band"] + ("" if r.get("seacache_reachable", True) else " †")
        H.append(f"<tr><td>{band_lab}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    sea_max = summ.get("seacache_max_speedup")
    H.append("</table><p class='sub'>* = 95% CI excludes 0. "
             + (f"† = beyond the swept SeaCache frontier (tops out at {sea_max:.2f}× — integer "
                f"refresh-count quantization); deltas there are conservative. " if sea_max else "")
             + "Highest CI-positive band vs SeaCache: "
             + ", ".join(f"{k}: <b>{v or 'none'}</b>" for k, v in highest.items()) + "</p>")
    H.append("<h2>4 · Closed loop vs fixed gain (Q1) — and the high-speed penalty (Q2)</h2>")
    if "cl_minus_fixed" in figs:
        H.append(f"<div class='fig'><img src='{du(figs['cl_minus_fixed'])}'><div class='cap'>Paired "
                 "β̂ − fixed-β ΔPSNR at identical compute — the pure value of adapting the gain."
                 "</div></div>")
    if "rm_penalty" in figs:
        H.append(f"<div class='fig'><img src='{du(figs['rm_penalty'])}'><div class='cap'>RM − plain "
                 "for fixed β (purple) vs closed loop: the fixed-β inversion at ≥4.3× (shaded) is the "
                 "failure the closed loop must remove.</div></div>")
    H.append("<h2>5 · β̂ behavior (prediction a) & innovation (prediction b, Q3)</h2>")
    if "beta" in figs:
        H.append(f"<div class='fig'><img src='{du(figs['beta'])}'><div class='cap'>β̂ vs achieved "
                 "speedup (left), 'RM off' fraction (middle), per-trajectory β̂ paths (right)."
                 "</div></div>")
    if "innovation" in figs:
        H.append(f"<div class='fig'><img src='{du(figs['innovation'])}'><div class='cap'>Per-image "
                 "innovation vs the per-image fixed-RM − plain gain.</div></div>")
    H.append(f"<div class='card'><p>{pa_txt}.</p><p>{pb_txt}.</p></div>")
    if "mid_minus_point" in figs:
        H.append("<h2>6 · Midpoint-λ ablation</h2>")
        H.append(f"<div class='fig'><img src='{du(figs['mid_minus_point'])}'><div class='cap'>Paired "
                 "midpoint − left-endpoint λ at identical compute (pure quadrature effect)."
                 "</div></div>")
    if grids:
        H.append("<h2>7 · Qualitative grids</h2>")
        for g, tau, label in grids:
            H.append(f"<div class='fig'><img src='{du(g)}'><div class='cap'>τ={tau:g} ({label})."
                     f"</div></div>")
    H.append("<h2>8 · Verdicts</h2><div class='card'><table><tr><th>question</th><th>verdict</th></tr>"
             f"<tr><td>closed-loop β̂ (online gain)</td><td>{badge(cl_verdict)}</td></tr>"
             f"<tr><td>midpoint-λ quadrature</td><td>{badge(mid_verdict) if mid_verdict != 'not_run' else 'not run'}</td></tr>"
             f"<tr><td>prediction (a): β̂ regime-dependence</td><td>{'PASS' if pred_a and pred_a['pass'] else 'FAIL/n-a'}</td></tr>"
             f"<tr><td>prediction (b): innovation ↔ gain</td><td>{'PASS' if pred_b.get('pass') else 'FAIL/n-a'}</td></tr>"
             "</table></div>")
    H.append("<h2>9 · Artifacts</h2><div class='card'><ul>"
             "<li>report: <code>reports/horizon_cache_closed_loop.html</code></li>"
             "<li>summary: <code>reports/horizon_cache_closed_loop_summary.{md,json}</code></li>"
             f"<li>run: <code>{gen}</code></li><li>figures: <code>{assets}</code></li>"
             f"<li>commit: <code>{sha}</code></li></ul></div></div>")

    style = (":root{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;--mut:#9aa7b4;--acc:#6ea8fe;"
             "--good:#7ee787;--warn:#f0b429;--bad:#ff7b72;--line:#283039}"
             "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);"
             "font:15px/1.6 -apple-system,Segoe UI,Roboto,Arial,sans-serif}.wrap{max-width:1020px;"
             "margin:0 auto;padding:32px 22px 80px}h1{font-size:25px}h2{font-size:20px;margin:1.6em 0 .4em;"
             "border-bottom:1px solid var(--line);padding-bottom:.3em}h3{color:var(--acc);font-size:15px}"
             ".sub{color:var(--mut);font-size:13px}code{background:#0b0e13;padding:.1em .35em;border-radius:4px;"
             "color:#d2a8ff;font-size:.9em}.badge{display:inline-block;padding:.2em .6em;border-radius:999px;"
             "font-weight:700;font-size:12.5px}.sk{background:rgba(126,231,135,.15);color:var(--good);"
             "border:1px solid var(--good)}.kill{background:rgba(255,123,114,.14);color:var(--bad);"
             "border:1px solid var(--bad)}.park{background:rgba(240,180,41,.13);color:var(--warn);"
             "border:1px solid var(--warn)}.card{background:var(--panel);border:1px solid var(--line);"
             "border-radius:12px;padding:16px 18px;margin:12px 0}table{border-collapse:collapse;width:100%;"
             "font-size:13px}th,td{border:1px solid var(--line);padding:6px 9px;text-align:right}"
             "td:first-child,th:first-child{text-align:left}th{color:var(--mut)}.pos{color:var(--good)}"
             ".neg{color:var(--bad)}img{max-width:100%;border:1px solid var(--line);border-radius:10px;"
             "background:#000}.fig{margin:14px 0}.cap{color:var(--mut);font-size:12.5px}"
             ".hl{background:rgba(110,168,254,.09);border-left:3px solid var(--acc);padding:.5em .8em;"
             "border-radius:0 8px 8px 0}ul{margin:.3em 0 .5em 1.1em}")
    html = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><style>{style}</style>"
            f"<title>E60 Closed-Loop Residual Motion</title></head><body>{''.join(H)}</body></html>")
    (reports_dir / "horizon_cache_closed_loop.html").write_text(html)

    sj = {
        "status": "DONE", "git_commit": sha,
        "run_dirs": [str(x) for x in (run_dirs or [gen])],
        "main_verdict": f"closed_loop={cl_verdict}; midpoint_lambda={mid_verdict}",
        "headline_claim": headline,
        "best_closed_loop_vs_seacache": ({
            "method": cl_sea_best["variant"], "tau": cl_sea_best["tau"],
            "speedup": round(cl_sea_best["speedup"], 3),
            "delta_psnr_vs_seacache": round(cl_sea_best["delta"], 3),
            "ci95": [round(cl_sea_best["ci"][0], 3), round(cl_sea_best["ci"][1], 3)],
            "win_rate": round(cl_sea_best["win"], 3)} if cl_sea_best else {}),
        "best_cl_minus_fixed": ({
            "variant": best_cf["rm_variant"], "tau": best_cf["tau"],
            "speedup": round(best_cf["rm_speedup"], 3),
            "delta_psnr": round(best_cf["mean_delta"], 3),
            "ci95": [round(best_cf["ci"][0], 3), round(best_cf["ci"][1], 3)],
            "excl0": best_cf["excl0"]} if best_cf else {}),
        "cl_minus_fixed_by_tau": [
            {"variant": r["rm_variant"], "tau": r["tau"], "speedup": round(r["rm_speedup"], 3),
             "delta_psnr": round(r["mean_delta"], 3),
             "ci95": [round(r["ci"][0], 3), round(r["ci"][1], 3)], "excl0": r["excl0"],
             "win": round(r["win"], 3)} for r in sorted(cf_rows, key=lambda r: (r["rm_variant"], r["tau"]))],
        "penalty_removed_at_high_speed": penalty_removed,
        "prediction_a_beta_regime": pred_a,
        "prediction_b_innovation_gain": {"corr": pred_b.get("corr"), "n": pred_b.get("n"),
                                          "pass": pred_b.get("pass")},
        "beta_by_tau": bb,
        "highest_positive_vs_seacache_band": highest,
        "seacache_max_speedup": summ.get("seacache_max_speedup"),
        "bands": bands,
        "verdicts": {"closed_loop": cl_verdict, "midpoint_lambda": mid_verdict,
                     "prediction_a": bool(pred_a and pred_a["pass"]),
                     "prediction_b": bool(pred_b.get("pass"))},
        "artifacts": {"html_report": "reports/horizon_cache_closed_loop.html",
                      "summary_md": "reports/horizon_cache_closed_loop_summary.md",
                      "summary_json": "reports/horizon_cache_closed_loop_summary.json",
                      "metrics_json": str(gen / "metrics.json"), "figures_dir": str(assets)},
    }
    (reports_dir / "horizon_cache_closed_loop_summary.json").write_text(
        json.dumps(sj, indent=2, default=float))

    md = [f"# E60 — Closed-Loop Residual Motion (online β̂ + midpoint-λ)", "",
          f"**Commit** `{sha}` · FLUX 512px/28 · cluster H100", "",
          f"**Headline.** {headline}", "",
          f"**Predictions.** {pa_txt}; {pb_txt}.", "",
          "## Δ vs SeaCache by speed band (best per family; * = CI excludes 0)", "",
          "| band | plain | fixed-β RM | closed-loop | closed-loop+mid |", "|---|---|---|---|---|"]
    for r in bands:
        cells = []
        for k in ("plain", "fixed_rm", "closed_loop", "closed_loop_mid"):
            d = r.get(k)
            cells.append("—" if not d else f"{d['delta_psnr_vs_seacache']:+.2f}"
                                           f"{'*' if d['excl0'] else ''} @{d['speedup']:.2f}×")
        md.append(f"| {r['band']}{'' if r.get('seacache_reachable', True) else ' †'} | "
                  + " | ".join(cells) + " |")
    md += ["", "## β̂ − fixed β (paired, per τ)", ""]
    for r in sorted(cf_rows, key=lambda r: (r["rm_variant"], r["tau"])):
        md.append(f"- {r['rm_variant']} τ{r['tau']:g} ({r['rm_speedup']:.2f}×): "
                  f"{r['mean_delta']:+.3f} {_fmt_ci(r['ci'])}{' *' if r['excl0'] else ''}")
    md += ["", "## Verdicts", "",
           f"- closed-loop β̂: **{cl_verdict}**", f"- midpoint-λ: **{mid_verdict}**",
           f"- prediction (a): **{'PASS' if pred_a and pred_a['pass'] else 'FAIL/n-a'}**",
           f"- prediction (b): **{'PASS' if pred_b.get('pass') else 'FAIL/n-a'}**"]
    (reports_dir / "horizon_cache_closed_loop_summary.md").write_text("\n".join(md))
    return sj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--samples-dir", default="")
    ap.add_argument("--run-dirs", nargs="*", default=None)
    ap.add_argument("--tau-grid", type=float, nargs="+",
                    default=[0.3, 0.5, 0.575, 0.65, 0.8, 1.0, 1.2, 1.4])
    ap.add_argument("--reports-dir", default=str(REPO / "reports"))
    a = ap.parse_args()
    reports = Path(a.reports_dir); reports.mkdir(parents=True, exist_ok=True)
    sj = build(Path(a.gen_dir), a.tau_grid, reports, git_hash(),
               samples_dir=a.samples_dir or None, run_dirs=a.run_dirs)
    print(json.dumps({"status": sj["status"], "main_verdict": sj["main_verdict"],
                      "best_closed_loop_vs_seacache": sj["best_closed_loop_vs_seacache"],
                      "best_cl_minus_fixed": sj["best_cl_minus_fixed"],
                      "penalty_removed_at_high_speed": sj["penalty_removed_at_high_speed"],
                      "prediction_a": sj["prediction_a_beta_regime"],
                      "prediction_b": sj["prediction_b_innovation_gain"],
                      "verdicts": sj["verdicts"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
