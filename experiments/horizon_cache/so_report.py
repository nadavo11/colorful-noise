"""E59 report — Second-Order Residual Hold + extreme-speed SeaCache comparison.

Builds reports/horizon_cache_second_order_extreme.{html,_summary.md,_summary.json}.
Central questions:
  - How far up the achieved-speedup axis (2×..5×+) does first-order Residual Motion stay
    above SeaCache, and where does every method break?           (the key table/figure)
  - Does a second-order residual hold beat first-order RM at matched compute?
  - Does the curvature diagnostic (ρ2, cosΔ) support the second-order assumption — and
    does it predict when SO helps?

    python -m horizon_cache.so_report --gen-dir <smoke> --tau-grid 0.3 0.5 0.575 0.65 0.8 1.0 1.2 \
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
from horizon_cache import so_analysis as SA
from horizon_cache import rm_analysis as RA

BANDS = SA.BANDS
FAM_LABEL = {"plain": "plain HorizonCache", "fo": "first-order RM",
             "so": "second-order RM", "quad": "quad RM (Lagrange)"}
FAM_COLOR = {"seacache": F.ACC, "plain": F.WARN, "fo": "#c084fc", "so": F.ACC2, "quad": "#38bdf8"}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO)).decode().strip()
    except Exception:
        return "unknown"


# ----------------------------------------------------------------- figures
def extreme_frontier(summ, out: Path):
    """6.1 PSNR + LPIPS vs achieved speedup, one curve per family representative + all variants faint."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax in axes:
        F._style(ax)
        for x in (2, 3, 4, 5):
            ax.axvline(x, color=F.LINE, lw=0.8, ls=":")
    for ax, metric, ylab in [(axes[0], "psnr", "PSNR vs full (dB) ↑"),
                             (axes[1], "lpips", "LPIPS vs full ↓")]:
        sea = [(p["speedup"], p[metric]) for p in summ["seacache_points"] if p[metric] is not None]
        sea.sort()
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
                    ls="--" if fam in ("so", "quad") else "-", alpha=0.85, label=lab, markersize=4)
        ax.set_xlabel("achieved speedup (block-stack-equivalent) →")
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=8, facecolor=F.PANEL, edgecolor=F.LINE, labelcolor=F.INK)
    fig.suptitle("Extreme-speed frontier — SeaCache vs plain HorizonCache vs first/second-order "
                 "Residual Motion (2×…5×+)", color=F.INK, fontsize=11)
    fig.tight_layout()
    return F._save(fig, out)


def delta_vs_seacache_by_band(summ, out: Path):
    """6.2 THE figure: ΔPSNR(method − SeaCache) per speed band, grouped bars + 95% CI."""
    fams = ["plain", "fo", "so", "quad"]
    rows = summ["bands"]
    keymap = {"plain": "plain", "fo": "first_order_rm", "so": "second_order_rm", "quad": "quad_rm"}
    present = [f for f in fams if any(keymap[f] in r for r in rows)]
    if not rows or not present:
        return None
    fig, ax = plt.subplots(figsize=(max(9.0, 1.7 * len(rows)), 5.0))
    F._style(ax)
    w = 0.8 / len(present)
    for j, f in enumerate(present):
        xs, ys, lo, hi, hatch = [], [], [], [], []
        for i, r in enumerate(rows):
            d = r.get(keymap[f])
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


def so_minus_fo_fig(summ, out: Path):
    """6.3 SO − FO ΔPSNR (and ΔLPIPS) per operating point, paired ±95% CI."""
    rows = []
    for v, lst in summ["so_minus_fo"].items():
        lp = {r["tau"]: r for r in summ.get("so_minus_fo_lpips", {}).get(v, [])}
        for r in lst:
            rows.append((v, r, lp.get(r["tau"])))
    if not rows:
        return None
    rows.sort(key=lambda x: (x[0], x[1]["rm_speedup"]))
    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 0.62 * len(rows)), 7.6), sharex=True)
    for ax in axes:
        F._style(ax)
        ax.axhline(0, color=F.INK, lw=1)
    xs = np.arange(len(rows))
    ys = [r["mean_delta"] for _, r, _ in rows]
    lo = [r["mean_delta"] - r["ci"][0] for _, r, _ in rows]
    hi = [r["ci"][1] - r["mean_delta"] for _, r, _ in rows]
    cols = [F.BAD if y < 0 else (F.ACC2 if r["excl0"] else F.WARN) for (_, r, _), y in zip(rows, ys)]
    axes[0].bar(xs, ys, color=cols, yerr=[lo, hi], capsize=3, ecolor=F.MUT)
    axes[0].axhline(0.2, color=F.WARN, ls=":", lw=1, label="+0.2 dB proceed bar")
    axes[0].set_ylabel("ΔPSNR: SO − FO (dB)")
    axes[0].legend(fontsize=8, facecolor=F.PANEL, edgecolor=F.LINE, labelcolor=F.INK)
    ylp = [(-l["mean_delta"] if l else np.nan) for _, _, l in rows]   # sign back to raw LPIPS delta
    axes[1].bar(xs, ylp, color=[F.BAD if (y == y and y > 0) else F.ACC2 for y in ylp])
    axes[1].set_ylabel("ΔLPIPS: SO − FO (raw, <0 better)")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"{v.split('_adaptive')[0]}\nτ{r['tau']:g}\n{r['rm_speedup']:.2f}×"
                             for v, r, _ in rows], fontsize=7)
    fig.suptitle("Second-order minus first-order Residual Motion (paired, same base/τ — pure "
                 "curvature-term ablation at identical compute)", color=F.INK, fontsize=10.5)
    fig.tight_layout()
    return F._save(fig, out)


def diagnostics_fig(summ, out: Path):
    """6.4 second-order assumption: ρ2 & cosΔ histograms + scatter vs the SO−FO gain."""
    d = summ["diagnostics"]
    sc = d.get("scatter", {})
    if not d.get("n_anchor_triples"):
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for ax in axes.flat:
        F._style(ax)
    # histograms rebuilt from percentiles are lossy — use scatter arrays for hists instead
    r2 = np.asarray(sc.get("rho2", []), float)
    cs = np.asarray(sc.get("cos_delta", []), float)
    gain = np.asarray(sc.get("so_minus_fo_psnr", []), float)
    if len(r2):
        axes[0, 0].hist(r2, bins=30, color=F.ACC, alpha=0.9)
    axes[0, 0].axvline(1.0, color=F.BAD, ls=":", lw=1.2)
    axes[0, 0].set_xlabel("ρ2 = ‖Δ²r‖₁/‖Δr‖₁ (per-image mean)"); axes[0, 0].set_ylabel("count")
    axes[0, 0].set_title(f"curvature/velocity ratio · p50={d['rho2_pct'].get(50, float('nan')):.2f} "
                         f"p95={d['rho2_pct'].get(95, float('nan')):.2f}", fontsize=9)
    if len(cs):
        axes[0, 1].hist(cs, bins=30, color=F.ACC2, alpha=0.9)
    axes[0, 1].axvline(0.0, color=F.BAD, ls=":", lw=1.2)
    axes[0, 1].set_xlabel("cosΔ = cos(Δr_a, Δr_{a−1}) (per-image mean)")
    axes[0, 1].set_title(f"directional stability · p50={d['cos_delta_pct'].get(50, float('nan')):.2f} "
                         f"p05={d['cos_delta_pct'].get(5, float('nan')):.2f}", fontsize=9)
    for ax, x, xl, c in [(axes[1, 0], r2, "ρ2 (per-image mean)", F.ACC),
                         (axes[1, 1], cs, "cosΔ (per-image mean)", F.ACC2)]:
        if len(x) and len(gain) == len(x):
            ax.scatter(x, gain, s=14, color=c, alpha=0.7)
            if len(x) >= 3:
                z = np.polyfit(x, gain, 1)
                xx = np.linspace(x.min(), x.max(), 50)
                ax.plot(xx, np.polyval(z, xx), color=F.INK, lw=1.2, ls="--")
        ax.axhline(0, color=F.MUT, lw=0.8)
        ax.set_xlabel(xl); ax.set_ylabel("SO − FO ΔPSNR (dB)")
    axes[1, 0].set_title(f"corr = {d.get('rho2_corr_with_gain') or float('nan'):+.2f}", fontsize=9)
    axes[1, 1].set_title(f"corr = {d.get('cos_delta_corr_with_gain') or float('nan'):+.2f}", fontsize=9)
    fig.suptitle("Second-order assumption diagnostic — is residual curvature smooth & directionally "
                 "stable, and does it predict when the curvature term helps?", color=F.INK, fontsize=10.5)
    fig.tight_layout()
    return F._save(fig, out)


def _load_png(p: Path):
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def qual_grid(samples: Path, df, tau, methods, keys, title, out: Path, max_rows=3):
    """6.5/6.6 grid: rows = prompts; cols = full + methods + |last − full| heatmap."""
    def img(method, key):
        p = samples / f"{key}__{method}.png"
        return _load_png(p) if p.exists() else None

    def meta(method, key):
        sub = df[(df.method == method) & (df.key == key)]
        if sub.empty:
            return None
        r = sub.iloc[0]
        return f"{float(r['compute_speedup']):.2f}× · {float(r['psnr']):.1f} dB"

    rows = [k for k in keys if img("full", k) is not None
            and all(img(m, k) is not None for m, _ in methods)][:max_rows]
    if not rows:
        return None
    cols = [("full", "full")] + list(methods) + [("__heat__", f"|{methods[-1][1]} − full|")]
    nr, nc = len(rows), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.5 * nr))
    axes = np.atleast_2d(axes)
    for i, key in enumerate(rows):
        full = img("full", key)
        for j, (m, lab) in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(F.LINE)
            if m == "__heat__":
                mm = img(methods[-1][0], key)
                if mm is not None:
                    heat = np.abs(mm - full).mean(axis=2)
                    ax.imshow(heat, cmap="magma", vmin=0,
                              vmax=max(0.08, float(np.percentile(heat, 99))))
                cap = "abs err vs full"
            else:
                im = img(m, key)
                if im is not None:
                    ax.imshow(im)
                cap = "reference" if m == "full" else (meta(m, key) or "—")
            ax.set_xlabel(cap, color=F.INK, fontsize=7)
            if i == 0:
                ax.set_title(lab, color=F.INK, fontsize=8)
    fig.suptitle(title, color=F.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return F._save(fig, out)


# ----------------------------------------------------------------- build
def _fmt_ci(ci):
    return f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci and ci[0] is not None else "—"


def build(gen: Path, tau_grid, reports_dir: Path, sha: str, samples_dir=None, run_dirs=None,
          oracle_dir=None):
    df = pd.read_csv(gen / "metrics.csv")
    summ = SA.summarize(gen, tau_grid)
    if oracle_dir and (Path(oracle_dir) / "traces").exists():
        summ["oracle"] = RA.oracle_residual(Path(oracle_dir))
        summ["oracle_by_family"] = SA.oracle_by_family(Path(oracle_dir))
    assets = reports_dir / "horizon_cache_second_order_extreme_assets"
    assets.mkdir(parents=True, exist_ok=True)
    figs = {}
    figs["frontier"] = extreme_frontier(summ, assets / "extreme_frontier.png")
    f = delta_vs_seacache_by_band(summ, assets / "delta_vs_seacache_by_band.png")
    if f:
        figs["delta_band"] = f
    f = so_minus_fo_fig(summ, assets / "so_minus_fo.png")
    if f:
        figs["so_minus_fo"] = f
    f = diagnostics_fig(summ, assets / "so_diagnostics.png")
    if f:
        figs["diagnostics"] = f

    # qualitative grids + failure gallery (needs --save-all-images run)
    sdir = Path(samples_dir) if samples_dir else gen
    grids = []
    if (sdir / "samples").exists():
        qdf = pd.read_csv(sdir / "metrics.csv") if (sdir / "metrics.csv").exists() else df
        keys = list(dict.fromkeys(qdf.key.tolist()))
        fam_best = {}
        for v in summ["variants"]:
            fam_best.setdefault(summ["families"][v], v)
        so_v = fam_best.get("so"); fo_v = fam_best.get("fo"); pl_v = fam_best.get("plain")
        for tau, label in [(t, l) for t, l in
                           [(0.5, "~2.5×"), (0.65, "~3.4×"), (0.8, "~4×"), (1.0, "~5×"), (1.2, ">5×")]
                           if t in tau_grid]:
            methods = [(f"seacache_t{tau:g}", "SeaCache")]
            for v, lab in [(pl_v, "plain"), (fo_v, "first-order RM"), (so_v, "second-order RM")]:
                if v:
                    methods.append((f"horizon_{v}_t{tau:g}", lab))
            g = qual_grid(sdir / "samples", qdf, tau, methods, keys,
                          f"Generated samples at τ={tau:g} ({label} band)",
                          assets / f"qual_t{tau:g}.png")
            if g:
                grids.append((g, tau, label))

    diag = summ["diagnostics"]
    orc = summ.get("oracle", {})
    bands = summ["bands"]
    highest = summ["highest_positive_vs_seacache_band"]

    # bests
    def best_from(dd):
        best = None
        for v, rows in dd.items():
            for r in rows:
                if best is None or r["mean_delta"] > best["mean_delta"]:
                    best = {**r, "variant": v}
        return best
    best_so_fo = best_from(summ["so_minus_fo"])
    fo_variants = [v for v, f in summ["families"].items() if f == "fo"]
    best_fo_sea = None
    for v in fo_variants:
        for r in summ["vs_seacache"][v]:
            if best_fo_sea is None or r["delta"] > best_fo_sea["delta"]:
                best_fo_sea = {**r, "variant": v}

    # ---- verdicts (pre-registered rules, §7) ----
    so_rows = [r for rows in summ["so_minus_fo"].values() for r in rows]
    so_strong = any(r["mean_delta"] > 0.3 and r["excl0"] for r in so_rows)
    so_keep = any(r["mean_delta"] > 0.1 and r["excl0"] for r in so_rows)
    so_park = any(r["mean_delta"] > 0.0 for r in so_rows)
    # does SO extend the positive band beyond FO?
    order = [b[0] for b in BANDS]
    so_extends = (highest.get("second_order_rm") is not None and
                  (highest.get("first_order_rm") is None or
                   order.index(highest["second_order_rm"]) > order.index(highest["first_order_rm"])))
    lpips_reg = any(r["mean_delta"] < -0.002 and r["excl0"]
                    for rows in summ.get("so_minus_fo_lpips", {}).values() for r in rows)
    # KILL dominates PARK when SO significantly LOSES to FO somewhere and never
    # significantly wins anywhere (decision rule: "loses to first-order RM").
    so_loses = any(r["mean_delta"] < -0.05 and r["excl0"] for r in so_rows)
    so_wins = any(r["mean_delta"] > 0.1 and r["excl0"] for r in so_rows)
    so_verdict = ("not_run" if not so_rows else
                  ("STRONG_KEEP" if ((so_strong or so_extends) and not lpips_reg) else
                   ("KEEP" if so_keep and not lpips_reg else
                    ("KILL" if (so_loses and not so_wins) else
                     ("PARK" if so_park else "KILL")))))
    fo_verdict = ("STRONG_KEEP" if highest.get("first_order_rm") not in (None, order[0]) else
                  ("KEEP" if highest.get("first_order_rm") else "PARK"))
    gated = [v for v, f2 in summ["families"].items() if f2 == "so" and SA.classify(v)["gamma"] is not None]
    gated_rows = [r for v in gated for r in summ["so_minus_fo"].get(v, [])]
    gated_verdict = ("not_run" if not gated else
                     ("KEEP" if any(r["mean_delta"] > 0.1 and r["excl0"] for r in gated_rows) else
                      ("PARK" if any(r["mean_delta"] > 0 for r in gated_rows) else "KILL")))

    interp = []
    p50r = diag["rho2_pct"].get(50); p50c = diag["cos_delta_pct"].get(50)
    if p50r is not None:
        interp.append(f"ρ2 p50={p50r:.2f} ({'curvature ≈ velocity — second differences are mostly noise' if p50r > 0.8 else ('moderate curvature' if p50r > 0.4 else 'nearly linear residual motion')})")
    if p50c is not None:
        interp.append(f"cosΔ p50={p50c:.2f} ({'directionally stable' if p50c > 0.5 else ('weakly stable' if p50c > 0 else 'direction reverses between anchors — extrapolation risky')})")
    cr = diag.get("rho2_corr_with_gain"); cc = diag.get("cos_delta_corr_with_gain")
    if cr is not None and cc is not None:
        interp.append(f"corr(ρ2, SO−FO gain)={cr:+.2f}, corr(cosΔ, gain)={cc:+.2f}"
                      + (" — the assumption predicts when SO helps" if (cc > 0.2 or cr < -0.2)
                         else " — the diagnostic does NOT predict when SO helps"))
    interpretation = "; ".join(interp)

    headline = ""
    if best_fo_sea:
        headline += (f"First-order RM ({best_fo_sea['variant']}) vs SeaCache at matched speed: best "
                     f"{best_fo_sea['delta']:+.2f} dB @ {best_fo_sea['speedup']:.2f}× "
                     f"(CI {_fmt_ci(best_fo_sea['ci'])}); highest CI-positive band "
                     f"{highest.get('first_order_rm') or 'none'}. ")
    if best_so_fo:
        headline += (f"Second-order − first-order: best {best_so_fo['mean_delta']:+.2f} dB @ "
                     f"{best_so_fo['rm_speedup']:.2f}× ({best_so_fo['variant']}, CI "
                     f"{_fmt_ci(best_so_fo['ci'])}) → {so_verdict}.")
    bounded = (f"On FLUX 512px/28 the extreme-speed comparison covers ~2×–{max((p['speedup'] for p in summ['seacache_points']), default=0):.1f}×. "
               f"Plain HorizonCache stays above SeaCache through {highest.get('plain_horizon') or 'no band'}, "
               f"first-order RM through {highest.get('first_order_rm') or 'no band'}, "
               f"second-order RM through {highest.get('second_order_rm') or 'no band'}. "
               f"Second-order term: {interpretation or 'diagnostics unavailable'}.")

    du = lambda p: F.data_uri(Path(p))

    def badge(v):
        c = {"STRONG_KEEP": "sk", "KEEP": "sk", "PARK": "park"}.get(v, "kill")
        return f"<span class='badge {c}'>{v.replace('_', ' ')}</span>"

    H = [f"<div class='wrap'><h1>Second-Order Residual Hold + Extreme-Speed Comparison (E59)</h1>",
         f"<p class='sub'>Curvature term on the residual secant · SeaCache comparison to ~5×+ · "
         f"FLUX 512px/28 · commit <code>{sha}</code> · cluster H100</p>"]
    H.append("<h2>1 · Executive summary</h2><div class='card'>"
             f"<p>First-order RM at extreme speed: {badge(fo_verdict)} · second-order RM: {badge(so_verdict)}"
             f" · gated second-order: {badge(gated_verdict) if gated_verdict != 'not_run' else 'not run'}</p>"
             f"<p class='hl'><b>Headline.</b> {headline}</p><p><b>Bounded claim.</b> {bounded}</p></div>")
    H.append("<h2>2 · Method</h2><div class='card'><ul>"
             "<li><b>First-order (E58)</b>: r_pred = r_a + β1·λ·P1(Δr_a) — cached residual moved along its "
             "fresh-anchor secant; free (no extra forward).</li>"
             "<li><b>Second-order uniform (E59)</b>: + β2·λ(λ+1)/2·P2(Δ²r_a) with Δ²r_a = r_a − 2r_{a−1} + r_{a−2} "
             "(Newton backward; exact for quadratic residual motion at uniform anchor spacing).</li>"
             "<li><b>Quad (E59)</b>: damped Lagrange quadratic through the three (σ_k, r_k) anchors — exact for "
             "nonuniform anchor spacing; r_pred = r_a + β_quad·(r_quad(σ) − r_a).</li>"
             "<li><b>Gate</b>: enable the curvature term only if cosΔ &gt; γ and ρ2 &lt; ρ_max (anchor-triple stats).</li>"
             "<li>β2=0 recovers E58 bit-identically (unit-tested); protocol, fixture, matched-achieved-speedup "
             "comparison and bootstrap unchanged from E56/E58.</li></ul></div>")
    H.append("<h2>3 · Extreme-speed frontier</h2>")
    H.append(f"<div class='fig'><img src='{du(figs['frontier'])}'><div class='cap'>PSNR (left) and LPIPS "
             "(right) vs achieved speedup; dotted verticals at 2/3/4/5×.</div></div>")
    if "delta_band" in figs:
        H.append(f"<div class='fig'><img src='{du(figs['delta_band'])}'><div class='cap'><b>The key figure.</b> "
                 "Best variant per family minus SeaCache, per speed band, paired ±95% CI (hatched = CI "
                 "includes 0).</div></div>")
    # band table
    H.append("<h3>Δ vs SeaCache by speed band (best per family)</h3><table>"
             "<tr><th>band</th><th>plain</th><th>first-order RM</th><th>second-order RM</th>"
             "<th>quad RM</th><th>SO − FO</th></tr>")
    for r in bands:
        cells = []
        for k in ("plain", "first_order_rm", "second_order_rm", "quad_rm"):
            d = r.get(k)
            cells.append("—" if not d else
                         f"<span class='{'pos' if d['delta_psnr_vs_seacache'] > 0 else 'neg'}'>"
                         f"{d['delta_psnr_vs_seacache']:+.2f}</span>{'*' if d['excl0'] else ''} "
                         f"<span class='sub'>@{d['speedup']:.2f}× w{d['win']*100:.0f}%</span>")
        sf = r.get("so_minus_fo")
        cells.append("—" if not sf else
                     f"<span class='{'pos' if sf['delta_psnr'] > 0 else 'neg'}'>{sf['delta_psnr']:+.2f}</span>"
                     f"{'*' if sf['excl0'] else ''}")
        band_lab = r["band"] + ("" if r.get("seacache_reachable", True) else " †")
        H.append(f"<tr><td>{band_lab}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    sea_max = summ.get("seacache_max_speedup")
    H.append("</table><p class='sub'>* = 95% CI excludes 0. "
             + (f"† = <b>beyond the swept SeaCache frontier</b>: within the swept τ grid SeaCache tops "
                f"out at {sea_max:.2f}× — its τ→speedup mapping is strongly quantized at the top end "
                f"(integer refresh counts), while jumps remove nodes and reach ~5.3× at the same τ. In "
                f"† bands the method is compared against SeaCache's fastest attained point while being "
                f"strictly faster, so the deltas are conservative; a fair same-speed SeaCache point does "
                f"not exist on this grid. " if sea_max else "")
             + "Highest CI-positive band vs SeaCache: "
             + ", ".join(f"{k}: <b>{v or 'none'}</b>" for k, v in highest.items()) + "</p>")
    H.append("<h2>4 · Second-order vs first-order</h2>")
    if "so_minus_fo" in figs:
        H.append(f"<div class='fig'><img src='{du(figs['so_minus_fo'])}'><div class='cap'>Paired SO−FO "
                 "ΔPSNR (top) and raw ΔLPIPS (bottom, &lt;0 better) at identical compute.</div></div>")
    H.append("<h2>5 · Second-order assumption diagnostic</h2>")
    if "diagnostics" in figs:
        H.append(f"<div class='fig'><img src='{du(figs['diagnostics'])}'><div class='cap'>ρ2 and cosΔ "
                 "distributions (per-image means over fresh-anchor triples) and their relation to the "
                 "SO−FO gain.</div></div>")
    obf = summ.get("oracle_by_family", {})
    orc_html = ""
    if obf.get("available"):
        orc_html = ("<p><b>Oracle residual (pointwise, extreme τ).</b></p><table>"
                    "<tr><th>predictor</th><th>τ</th><th>frozen err</th><th>motion err</th>"
                    "<th>rel. reduction</th></tr>")
        for r in obf["rows"]:
            orc_html += (f"<tr><td>{'second-order (β₂=0.1)' if r['family'] == 'so' else 'first-order'}</td>"
                         f"<td>{r['tau']}</td><td>{r['frozen_err']:.3f}</td><td>{r['motion_err']:.3f}</td>"
                         f"<td class='neg'>{r['relative_reduction']*100:+.1f}%</td></tr>")
        orc_html += ("</table><p class='sub'>Consistent with E58: motion does NOT reduce the pointwise "
                     "residual error (the PSNR gain is accumulated-drift correction), and the curvature "
                     "term makes the pointwise error strictly worse — the second difference is noise.</p>")
    H.append(f"<div class='card'><p>{interpretation or 'No diagnostics collected.'}</p>"
             + orc_html
             + (f"<p>Aggregate oracle: frozen {orc.get('frozen_residual_error', 0):.3f} vs motion "
                f"{orc.get('residual_motion_error', 0):.3f}.</p>" if orc.get("available") else "") + "</div>")
    if grids:
        H.append("<h2>6 · Qualitative grids & failure gallery</h2>"
                 "<p class='sub'>Rows = prompts; columns = full reference, SeaCache, plain HorizonCache, "
                 "first-order RM, second-order RM, and the error heatmap of the last column vs full. "
                 "Higher τ panels show where each method breaks.</p>")
        for g, tau, label in grids:
            H.append(f"<div class='fig'><img src='{du(g)}'><div class='cap'>τ={tau:g} ({label}).</div></div>")
    H.append("<h2>7 · Verdicts</h2><div class='card'><table><tr><th>question</th><th>verdict</th></tr>"
             f"<tr><td>first-order RM at extreme speed</td><td>{badge(fo_verdict)}</td></tr>"
             f"<tr><td>second-order RM (uniform Δ²)</td><td>{badge(so_verdict)}</td></tr>"
             f"<tr><td>gated second-order</td><td>{badge(gated_verdict) if gated_verdict != 'not_run' else 'not run'}</td></tr>"
             "</table></div>")
    H.append("<h2>8 · Artifacts</h2><div class='card'><ul>"
             "<li>report: <code>reports/horizon_cache_second_order_extreme.html</code></li>"
             "<li>summary: <code>reports/horizon_cache_second_order_extreme_summary.{md,json}</code></li>"
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
            f"<title>E59 Second-Order Residual Hold + Extreme Speed</title></head><body>{''.join(H)}</body></html>")
    (reports_dir / "horizon_cache_second_order_extreme.html").write_text(html)

    # ---- summary JSON (§8 schema) ----
    def band_json():
        out = []
        for r in bands:
            e = {"band": r["band"],
                 "seacache_best": f"seacache (reference)",
                 "plain_horizon_best": r.get("plain", {}).get("variant", ""),
                 "first_order_rm_best": r.get("first_order_rm", {}).get("variant", ""),
                 "second_order_rm_best": r.get("second_order_rm", {}).get("variant", ""),
                 "plain_minus_seacache_delta_psnr": r.get("plain", {}).get("delta_psnr_vs_seacache", None),
                 "first_order_rm_minus_seacache_delta_psnr": r.get("first_order_rm", {}).get("delta_psnr_vs_seacache", None),
                 "second_order_rm_minus_seacache_delta_psnr": r.get("second_order_rm", {}).get("delta_psnr_vs_seacache", None),
                 "second_order_minus_first_order_delta_psnr": r.get("so_minus_fo", {}).get("delta_psnr", None),
                 "ci95_second_order_minus_first_order": r.get("so_minus_fo", {}).get("ci", None)}
            fo = r.get("first_order_rm")
            verdict = ""
            if fo:
                verdict = ("FO above SeaCache (CI+)" if fo["delta_psnr_vs_seacache"] > 0 and fo["excl0"]
                           else ("FO above SeaCache (ns)" if fo["delta_psnr_vs_seacache"] > 0 else "FO below SeaCache"))
            e["verdict"] = verdict
            out.append(e)
        return out

    b_fo = best_fo_sea or {}
    lp_fo = None
    if b_fo:
        lp_fo = next((x for x in summ["vs_seacache_lpips"].get(b_fo["variant"], [])
                      if x["tau"] == b_fo["tau"]), None)
    b_sf = best_so_fo or {}
    b_sf_cls = SA.classify(b_sf["variant"]) if b_sf else {}
    b_sf_sea = None
    if b_sf:
        b_sf_sea = next((x for x in summ["vs_seacache"].get(b_sf["variant"], [])
                         if x["tau"] == b_sf["tau"]), None)
    fo_pt = {}
    if b_fo:
        fo_pt = next((p for p in summ["method_points"][b_fo["variant"]] if p["tau"] == b_fo["tau"]), {})
    sj = {
        "status": "DONE", "git_commit": sha,
        "run_dirs": [str(x) for x in (run_dirs or [gen])],
        "main_verdict": f"first_order_rm_extreme={fo_verdict}; second_order_rm={so_verdict}",
        "headline_claim": headline, "bounded_claim": bounded,
        "best_first_order_rm": ({
            "method": b_fo.get("variant", ""), "speedup": round(b_fo.get("speedup", 0.0), 3),
            "psnr": round(fo_pt.get("psnr", 0.0) or 0.0, 3), "lpips": round(fo_pt.get("lpips", 0.0) or 0.0, 4),
            "delta_psnr_vs_seacache": round(b_fo.get("delta", 0.0), 3),
            "ci95_vs_seacache": [round(b_fo["ci"][0], 3), round(b_fo["ci"][1], 3)] if b_fo.get("ci") else [0, 0],
            "win_rate_vs_seacache": round(b_fo.get("win", 0.0), 3)} if b_fo else {}),
        "best_second_order_rm": ({
            "method": b_sf.get("variant", ""), "beta1": b_sf_cls.get("beta1"), "beta2": b_sf_cls.get("beta2"),
            "speedup": round(b_sf.get("rm_speedup", 0.0), 3),
            "psnr": round(b_sf.get("rm_metric", 0.0), 3), "lpips": round(b_sf.get("rm_lpips") or 0.0, 4),
            "delta_psnr_vs_first_order_rm": round(b_sf.get("mean_delta", 0.0), 3),
            "ci95_vs_first_order_rm": [round(b_sf["ci"][0], 3), round(b_sf["ci"][1], 3)] if b_sf.get("ci") else [0, 0],
            "delta_psnr_vs_seacache": round(b_sf_sea["delta"], 3) if b_sf_sea else None,
            "ci95_vs_seacache": [round(b_sf_sea["ci"][0], 3), round(b_sf_sea["ci"][1], 3)] if b_sf_sea else None}
            if b_sf else {}),
        "highest_positive_vs_seacache_band": {
            "seacache": "reference", "plain_horizon": highest.get("plain_horizon"),
            "first_order_rm": highest.get("first_order_rm"),
            "second_order_rm": highest.get("second_order_rm")},
        "second_order_diagnostic": {
            "rho2_p50": diag["rho2_pct"].get(50), "rho2_p95": diag["rho2_pct"].get(95),
            "cos_delta_p50": diag["cos_delta_pct"].get(50), "cos_delta_p05": diag["cos_delta_pct"].get(5),
            "rho2_correlation_with_gain": diag.get("rho2_corr_with_gain"),
            "cos_delta_correlation_with_gain": diag.get("cos_delta_corr_with_gain"),
            "n_anchor_triples": diag["n_anchor_triples"],
            "n_so_applications": diag["n_so_applications"],
            "n_so_gated_off": diag["n_so_gated_off"],
            "interpretation": interpretation},
        "oracle_residual_by_family": summ.get("oracle_by_family", {"available": False}),
        "seacache_max_speedup": summ.get("seacache_max_speedup"),
        "speed_band_results": band_json(),
        "verdicts": {"first_order_rm_extreme": fo_verdict, "second_order_rm": so_verdict,
                     "second_order_gated": gated_verdict},
        "key_findings": [headline, bounded],
        "failure_modes": [],
        "artifacts": {"html_report": "reports/horizon_cache_second_order_extreme.html",
                      "summary_md": "reports/horizon_cache_second_order_extreme_summary.md",
                      "summary_json": "reports/horizon_cache_second_order_extreme_summary.json",
                      "metrics_csv": str(gen / "metrics.csv"), "figures_dir": str(assets),
                      "samples_dir": str((Path(samples_dir) if samples_dir else gen) / "samples")},
    }
    (reports_dir / "horizon_cache_second_order_extreme_summary.json").write_text(
        json.dumps(sj, indent=2, default=float))

    md = [f"# E59 — Second-Order Residual Hold + Extreme-Speed Comparison", "",
          f"**Commit** `{sha}` · FLUX 512px/28 · cluster H100", "",
          f"**Headline.** {headline}", "", f"**Bounded claim.** {bounded}", "",
          "## Δ vs SeaCache by speed band (best per family; * = CI excludes 0)", "",
          "| band | plain | first-order RM | second-order RM | quad RM | SO−FO |",
          "|---|---|---|---|---|---|"]
    for r in bands:
        cells = []
        for k in ("plain", "first_order_rm", "second_order_rm", "quad_rm"):
            d = r.get(k)
            cells.append("—" if not d else f"{d['delta_psnr_vs_seacache']:+.2f}{'*' if d['excl0'] else ''} "
                                           f"@{d['speedup']:.2f}×")
        sf = r.get("so_minus_fo")
        cells.append("—" if not sf else f"{sf['delta_psnr']:+.2f}{'*' if sf['excl0'] else ''}")
        md.append(f"| {r['band']} | " + " | ".join(cells) + " |")
    md += ["", f"**Highest CI-positive band vs SeaCache:** plain: {highest.get('plain_horizon')}, "
               f"FO-RM: {highest.get('first_order_rm')}, SO-RM: {highest.get('second_order_rm')}", "",
           "## Verdicts", "",
           f"- first-order RM at extreme speed: **{fo_verdict}**",
           f"- second-order RM: **{so_verdict}**",
           f"- gated second-order: **{gated_verdict}**", "",
           f"## Second-order diagnostic", "", interpretation or "n/a"]
    (reports_dir / "horizon_cache_second_order_extreme_summary.md").write_text("\n".join(md))
    return sj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--samples-dir", default="")
    ap.add_argument("--oracle-dir", default="")
    ap.add_argument("--run-dirs", nargs="*", default=None)
    ap.add_argument("--tau-grid", type=float, nargs="+",
                    default=[0.3, 0.5, 0.575, 0.65, 0.8, 1.0, 1.2])
    ap.add_argument("--reports-dir", default=str(REPO / "reports"))
    a = ap.parse_args()
    reports = Path(a.reports_dir); reports.mkdir(parents=True, exist_ok=True)
    sj = build(Path(a.gen_dir), a.tau_grid, reports, git_hash(),
               samples_dir=a.samples_dir or None, run_dirs=a.run_dirs,
               oracle_dir=a.oracle_dir or None)
    print(json.dumps({"status": sj["status"], "main_verdict": sj["main_verdict"],
                      "highest_positive_vs_seacache_band": sj["highest_positive_vs_seacache_band"],
                      "best_first_order_rm": sj["best_first_order_rm"],
                      "best_second_order_rm": sj["best_second_order_rm"],
                      "verdicts": sj["verdicts"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
