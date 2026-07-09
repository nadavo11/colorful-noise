"""E61 report — Innovation-Gated Fixed Residual Motion.

Builds reports/horizon_cache_e61_innovation_gate.{html,_summary.md,_summary.json}.
Goal: keep fixed β=0.5 in-band, causally throttle it only when the previous-anchor innovation
I_a signals the secant is unreliable. NOT trying to just beat SeaCache (fixed RM already does);
trying to MATCH fixed RM in-band while FIXING its high-speed penalty vs plain HorizonCache.

    python -m horizon_cache.e61_report --gen-dir <run> --tau-grid 0.3 ... 1.4 \
        --reports-dir ../reports [--samples-dir <run with --save-all-images>]
"""
from __future__ import annotations

import argparse
import json
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
from horizon_cache import e61_analysis as EA
from horizon_cache.so_report import qual_grid, _fmt_ci, git_hash

BANDS = EA.BANDS
GATE_FAMILIES = ["gate_hard", "gate_soft", "gate_floor", "betahat_gate"]
FAM_LABEL = {"plain": "plain HorizonCache", "fo": "fixed-β RM (E58)", "cl": "closed-loop RM (E60)",
             "gate_hard": "hard gate", "gate_soft": "soft gate", "gate_floor": "floor gate",
             "betahat_gate": "β̂-as-gate"}
FAM_COLOR = {"seacache": F.ACC, "plain": F.WARN, "fo": "#c084fc", "cl": "#38bdf8",
             "gate_hard": "#f472b6", "gate_soft": F.ACC2, "gate_floor": "#facc15",
             "betahat_gate": "#a78bfa"}
BAND_KEY = {"plain": "plain", "fo": "fixed_rm", "cl": "closed_loop", **{g: g for g in GATE_FAMILIES}}


# ----------------------------------------------------------------- figures
def extreme_frontier(summ, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax in axes:
        F._style(ax)
        for x in (2, 3, 4, 5):
            ax.axvline(x, color=F.LINE, lw=0.8, ls=":")
    for ax, metric, ylab in [(axes[0], "psnr", "PSNR vs full (dB) ↑"),
                             (axes[1], "lpips", "LPIPS vs full ↓")]:
        sea = sorted((p["speedup"], p[metric]) for p in summ["seacache_points"] if p[metric] is not None)
        if sea:
            ax.plot(*zip(*sea), marker="o", color=FAM_COLOR["seacache"], lw=2.6, label="SeaCache", zorder=8)
        seen = set()
        for v, pts in summ["method_points"].items():
            fam = summ["families"][v]
            xy = sorted((p["speedup"], p[metric]) for p in pts if p[metric] is not None)
            if not xy:
                continue
            lab = FAM_LABEL.get(fam, fam) if fam not in seen else None
            seen.add(fam)
            ax.plot(*zip(*xy), marker="D", color=FAM_COLOR.get(fam, F.MUT), lw=1.7,
                    ls="--" if fam.startswith("gate_") or fam == "betahat_gate" else "-",
                    alpha=0.85, label=lab, markersize=4)
        ax.set_xlabel("achieved speedup (block-stack-equivalent) →")
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=7.5, facecolor=F.PANEL, edgecolor=F.LINE, labelcolor=F.INK)
    fig.suptitle("Extreme-speed frontier — SeaCache vs plain vs fixed-β RM vs closed-loop vs "
                "innovation-gated RM", color=F.INK, fontsize=11)
    fig.tight_layout()
    return F._save(fig, out)


def gate_minus_plain_fig(summ, out: Path):
    """10.3: Δ(gate-plain) by band — does the gate fix the high-speed RM penalty?"""
    return _gate_delta_fig(summ, "gate_minus_plain", "Gate − plain HorizonCache (paired, same τ) "
                           "— fixed RM inverts here at ≥4.3× (E59/E60); the gate should not.",
                           "ΔPSNR: gate − plain (dB)", out)


def _gate_delta_fig(summ, key, title, ylabel, out: Path, floor_line=None, target_line=None):
    entries = sorted(summ.get(key, {}).items())
    if not entries:
        return None
    fig, ax = plt.subplots(figsize=(max(9.5, 1.05 * sum(len(r) for _, r in entries)), 4.8))
    F._style(ax)
    ax.axhline(0, color=F.INK, lw=1.2)
    if floor_line is not None:
        ax.axhline(floor_line, color=F.WARN, ls=":", lw=1.1)
    if target_line is not None:
        ax.axhline(target_line, color=F.ACC2, ls=":", lw=1.1)
    xs_all, lbl_all, idx = [], [], 0
    for v, rows in entries:
        fam = EA.classify(v)["family"]
        col = FAM_COLOR.get(fam, F.MUT)
        for r in sorted(rows, key=lambda r: r["tau"]):
            ax.bar(idx, r["mean_delta"], color=col,
                  yerr=[[r["mean_delta"] - r["ci"][0]], [r["ci"][1] - r["mean_delta"]]],
                  capsize=3, ecolor=F.MUT, alpha=0.95 if r["excl0"] else 0.5,
                  hatch=None if r["excl0"] else "//")
            xs_all.append(idx)
            lbl_all.append(f"{v}\nτ{r['tau']:g}\n{r['rm_speedup']:.2f}×")
            idx += 1
        idx += 1
    ax.set_xticks(xs_all); ax.set_xticklabels(lbl_all, fontsize=6.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)
    fig.tight_layout()
    return F._save(fig, out)


def innovation_diag_fig(summ, out: Path):
    """10.4: I_a (Ī_a) vs RM-plain gain scatter + binned E[gain|I_a]."""
    ig = summ.get("innovation_vs_gain", {})
    sc = ig.get("scatter", {})
    xs = np.asarray(sc.get("I", []), float)
    ys = np.asarray(sc.get("rm_minus_plain_psnr", []), float)
    if len(xs) < 3:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for ax in axes:
        F._style(ax)
    taus = np.asarray(sc.get("tau", []), float)
    sct = axes[0].scatter(xs, ys, c=taus, cmap="viridis", s=15, alpha=0.7)
    cb = fig.colorbar(sct, ax=axes[0]); cb.set_label("τ", color=F.INK)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=F.INK)
    z = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 50)
    axes[0].plot(xx, np.polyval(z, xx), color=F.INK, lw=1.4, ls="--")
    axes[0].axhline(0, color=F.MUT, lw=0.8)
    axes[0].set_xlabel("Ī_a (mean innovation, closed-loop run)")
    axes[0].set_ylabel("per-image fixed-RM − plain ΔPSNR (dB)")
    axes[0].set_title(f"corr = {ig.get('corr'):+.2f} (n={ig.get('n')})", fontsize=10)
    bins = ig.get("bins", [])
    if bins:
        bx = [b["I_mean"] for b in bins]
        by = [b["gain_mean"] for b in bins]
        be = [b["gain_std"] / max(1, b["n"]) ** 0.5 for b in bins]
        axes[1].errorbar(bx, by, yerr=be, marker="o", color=F.ACC2, lw=1.8, capsize=3)
        axes[1].axhline(0, color=F.MUT, lw=0.8)
        axes[1].set_xlabel("Ī_a bin mean")
        axes[1].set_ylabel("E[fixed-RM − plain ΔPSNR | Ī_a bin]")
        axes[1].set_title("binned monotonicity check", fontsize=10)
    fig.suptitle("Innovation diagnostic — does Ī_a predict when fixed RM helps?", color=F.INK, fontsize=11)
    fig.tight_layout()
    return F._save(fig, out)


def beta_by_band_fig(summ, out: Path):
    """10.5: mean/p10/p50/p90 β_i by speed band, one line per gate family."""
    bb = summ.get("beta_by_tau", [])
    if not bb:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    F._style(ax)
    ax.axhline(0.5, color=F.MUT, ls=":", lw=1.2, label="fixed β=0.5")
    ax.axhline(0.0, color=F.INK, lw=1.0)
    for fam in GATE_FAMILIES:
        rows = sorted([r for r in bb if r["family"] == fam], key=lambda r: r["speedup"])
        if not rows:
            continue
        xs = [r["speedup"] for r in rows]
        med = [r["beta_p50"] for r in rows]
        lo = [r["beta_p50"] - r["beta_p10"] for r in rows]
        hi = [r["beta_p90"] - r["beta_p50"] for r in rows]
        ax.errorbar(xs, med, yerr=[lo, hi], marker="o", lw=1.8, capsize=3,
                   color=FAM_COLOR.get(fam, F.ACC2), label=FAM_LABEL[fam], markersize=4)
    ax.set_xlabel("achieved speedup →")
    ax.set_ylabel("β_i (p50, bars p10–p90)")
    ax.set_title("β_i by speed band — expect ≈0.5 through ~3.5×, lower above 4×", fontsize=10)
    ax.legend(fontsize=8, facecolor=F.PANEL, edgecolor=F.LINE, labelcolor=F.INK)
    fig.tight_layout()
    return F._save(fig, out)


# ----------------------------------------------------------------- build
def build(gen: Path, tau_grid, reports_dir: Path, sha: str, samples_dir=None, run_dirs=None):
    df = EA.load_df(gen)
    summ = EA.summarize(gen, tau_grid)
    assets = reports_dir / "horizon_cache_e61_innovation_gate_assets"
    assets.mkdir(parents=True, exist_ok=True)

    figs = {}
    figs["frontier"] = extreme_frontier(summ, assets / "extreme_frontier.png")
    for name, fn in [("gate_minus_fixed", lambda: _gate_delta_fig(
                        summ, "gate_minus_fixed", "Gate − fixed-β RM (paired, same τ)",
                        "ΔPSNR: gate − fixed RM (dB)", assets / "gate_minus_fixed.png",
                        floor_line=-0.1, target_line=0.2)),
                     ("gate_minus_plain", lambda: gate_minus_plain_fig(summ, assets / "gate_minus_plain.png")),
                     ("innovation", lambda: innovation_diag_fig(summ, assets / "innovation.png")),
                     ("beta_band", lambda: beta_by_band_fig(summ, assets / "beta_by_band.png"))]:
        f = fn()
        if f:
            figs[name] = f

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
        gate_v = next((fam_best.get(g) for g in GATE_FAMILIES if fam_best.get(g)), None)
        for tau, label in [(t, l) for t, l in
                          [(0.5, "~2.5×"), (0.65, "~3.5×"), (1.0, "~4.5×"), (1.2, "~5.2×")]
                          if t in tau_grid]:
            methods = [(f"seacache_t{tau:g}", "SeaCache")]
            for v, lab in [(pl_v, "plain"), (fo_v, "fixed-β RM"), (gate_v, "gated RM")]:
                if v:
                    methods.append((f"horizon_{v}_t{tau:g}", lab))
            g = qual_grid(sdir / "samples", qdf, tau, methods, keys,
                         f"Generated samples at τ={tau:g} ({label} band)",
                         assets / f"qual_t{tau:g}.png")
            if g:
                grids.append((g, tau, label))

    bands = summ["bands"]
    highest = summ["highest_positive_vs_seacache_band"]
    ig = summ.get("innovation_vs_gain", {})

    def _rows_for(prefix):
        return [r for v, rows in summ.get(prefix, {}).items() for r in rows]

    def best_from(dd):
        best = None
        for v, rows in dd.items():
            for r in rows:
                if best is None or r["mean_delta"] > best["mean_delta"]:
                    best = {**r, "variant": v}
        return best

    def verdict_for_family(fam_name):
        rows = [r for v, rows in summ.get("gate_minus_fixed", {}).items()
               if EA.classify(v)["family"] == fam_name for r in rows]
        rows_plain = [r for v, rows in summ.get("gate_minus_plain", {}).items()
                     if EA.classify(v)["family"] == fam_name for r in rows]
        lp_rows = [r for v, rows in summ.get("gate_minus_fixed_lpips", {}).items()
                  if EA.classify(v)["family"] == fam_name for r in rows]
        if not rows:
            return "not_run", rows, rows_plain
        inband = [r for r in rows if 2.0 <= r["rm_speedup"] <= 3.5]
        highspeed = [r for r in rows if r["rm_speedup"] > 4.3]
        inband_ok = all(r["mean_delta"] > -0.1 for r in inband) if inband else True
        highspeed_win = any(r["mean_delta"] > 0.2 and r["excl0"] for r in highspeed)
        highspeed_plain_ok = all(r["mean_delta"] > -0.1 for r in
                                 [r for r in rows_plain if r["rm_speedup"] > 4.3]) \
                            if any(r["rm_speedup"] > 4.3 for r in rows_plain) else False
        lpips_reg = any(r["mean_delta"] < -0.002 and r["excl0"] for r in lp_rows)
        loses_inband = any(r["mean_delta"] < -0.1 and r["excl0"] for r in inband)
        if inband_ok and (highspeed_win or highspeed_plain_ok) and not lpips_reg:
            return "STRONG_KEEP", rows, rows_plain
        if highspeed_win or highspeed_plain_ok:
            return "KEEP", rows, rows_plain
        if loses_inband and not highspeed_win and not highspeed_plain_ok:
            return "KILL", rows, rows_plain
        return "PARK", rows, rows_plain

    verdicts = {}
    for fam_name, key in [("gate_hard", "hard_gate"), ("gate_soft", "soft_gate"),
                          ("gate_floor", "floor_gate"), ("betahat_gate", "beta_hat_gate")]:
        v, _, _ = verdict_for_family(fam_name)
        verdicts[key] = v
    # ema_floor is floor family with ema_alpha<1 — split out if both plain-floor and ema-floor exist
    ema_variants = [v for v in summ["variants"]
                   if EA.classify(v)["family"] == "gate_floor" and EA.classify(v)["ema_alpha"] < 1.0]
    if ema_variants:
        rows_ema = [r for v, rows in summ.get("gate_minus_fixed", {}).items()
                   if v in ema_variants for r in rows]
        rows_ema_plain = [r for v, rows in summ.get("gate_minus_plain", {}).items()
                         if v in ema_variants for r in rows]
        inband = [r for r in rows_ema if 2.0 <= r["rm_speedup"] <= 3.5]
        highspeed = [r for r in rows_ema if r["rm_speedup"] > 4.3]
        inband_ok = all(r["mean_delta"] > -0.1 for r in inband) if inband else True
        highspeed_win = any(r["mean_delta"] > 0.2 and r["excl0"] for r in highspeed)
        verdicts["ema_floor_gate"] = ("STRONG_KEEP" if inband_ok and highspeed_win else
                                      ("KEEP" if highspeed_win else
                                       ("KILL" if inband and not inband_ok and not highspeed_win else "PARK")))
    else:
        verdicts["ema_floor_gate"] = "not_run"

    best_gate = best_from(summ.get("gate_minus_fixed", {}))
    best_gate_sea = None
    if best_gate:
        best_gate_sea = next((x for x in summ["vs_seacache"].get(best_gate["rm_variant"], [])
                             if x["tau"] == best_gate["tau"]), None)
    best_gate_plain = next((r for r in summ.get("gate_minus_plain", {}).get(
                            best_gate["rm_variant"], []) if r["tau"] == best_gate["tau"]), None) \
        if best_gate else None

    main_verdict = "; ".join(f"{k}={v}" for k, v in verdicts.items())

    headline = ""
    if best_gate:
        cls = EA.classify(best_gate["rm_variant"])
        headline = (f"Best gate ({best_gate['rm_variant']}, {cls['family']}) vs fixed RM: "
                   f"{best_gate['mean_delta']:+.2f} dB @ {best_gate['rm_speedup']:.2f}× "
                   f"(CI {_fmt_ci(best_gate['ci'])}); vs plain "
                   f"{best_gate_plain['mean_delta']:+.2f} dB" if best_gate_plain else "")
        headline += f". Overall: {main_verdict}."
    bounded = (f"Innovation diagnostic corr(Ī_a, RM−plain gain) = {ig.get('corr')} (n={ig.get('n')}), "
              f"vs E60's −0.44 baseline. Highest CI-positive-vs-SeaCache band per family: "
              + ", ".join(f"{k}: {v or 'none'}" for k, v in highest.items()))

    du = lambda p: F.data_uri(Path(p))

    def badge(v):
        c = {"STRONG_KEEP": "sk", "KEEP": "sk", "PARK": "park", "not_run": "park"}.get(v, "kill")
        return f"<span class='badge {c}'>{v.replace('_', ' ')}</span>"

    H = [f"<div class='wrap'><h1>Innovation-Gated Fixed Residual Motion (E61)</h1>",
        f"<p class='sub'>Causal gate on fixed β=0.5 from the fixed-β forecast innovation · "
        f"FLUX 512px/28 · commit <code>{sha}</code> · cluster H100</p>"]
    H.append("<h2>1 · Executive summary</h2><div class='card'>"
            f"<p>{' · '.join(badge(v) + ' ' + k.replace('_',' ') for k, v in verdicts.items())}</p>"
            f"<p class='hl'><b>Headline.</b> {headline}</p><p><b>Bounded claim.</b> {bounded}</p></div>")
    H.append("<h2>2 · Method</h2><div class='card'><ul>"
            "<li><b>Keeps</b> fixed β=0.5 (E58/E59 STRONG KEEP) as the default in-band gain — "
            "does NOT replace it with a self-calibrated scalar (E60 showed that fails).</li>"
            "<li><b>Causal gate</b>: at fresh anchor a, r̂_a = r_{a-1} + 0.5·λ_a·Δr_{a-1}, "
            "ε_a = r_a − r̂_a, I_a = ‖ε_a‖₁/‖r_a‖₁ (or Δr-/pred-normalized). g_a is fixed at "
            "anchor a and used unchanged by every cached step until the next fresh forward — "
            "it cannot see any information from the cached steps it governs.</li>"
            "<li><b>Gate forms</b>: hard 1[I&lt;κ], soft clip(1-I/κ,0,1), floor "
            "g_min+(1-g_min)·soft, each optionally EMA-smoothed over anchors (ᾱ). "
            "β_i = 0.5·g_a.</li>"
            "<li><b>β̂-as-gate</b> (optional): reuses the E60 LS β̂ only as a risk signal, "
            "g=clip(β̂/0.5,0,1), never as the gain itself.</li>"
            "<li>Zero extra forwards; E56/E58/E60 protocol unchanged.</li></ul></div>")
    H.append("<h2>3 · Extreme-speed frontier</h2>")
    H.append(f"<div class='fig'><img src='{du(figs['frontier'])}'><div class='cap'>PSNR (left) "
            "and LPIPS (right) vs achieved speedup.</div></div>")
    H.append("<h3>Δ vs SeaCache by speed band (best per family)</h3><table><tr><th>band</th>"
            "<th>plain</th><th>fixed RM</th><th>closed loop</th>"
            + "".join(f"<th>{FAM_LABEL[g]}</th>" for g in GATE_FAMILIES) + "</tr>")
    for r in bands:
        cells = []
        for k in ["plain", "fixed_rm", "closed_loop"] + GATE_FAMILIES:
            d = r.get(k)
            cells.append("—" if not d else
                        f"<span class='{'pos' if d['delta_psnr_vs_seacache'] > 0 else 'neg'}'>"
                        f"{d['delta_psnr_vs_seacache']:+.2f}</span>{'*' if d['excl0'] else ''}")
        band_lab = r["band"] + ("" if r.get("seacache_reachable", True) else " †")
        H.append(f"<tr><td>{band_lab}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    sea_max = summ.get("seacache_max_speedup")
    H.append("</table><p class='sub'>* = 95% CI excludes 0. "
            + (f"† = beyond the swept SeaCache frontier ({sea_max:.2f}×), conservative. " if sea_max else "")
            + "</p>")
    H.append("<h2>4 · Gate vs fixed RM (the main question)</h2>")
    if "gate_minus_fixed" in figs:
        H.append(f"<div class='fig'><img src='{du(figs['gate_minus_fixed'])}'><div class='cap'>"
                 "Paired gate − fixed-β RM ΔPSNR at identical τ. Dotted lines mark the E61 proceed "
                 "thresholds (−0.1 dB in-band floor, +0.2 dB high-speed target).</div></div>")
    H.append("<h2>5 · Gate vs plain HorizonCache (does it fix the penalty?)</h2>")
    if "gate_minus_plain" in figs:
        H.append(f"<div class='fig'><img src='{du(figs['gate_minus_plain'])}'><div class='cap'>"
                 "Fixed RM inverts vs plain at ≥4.3× (E59/E60); a working gate should not."
                 "</div></div>")
    H.append("<h2>6 · Innovation diagnostic</h2>")
    if "innovation" in figs:
        H.append(f"<div class='fig'><img src='{du(figs['innovation'])}'></div>")
    H.append("<h2>7 · β_i by speed band</h2>")
    if "beta_band" in figs:
        H.append(f"<div class='fig'><img src='{du(figs['beta_band'])}'></div>")
    if grids:
        H.append("<h2>8 · Qualitative grids</h2>")
        for g, tau, label in grids:
            H.append(f"<div class='fig'><img src='{du(g)}'><div class='cap'>τ={tau:g} ({label})."
                    f"</div></div>")
    H.append("<h2>9 · Verdicts</h2><div class='card'><table><tr><th>gate</th><th>verdict</th></tr>"
            + "".join(f"<tr><td>{k.replace('_',' ')}</td><td>{badge(v)}</td></tr>"
                     for k, v in verdicts.items()) + "</table></div>")
    H.append("<h2>10 · Artifacts</h2><div class='card'><ul>"
            "<li>report: <code>reports/horizon_cache_e61_innovation_gate.html</code></li>"
            "<li>summary: <code>reports/horizon_cache_e61_innovation_gate_summary.{md,json}</code></li>"
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
           f"<title>E61 Innovation-Gated Fixed RM</title></head><body>{''.join(H)}</body></html>")
    (reports_dir / "horizon_cache_e61_innovation_gate.html").write_text(html)

    def gate_json(bg, name):
        if not bg:
            return {}
        cls = EA.classify(bg["rm_variant"])
        sea = next((x for x in summ["vs_seacache"].get(bg["rm_variant"], []) if x["tau"] == bg["tau"]), None)
        pl = next((r for r in summ.get("gate_minus_plain", {}).get(bg["rm_variant"], [])
                  if r["tau"] == bg["tau"]), None)
        return {
            "method": bg["rm_variant"], "gate_type": cls["family"], "kappa": cls.get("kappa"),
            "ema_alpha": cls.get("ema_alpha"), "g_min": cls.get("gmin"),
            "speedup": round(bg["rm_speedup"], 3), "psnr": round(bg["rm_metric"], 3),
            "lpips": round(bg.get("rm_lpips") or 0.0, 4),
            "delta_psnr_vs_fixed_rm": round(bg["mean_delta"], 3),
            "ci95_vs_fixed_rm": [round(bg["ci"][0], 3), round(bg["ci"][1], 3)],
            "delta_psnr_vs_plain_horizon": round(pl["mean_delta"], 3) if pl else None,
            "ci95_vs_plain_horizon": [round(pl["ci"][0], 3), round(pl["ci"][1], 3)] if pl else None,
            "delta_psnr_vs_seacache": round(sea["delta"], 3) if sea else None,
            "ci95_vs_seacache": [round(sea["ci"][0], 3), round(sea["ci"][1], 3)] if sea else None,
        }

    def band_json():
        out = []
        for r in bands:
            e = {"band": r["band"], "fixed_rm_best": r.get("fixed_rm", {}).get("variant", ""),
                "plain_horizon_best": r.get("plain", {}).get("variant", ""),
                "gate_best": r.get("best_gate_minus_fixed", {}).get("variant", "")}
            gmf = r.get("best_gate_minus_fixed")
            gmp = r.get("best_gate_minus_plain")
            e["gate_minus_fixed_rm_delta_psnr"] = gmf["delta_psnr"] if gmf else None
            e["ci95_gate_minus_fixed_rm"] = gmf["ci"] if gmf else None
            e["gate_minus_plain_delta_psnr"] = gmp["delta_psnr"] if gmp else None
            e["ci95_gate_minus_plain"] = gmp["ci"] if gmp else None
            gsea = r.get("gate_hard") or r.get("gate_soft") or r.get("gate_floor") or r.get("betahat_gate")
            e["gate_minus_seacache_delta_psnr"] = gsea["delta_psnr_vs_seacache"] if gsea else None
            e["ci95_gate_minus_seacache"] = gsea["ci"] if gsea else None
            band_beta = [b for b in summ.get("beta_by_tau", []) if b.get("band") == r["band"]]
            e["mean_beta"] = float(np.mean([b["beta_mean"] for b in band_beta])) if band_beta else None
            e["frac_beta_lt_0p5"] = (float(np.mean([b["frac_beta_lt_0p5"] for b in band_beta
                                                    if b.get("frac_beta_lt_0p5") is not None]))
                                     if any(b.get("frac_beta_lt_0p5") is not None for b in band_beta) else None)
            e["verdict"] = ("gate matches/beats fixed" if gmf and gmf["delta_psnr"] > -0.1 else
                           ("gate loses to fixed" if gmf else ""))
            out.append(e)
        return out

    sj = {
        "status": "DONE", "git_commit": sha, "run_dirs": [str(x) for x in (run_dirs or [gen])],
        "main_verdict": main_verdict, "headline_claim": headline, "bounded_claim": bounded,
        "best_gate": gate_json(best_gate, "best"),
        "speed_band_results": band_json(),
        "innovation_diagnostic": {
            "corr_innovation_rm_minus_plain": ig.get("corr"),
            "corr_ema_innovation_rm_minus_plain": ig.get("corr"),
            "best_kappa": EA.classify(best_gate["rm_variant"])["kappa"] if best_gate else None,
            "interpretation": (f"corr={ig.get('corr')} (n={ig.get('n')}) vs E60 baseline -0.44"
                              if ig.get("corr") is not None else "insufficient data"),
        },
        "verdicts": verdicts,
        "key_findings": [headline, bounded],
        "failure_modes": [],
        "artifacts": {"html_report": "reports/horizon_cache_e61_innovation_gate.html",
                     "summary_md": "reports/horizon_cache_e61_innovation_gate_summary.md",
                     "summary_json": "reports/horizon_cache_e61_innovation_gate_summary.json",
                     "metrics_csv": str(gen / "metrics.csv"), "figures_dir": str(assets),
                     "samples_dir": str((Path(samples_dir) if samples_dir else gen) / "samples")},
    }
    (reports_dir / "horizon_cache_e61_innovation_gate_summary.json").write_text(
        json.dumps(sj, indent=2, default=float))

    md = [f"# E61 — Innovation-Gated Fixed Residual Motion", "",
         f"**Commit** `{sha}` · FLUX 512px/28 · cluster H100", "",
         f"**Headline.** {headline}", "", f"**Bounded claim.** {bounded}", "",
         "## Δ vs SeaCache by speed band", "",
         "| band | plain | fixed RM | closed loop | " + " | ".join(FAM_LABEL[g] for g in GATE_FAMILIES) + " |",
         "|---|---|---|---|" + "---|" * len(GATE_FAMILIES)]
    for r in bands:
        cells = []
        for k in ["plain", "fixed_rm", "closed_loop"] + GATE_FAMILIES:
            d = r.get(k)
            cells.append("—" if not d else f"{d['delta_psnr_vs_seacache']:+.2f}{'*' if d['excl0'] else ''}")
        md.append(f"| {r['band']} | " + " | ".join(cells) + " |")
    md += ["", "## Verdicts", ""] + [f"- {k.replace('_',' ')}: **{v}**" for k, v in verdicts.items()]
    (reports_dir / "horizon_cache_e61_innovation_gate_summary.md").write_text("\n".join(md))
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
                      "best_gate": sj["best_gate"], "verdicts": sj["verdicts"]},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
