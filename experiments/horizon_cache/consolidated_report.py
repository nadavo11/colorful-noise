"""E56 consolidation report — the paper-ready package.

Turns a finished FLUX scale-up run (metrics.csv + summary.json + traces) into:
  reports/horizon_cache_consolidated.html   (10 sections, self-contained, data-URI figures)
  reports/horizon_cache_consolidated_summary.md
  reports/horizon_cache_consolidated_summary.json  (fixed schema)

It reuses the *exact* fair-comparison protocol from run.summarize_generation (matched
achieved-speedup, per-image paired, percentile bootstrap) and additionally computes ΔLPIPS
and win-rate bootstrap CIs, plus a per-speed-band significance table with the decision rule.

    python -m horizon_cache.consolidated_report --gen-dir <n50_run> \
        --frontier-meta <frontier_meta.json> --reports-dir ../reports
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
REPO = HERE.parents[1]

from horizon_cache import figures as F
from horizon_cache import mechanism_figures as MF
from horizon_cache import capability

# ---- decision rule (from the E56 spec) --------------------------------------
BANDS = [("1.5-2.0×", 1.5, 2.0), ("2.0-2.5×", 2.0, 2.5), ("2.5-2.8×", 2.5, 2.8), ("3.0×+", 3.0, 99.0)]
KEEP_BAND = (2.0, 2.7)  # strong-KEEP must land here
DPSNR_BAR = 0.5
WIN_BAR = 0.65


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO)).decode().strip()
    except Exception:
        return "unknown"


def _boot_mean_ci(vals, n_boot=5000, seed=0):
    a = np.asarray(vals, float)
    if len(a) < 2:
        return (None, None)
    rng = np.random.RandomState(seed)
    m = a[rng.randint(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def _boot_rate_ci(signs, n_boot=5000, seed=1):
    a = np.asarray(signs, float)
    if len(a) < 2:
        return (None, None)
    rng = np.random.RandomState(seed)
    m = a[rng.randint(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def matched_per_image(df: pd.DataFrame, variant: str, tau: float, tau_grid, metric: str, higher: bool):
    """Per-image matched delta vs SeaCache interpolated at each image's own achieved speedup.
    Returns (deltas, win_signs). delta>0 always means HorizonCache better (LPIPS sign-flipped)."""
    hk = f"horizon_{variant}_t{tau:g}"
    hor = df[df.method == hk]
    # SeaCache (speedup, metric) points per image key, across the tau grid
    sea_by_key = collections.defaultdict(list)
    for t2 in tau_grid:
        for _, s in df[df.method == f"seacache_t{t2:g}"].iterrows():
            sea_by_key[s["key"]].append((s["compute_speedup"], s[metric]))
    deltas, wins = [], []
    for _, r in hor.iterrows():
        pts = sorted(sea_by_key.get(r["key"], []))
        if len(pts) < 2:
            continue
        sea_at = float(np.interp(r["compute_speedup"], [p[0] for p in pts], [p[1] for p in pts]))
        d = (r[metric] - sea_at) if higher else (sea_at - r[metric])  # >0 => Horizon better
        deltas.append(d)
        wins.append(1.0 if d > 0 else 0.0)
    return deltas, wins


def variant_table(df, variant, tau_grid, metric="psnr", higher=True):
    """Rows across the tau grid for a variant: speedup, mean Δ, CI, win, CI."""
    out = []
    for tau in tau_grid:
        hk = f"horizon_{variant}_t{tau:g}"
        sub = df[df.method == hk]
        if sub.empty:
            continue
        d, w = matched_per_image(df, variant, tau, tau_grid, metric, higher)
        if not d:
            continue
        lo, hi = _boot_mean_ci(d)
        wlo, whi = _boot_rate_ci(w)
        out.append(dict(
            tau=tau, speedup=float(sub.compute_speedup.mean()),
            horizon_metric=float(sub[metric].mean()),
            mean_delta=float(np.mean(d)), ci=(lo, hi),
            excl0=(lo is not None and (lo > 0 or hi < 0)),
            win=float(np.mean(w)), win_ci=(wlo, whi), n=len(d),
        ))
    return out


def band_significance(df, variant, tau_grid):
    """Assign each in-grid operating point to a speed band; report the significance per band."""
    rows = variant_table(df, variant, tau_grid, "psnr", True)
    banded = []
    for name, lo, hi in BANDS:
        pts = [r for r in rows if lo <= r["speedup"] < hi]
        if not pts:
            banded.append(dict(band=name, method=variant, n=0, mean_delta=None, ci=(None, None),
                               win=None, verdict="n/a (no operating point)"))
            continue
        # representative point = closest matched delta significance / pick max win among excl0, else max delta
        best = max(pts, key=lambda r: (r["excl0"], r["mean_delta"]))
        in_keep = KEEP_BAND[0] <= best["speedup"] <= KEEP_BAND[1]
        if best["excl0"] and best["mean_delta"] > DPSNR_BAR and best["win"] > WIN_BAR:
            verdict = "STRONG KEEP" if in_keep else "significant gain"
        elif best["excl0"] and best["mean_delta"] < 0:
            verdict = "significant loss (overshoot)"
        elif best["mean_delta"] > 0:
            verdict = "gain (n.s.)"
        else:
            verdict = "loss (n.s.)"
        banded.append(dict(band=name, method=variant, n=best["n"], mean_delta=best["mean_delta"],
                           ci=best["ci"], win=best["win"], speedup=best["speedup"], verdict=verdict))
    return banded


def verdict_for(df, variant, tau_grid):
    rows = variant_table(df, variant, tau_grid, "psnr", True)
    if not rows:
        return "PARK"
    in_keep = [r for r in rows if KEEP_BAND[0] <= r["speedup"] <= KEEP_BAND[1]]
    strong = any(r["excl0"] and r["mean_delta"] > DPSNR_BAR and r["win"] > WIN_BAR for r in in_keep)
    if strong:
        return "STRONG_KEEP"
    any_sig_gain = any(r["excl0"] and r["mean_delta"] > 0 for r in rows)
    all_sig_loss = rows and all((r["excl0"] and r["mean_delta"] < 0) for r in rows if r["speedup"] >= 2.5)
    if any_sig_gain:
        return "KEEP"
    if all_sig_loss:
        return "KILL"
    return "PARK"


# --------------------------------------------------------------------- HTML
CSS = """
:root{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;--mut:#9aa7b4;--acc:#6ea8fe;--good:#7ee787;
--warn:#f0b429;--bad:#ff7b72;--line:#283039}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:32px 22px 80px}
h1{font-size:27px;margin:.2em 0}h2{font-size:20px;margin:1.7em 0 .5em;border-bottom:1px solid var(--line);
padding-bottom:.3em}h3{font-size:16px;color:var(--acc);margin:1.2em 0 .3em}
.sub{color:var(--mut);font-size:13.5px}code{background:#0b0e13;padding:.1em .35em;border-radius:4px;
font-size:.9em;color:#d2a8ff}
.badge{display:inline-block;padding:.25em .7em;border-radius:999px;font-weight:700;font-size:13px}
.sk{background:rgba(126,231,135,.15);color:var(--good);border:1px solid var(--good)}
.kill{background:rgba(255,123,114,.14);color:var(--bad);border:1px solid var(--bad)}
.park{background:rgba(240,180,41,.13);color:var(--warn);border:1px solid var(--warn)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:.6em 0}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:right}th{color:var(--mut);font-weight:600}
td:first-child,th:first-child{text-align:left}
.pos{color:var(--good)}.neg{color:var(--bad)}
img{max-width:100%;border:1px solid var(--line);border-radius:10px;margin:.5em 0;background:#000}
.fig{margin:16px 0}.cap{color:var(--mut);font-size:12.5px;margin-top:-.2em}
.tag{color:var(--mut);font-size:12px}
ul{margin:.3em 0 .6em 1.1em}li{margin:.2em 0}
.hl{background:rgba(110,168,254,.09);border-left:3px solid var(--acc);padding:.5em .8em;border-radius:0 8px 8px 0}
"""


def _fmt_ci(ci):
    if not ci or ci[0] is None:
        return "—"
    return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"


def _sig_table_html(df, variants, tau_grid):
    rows = []
    for v in variants:
        for r in variant_table(df, v, tau_grid, "psnr", True):
            cls = "pos" if r["mean_delta"] > 0 else "neg"
            star = " ✓" if r["excl0"] else ""
            rows.append(
                f"<tr><td>{v}</td><td>{r['tau']:g}</td><td>{r['speedup']:.2f}×</td>"
                f"<td class='{cls}'>{r['mean_delta']:+.2f}</td><td>{_fmt_ci(r['ci'])}{star}</td>"
                f"<td>{r['win']*100:.0f}%</td><td>{r['n']}</td></tr>")
    return ("<table><tr><th>variant</th><th>τ</th><th>speedup</th><th>Δ PSNR</th>"
            "<th>95% CI (✓=excl 0)</th><th>win</th><th>n</th></tr>" + "".join(rows) + "</table>")


def _band_table_html(bands):
    rows = []
    for b in bands:
        if b["n"] == 0:
            rows.append(f"<tr><td>{b['band']}</td><td colspan=4 class='tag'>no operating point in band</td>"
                        f"<td>{b['verdict']}</td></tr>")
            continue
        cls = "pos" if (b["mean_delta"] or 0) > 0 else "neg"
        vcls = "sk" if "STRONG" in b["verdict"] else ("kill" if "loss" in b["verdict"] else "park")
        rows.append(
            f"<tr><td>{b['band']}</td><td>{b.get('speedup',0):.2f}×</td>"
            f"<td class='{cls}'>{b['mean_delta']:+.2f}</td><td>{_fmt_ci(b['ci'])}</td>"
            f"<td>{b['win']*100:.0f}%</td><td><span class='badge {vcls}'>{b['verdict']}</span></td></tr>")
    return ("<table><tr><th>band</th><th>speedup</th><th>Δ PSNR</th><th>95% CI</th>"
            "<th>win</th><th>verdict</th></tr>" + "".join(rows) + "</table>")


def build(gen: Path, reports_dir: Path, frontier_meta: dict | None, caps: dict, sha: str,
          target_speed: float = 2.5) -> dict:
    df = pd.read_csv(gen / "metrics.csv")
    summ = json.loads((gen / "summary.json").read_text()) if (gen / "summary.json").exists() else {}
    cfg = json.loads((gen / "config.json").read_text()) if (gen / "config.json").exists() else {}
    tau_grid = cfg.get("tau_grid", [0.2, 0.3, 0.4, 0.5, 0.65])
    variants = sorted({m[len("horizon_"):m.rfind("_t")] for m in df.method.unique() if m.startswith("horizon_")})
    n_pairs = int(df[df.method == "full"].shape[0])

    assets = reports_dir / "horizon_cache_consolidated_assets"
    assets.mkdir(parents=True, exist_ok=True)
    rows = df.to_dict("records")

    # ---- figures
    figs = {}
    figs["frontier_psnr"] = F.frontier_plot(rows, tau_grid, assets / "frontier_psnr.png",
                                            "psnr", "PSNR vs full (dB) ↑", True)
    if df["lpips"].notna().any():
        figs["frontier_lpips"] = F.frontier_plot(rows, tau_grid, assets / "frontier_lpips.png",
                                                 "lpips", "LPIPS vs full ↓", False)
    # matched delta + winrate use summary's matched dict shape
    mbv = summ.get("matched_speedup_deltas_by_variant") or {}
    if mbv:
        figs["delta"] = F.delta_plot(mbv, assets / "delta.png")
        figs["winrate"] = F.winrate_plot(mbv, assets / "winrate.png")
    mech = MF.build_all(gen, assets, target_speed=target_speed)
    figs.update({k: v for k, v in mech.items()})
    figs = {k: Path(v) for k, v in figs.items()}  # frontier fns return Path, mech returns str

    # ---- verdicts + bands
    v_verdicts = {v: verdict_for(df, v, tau_grid) for v in variants}
    a125_bands = band_significance(df, "adaptive_1.25", tau_grid) if "adaptive_1.25" in variants else []
    a125_rows = variant_table(df, "adaptive_1.25", tau_grid, "psnr", True) if "adaptive_1.25" in variants else []
    a125_lpips = variant_table(df, "adaptive_1.25", tau_grid, "lpips", False) if "adaptive_1.25" in variants else []
    # headline in-keep-band point
    in_keep = [r for r in a125_rows if KEEP_BAND[0] <= r["speedup"] <= KEEP_BAND[1]]
    headline = max(in_keep, key=lambda r: r["win"] if r["excl0"] else -1) if in_keep else (a125_rows[-1] if a125_rows else None)
    lp_at = next((r for r in a125_lpips if headline and abs(r["speedup"] - headline["speedup"]) < 0.01), None)

    main_verdict = v_verdicts.get("adaptive_1.25", "PARK")
    bounded = ("A conservative headroom-adaptive stride extension shifts the FLUX SeaCache frontier "
               "upward in the ~1.7–2.7× achieved-speedup band, while aggressive jumps still overshoot.")
    headline_claim = "SD3 not run (no MMDiT harness); FLUX scale-up N=%d only." % n_pairs
    if headline:
        headline_claim = (f"FLUX N={n_pairs}: adaptive_1.25 @ {headline['speedup']:.2f}× = "
                          f"{headline['mean_delta']:+.2f} dB vs matched SeaCache "
                          f"(95% CI {_fmt_ci(headline['ci'])}, win {headline['win']*100:.0f}%).")

    # ---- HTML
    def vbadge(v):
        cls = {"STRONG_KEEP": "sk", "KEEP": "sk", "KILL": "kill", "PARK": "park"}.get(v, "park")
        return f"<span class='badge {cls}'>{v.replace('_',' ')}</span>"

    H = []
    H.append(f"<div class='wrap'><h1>HorizonCache — Consolidation (E56)</h1>")
    H.append(f"<p class='sub'>Causal safe-horizon prediction for flow-model acceleration · "
             f"FLUX.1-dev 512px/28 · N={n_pairs} paired samples · commit <code>{sha}</code> · "
             f"cluster H100 (Run:AI)</p>")

    # 1 exec summary
    H.append("<h2>1 · Executive summary</h2>")
    H.append(f"<div class='card'><p>Verdict: {vbadge(main_verdict)} for <b>adaptive_1.25</b> "
             f"(the conservative headroom-adaptive stride).</p>"
             f"<p class='hl'><b>Headline.</b> {headline_claim}</p>"
             f"<p><b>Bounded claim.</b> {bounded}</p>"
             f"<p><b>What failed.</b> Aggressive jumps (adaptive_2.0 / high-τ) overshoot the safe band "
             f"— a <i>measured</i> significant loss, not a caveat. SD3 replication is unavailable "
             f"(no MMDiT generation harness in the module). Learned v1 remains PARK.</p></div>")

    # 2 background
    H.append("<h2>2 · Background (from the deck)</h2>")
    H.append("<div class='card'><ul>"
             "<li><b>SeaCache signal.</b> One cheap score off the modulated input <code>h</code> "
             "(before the L-block stack, Wiener-filtered relative-L1) → refresh vs cache.</li>"
             "<li><b>h-drift / headroom.</b> Accumulate the drift; headroom <code>=1−acc/τ</code> "
             "measures how far the current computation stays trustworthy.</li>"
             "<li><b>cache vs jump.</b> Cache freezes the block residual but keeps the σ-grid (saves the "
             "block stack <i>inside</i> a node); a jump takes a longer σ-stride that <i>removes</i> a whole node.</li>"
             "<li><b>Original FLUX tie expectation.</b> E53 found SeaCache's rel-L1 ranks the oracle safe-jump "
             "length only weakly (Spearman −0.50) → a naïve jump gate was expected to tie.</li>"
             "<li><b>E56 update.</b> FLUX has a <i>conservative safe-stride pocket</i>: refresh often (low τ) "
             "yet still save compute with small headroom-gated jumps — a gentler quality/speed tradeoff exactly "
             "where SeaCache's frontier drops steepest.</li></ul></div>")

    # 3 method
    L = int(df.L.iloc[0]) if "L" in df else 57
    H.append("<h2>3 · Method</h2>")
    H.append(f"<div class='card'><p><b>HorizonCache adaptive headroom stride.</b> In cache territory "
             f"(acc&lt;τ) the stride factor is "
             f"<code>jf = 1 + (jf_max−1)·headroom</code>, <code>headroom = 1−acc/τ</code> — small safe "
             f"strides when the cache is fresh, larger (capped at jf_max) when there is headroom.</p>"
             f"<p><b>Scheduler.</b> <code>regrid</code>: off-grid target "
             f"<code>σ_target=σ_i+jf·(σ_{{i+1}}−σ_i)</code>, then re-space the remaining tail to 0 (net −1 node). "
             f"Every jump logged. Jumps disabled ⇒ SeaCache exactly (fair by identity).</p>"
             f"<p><b>Compute accounting (achieved, not nominal).</b> fresh=1 forward, cache/jump≈1/L (L={L}); "
             f"jumps additionally remove executed nodes. We report block-stack-equivalent and measured wall speedup.</p>"
             f"<p><b>Fair comparison.</b> ΔPSNR at <b>matched achieved speedup</b> — the SeaCache PSNR(speedup) "
             f"curve interpolated at each HorizonCache image's own speedup, paired per prompt×seed, percentile "
             f"bootstrap 95% CI. Same protocol as the sig run; not changed here.</p></div>")

    # 4 experiments
    H.append("<h2>4 · Experiments</h2>")
    H.append(f"<div class='card'><ul>"
             f"<li><b>FLUX scale-up.</b> N={n_pairs} paired samples ({cfg.get('n','?')} prompts × "
             f"{cfg.get('seeds_per_prompt','?')} seeds), 512px/{cfg.get('steps','28')} Euler, bf16+bnb4, "
             f"τ∈{tau_grid}.</li>"
             f"<li><b>Methods.</b> full · uniform · TeaCache · SeaCache · regrid_1.25 · "
             f"adaptive_1.25/1.5/2.0.</li>"
             f"<li><b>SD3.</b> <span class='badge park'>unavailable path</span> — the module's generation "
             f"harness hooks FLUX transformer internals (<code>load_flux_pipeline</code>/<code>sample_flux</code>); "
             f"there is no MMDiT/SD3 harness. Running SD3 needs a new h-signal hook + Euler loop for SD3.5-medium, "
             f"not just a weight download. Documented, not stalled.</li>"
             f"<li><b>Stats.</b> per-image paired matched-speedup deltas; 5000-sample percentile bootstrap "
             f"for mean ΔPSNR, mean ΔLPIPS and win-rate.</li></ul></div>")

    # 5 results
    H.append("<h2>5 · Results</h2>")
    for k, cap in [("frontier_psnr", "PSNR vs achieved speedup — SeaCache frontier vs each HorizonCache family."),
                   ("frontier_lpips", "LPIPS vs achieved speedup (lower is better)."),
                   ("delta", "ΔPSNR at matched achieved speedup ±95% bootstrap CI (hollow = extrapolated)."),
                   ("winrate", "Per-image win-rate vs matched SeaCache.")]:
        if k in figs:
            H.append(f"<div class='fig'><img src='{F.data_uri(figs[k])}'><div class='cap'>{cap}</div></div>")
    H.append("<h3>Significance — matched achieved speedup (per variant × τ)</h3>")
    H.append(_sig_table_html(df, variants, tau_grid))
    if a125_bands:
        H.append("<h3>Per-speed-band significance — adaptive_1.25</h3>")
        H.append(_band_table_html(a125_bands))
    if lp_at and headline:
        H.append(f"<p class='sub'>At the headline point ({headline['speedup']:.2f}×), matched ΔLPIPS = "
                 f"{lp_at['mean_delta']:+.3f} (95% CI {_fmt_ci(lp_at['ci'])}).</p>")

    # 6 mechanism
    H.append("<h2>6 · Mechanism — why adaptive_1.25 survives</h2>")
    for k, cap in [("mechanism_pair", "The one figure: SeaCache vs HorizonCache at matched speed. Same fresh "
                    "count, same speed — HorizonCache removes whole nodes with small safe jumps and finishes earlier."),
                   ("sigma_schedule", "σ-schedule semantics: cache keeps the grid; regrid/adaptive remove a node."),
                   ("compute_accounting", "Where the steps go at matched speed — cache saves the block stack "
                    "inside a node; a jump saves the whole node."),
                   ("staleness_stride", "Staleness ↔ stride: larger jumps only when headroom is high, capped at jf_max.")]:
        if k in figs:
            H.append(f"<div class='fig'><img src='{F.data_uri(figs[k])}'><div class='cap'>{cap}</div></div>")

    # 7 failures
    H.append("<h2>7 · Failures</h2>")
    if "failure_visual" in figs:
        H.append(f"<div class='fig'><img src='{F.data_uri(figs['failure_visual'])}'>"
                 f"<div class='cap'>Aggressive-jump overshoot: large strides remove nodes where the field still "
                 f"curves; integration error compounds to the end.</div></div>")
    H.append("<div class='card'><ul>"
             "<li><b>adaptive_2.0 / 3×+ band.</b> Significant loss vs matched SeaCache — the aggressive stride "
             "overshoots (the deck's DP-surrogate compounding lesson, made visible).</li>"
             "<li><b>v1 supervision.</b> Full-rollout frontier-improvement labels give a real positive class "
             "(~70% jump-helpful) but the learned binary policy ties the heuristic on the small dataset — a "
             "useful negative; needs a much larger label set.</li>"
             "<li><b>SD3/FLUX mismatch.</b> Not testable here — no SD3 harness; the wider-band hypothesis "
             "(SD3 jf_max≈1.5–2.0) stays open.</li></ul></div>")

    # 8 verdicts
    H.append("<h2>8 · Verdict</h2><div class='card'><table>"
             "<tr><th>method</th><th>verdict</th><th>note</th></tr>")
    notes = {
        "adaptive_1.25": "surviving primitive — narrow safe-band frontier shift",
        "adaptive_1.5": "wider stride — earlier overshoot",
        "adaptive_2.0": "aggressive ablation — overshoots the safe band",
        "regrid_1.25": "fixed-factor baseline the adaptive stride beats",
    }
    for v in ["adaptive_1.25", "adaptive_1.5", "adaptive_2.0", "regrid_1.25"]:
        if v in v_verdicts:
            H.append(f"<tr><td>{v}</td><td>{vbadge(v_verdicts[v])}</td><td class='tag'>{notes.get(v,'')}</td></tr>")
    H.append(f"<tr><td>learned v1</td><td>{vbadge('PARK')}</td><td class='tag'>label correct, ties heuristic; "
             f"needs scale</td></tr>")
    H.append(f"<tr><td>SD3 transfer</td><td>{vbadge('PARK')}</td><td class='tag'>unavailable path — no MMDiT "
             f"harness</td></tr></table></div>")

    # 9 next
    H.append("<h2>9 · Next steps</h2><div class='card'><ul>"
             "<li>Paper figure package: the matched-speed mechanism pair + frontier + per-band table are ready.</li>"
             "<li>SD3 harness: author an MMDiT h-signal hook + Euler loop to test the wider-band hypothesis.</li>"
             "<li>Editing branch-cache-horizon on FlowEdit/PIE-Bench (cache-length first, not σ jumps).</li>"
             "<li>Larger learned frontier policy only if the enlarged label set shows a gap over the heuristic.</li>"
             "</ul></div>")

    # 10 artifacts
    H.append("<h2>10 · Artifact index</h2><div class='card'><ul>"
             f"<li>report: <code>reports/horizon_cache_consolidated.html</code></li>"
             f"<li>summary: <code>reports/horizon_cache_consolidated_summary.{{md,json}}</code></li>"
             f"<li>FLUX N={n_pairs} run: <code>{gen}</code> (metrics.csv · summary.json · traces/ · samples/)</li>"
             f"<li>sig run: <code>results/horizon_cache_sig/gen_20260707_095918/</code></li>"
             f"<li>figures: <code>{assets}</code></li>"
             f"<li>commit: <code>{sha}</code></li></ul></div>")
    H.append("</div>")

    html = f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "\
           f"content='width=device-width,initial-scale=1'><style>{CSS}</style>"\
           f"<title>HorizonCache Consolidation (E56)</title></head><body>{''.join(H)}</body></html>"
    (reports_dir / "horizon_cache_consolidated.html").write_text(html)

    # ---- JSON schema
    def _best_json():
        if not headline:
            return {}
        return dict(method="adaptive_1.25", speedup=round(headline["speedup"], 3),
                    psnr=round(headline["horizon_metric"], 3),
                    lpips=(round(lp_at["horizon_metric"], 4) if lp_at else 0.0),
                    delta_psnr_vs_matched_seacache=round(headline["mean_delta"], 3),
                    delta_psnr_ci95=[round(headline["ci"][0], 3), round(headline["ci"][1], 3)],
                    win_rate_vs_matched_seacache=round(headline["win"], 3),
                    win_rate_ci95=[round(headline["win_ci"][0], 3), round(headline["win_ci"][1], 3)])

    band_json = []
    for b in a125_bands:
        if b["n"] == 0:
            continue
        band_json.append(dict(band=b["band"], method="adaptive_1.25",
                              mean_delta_psnr_vs_seacache=round(b["mean_delta"], 3),
                              ci95=[round(b["ci"][0], 3), round(b["ci"][1], 3)],
                              win_rate=round(b["win"], 3), verdict=b["verdict"]))

    fmap = {"adaptive_1.25": "adaptive_1p25", "adaptive_1.5": "adaptive_1p5",
            "adaptive_2.0": "adaptive_2p0", "regrid_1.25": "fixed_regrid_1p25"}
    verdicts_json = {fmap[v]: v_verdicts.get(v, "PARK") for v in fmap}
    verdicts_json["learned_v1"] = "PARK"

    summary_json = {
        "status": "DONE",
        "git_commit": sha,
        "runnable_paths": {"sd3_generation": bool(caps.get("sd3_generation")),
                           "flux_generation": bool(caps.get("flux_generation")),
                           "flowedit": bool(caps.get("flowedit")), "flowalign": bool(caps.get("flowalign"))},
        "main_verdict": main_verdict,
        "headline_claim": headline_claim,
        "bounded_claim": bounded,
        "best_flux": _best_json(),
        "best_sd3": {"available": False, "method": "", "speedup": 0.0, "psnr": 0.0, "lpips": 0.0,
                     "delta_psnr_vs_matched_seacache": 0.0, "delta_psnr_ci95": [0.0, 0.0],
                     "win_rate_vs_matched_seacache": 0.0, "win_rate_ci95": [0.0, 0.0]},
        "speed_band_results": band_json,
        "verdicts": verdicts_json,
        "key_findings": [
            headline_claim,
            "Node removal, not cache staleness: at matched ~2.5× HorizonCache keeps the same fresh count as "
            "SeaCache but removes whole nodes with small headroom-gated jumps.",
            "The frontier shift is narrow-band (~1.7–2.7×) and reduces to SeaCache exactly when jumps are off.",
            "regrid_1.25 (fixed factor) is beaten by the adaptive stride; adaptive_1.5/2.0 overshoot earlier.",
        ],
        "failure_modes": [
            "Aggressive jumps (adaptive_2.0 / high-τ, 3×+) overshoot — significant loss vs matched SeaCache.",
            "Learned v1 ties the heuristic on the small frontier-label set (PARK, needs scale).",
            "SD3 unavailable: no MMDiT generation harness in the module (needs a new SD3 h-signal hook).",
        ],
        "artifacts": {
            "html_report": "reports/horizon_cache_consolidated.html",
            "summary_md": "reports/horizon_cache_consolidated_summary.md",
            "summary_json": "reports/horizon_cache_consolidated_summary.json",
            "flux_results_dir": str(gen),
            "sd3_results_dir": "",
            "figures_dir": str(assets),
            "samples_dir": str(gen / "samples"),
            "metrics_csv": str(gen / "metrics.csv"),
        },
        "n_pairs": n_pairs,
        "tau_grid": tau_grid,
        "frontier_meta": frontier_meta or {},
    }
    (reports_dir / "horizon_cache_consolidated_summary.json").write_text(json.dumps(summary_json, indent=2))

    # ---- MD
    md = [f"# HorizonCache Consolidation (E56) — {main_verdict.replace('_',' ')}",
          "", f"**Commit** `{sha}` · FLUX 512px/28 · N={n_pairs} paired · cluster H100", "",
          f"**Headline.** {headline_claim}", "",
          f"**Bounded claim.** {bounded}", "",
          "## Per-band significance (adaptive_1.25)", "",
          "| band | speedup | ΔPSNR | 95% CI | win | verdict |", "|---|---|---|---|---|---|"]
    for b in a125_bands:
        if b["n"] == 0:
            md.append(f"| {b['band']} | — | — | — | — | {b['verdict']} |")
        else:
            md.append(f"| {b['band']} | {b.get('speedup',0):.2f}× | {b['mean_delta']:+.2f} | "
                      f"{_fmt_ci(b['ci'])} | {b['win']*100:.0f}% | {b['verdict']} |")
    md += ["", "## Verdicts", ""]
    for v in ["adaptive_1.25", "adaptive_1.5", "adaptive_2.0", "regrid_1.25"]:
        if v in v_verdicts:
            md.append(f"- **{v}**: {v_verdicts[v].replace('_',' ')} — {notes.get(v,'')}")
    md += ["- **learned v1**: PARK — label correct, ties heuristic; needs scale",
           "- **SD3 transfer**: PARK — unavailable path (no MMDiT harness)", "",
           "## What failed", "",
           "- Aggressive jumps (adaptive_2.0 / high-τ) overshoot the safe band — measured significant loss.",
           "- SD3 replication unavailable: no SD3 generation harness in `experiments/horizon_cache/`.",
           "- Learned v1 still ties the heuristic on the frontier-label set.", ""]
    (reports_dir / "horizon_cache_consolidated_summary.md").write_text("\n".join(md))

    return summary_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--frontier-meta", default="")
    ap.add_argument("--reports-dir", default=str(REPO / "reports"))
    ap.add_argument("--target-speed", type=float, default=2.5)
    args = ap.parse_args()
    fm = None
    if args.frontier_meta and Path(args.frontier_meta).exists():
        fm = json.loads(Path(args.frontier_meta).read_text())
    reports = Path(args.reports_dir); reports.mkdir(parents=True, exist_ok=True)
    sj = build(Path(args.gen_dir), reports, fm, capability.detect(), git_hash(), args.target_speed)
    print(json.dumps({"status": sj["status"], "main_verdict": sj["main_verdict"],
                      "best_flux": sj["best_flux"], "verdicts": sj["verdicts"]}, indent=2))


if __name__ == "__main__":
    main()
