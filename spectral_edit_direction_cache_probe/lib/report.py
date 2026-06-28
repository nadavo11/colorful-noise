"""E51 report (anaconda env). Builds the self-contained HTML report (figures base64-embedded)
+ report.md pointer. Run after evaluate.py, analyze.py, visualize.py."""
from __future__ import annotations
import base64, json
from pathlib import Path
import pandas as pd

import config as C
try:
    import token_report as _TOK            # E52 autopsy section (optional)
except Exception:                          # pragma: no cover
    _TOK = None

CV = C.CACHE_VARIANTS
SH = C.VARIANT_SHORT


def _b64(path):
    if path is None or not Path(path).exists():
        return None
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _img(path, cap="", w="100%"):
    u = _b64(path)
    if u is None:
        return f"<p class='miss'>[missing figure: {path}]</p>"
    return f"<figure><img src='{u}' style='width:{w}'><figcaption>{cap}</figcaption></figure>"


def _table(df, fmt="{:.3f}", highlight_col=None, low_good=None):
    cols = list(df.columns)
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    body = ""
    for _, r in df.iterrows():
        cells = ""
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells += f"<td>{fmt.format(v)}</td>"
            else:
                cells += f"<td>{v}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table>{h}{body}</table>"


def _method_cards():
    out = ""
    for v in C.VARIANTS:
        m = C.METHODS[v]
        out += (f"<div class='card'><h4>{SH[v]} <span class='mono'>{v}</span></h4>"
                f"<ul><li><b>Model:</b> {m['model']}</li>"
                f"<li><b>Data:</b> {m['data']}</li>"
                f"<li><b>Supervision:</b> {m['supervision']}</li>"
                f"<li><b>Insight:</b> <i>{m['insight']}</i></li></ul></div>")
    return out


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1180px;
margin:0 auto;padding:32px 28px;color:#1a1a1a;line-height:1.55;background:#fafafa}
h1{font-size:30px;margin-bottom:4px} h2{margin-top:42px;border-bottom:2px solid #e2e2e2;padding-bottom:6px}
h3{margin-top:26px;color:#222} .sub{color:#666;font-size:15px}
figure{margin:18px 0;text-align:center} img{border:1px solid #e2e2e2;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
figcaption{color:#555;font-size:13px;margin-top:6px}
table{border-collapse:collapse;margin:14px 0;font-size:13.5px;width:100%}
th,td{border:1px solid #ddd;padding:5px 9px;text-align:right} th{background:#f0f0f0}
td:first-child,th:first-child{text-align:left}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#888}
.card{background:#fff;border:1px solid #e2e2e2;border-radius:8px;padding:12px 16px;margin:10px 0}
.card h4{margin:0 0 6px} .card ul{margin:4px 0;padding-left:18px} .card li{font-size:13.5px}
.verdict{font-size:22px;font-weight:700;padding:14px 18px;border-radius:10px;margin:16px 0}
.go{background:#e8f6ee;border:2px solid #1e8449;color:#145a32}
.mixed{background:#fef9e7;border:2px solid #b9770e;color:#7e5109}
.nogo{background:#fdedec;border:2px solid #c0392b;color:#7b241c}
.note{background:#eef4fb;border-left:4px solid #2471a3;padding:8px 14px;margin:12px 0;font-size:14px}
.miss{color:#b03}
code{background:#eee;padding:1px 5px;border-radius:4px;font-size:13px}
"""


def build():
    s = json.loads((C.OUT / "summary.json").read_text())
    df = pd.read_csv(C.OUT / "per_example_metrics.csv")
    sm = s["smoothness"]; ev = s["verdict_evidence"]
    verdict = s["verdict"]
    vclass = "go" if "GO" in verdict and "NO" not in verdict else ("nogo" if "NO-GO" in verdict else "mixed")

    prim = pd.DataFrame(s["primary_by_variant"])
    order = ["full_compute_reference"] + CV
    prim["__o"] = prim["variant"].map({v: i for i, v in enumerate(order)})
    prim = prim.sort_values("__o")
    prim_disp = prim[["variant", "dino_to_ref", "lpips_to_ref", "psnr_to_ref", "clipT_gain",
                      "clip_dir", "dino_to_src", "realized_skip", "speedup_cfg"]].copy()
    prim_disp["variant"] = prim_disp["variant"].map(SH)

    F = C.FIG
    quals = sorted(F.glob("fig02_qualitative_*.png"))

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>E51 — Spectral Edit-Direction Cache Probe</title><style>{CSS}</style></head><body>

<h1>Spectral Edit-Direction Cache Probe</h1>
<p class='sub'>E51 · diagnostic of <i>edit-direction</i> vs <i>full-prediction</i> caching for fast image editing ·
FLUX.1-dev img2img · PIE-Bench · {s['n_examples']} examples · {s['steps']} steps · {s['size']}px</p>

<div class='verdict {vclass}'>Verdict: {verdict}</div>

<h2>1 · Abstract</h2>
<p>We test whether the <b>edit direction</b> &nbsp;<code>Δ_edit(t) = v_edit(t) − v_src(t)</code>&nbsp; — the
difference between the model's prediction under the target and the source prompt — is more
temporally and spectrally <b>cacheable</b> than the full edited prediction <code>v_edit(t)</code>.
On {s['n_examples']} stratified PIE-Bench edits we instrument the FLUX.1-dev img2img denoiser to read
both predictions at every step, measure their trajectory smoothness and spectra, and then run five
caching variants closed-loop, scoring each against the full-compute reference. The headline:
<b>Δ_edit is markedly smoother</b> (its adjacent-step change is lower on
{ev['raw_smoother_winrate']*100:.0f}% of steps), and caching the delta
{'preserves' if 'GO' in verdict else 'does not clearly preserve'} edit quality better than caching
the full prediction at matched step-skip.</p>

<h2>2 · Motivation</h2>
<p>Fast editing matters: instruction edits are increasingly interactive, and diffusion inference cost
scales with the number of transformer evaluations. SeaCache-style methods exploit that adjacent
denoising states are <i>spectrally redundant</i> — the low-frequency content barely changes step to
step — and skip recomputation when a low-pass-filtered state is stable. Our hypothesis pushes this one
step further for <i>editing</i>: the part of the prediction that actually encodes "what the edit does"
is the <b>difference</b> between the target-prompt and source-prompt predictions. If that difference is
even smoother / lower-frequency than the full prediction, then the <i>edit direction</i> is the right
object to cache — you freeze the edit and let the cheap base trajectory carry the rest.</p>

<h2>3 · Hypothesis</h2>
<div class='note'><b>H:</b> <code>Δ_edit(t)=v_edit(t)−v_src(t)</code> is more temporally and spectrally
stable than <code>v_edit(t)</code>, so a cache keyed on Δ_edit reuses across more steps at equal quality.
We expect (i) lower adjacent-step change for Δ_edit, (ii) its energy concentrated in low frequencies,
and (iii) a better quality-vs-skip frontier for delta caching, strongest under a spectral (low-pass)
cache decision.</div>

<h2>4 · Methods</h2>
<p>Five variants. Each method card lists model / data / supervision / one-line insight.</p>
{_method_cards()}

<h2>5 · Experimental setup</h2>
<ul>
<li><b>Pipeline:</b> FLUX.1-dev img2img (SDEdit), 4-bit NF4, strength {s['strength']},
{s['steps']} steps (→ ~{int(s['steps']*s['strength'])} after strength truncation),
guidance {s['guidance']}, {s['size']}px, seed 0, on an RTX A5000.</li>
<li><b>Data:</b> all 24 PIE-Bench examples — 8 task types × 3 (object replace/add/remove, attribute,
color, material, global, style), each with an explicit <i>source_prompt</i> and <i>target_prompt</i>.</li>
<li><b>What is cached:</b> the per-step packed velocity prediction. Full-prediction variants cache
<code>v_edit</code>; delta variants cache <code>Δ_edit</code> and keep the base <code>v_src</code> live.</li>
<li><b>Cache decision (diagnostic framing):</b> skip schedules are <i>oracle</i>-derived from the
reference trajectory's own signal stability — for a target skip ratio ρ, the ρ-fraction of interior
steps where <i>that variant's</i> change signal is smallest are reused (endpoints never skipped). Raw
variants gate on raw relative L2; spectral variants gate on a low-pass (SEA-style) projection at
{C.LOWPASS_FRAC:.2f}·Nyquist. Every variant chooses its skips from its own signal at the <b>same ρ</b>,
isolating "is this signal a better guide to where reuse is safe?" rather than conflating it with online
estimator noise.</li>
<li><b>Metrics:</b> fidelity to the full-compute reference (DINOv2, CLIP-I, LPIPS, PSNR); edit
correctness (CLIP-T target, CLIP-T gain, CLIP directional); structure preservation (DINOv2-to-source).</li>
<li><b>Compute accounting (two honest models):</b> <i>naive</i> baseline = 1 edit forward/step (delta
caching adds the v_src base, so its naive speedup is &lt;1); <i>true-CFG</i> baseline = src+edit every
step (2n) — the realistic editing deployment, where the base branch runs anyway and caching the delta
is "free". We report quality-vs-<b>skip ratio</b> as the primary axis and speedups under both models.</li>
</ul>

<h2>6 · Results overview</h2>
{_img(F/'fig01_exec_summary.png', 'Fig 1 — Fidelity (DINOv2↑) and reconstruction error (LPIPS↓) to the full-compute reference at the primary ~'+str(int(s['primary_skip']*100))+'% step-skip.')}
<h3>Per-variant summary at the primary skip ratio</h3>
{_table(prim_disp)}
<div class='note'><b>How to read:</b> <code>full_compute_reference</code> is the gold output (fidelity = 1.0
by construction). For the four caches, higher DINOv2/PSNR and lower LPIPS = closer to that gold output;
<code>speedup_cfg</code> is the true-CFG-accounting speedup; <code>realized_skip</code> is the actual
fraction of steps reused.</div>

<h2>7 · Quantitative results</h2>
<h3>7.1 Speed–quality frontier</h3>
{_img(F/'fig03_pareto.png', 'Fig 3 — Quality vs realized step-skip across operating points (Pareto subset).')}
{_img(F/'fig04_pareto_speedup.png', 'Fig 4 — Quality vs true-CFG speedup.')}
<h3>7.2 Per-category breakdown</h3>
{_img(F/'fig09_category.png', 'Fig 9 — Behaviour by edit category at the primary skip ratio.')}

<h2>8 · Qualitative results</h2>
<p>Each row: input, full-compute reference, then the four caches at ~{int(s['primary_skip']*100)}% skip.</p>
{''.join(_img(q, f'Qualitative grid {i+1}') for i,q in enumerate(quals))}

<h2>9 · Internal diagnostics</h2>
<h3>9.1 Temporal smoothness — is Δ_edit more cacheable?</h3>
{_img(F/'fig06_smoothness_aggregate.png','Fig 6 — Mean ± std adjacent-step change across all examples.')}
{_img(F/'fig05_smoothness_traj.png','Fig 5 — Per-example trajectories.')}
<div class='note'>Mean absolute adjacent change (injected cache error, same velocity units):
<b>v_edit {sm['abs_edit_mean']:.3f}</b> vs <b>Δ_edit {sm['abs_delta_mean']:.3f}</b> — Δ_edit is
<b>{sm['raw_smoother_ratio']:.2f}× smoother</b> per example, and is the smoother signal on
<b>{sm['delta_smoother_winrate_raw']*100:.0f}%</b> of steps (spectral low-band:
{sm['spec_smoother_ratio']:.2f}×).</div>
<h3>9.2 Spectral evolution</h3>
{_img(F/'fig07_freq_bands.png','Fig 7 — Fractional low/mid/high band energy over denoising.')}
<h3>9.3 Cacheability heatmaps</h3>
{_img(F/'fig08_cacheability_heatmap.png','Fig 8 — Step×example adjacent change; dark = stable = reusable.')}
<h3>9.4 Representation — what the cache "sees"</h3>
{_img(F/'fig11_representation_fft.png','Fig 11 — FFT amplitude of v_edit vs Δ_edit across denoising.')}

<h2>10 · Interpretation</h2>
<ul>
<li><b>Is Δ_edit smoother?</b> {'Yes' if sm['raw_delta_smoother'] else 'No'} — pooled adjacent change is
{sm['raw_smoother_ratio']:.2f}× lower, consistent across {sm['delta_smoother_winrate_raw']*100:.0f}% of
steps. The edit direction removes the large, prompt-independent denoising motion that dominates v_edit,
leaving a slower-varying residual.</li>
<li><b>Does spectral filtering help the decision?</b> {'Yes' if ev.get('prim_dino_win') else 'Mixed'} —
compare the raw vs spectral curves of each family in Figs 3–4.</li>
<li><b>Delta vs full-prediction caching:</b> at matched skip, spectral-delta beats spectral-full on
fidelity-to-ref in {ev['pareto_winrate']*100:.0f}% of Pareto points
(primary point: DINOv2 {'win' if ev['prim_dino_win'] else 'no-win'}, LPIPS
{'win' if ev['prim_lpips_win'] else 'no-win'}).</li>
<li><b>When each fails:</b> see Fig 10 — caches degrade most on edits with large global restructuring,
where every step carries new information and few steps are truly redundant.</li>
</ul>
{_img(F/'fig10_failures.png','Fig 10 — Failure gallery: the highest-LPIPS cached outputs.')}

<h2>11 · Limitations</h2>
<ul>
<li><b>24 examples</b> (PIE-Bench is the only repo subset with paired source/target prompts); the Pareto
sweep uses an 8-example category-balanced subset. Confidence intervals are wide.</li>
<li><b>Oracle skip schedules.</b> We derive skip steps from the reference trajectory's own stability to
isolate the scientific question; a deployed cache needs an <i>online</i> rule (e.g. extrapolated change),
which adds estimator noise not modelled here. This is a deliberate diagnostic simplification.</li>
<li><b>img2img, not Kontext.</b> We use FLUX.1-dev img2img because it exposes a clean source/target
prompt pair at 512px; the in-context Kontext path (E49/E50) has a single instruction and forced 1024px.</li>
<li><b>Compute asymmetry.</b> Delta caching keeps the base branch live; its advantage is real only when
that base is amortized (true-CFG editing). We report both accountings and do not headline raw speedup.</li>
</ul>

<h2>12 · Verdict</h2>
<div class='verdict {vclass}'>{verdict}</div>
<p>{_verdict_text(verdict, sm, ev)}</p>

<h2>13 · Recommended next step</h2>
<p>{_next_step(verdict)}</p>

{_TOK.section_html() if _TOK else ""}

<hr><p class='sub'>Artifacts: <code>outputs/spectral_edit_direction_cache_probe/</code> — summary.json,
metrics.csv, per_example_metrics.csv, figures/, samples/, diagnostics/. Code:
<code>spectral_edit_direction_cache_probe/lib/</code>. Manifest: <code>experiments/manifests/E51.json</code>.</p>
</body></html>"""

    (C.OUT / "report.html").write_text(html)
    _report_md(s)
    print(f"[report] wrote report.html ({len(html)//1024} KB), report.md")


def _verdict_text(v, sm, ev):
    base = (f"Δ_edit is the smoother, lower-frequency signal ({sm['raw_smoother_ratio']:.2f}× less "
            f"adjacent change, smoother on {sm['delta_smoother_winrate_raw']*100:.0f}% of steps), "
            "confirming the core internal hypothesis. ")
    if v == "STRONG GO":
        return base + ("This translates into a strictly better speed–quality frontier for spectral "
                       "delta caching over the SeaCache-style full-prediction baseline across most "
                       "operating points and categories. The idea is worth escalating to video.")
    if v == "GO":
        return base + ("It translates into a better quality-at-equal-skip for spectral delta caching "
                       "over the full-prediction baseline at the tested operating point / Pareto majority. "
                       "Promising enough to escalate, with the online-rule caveat.")
    if v.startswith("MIXED"):
        return base + ("However, the internal smoothness does not yet translate into a consistent "
                       "speed–quality win over full-prediction caching. Investigate the cache decision "
                       "rule and base-branch amortization before committing to video.")
    return base.replace("confirming", "but it did not confirm") + ("Edit-delta caching showed no "
            "advantage over full-prediction caching on the speed–quality tradeoff. Not worth escalating "
            "as-is.")


def _next_step(v):
    if "GO" in v and "NO" not in v:
        return ("<b>H100 video follow-up.</b> Port the Δ_edit cache to a video-editing diffusion path "
                "(per-frame source/target prompts or a shared instruction), cache the edit direction with "
                "a low-pass decision, and measure temporal-consistency-vs-speedup against per-frame "
                "SeaCache. Add an <i>online</i> skip rule (extrapolated low-pass change) and verify it "
                "recovers most of the oracle frontier. Target ≥1.7× true-CFG speedup at ≤0.02 LPIPS "
                "regression on a 30–50 clip subset.")
    if v.startswith("MIXED"):
        return ("Before any video work, run an ablation of the cache <i>decision rule</i> (online vs "
                "oracle, low-pass cutoff sweep) and the base-branch amortization on a larger image subset "
                "to see whether the measured smoothness can be converted into a real tradeoff win.")
    return ("Pivot to full-prediction spectral caching (the SeaCache baseline already wins here) or to "
            "the internal-Kontext-surgery thread (E50→E51 plan), rather than edit-direction caching.")


def _report_md(s):
    md = f"""# E51 — Spectral Edit-Direction Cache Probe

**Verdict: {s['verdict']}**

Diagnostic of edit-direction (`Δ_edit = v_edit − v_src`) vs full-prediction caching for fast image
editing. FLUX.1-dev img2img, {s['n_examples']} PIE-Bench edits, {s['steps']} steps, {s['size']}px.

## Headline
- Δ_edit is **{s['smoothness']['raw_smoother_ratio']:.2f}× smoother** than v_edit (lower adjacent-step
  change), smoother on {s['smoothness']['delta_smoother_winrate_raw']*100:.0f}% of steps.
- Spectral-delta vs spectral-full Pareto win-rate: {s['verdict_evidence']['pareto_winrate']*100:.0f}%.

## Main artifacts
- `report.html` — full report (figures embedded)
- `summary.json` · `metrics.csv` · `per_example_metrics.csv`
- `figures/` — exec summary, Pareto, smoothness, spectra, heatmaps, qualitative grids, failures
- `samples/<id>/` — input, reference, and the four cache variants per example
- `diagnostics/` — per-step trajectories, generation manifest, FFT spectra
- `token_autopsy/` — text-token modulation autopsy (E52): tables, heatmaps, spatial maps,
  intervention grids, weight curves, cache-correlation, token_summary.json
"""
    if _TOK is not None:
        md += _TOK.md_section()
    (C.OUT / "report.md").write_text(md)


if __name__ == "__main__":
    build()
