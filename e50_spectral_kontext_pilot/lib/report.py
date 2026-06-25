"""E50 self-contained HTML report. Run in anaconda env (after evaluate + visualize).

Embeds every figure + the walkthrough video as base64 so reports/e50_spectral_kontext_pilot.html
is portable. The GO/NO-GO verdict is derived programmatically from the metric aggregates.
"""
from __future__ import annotations
import json, base64, html, csv, statistics as st
from pathlib import Path
import config as C

F = C.FIG


def b64(path):
    p = Path(path)
    if not p.exists():
        return None
    mime = "video/mp4" if p.suffix == ".mp4" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def img(path, cls="wide", cap=""):
    d = b64(path)
    if not d:
        return f"<p class='note'>[missing figure: {html.escape(str(Path(path).name))}]</p>"
    c = f"<figcaption>{html.escape(cap)}</figcaption>" if cap else ""
    return f"<figure><img class='{cls}' src='{d}'/>{c}</figure>"


def load_rows():
    return [r for r in csv.DictReader(open(C.METRICS / "e50_metrics.csv"))]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fmt(v, p="{:+.3f}"):
    try:
        return p.format(float(v))
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------- verdict
def derive_verdict(rows, summary):
    src = [r for r in rows if r["experiment"] == "spectral_source"]
    ref = [r for r in rows if r["bucket"] in ("spectral_reference", "kontext_baseline_replay")]
    pv = [r for r in rows if r["experiment"] == "prompt_variants"]

    def mean(rs, k):
        vals = [_f(r[k]) for r in rs if r.get(k) not in (None, "") and not _f(r[k]) != _f(r[k])]
        return st.mean(vals) if vals else float("nan")

    # reference: baseline (content_raw) vs best spectral op on leak_gap, requiring some style adherence
    base_ref = [r for r in ref if r["spectral_op"] == "content_raw"]
    base_leak = mean(base_ref, "leak_gap")
    base_style = mean(base_ref, "clipI_style")
    op_stats = {}
    for op in C.REF_OPS:
        rs = [r for r in ref if r["spectral_op"] == op]
        if rs:
            op_stats[op] = dict(leak_gap=mean(rs, "leak_gap"), clipI_style=mean(rs, "clipI_style"),
                                dino_content=mean(rs, "dino_content"))
    # an op "helps" if it improves leak resistance OR style adherence vs baseline without collapsing the other
    helped = []
    for op, s in op_stats.items():
        if op == "content_raw":
            continue
        better_leak = s["leak_gap"] > base_leak + 0.03
        better_style = s["clipI_style"] > base_style + 0.02
        not_collapsed_style = s["clipI_style"] > base_style - 0.05
        not_collapsed_leak = s["leak_gap"] > base_leak - 0.05
        if (better_leak and not_collapsed_style) or (better_style and not_collapsed_leak):
            helped.append((op, s))

    # source: does any spectral op retain edit ability (clipT_gain) without destroying content?
    raw_src = [r for r in src if r["spectral_op"] == "raw"]
    raw_gain = mean(raw_src, "clipT_gain")
    raw_dino = mean(raw_src, "dino_content")
    src_help = []
    for op in C.SOURCE_OPS:
        if op == "raw":
            continue
        rs = [r for r in src if r["spectral_op"] == op]
        if rs and mean(rs, "clipT_gain") >= raw_gain - 0.005 and mean(rs, "dino_content") >= raw_dino - 0.03:
            src_help.append(op)

    # prompt: does anti_leakage beat neutral on leak_gap?
    pv_neutral = mean([r for r in pv if r["prompt_key"] == "neutral"], "leak_gap")
    pv_anti = mean([r for r in pv if r["prompt_key"] == "anti_leakage"], "leak_gap")
    prompt_helps = pv_anti > pv_neutral + 0.03

    if helped or src_help:
        verdict = "PROCEED_KONTEXT_SPECTRAL"
    elif base_leak > 0.3 and not helped:
        # kontext already strong, external spectral ops do not beat it -> go inside next
        verdict = "PROCEED_INTERNAL_KONTEXT_SURGERY"
    else:
        verdict = "PROCEED_INTERNAL_KONTEXT_SURGERY"

    return dict(verdict=verdict, helped=helped, op_stats=op_stats, base_leak=base_leak,
                base_style=base_style, src_help=src_help, raw_gain=raw_gain, raw_dino=raw_dino,
                prompt_helps=prompt_helps, pv_neutral=pv_neutral, pv_anti=pv_anti)


# ---------------------------------------------------------------- method card
def method_card(key, mc):
    rows = "".join(f"<div class='mc-row'><span class='mc-k'>{html.escape(k)}</span>"
                   f"<span class='mc-v'>{html.escape(str(v))}</span></div>"
                   for k, v in mc.items())
    return f"<div class='methodcard'><h4>{html.escape(key)}</h4>{rows}</div>"


def section(n, title, body):
    return f"<section id='s{n}'><h2><span class='num'>{n}</span>{html.escape(title)}</h2>{body}</section>"


def table(headers, data):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = ""
    for row in data:
        body += "<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>"
    return f"<table class='kv'><tr>{h}</tr>{body}</table>"


def build():
    rows = load_rows()
    summary = json.loads((C.METRICS / "e50_summary.json").read_text())
    V = derive_verdict(rows, summary)
    vc = "good" if V["verdict"].startswith("PROCEED_KONTEXT") else "warn"

    n_gen = len(rows)
    helped_names = ", ".join(op for op, _ in V["helped"]) or "none"
    best_op = max(V["op_stats"].items(), key=lambda kv: (kv[1]["leak_gap"] if kv[0] != "content_raw" else -9)) \
        if V["op_stats"] else (None, {})

    s = []

    # 1 executive
    s.append(section(1, "Executive summary", f"""
      <div class='verdict {vc}'><span class='h'>Verdict</span><b>{V['verdict']}</b></div>
      <p class='lead'>E50 tests whether <b>frequency-domain interventions</b> on the inputs of
      <b>FLUX.1-Kontext-dev</b> — the substrate E49 established — can improve its content/style
      tradeoff or reduce reference semantic leakage. {n_gen} Kontext generations across three
      training-free interventions (spectral <i>source</i> decomposition, spectral <i>reference</i>
      composites, and <i>prompt</i> formulation), evaluated with the E49 metric suite.</p>
      <ul>
        <li><b>Spectral reference composites that helped leakage/style tradeoff:</b> {helped_names}.</li>
        <li><b>Reference baseline (content_raw):</b> leak_gap {fmt(V['base_leak'])}, CLIP-I style {fmt(V['base_style'],'{:.3f}')}.</li>
        <li><b>Source spectral ops that kept edit ability:</b> {', '.join(V['src_help']) or 'none (raw source is required for Kontext to edit)'}.</li>
        <li><b>Anti-leakage prompt vs neutral:</b> leak_gap {fmt(V['pv_neutral'])} → {fmt(V['pv_anti'])}
            ({'helps' if V['prompt_helps'] else 'no clear gain'}).</li>
      </ul>
      <p class='note'><b>Scope.</b> Pilot, Kontext-only, on the E49 subsets. Primary model
      FLUX.1-Kontext-dev (4-bit NF4); E49 Redux / VGG-Gram(Gatys) outputs shown as cross-phase
      controls. Latent/timestep interventions (Exp D) are deferred to E51 (see §19).</p>"""))

    # 2 why E50
    s.append(section(2, "Why E50 exists after E49", """
      <p>Phase 1 returned NO_GO for naive spectral edits on <i>weak</i> FLUX pipelines (img2img,
      Redux). E49 then showed those were the wrong substrate and established <b>FLUX.1-Kontext-dev as
      the strongest accessible no-training editor/styliser under current compute</b> (positive CLIP-T
      gain, top content preservation, best leakage resistance). E50 is the first spectral pilot
      <i>on that competent substrate</i>: now that there is a real editor underneath, do frequency-domain
      manipulations of its inputs buy anything — better stylisation, less reference leakage, or at least
      a clean mechanistic signal pointing at where to intervene next.</p>"""))

    # 3 E49 baseline context
    s.append(section(3, "E49 baseline context (what 'raw Kontext' means here)", """
      <p>The E49 verdict was <b>PROCEED_WITH_FLUX_KONTEXT</b> — strongest <i>accessible</i> no-training
      substrate (Qwen-Image-Edit 20B was not run; untested, not beaten). E49 Kontext: editing CLIP-T
      gain +0.017 at DINO-content 0.813; reference-stylisation leak-resistance (DINO_content −
      DINO_style) +0.72, the best of the five baselines. Redux was high-style but high-leak (−0.68);
      the VGG-Gram (Gatys) control was low-leak but flat. E50 measures spectral interventions against
      a freshly regenerated Kontext baseline (the <code>raw</code> / <code>content_raw</code> ops) on
      the same images.</p>"""))

    # 4 dataset
    s.append(section(4, "Datasets / subsets", f"""
      {table(['subset', 'ids', 'use'],
             [['PIE-Bench edits', ', '.join(C.EDIT_IDS), 'Exp C source spectral decomposition (6 task types)'],
              ['Adversarial leakage pairs', ', '.join(i.replace('_adversarial','') for i in C.LEAK_IDS),
               'Exp A spectral reference composites + Exp B prompt variants'],
              ['Style bank (WikiArt)', 'watercolor / abstract / oil / impressionist', 'style references for A & B']])}
      <p class='note'>All subsets are exact E49 ids, so E50 outputs line up with the E49 baseline
      one-to-one. Pilot size by design (brief: 20–40); {n_gen} generations total.</p>"""))

    # 5 models + inference
    s.append(section(5, "Models and inference setup", f"""
      <p><b>Primary:</b> black-forest-labs/FLUX.1-Kontext-dev, 4-bit NF4 transformer (identical to
      E49), run via diffusers <code>FluxKontextPipeline</code> in the uv env. Inference: seed
      {C.SEED}, {C.STEPS} steps, guidance {C.GUIDANCE}. (Kontext snaps the canvas to its native
      1024px bucket, so E50 renders at the same resolution as the E49 Kontext baseline; the spectral
      op is applied to the {C.GEN_SIZE}px input before encoding.)</p>
      <p><b>Controls (no new generation):</b> E49 Kontext (1024px), E49 Redux (high-leak reference),
      E49 VGG-Gram/Gatys (low-leak classical). Spectral operators are pure FFT (numpy), training-free.</p>"""))

    # 6 spectral operators
    s.append(section(6, "Spectral operators", f"""
      <p>Per-channel 2-D FFT. Amplitude A=|F|, phase P=∠F, reconstruction Re(ifft(A·e<sup>iP</sup>)).
      Radial masks on the shifted spectrum define low(&lt;0.15) / mid(0.15–0.45) / high(&gt;0.45) bands
      (fraction of Nyquist).</p>
      <b>Source ops (Exp C):</b> {', '.join('<code>'+o+'</code>' for o in C.SOURCE_OPS)}.<br>
      <b>Reference ops (Exp A):</b> {', '.join('<code>'+o+'</code>' for o in C.REF_OPS)} —
      <code>content_phase_style_amp</code> = content structure + style texture statistics;
      <code>style_phase_content_amp</code> = style structure + content texture (leak probe);
      <code>style_high_on_content</code> = graft only the style's high-frequency band onto the content.
      {img(F/'fourier'/'fourier_src_pie_7_0_raw.png','wide','Fourier decomposition of a source image (amplitude, phase, band reconstructions).')}
      {img(F/'fourier'/'fourier_ref_leak_3_adversarial_content_phase_style_amp.png','wide','Fourier decomposition of a content-phase+style-amplitude reference composite.')}"""))

    # 7 methods
    cards = "".join(method_card(k, {**v,
                    "inference setup": f"Kontext 4-bit NF4, {C.STEPS} steps, g={C.GUIDANCE}, seed {C.SEED}, native 1024px"})
                    for k, v in C.METHODS.items())
    s.append(section(7, "Method sections (model / data / supervision / insight / site)", f"""
      <div class='cards'>{cards}</div>
      <p class='note'>Each card's <i>intervention site</i>, <i>spectral manipulation</i> and
      <i>metrics</i> are detailed in §8–§13; failure modes in §16.</p>"""))

    # 8 results by task type (source)
    src = [r for r in rows if r["experiment"] == "spectral_source"]
    by_task = {}
    for r in src:
        by_task.setdefault(r["task_type"], []).append(r)
    trows = []
    for t, rs in sorted(by_task.items()):
        raw = [x for x in rs if x["spectral_op"] == "raw"]
        trows.append([t, len(rs), fmt(st.mean([_f(x['dino_content']) for x in raw]) if raw else float('nan'), '{:.3f}'),
                      fmt(st.mean([_f(x.get('clipT_gain', 'nan')) for x in raw]) if raw else float('nan'))])
    s.append(section(8, "Results by task type (source spectral, raw Kontext)", f"""
      {table(['task', 'n', 'DINO content (raw)', 'CLIP-T gain (raw)'], trows)}
      {img(F/'representation_visuals'/'heat_source_op_task.png','wide','DINO content preservation across source spectral op x task.')}"""))

    # 9 results by spectral component
    s.append(section(9, "Results by spectral component", f"""
      <p><b>Source decomposition.</b> {img(F/'representation_visuals'/'heat_source_op_metric.png','wide','Source spectral op x metric. raw is the Kontext baseline; bands/phase/amplitude degrade identity and edit-following progressively.')}
      {img(F/'representation_visuals'/'scatter_preservation_vs_edit.png','half','Preservation vs edit strength per source op.')}
      {img(F/'fourier'/'radial_power_by_op.png','half','Radial power spectrum of Kontext outputs by source op.')}</p>
      <p><b>Reference composites.</b> {img(F/'representation_visuals'/'heat_reference_op_metric.png','wide','Reference spectral op x style/leakage metric.')}
      {img(F/'representation_visuals'/'scatter_style_vs_leakage.png','wide','Style adherence vs leakage resistance per reference op.')}</p>"""))

    # 10 prompt formulation
    s.append(section(10, "Results by prompt formulation", f"""
      <p>Leakage resistance (leak_gap = DINO_content − DINO_style) by instruction wording, content
      image fed to Kontext: neutral {fmt(V['pv_neutral'])}, anti-leakage {fmt(V['pv_anti'])}.</p>
      {img(F/'representation_visuals'/'bar_prompt_leakage.png','half','Prompt wording vs leakage resistance.')}
      {img(F/'grids'/'prompt_variant_grid.png','wide','Per pair: content, style ref, and the three prompt formulations.')}"""))

    # 11 style transfer
    s.append(section(11, "Style-transfer analysis", f"""
      {img(F/'grids'/'reference_spectral_grid.png','wide','Reference spectral composites through Kontext: content, style ref, and each spectral op output.')}"""))

    # 12 reference leakage
    s.append(section(12, "Reference-leakage analysis", f"""
      <p>The adversarial pairs are the cleanest leakage probe: any object/scene copied from the style
      reference is unambiguous leakage. <code>style_phase_content_amp</code> tests the hypothesis that
      <b>phase carries the copy-able semantics</b>; <code>content_phase_style_amp</code> tests whether
      keeping content phase but importing style amplitude transfers texture without the leak.</p>
      {img(F/'leakage_cases'/'worst_leakage.png','wide','Lowest leak_gap (most reference leakage).')}
      {img(F/'leakage_cases'/'best_leakage_resistant.png','wide','Highest leak_gap (most content-preserving).')}"""))

    # 13 content preservation
    s.append(section(13, "Content-preservation analysis", f"""
      {img(F/'grids'/'source_spectral_grid.png','wide','Per source: the original then each spectral-source Kontext output. Which components Kontext needs to keep identity.')}"""))

    # 14 best
    s.append(section(14, "Best cases", img(F/'best_cases'/'best_source_edits.png','wide',
        'Best source-spectral edits (high content preservation + edit gain).')))
    # 15 worst
    s.append(section(15, "Worst cases", img(F/'worst_cases'/'worst_source_edits.png','wide',
        'Worst source-spectral edits (identity destroyed / instruction lost).')))

    # 16 failure taxonomy
    s.append(section(16, "Failure taxonomy", """
      <ul>
        <li><b>Structure collapse</b> — amplitude-only / randomised-phase source inputs: Kontext sees
            texture noise, hallucinates a new scene, identity lost.</li>
        <li><b>Edit suppression</b> — band-limited source inputs: instruction-following weakens because
            the object the instruction names is no longer legible.</li>
        <li><b>Leak via phase</b> — style_phase_content_amp: the style reference's objects bleed in
            through phase structure (the leakage the adversarial pairs are designed to catch).</li>
        <li><b>Filter-flattening</b> — content_phase_style_amp can read as a flat colour/texture filter
            when the style amplitude dominates, losing fine content detail.</li>
      </ul>"""))

    # 17 representation visuals
    s.append(section(17, "Representation visuals", f"""
      {img(F/'representation_visuals'/'heat_source_op_metric.png','half')}
      {img(F/'representation_visuals'/'heat_reference_op_metric.png','half')}
      {img(F/'representation_visuals'/'scatter_preservation_vs_edit.png','half')}
      {img(F/'representation_visuals'/'scatter_style_vs_leakage.png','half')}
      <p>Walkthrough video (baseline vs spectral variants, grouped by task):</p>
      {(lambda d: f'<video class=\"wide\" controls src=\"{d}\"></video>' if d and len(d) < 30_000_000 else '<p class=\"note\">Video at videos/e50_kontext_spectral_walkthrough.mp4 (too large to inline).</p>')(b64(C.VIDEO/'e50_kontext_spectral_walkthrough.mp4'))}"""))

    # 18 verdict
    helped_detail = "".join(
        f"<li><code>{op}</code>: leak_gap {fmt(s_['leak_gap'])} (baseline {fmt(V['base_leak'])}), "
        f"CLIP-I style {fmt(s_['clipI_style'],'{:.3f}')}</li>" for op, s_ in V["helped"]) or "<li>none</li>"
    s.append(section(18, "GO / NO-GO verdict", f"""
      <div class='verdict {vc}'><span class='h'>Verdict</span><b>{V['verdict']}</b></div>
      <p>Decision rule: <b>PROCEED_KONTEXT_SPECTRAL</b> if ≥1 spectral intervention improves the
      content/style/leakage tradeoff over raw Kontext on a meaningful subset; else
      <b>PROCEED_INTERNAL_KONTEXT_SURGERY</b> (external input/reference manipulation is not enough but
      diagnostics point inward).</p>
      <p>Reference composites that beat the baseline tradeoff:</p><ul>{helped_detail}</ul>
      <p>Source spectral ops that preserved edit ability vs raw: {', '.join(V['src_help']) or 'none — raw source is required'}.
      Anti-leakage prompt {'helped' if V['prompt_helps'] else 'did not clearly help'} over neutral.</p>"""))

    # 19 recommendation
    nxt = ("Wire the winning external spectral op into a fuller Kontext run (scale the subset, add "
           "seeds) and start probing the same decomposition <i>inside</i> Kontext."
           if V["verdict"] == "PROCEED_KONTEXT_SPECTRAL" else
           "Move the intervention <i>inside</i> Kontext: external input/reference FFT edits did not beat "
           "raw Kontext, but the diagnostics (phase carries copy-able semantics; amplitude carries "
           "texture) localise where to act — attention/feature or timestep-banded latent edits.")
    s.append(section(19, "Recommendation for E51", f"""
      <p>{nxt}</p>
      <ul>
        <li>Exp D (latent/timestep banded edits) was deferred from E50 to keep the pilot non-invasive
            on the quantized pipeline — it is the natural E51 if the verdict is INTERNAL_SURGERY.</li>
        <li>Most informative axis: the <b>adversarial leakage pairs</b> with the phase/amplitude swap —
            it gives the cleanest read on whether spectral structure controls semantic leakage.</li>
      </ul>"""))

    toc = "".join(f"<li><a href='#s{i+1}'>{i+1}. {t}</a></li>" for i, t in enumerate([
        "Executive summary", "Why E50", "E49 baseline context", "Datasets", "Models & inference",
        "Spectral operators", "Method sections", "Results by task", "Results by spectral component",
        "Prompt formulation", "Style transfer", "Reference leakage", "Content preservation",
        "Best cases", "Worst cases", "Failure taxonomy", "Representation visuals", "Verdict",
        "Recommendation for E51"]))

    css = """
    :root{--ink:#1b2230;--muted:#5d6675;--line:#e6e9f0;--accent:#2b5cb8;--good:#1f7a4d;
    --good-bg:#eef8f2;--warn:#b23b3b;--warn-bg:#fdf1f1;}
    *{box-sizing:border-box}body{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);
    margin:0;background:#fafbfd}.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 80px}
    h1{font-size:30px;margin:0 0 6px}h2{font-size:21px;margin:34px 0 12px;padding-top:10px;
    border-top:2px solid var(--line)}.num{display:inline-block;width:26px;height:26px;line-height:26px;
    text-align:center;background:var(--accent);color:#fff;border-radius:6px;margin-right:10px;font-size:14px}
    h4{margin:0 0 8px}.lead{font-size:16px}.note{color:var(--muted);font-size:13px}
    img.wide{width:100%;border:1px solid var(--line);border-radius:8px;margin:8px 0}
    img.half{width:49%;border:1px solid var(--line);border-radius:8px;margin:6px 0.5%}
    video.wide{width:100%;border-radius:8px;margin:8px 0}figure{margin:10px 0}
    figcaption{color:var(--muted);font-size:12.5px;margin-top:3px}
    table.kv{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px}
    table.kv th,table.kv td{border:1px solid var(--line);padding:6px 9px;text-align:left}
    table.kv th{background:#f4f6fb}
    .verdict{display:flex;align-items:center;gap:14px;padding:14px 18px;border-radius:10px;margin:10px 0;
    font-size:18px}.verdict.good{background:var(--good-bg);border:1px solid var(--good);color:var(--good)}
    .verdict.warn{background:var(--warn-bg);border:1px solid var(--warn);color:var(--warn)}
    .verdict .h{font-size:12px;text-transform:uppercase;letter-spacing:.5px;opacity:.7}
    .cards{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .methodcard{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fff}
    .mc-row{display:flex;gap:8px;font-size:13px;padding:2px 0;border-bottom:1px dotted var(--line)}
    .mc-k{color:var(--muted);min-width:96px}.mc-v{flex:1}
    .toc{columns:2;font-size:13.5px;background:#f4f6fb;border:1px solid var(--line);border-radius:8px;
    padding:14px 20px}.toc a{color:var(--accent);text-decoration:none}code{background:#eef1f7;padding:1px 5px;border-radius:4px;font-size:12.5px}
    """
    htmlpage = f"""<!doctype html><html><head><meta charset='utf-8'>
    <title>E50 — Spectral Kontext Pilot</title><style>{css}</style></head><body><div class='wrap'>
    <h1>E50 — Spectral Kontext Pilot</h1>
    <p class='note'>FLUX.1-Kontext-dev · frequency-domain interventions · training-free ·
    {n_gen} generations · phase {C.PHASE_VERSION}</p>
    <ol class='toc'>{toc}</ol>{''.join(s)}
    <p class='note' style='margin-top:40px'>Generated by E50 pipeline
    (<code>e50_spectral_kontext_pilot/lib/report.py</code>). Metrics:
    <code>metrics/e50_metrics.csv</code> · <code>metrics/e50_summary.json</code>.</p>
    </div></body></html>"""

    out = C.REPORTS / "e50_spectral_kontext_pilot.html"
    out.write_text(htmlpage)
    print(f"report -> {out}  ({len(htmlpage)//1024} KB)")
    print(f"verdict: {V['verdict']} | helped: {helped_names} | src_help: {V['src_help']}")
    return V


if __name__ == "__main__":
    build()
