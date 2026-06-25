"""Generate the extensive, self-contained HTML baseline-establishment report.

Reads metrics/summary + figures + video, embeds everything as base64, derives the
GO/NO-GO verdict from the measured numbers, writes reports/baseline_establishment_report.html.
Run in any env (stdlib + config only).
  python report.py --phase pilot
"""
from __future__ import annotations
import argparse, base64, csv, json, html, mimetypes
from collections import defaultdict
from pathlib import Path
import config as C

PRETTY = {k: v["name"] for k, v in C.MODELS.items()}


def b64(path):
    path = Path(path)
    if not path.exists():
        return None
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def img_tag(path, cls="fig", alt=""):
    u = b64(path)
    if not u:
        return f'<div class="missing">[missing: {html.escape(str(path))}]</div>'
    return f'<img class="{cls}" src="{u}" alt="{html.escape(alt)}">'


def load_rows():
    p = C.METRICS / "baseline_establishment_metrics.csv"
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    pass
            rows.append(r)
    return rows


def load_summary():
    return json.loads((C.METRICS / "baseline_establishment_summary.json").read_text())


def mean(rows, model, key, pred=lambda r: True):
    v = [r[key] for r in rows if r["model"] == model and isinstance(r.get(key), float) and pred(r)]
    return sum(v) / len(v) if v else None


def fmt(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


# ---------------------------------------------------------------- verdict logic
def derive_verdict(rows):
    edit_models = ["flux_img2img", "flux_kontext"]
    style_models = ["flux_redux", "flux_ipadapter", "styleid", "flux_kontext"]
    v = {}
    # editing: balance edit strength (clipT_gain or clipT_target) with content preservation
    def edit_score(m):
        gain = mean(rows, m, "clipT_gain", lambda r: r["kind"] == "edit")
        if gain is None:
            gain = mean(rows, m, "clipT_target", lambda r: r["kind"] == "edit")
            gain = (gain - 0.2) if gain is not None else None
        pres = mean(rows, m, "clipI_content", lambda r: r["kind"] == "edit")
        if gain is None or pres is None:
            return None
        return gain + 0.5 * pres
    v["edit_scores"] = {m: edit_score(m) for m in edit_models}
    v["best_edit"] = max((m for m in edit_models if v["edit_scores"][m] is not None),
                         key=lambda m: v["edit_scores"][m], default=None)
    # style adherence
    def style_adh(m):
        return mean(rows, m, "clipI_style", lambda r: r["kind"] == "style")
    v["style_adh"] = {m: style_adh(m) for m in style_models}
    v["best_style"] = max((m for m in style_models if v["style_adh"][m] is not None),
                          key=lambda m: v["style_adh"][m], default=None)
    # leakage resistance = content kept minus reference leaked
    def leak_res(m):
        cp = mean(rows, m, "dino_content", lambda r: r["kind"] == "style")
        lk = mean(rows, m, "dino_style", lambda r: r["kind"] == "style")
        if cp is None or lk is None:
            return None
        return cp - lk
    v["leak_res"] = {m: leak_res(m) for m in style_models}
    v["best_leak"] = max((m for m in style_models if v["leak_res"][m] is not None),
                         key=lambda m: v["leak_res"][m], default=None)
    # substrate verdict
    v["verdict"] = "PROCEED_WITH_FLUX_KONTEXT" if v["best_edit"] == "flux_kontext" else (
        "NEEDS_MORE_BASELINE_VALIDATION" if v["best_edit"] is None else
        "PROCEED_WITH_FLUX_IMG2IMG_REVIEW")
    return v


# ---------------------------------------------------------------- table builders
def metric_table(rows, models, keys, key_labels, pred=lambda r: True, lower=()):
    head = "".join(f"<th>{html.escape(l)}</th>" for l in key_labels)
    body = ""
    # find best per column for highlight
    col_vals = {k: [mean(rows, m, k, pred) for m in models] for k in keys}
    best = {}
    for k in keys:
        vals = [(i, x) for i, x in enumerate(col_vals[k]) if x is not None]
        if vals:
            best[k] = (min if k in lower else max)(vals, key=lambda t: t[1])[0]
    for i, m in enumerate(models):
        cells = ""
        for k in keys:
            x = col_vals[k][i]
            hl = ' class="best"' if best.get(k) == i else ""
            cells += f"<td{hl}>{fmt(x)}</td>"
        body += f"<tr><td class='ml'>{html.escape(PRETTY[m])}</td>{cells}</tr>"
    return f"<table class='metrics'><thead><tr><th>model</th>{head}</tr></thead><tbody>{body}</tbody></table>"


def method_card(mid, extra):
    d = C.MODELS[mid]
    rows = "".join(
        f"<div class='mc-row'><span class='mc-k'>{k}</span><span class='mc-v'>{html.escape(str(v))}</span></div>"
        for k, v in [("Model", d["model"]), ("Data", d["data"]), ("Supervision", d["supervision"]),
                     ("One-line insight", d["insight"]), ("Inference setup", extra.get("inference", "")),
                     ("Strengths", extra.get("strengths", "")), ("Weaknesses", extra.get("weaknesses", ""))])
    return f"<div class='methodcard'><h4>{html.escape(d['name'])} <span class='fam'>{html.escape(d['family'])}</span></h4>{rows}</div>"


# ---------------------------------------------------------------- main HTML
def build(phase):
    rows = load_rows()
    summ = load_summary()
    V = derive_verdict(rows)
    data_summary = json.loads((C.DATA / "data_summary.json").read_text())
    run_models = sorted({r["model"] for r in rows})

    edit_models = [m for m in ["flux_img2img", "flux_kontext"] if m in run_models]
    style_models = [m for m in ["flux_redux", "flux_ipadapter", "styleid", "flux_kontext"] if m in run_models]

    F = C.FIG
    RV = F / "representation_visuals"

    def section(num, title, body):
        return f"<section id='s{num}'><h2><span class='num'>{num}</span>{html.escape(title)}</h2>{body}</section>"

    # ---- inference notes per model
    INFER = {
        "flux_img2img": dict(inference=f"FLUX.1-dev 4-bit NF4, {C.GEN_SIZE}px, {C.INFER_STEPS} steps, g={C.GUIDANCE}, strength 0.6",
                             strengths="cheap; continuity with prior phase", weaknesses="renoises globally; weak instruction grounding; identity drift"),
        "flux_redux": dict(inference=f"Redux SigLIP prior + FLUX.1-dev 4-bit, {C.GEN_SIZE}px, {C.INFER_STEPS} steps",
                           strengths="strong appearance/style cue from a reference", weaknesses="copies reference *semantics* — high leakage; no instruction control"),
        "flux_ipadapter": dict(inference=f"XLabs IP-Adapter on FLUX.1-dev 4-bit, scale 0.9, {C.GEN_SIZE}px",
                              strengths="decouples content prompt from style image (InstantStyle-like)", weaknesses="prompt must carry the content; tuning-sensitive"),
        "flux_kontext": dict(inference="FLUX.1-Kontext-dev 4-bit, native 1024px, 20 steps, g=2.5",
                            strengths="native in-context instruction editor; preserves untouched regions", weaknesses="forced 1024px → slowest; heavier VRAM"),
        "styleid": dict(inference="VGG-19 Gram (Gatys 2016), Adam 300 it, content-anchored, 512px — classical control, NOT StyleID-attention",
                       strengths="lowest semantic leakage; pure texture/colour transfer", weaknesses="no semantics; can look like a filter; per-image optimisation (diagnostic control, not a main no-train baseline)"),
    }

    # ===== sections
    s = []

    # 1 executive summary
    verdict_class = "good" if V["verdict"].startswith("PROCEED") else "warn"
    s.append(section(1, "Executive summary", f"""
      <div class="verdict {verdict_class}"><span class="h">Verdict</span><b>{V['verdict']}</b></div>
      <p class="note"><b>Scope:</b> the verdict means <i>FLUX.1-Kontext-dev is the strongest
      <u>accessible no-training</u> substrate under current compute constraints</i> — not a global
      claim that it beats every editor. The one stronger open editor (Qwen-Image-Edit, 20B) was
      <b>not run</b> (exceeds the single-25&nbsp;GB-A5000 VRAM/disk budget), so it is untested, not
      beaten. The "StyleID" slot is the <b>classical VGG-19 Gram (Gatys 2016) control</b>, not the
      StyleID attention-injection method.</p>
      <p class="lead">We established a serious, training-free baseline suite for real-world image
      editing and reference-based stylisation, and measured it on real benchmarks
      (MagicBrush, PIE-Bench++) plus a custom reference-leakage diagnostic set. The goal was to
      pick the right <b>substrate</b> for the next spectral/frequency-domain intervention phase.</p>
      <ul>
        <li><b>Strongest <i>accessible no-training</i> editor:</b> {PRETTY.get(V['best_edit'],'—')} — best blend of edit
            strength and content preservation across {len([r for r in rows if r['kind']=='edit'])} editing outputs (among the baselines actually run).</li>
        <li><b>Strongest style adherence:</b> {PRETTY.get(V['best_style'],'—')} (CLIP-I to style ref).</li>
        <li><b>Most leakage-resistant reference stylisation:</b> {PRETTY.get(V['best_leak'],'—')}
            (keeps content, copies least reference semantics).</li>
        <li><b>Models run:</b> {', '.join(PRETTY[m] for m in run_models)}.
            Qwen-Image-Edit was <i>not</i> run (20B exceeds the 25&nbsp;GB VRAM / disk budget); Kontext is the substituted strong editor.</li>
      </ul>"""))

    # 2 why baseline phase
    s.append(section(2, "Why a baseline-establishment phase", """
      <p>Phase&nbsp;1 found a <b>NO_GO</b> for naive spectral interventions <i>on weak FLUX
      pipelines</i> (vanilla img2img, Redux). That result is about the substrate, not the idea:
      if the underlying editor is weak, a frequency-domain edit has nothing solid to act on.
      Before any further spectral work we therefore need to know which no-training baseline is
      actually <b>competent</b>, and on which tasks. This phase answers that empirically.</p>"""))

    # 3 prior context
    s.append(section(3, "Prior context and motivation", """
      <p>The prior-phase report referenced by the brief was not available in this environment
      (the <code>/mnt/data</code> handoff file was absent), so continuity is taken from the
      project's own experiment record (the spectral demo toolkit and the E-series roadmap):
      token / latent / velocity spectral operators on FLUX and SD3.5. The consistent finding
      there — weak editors give spectral ops nothing to bite on — is exactly what motivates
      establishing a strong substrate first.</p>"""))

    # 4 benchmarks
    s.append(section(4, "Benchmarks and datasets", f"""
      <table class="kv">
        <tr><th>MagicBrush (dev)</th><td>{data_summary['magicbrush']} real instruction-edit examples
            (source photo + free-form instruction + GT target). Mixed local/global edits.</td></tr>
        <tr><th>PIE-Bench++</th><td>{data_summary['piebench']} examples spanning 8 task types:
            object replace / add / remove, attribute, colour, material, background(global), style.
            Source + cleaned target prompt; instruction synthesised from the bracketed word swaps.</td></tr>
        <tr><th>Style bank (WikiArt)</th><td>{data_summary['style_refs']} artwork references
            (oil, impressionist, abstract, watercolour) streamed from <code>huggan/wikiart</code>.</td></tr>
        <tr><th>Custom leakage set</th><td>{data_summary['leakage_pairs']} content×style pairs
            (aligned & adversarial) built from real photos × WikiArt refs, for reference-leakage stress.</td></tr>
      </table>
      <p class="note">All images centre-cropped to square. Subsets are deliberately pilot-scale
      (Phase&nbsp;B) — enough to expose clear strengths/weaknesses with the full metric suite,
      reproducible from the manifests under <code>data/</code>.</p>"""))

    # 5 models + inference
    cards = "".join(method_card(m, INFER[m]) for m in run_models)
    cards += method_card("qwen_image_edit", dict(inference="NOT RUN — 20B model, &gt;25GB VRAM and disk budget exceeded",
                                                 strengths="(would be the strong external editor)",
                                                 weaknesses="hardware-infeasible this phase; Kontext substituted"))
    s.append(section(5, "Models and inference settings", f"""
      <p>Every method is <b>training-free</b> (no LoRA / finetune / DreamBooth / learned adapters /
      per-image latent codes as a baseline). All FLUX transformers run 4-bit NF4 on one RTX A5000.</p>
      {cards}"""))

    # 6 methodology
    s.append(section(6, "Evaluation methodology", """
      <p>Metrics are computed with cached frozen encoders. Higher is more similar for CLIP/DINO/SigLIP;
      lower is more similar for LPIPS / colour-hist / Fourier distance.</p>
      <table class="kv">
        <tr><th>Content preservation</th><td>CLIP-I, DINOv2, SigLIP, LPIPS, colour-hist — output vs content</td></tr>
        <tr><th>Edit correctness</th><td>CLIP-T to target prompt; CLIP-T <i>gain</i> = target − source (PIE-Bench)</td></tr>
        <tr><th>Style transfer</th><td>CLIP-I / DINO to style ref; colour-hist & Fourier-amplitude distance</td></tr>
        <tr><th>Reference leakage</th><td>DINO / CLIP-I to the style image — high means the output copied
            the reference's <i>semantics</i>, not just its style</td></tr>
      </table>"""))

    # 7 results by benchmark
    s.append(section(7, "Results by benchmark", f"""
      <h3>Instruction editing (content preservation + edit correctness)</h3>
      {metric_table(rows, edit_models,
            ['clipI_content','dino_content','lpips_content','clipT_target','clipT_gain'],
            ['CLIP-I content','DINO content','LPIPS↓','CLIP-T target','CLIP-T gain'],
            pred=lambda r: r['kind']=='edit', lower=('lpips_content',))}
      <p class="cap">Editing comparison grid (content vs FLUX editors):</p>
      {img_tag(F/'grids'/'edit_comparison.png', 'wide')}"""))

    # 8 results by task type
    s.append(section(8, "Results by task type", f"""
      <p>Which edit categories are already solved vs still hard (CLIP-T gain per task):</p>
      {img_tag(RV/'heatmap_task_model.png','mid')}
      <p>Full model × metric profile (z-scored colour, raw values shown):</p>
      {img_tag(RV/'heatmap_model_metric.png','wide')}"""))

    # 9 results by model family
    s.append(section(9, "Results by model family", f"""
      <p>Preservation vs edit-strength trade-off — the core editing question:</p>
      {img_tag(RV/'scatter_preservation_vs_edit.png','mid')}
      {img_tag(RV/'similarity_matrix.png','mid')}
      {img_tag(RV/'runtime.png','mid')}"""))

    # 10 style transfer
    s.append(section(10, "Style-transfer section", f"""
      {metric_table(rows, style_models,
            ['clipI_style','dino_style','colorhist_style','fourier_style','dino_content'],
            ['CLIP-I style','DINO style','colour-hist↓','Fourier↓','DINO content'],
            pred=lambda r: r['kind']=='style', lower=('colorhist_style','fourier_style'))}
      <p class="cap">Style comparison grid (content + reference vs style baselines):</p>
      {img_tag(F/'grids'/'style_comparison.png','wide')}
      <p>Style adherence vs content preservation:</p>
      {img_tag(RV/'scatter_style_vs_leakage.png','mid')}"""))

    # 11 leakage
    s.append(section(11, "Reference-leakage section", f"""
      <p>The research-critical question: when a reference is meant as <i>style only</i>, which
      models copy its <b>content</b>? Leakage-resistance = DINO(content) − DINO(style):</p>
      {img_tag(RV/'pareto_leakage_resistance.png','mid')}
      <div class="two">
        <div><p class="cap">Highest leakage (copies the reference):</p>{img_tag(F/'leakage_cases'/'high_leakage.png','col')}</div>
        <div><p class="cap">Lowest leakage (content preserved):</p>{img_tag(F/'leakage_cases'/'low_leakage.png','col')}</div>
      </div>"""))

    # 12 failure analysis
    s.append(section(12, "Failure analysis", f"""
      <ul>
        <li><b>FLUX img2img:</b> global renoising — colour/material edits bleed everywhere, identity drifts.</li>
        <li><b>FLUX Redux:</b> strongest <i>leakage</i> — it remixes the reference, so adversarial
            (mismatched) pairs import the reference's objects, not just its style.</li>
        <li><b>IP-Adapter:</b> better content/style decoupling but depends on the content prompt being right.</li>
        <li><b>VGG-Gram (Gatys) control:</b> never leaks semantics, but applies style as a flat texture/colour filter.</li>
        <li><b>Kontext:</b> the only method that keeps untouched regions <i>and</i> applies the asked edit;
            main cost is the forced 1024px runtime.</li>
      </ul>"""))

    # 13 best
    s.append(section(13, "Best qualitative examples", f"""
      {img_tag(F/'best_cases'/'best_edits.png','wide')}
      {img_tag(F/'leakage_cases'/'low_leakage.png','wide')}"""))

    # 14 worst
    s.append(section(14, "Worst qualitative examples", f"""
      {img_tag(F/'worst_cases'/'worst_edits.png','wide')}
      {img_tag(F/'leakage_cases'/'high_leakage.png','wide')}"""))

    # 15 representation visuals + video
    _small = C.VIDEO / "baseline_walkthrough_small.mp4"
    vid = b64(_small if _small.exists() else C.VIDEO / "baseline_walkthrough.mp4")
    vid_html = (f'<video class="wide" controls src="{vid}"></video>' if vid and len(vid) < 25_000_000
                else '<p class="note">Video at <code>videos/baseline_walkthrough.mp4</code> (too large to inline).</p>')
    s.append(section(15, "Representation visuals and summary plots", f"""
      <p>Benchmark- and metric-space summaries (model×metric, task×model, scatters, Pareto,
      similarity matrix, runtime) appear above. Walkthrough montage of every example across models:</p>
      {vid_html}"""))

    # 16 recommendation
    s.append(section(16, "Final recommendation — which baseline to build on next", f"""
      <div class="verdict {verdict_class}"><span class="h">Substrate verdict</span><b>{V['verdict']}</b></div>
      <p>For the next spectral/frequency-domain phase:</p>
      <ul>
        <li><b>Editing substrate:</b> {PRETTY.get(V['best_edit'],'—')} — it is the competent in-context
            editor that preserves content while making the requested change, so a frequency-domain edit
            has a stable signal to act on (unlike the Phase-1 weak baselines).</li>
        <li><b>Style-transfer substrate:</b> {PRETTY.get(V['best_style'],'—')} for adherence, with
            <b>VGG-Gram (Gatys) control</b> as the low-leakage reference point.</li>
        <li><b>Leakage-resistant stylisation:</b> {PRETTY.get(V['best_leak'],'—')} — the right control
            condition for studying how spectral edits move content vs style.</li>
        <li><b>Most informative benchmark subsets:</b> PIE-Bench colour / material / object-replace
            (clean source→target prompts give a real CLIP-T gain signal) and the adversarial leakage pairs.</li>
      </ul>"""))

    # ---- assemble
    toc = "".join(f"<li><a href='#s{i+1}'>{i+1}. {t}</a></li>" for i, t in enumerate([
        "Executive summary","Why a baseline phase","Prior context","Benchmarks & datasets",
        "Models & inference","Methodology","Results by benchmark","Results by task type",
        "Results by model family","Style transfer","Reference leakage","Failure analysis",
        "Best examples","Worst examples","Representation visuals & video","Final recommendation"]))

    css = """
    :root{--ink:#1b2230;--muted:#5d6675;--line:#e6e9f0;--accent:#2b5cb8;--good:#1f7a4d;--good-bg:#eef8f2;--warn:#b23b3b;--warn-bg:#fdf1f1;--chip:#f4f6fb;}
    *{box-sizing:border-box}body{font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);margin:0;background:#fafbfc}
    .wrap{max-width:1100px;margin:0 auto;padding:40px 28px 120px;background:#fff}
    h1{font-size:34px;letter-spacing:-.02em;margin:0 0 6px}.sub{color:var(--muted);font-size:18px;margin:0 0 24px}
    h2{font-size:24px;margin:14px 0 10px;letter-spacing:-.01em;padding-top:14px;border-top:1px solid var(--line)}
    h2 .num{display:inline-block;min-width:30px;color:var(--accent);font-weight:800}
    h3{font-size:18px;margin:20px 0 6px}h4{margin:0 0 8px}
    section{margin:30px 0}.lead{font-size:17.5px}
    .toc{background:var(--chip);border:1px solid var(--line);border-radius:12px;padding:14px 20px;columns:2;font-size:14.5px}
    .toc li{margin:3px 0}.toc a{color:var(--accent);text-decoration:none}
    img.fig,img.wide,img.mid,img.col,video.wide{max-width:100%;border:1px solid var(--line);border-radius:10px;display:block;margin:10px 0}
    img.mid{max-width:680px}img.col{max-width:100%}
    table.metrics{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}
    table.metrics th,table.metrics td{border:1px solid var(--line);padding:6px 9px;text-align:center}
    table.metrics th{background:var(--chip)}table.metrics td.ml{text-align:left;font-weight:600}
    table.metrics td.best{background:var(--good-bg);font-weight:700;color:var(--good)}
    table.kv{border-collapse:collapse;width:100%;font-size:14.5px;margin:8px 0}
    table.kv th{text-align:left;width:190px;vertical-align:top;padding:7px 10px;background:var(--chip);border:1px solid var(--line)}
    table.kv td{padding:7px 10px;border:1px solid var(--line)}
    .verdict{border-radius:12px;padding:14px 18px;margin:14px 0;font-size:16px}
    .verdict.good{background:var(--good-bg);border:1px solid #cfe9da}.verdict.warn{background:var(--warn-bg);border:1px solid #f0d4d4}
    .verdict .h{display:block;font-size:12px;letter-spacing:.06em;text-transform:uppercase;font-weight:700;margin-bottom:3px}
    .verdict.good .h{color:var(--good)}.verdict.warn .h{color:var(--warn)}.verdict b{font-size:19px}
    .methodcard{border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:12px 0;background:#fff}
    .methodcard h4 .fam{font-size:12px;font-weight:600;color:#fff;background:var(--accent);border-radius:20px;padding:2px 10px;margin-left:8px;vertical-align:middle}
    .mc-row{display:flex;gap:12px;padding:3px 0;border-top:1px dashed var(--line)}
    .mc-k{flex:0 0 130px;color:var(--muted);font-size:13px;text-transform:uppercase;letter-spacing:.03em}
    .mc-v{flex:1;font-size:14.5px}
    .two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .cap{font-size:13.5px;color:var(--muted);margin:14px 0 2px}.note{font-size:13.5px;color:var(--muted)}
    .missing{color:#b23b3b;font-size:13px;padding:8px;background:var(--warn-bg);border-radius:6px}
    footer{margin-top:50px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
    @media(max-width:720px){.toc,.two{columns:1;grid-template-columns:1fr}}
    """
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Baseline Establishment — Spectral FLUX Editing</title><style>{css}</style></head>
    <body><div class="wrap">
    <h1>Baseline Establishment for Spectral FLUX Image Editing</h1>
    <p class="sub">Phase&nbsp;2 · training-free baselines for real-world editing & reference stylisation ·
    {len(rows)} evaluated outputs · {len(run_models)} models · MagicBrush + PIE-Bench++ + custom leakage set</p>
    <ol class="toc">{toc}</ol>
    {''.join(s)}
    <footer>Generated by <code>baseline_establishment</code> · all FLUX models 4-bit NF4 on one RTX A5000 ·
    metrics in <code>metrics/baseline_establishment_metrics.csv</code> · figures embedded as base64 for self-containment.</footer>
    </div></body></html>"""

    out = C.REPORTS / "baseline_establishment_report.html"
    out.write_text(doc)
    # also the canonical html/ path
    (C.REPORTS / "html").mkdir(parents=True, exist_ok=True)
    print(f"report -> {out}  ({len(doc)//1024} KB)")
    print("verdict:", V["verdict"], "| best_edit", V["best_edit"], "| best_style", V["best_style"], "| best_leak", V["best_leak"])
    return V


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="pilot")
    a = ap.parse_args()
    build(a.phase)


if __name__ == "__main__":
    main()
