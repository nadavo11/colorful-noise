"""E52 — Text-Token Modulation Autopsy report section (anaconda env). Exposes
section_html() and md_section(), which report.py injects into the integrated E51+E52 report.
Self-contained (base64-embeds its own figures); guarded against missing data."""
from __future__ import annotations
import base64, json
from pathlib import Path

import config as C


def _b64(path):
    p = Path(path)
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def _img(path, cap="", w="100%"):
    u = _b64(path)
    if u is None:
        return ""
    return f"<figure><img src='{u}' style='width:{w}'><figcaption>{cap}</figcaption></figure>"


def _summary():
    fp = C.TOK / "token_summary.json"
    return json.loads(fp.read_text()) if fp.exists() else None


def _token_table(eid):
    fp = C.DIAG / f"token_align_{eid}.json"
    if not fp.exists():
        return ""
    a = json.loads(fp.read_text())
    tags = a["align"]["tags"]
    cols = {"same": "#eef", "changed": "#fde8e8", "inserted": "#e8f6ee"}

    def chips(tokens, tagged=None):
        out = ""
        for j, t in enumerate(tokens):
            bg = cols.get(tagged[j], "#f4f4f4") if tagged else "#eef"
            out += f"<span style='background:{bg};border:1px solid #ddd;border-radius:4px;padding:1px 5px;margin:1px;display:inline-block;font-size:12px'>{t}</span> "
        return out
    roles = a.get("roles", {})
    role_rows = "".join(
        f"<tr><td>{r}</td><td>{(v or {}).get('word','—') if v else '—'}</td>"
        f"<td>{(v or {}).get('index','—') if v else '—'}</td></tr>" for r, v in roles.items())
    return (f"<div class='card'><h4>{eid} <span class='mono'>{a['task_type']}</span></h4>"
            f"<p><b>source:</b> {chips(a['source_tokens'])}</p>"
            f"<p><b>target:</b> {chips(a['target_tokens'], tags)} "
            f"<span class='sub'>(<span style='background:#fde8e8'>changed</span> · "
            f"<span style='background:#e8f6ee'>inserted</span>)</span></p>"
            f"<table style='width:auto'><tr><th>role</th><th>token</th><th>idx</th></tr>{role_rows}</table></div>")


def _vclass(v):
    return "go" if "GO" in v and "NO" not in v else ("nogo" if "NO-GO" in v else "mixed")


def section_html():
    s = _summary()
    if s is None:
        return ("<h2>14 · Text-Token Modulation Autopsy</h2>"
                "<p class='miss'>No token-autopsy artifacts found — run "
                "<code>token_autopsy.py</code> + <code>token_evaluate.py</code> + "
                "<code>token_analyze.py</code> + <code>token_visualize.py</code>.</p>")
    te = s["text_entry"]; ev = s["verdict_evidence"]; iv = s.get("interventions", {})
    cc = s.get("cache_connection", {})
    ids = [pe["id"] for pe in s["per_example"]]
    vclass = _vclass(s["verdict"])

    tables = "".join(_token_table(eid) for eid in ids[:6])
    attn = "".join(
        _img(C.TOK_HEAT / f"per_token_attention_{eid}.png", f"{eid}: per-token attention over denoising")
        + _img(C.TOK_HEAT / f"token_timestep_heatmap_{eid}.png", f"{eid}: token × timestep attention mass")
        + _img(C.TOK_HEAT / f"token_layer_heatmap_{eid}.png", f"{eid}: token × block contribution")
        + _img(C.TOK_HEAT / f"edit_influence_{eid}.png", f"{eid}: per-token causal influence on Δ_edit")
        for eid in ids[:3])
    spatial = "".join(_img(C.TOK_SPATIAL / f"spatial_{eid}.png", f"{eid}: per-token spatial attention maps + overlay")
                      for eid in ids[:3])
    freq = "".join(_img(C.TOK_HEAT / f"freq_coupling_{eid}.png", f"{eid}: frequency/token coupling")
                   for eid in ids[:3])
    curves = (_img(C.TOK_CURVES / "weight_edit_strength.png", "Token weight vs edit strength, per mechanism")
              + _img(C.TOK_CURVES / "weight_preservation.png", "Token weight vs preservation (drift from the unmodified edit)")
              + _img(C.TOK_CURVES / "weight_delta_smoothness.png", "Token weight vs Δ_edit spectral stability"))
    montages = ""
    md = C.TOK_GRIDS / "_montage"
    if md.exists():
        for fp in sorted(md.glob("*.png"))[:8]:
            montages += _img(fp, fp.stem.replace("__", " · "))
    cache = "".join(_img(C.TOK_CACHE / f, "") for f in
                    ["stability_vs_cache.png", "entropy_vs_cache.png", "peakstep_vs_smoothness.png"])

    # mechanism summary table
    mech_rows = ""
    for m, md_ in iv.get("mechanisms", {}).items():
        mech_rows += (f"<tr><td>{m}</td><td>{md_['edit_response']:.4f}</td>"
                      f"<td>{md_['preservation_cost']:.4f}</td>"
                      f"<td>{md_.get('smoothness_response', float('nan')):.4f}</td></tr>")
    best = iv.get("best_mechanism", "—")

    def yn(b):
        return "✅ yes" if b else "❌ no"
    decision = f"""
    <table style='width:auto'>
    <tr><th>Decision question</th><th>Answer</th></tr>
    <tr><td>1. Do edit-relevant tokens show identifiable attention/influence patterns?</td><td>{yn(ev.get('identifiable'))} (edit/non-edit attention ratio {ev.get('edit_attention_ratio', float('nan')):.2f}×)</td></tr>
    <tr><td>2. Can we causally amplify/suppress individual token effects?</td><td>{yn(ev.get('controllable'))} (best edit response {ev.get('best_edit_response', float('nan')):.4f}/×)</td></tr>
    <tr><td>3. Is attention-space weighting better than embedding-space?</td><td>{yn(ev.get('attention_better_than_embedding'))} (attn range {ev.get('attn_space_range', float('nan')):.3f} vs embed {ev.get('embed_space_range', float('nan')):.3f})</td></tr>
    <tr><td>4. Does token weighting improve controllability?</td><td>{yn(ev.get('controllable'))}</td></tr>
    <tr><td>5. Does token weighting make Δ_edit caching easier or harder?</td><td>{('easier' if ev.get('cache_helps') else 'unclear / preliminary')}</td></tr>
    <tr><td>6. Is this a promising research direction?</td><td>{yn('GO' in s['verdict'] and 'NO' not in s['verdict'])}</td></tr>
    </table>"""

    return f"""
<h2>14 · Text-Token Modulation Autopsy</h2>
<div class='verdict {vclass}'>Token-autopsy verdict: {s['verdict']}</div>
<p class='sub'>E52 · {s['n_examples']} PIE-Bench examples · attention taps on double-stream blocks
{s['tap_blocks']} · core question: <i>can we identify, visualize and causally manipulate the
text-token components that drive image edits?</i></p>

<h3>14.1 What text representation the model uses &amp; where text enters</h3>
<div class='note'><b>Representation:</b> {te['representation']}<br><br>
<b>Mechanism (where text enters):</b> {te['mechanism']}<br><br>
<b>Pooled path:</b> {te['pooled']}<br><br>
<b>Implication:</b> {te['note']}</div>

<h3>14.2 Tokenized prompts, alignment &amp; role assignment</h3>
<p>Per example we tokenize source and target (T5), align them (difflib) to mark
changed/inserted tokens, and assign five probe roles. The autopsy then tracks each token's
attention and causal influence.</p>
{tables}

<h3>14.3 Which tokens dominate the edit (observational)</h3>
<p>Edited tokens attract <b>{ev.get('edit_attention_ratio', float('nan')):.2f}×</b> the
attention mass of unedited tokens; dominant causal roles across examples:
<b>{', '.join(ev.get('dominant_roles', [])) or '—'}</b>. The heatmaps show where (which step,
which block) each token enters; the influence curves are token-ablation Δ_edit contributions.</p>
{attn}

<h3>14.4 Per-token spatial attention maps</h3>
<p>For the object / attribute / style / background / control tokens, the image-query attention
reshaped to the latent grid, and overlaid on the input.</p>
{spatial}

<h3>14.5 Frequency / token coupling</h3>
<p>Decomposing each token's causal Δ_edit effect into radial low (layout) / mid (shape) /
high (texture) bands — does a token act globally or on fine detail?</p>
{freq}

<h3>14.6 Causal token interventions</h3>
<p>Four internal mechanisms, swept over weights {C.TOK_WEIGHTS}, applied to each role token.
Embedding-space scales <code>e_i</code>; the other three act inside joint attention
(logit bias, post-softmax reweight, value scaling).</p>
<table style='width:auto'><tr><th>mechanism</th><th>edit response (Δgain/×)</th>
<th>preservation cost (ΔLPIPS/×)</th><th>Δ_edit smoothness response</th></tr>{mech_rows}</table>
<div class='note'><b>Best mechanism by edit response:</b> <code>{best}</code>. Attention-space
controllable edit range {ev.get('attn_space_range', float('nan')):.3f} vs embedding-space
{ev.get('embed_space_range', float('nan')):.3f}.</div>
{curves}
<h4>Intervention grids (token weight sweeps)</h4>
{montages or "<p class='miss'>[no intervention montages]</p>"}

<h3>14.7 Connection to the edit-direction cache</h3>
<p>Relating token-attention behaviour (this autopsy) to the spectral-delta-cache quality (§9)
per example. Stable, peaky token attention should coincide with smoother, more cacheable
Δ_edit.</p>
{cache or "<p class='miss'>[too few paired examples for cache correlation]</p>"}
<div class='note'>stability↔cache-LPIPS r =
{cc.get('stability_vs_cache_lpips', float('nan')):.2f} · entropy↔cache-LPIPS r =
{cc.get('entropy_vs_cache_lpips', float('nan')):.2f} · peak-step↔Δ_edit-smoothness r =
{cc.get('peakstep_vs_smoothness', float('nan')):.2f} (n={cc.get('n', 0)}).</div>

<h3>14.8 Decision criteria</h3>
{decision}
<div class='verdict {vclass}'>{s['verdict']}</div>
<p>{_verdict_text(s['verdict'])}</p>
"""


def _verdict_text(v):
    if v == "STRONG GO":
        return ("Token influence is measurable (edit tokens are clearly the attention/contribution "
                "hotspots), causal (internal interventions move the edit monotonically), controllable, "
                "and the stable-attention examples are the more cacheable ones — a combined "
                "token-modulation + edit-direction-cache method is worth building.")
    if v == "GO":
        return ("Token influence is measurable and causally controllable via internal interventions; "
                "the cache connection is suggestive but underpowered at this sample size. Worth "
                "escalating token-attention modulation, with a larger cache-correlation follow-up.")
    if v.startswith("MIXED"):
        return ("Token effects exist but are unstable or mechanism/architecture-dependent — some "
                "interventions move the edit, others mostly add artifacts. Tighten the intervention "
                "and re-test before committing.")
    return ("Internal token weighting did not reliably steer the edit beyond artifacts at this "
            "setting; not a promising standalone direction as tested.")


def md_section():
    s = _summary()
    if s is None:
        return "\n## E52 — Text-Token Modulation Autopsy\n_(not yet run)_\n"
    ev = s["verdict_evidence"]
    return f"""
## E52 — Text-Token Modulation Autopsy (integrated)
**Token-autopsy verdict: {s['verdict']}**
- Where text enters: FLUX MMDiT joint attention (T5 tokens are the leading key/value columns) + pooled-CLIP AdaLN.
- Edit tokens attract {ev.get('edit_attention_ratio', float('nan')):.2f}× the attention of unedited tokens.
- Best intervention mechanism: `{s.get('interventions', {}).get('best_mechanism', '—')}`; best edit response {ev.get('best_edit_response', float('nan')):.4f}/×.
- Attention-space vs embedding-space controllable range: {ev.get('attn_space_range', float('nan')):.3f} vs {ev.get('embed_space_range', float('nan')):.3f}.
- Artifacts under `outputs/spectral_edit_direction_cache_probe/token_autopsy/`.
"""
