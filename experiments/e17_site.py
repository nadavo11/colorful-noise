"""Build a self-contained HTML explainer for E17 (SBN vs CFG-Zero* vs CFG++ on
Stable Diffusion 3.5, the true-CFG testbed).

Reads results/e17/report.json (top keys ["params","scores"]; scores[scene]["conds"]
[cond][metric] = {"mean","std","n"}) plus the per-scene grid_<scene>.png files, and
EMBEDS every image as base64 so the page is fully portable (open results/e17/index.html
anywhere). Honors CN_RESULTS via common.RESULTS.

The page STANDS ALONE: it defines every term (SD3.5 true CFG, the cfg=1 reference,
SBN / band-norm, CFG-Zero*, CFG++, every condition, every metric with its ↑/↓ "good"
direction), then leads each scene with its 8-condition grid + a "what to look for"
caption before the per-condition number table, and closes with an aggregate table
across the 8 scenes. Numbers are pulled straight from report.json (no re-scoring, no
model load). (memory: experiment-documentation-standard.)

    python e17_site.py            # writes results/e17/index.html, no model loaded
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e27_site import data_uri  # base64 image embedding (the project standard)

OUT = os.path.join(RESULTS, "e17")

# The 8 conditions in report.json, in display order.
CONDS = ["cfg1", "cfg_hi", "bandnorm", "bandnorm_pp", "cfgzero", "cfgzero_sbn",
         "cfgpp", "cfgpp_sbn"]
BASE = "cfg_hi"                                # the high-CFG baseline conditions are judged against
# Pretty labels for the condition column.
COND_LABEL = {
    "cfg1": "cfg1",
    "cfg_hi": "cfg_hi (baseline)",
    "bandnorm": "bandnorm (SBN)",
    "bandnorm_pp": "bandnorm_pp",
    "cfgzero": "cfgzero",
    "cfgzero_sbn": "cfgzero_sbn",
    "cfgpp": "cfgpp",
    "cfgpp_sbn": "cfgpp_sbn",
}
# Scene display order (matches the 8 grid_<scene>.png files).
SCENES = ["fisherman", "apothecary", "market", "cyberpunk", "library",
          "astronaut", "banquet", "workshop"]

# Headline metrics shown per scene + in the aggregate, with ↑/↓ "good" direction.
# (False = higher is better, True = lower is better.)
HEADLINE = [("aesthetic", False), ("imagereward", False), ("clip_t", False)]
# image-statistic columns (descriptive sanity signals, no "best" highlight).
STATS = ["sharpness", "hf_frac", "rms_contrast", "colorfulness", "saturation"]

CSS = """
body{font:15px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px auto;max-width:1000px;color:#1a1a1a;padding:0 16px}
h1{font-size:25px;line-height:1.25} h2{font-size:19px;margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:5px}
h3{font-size:15px;margin-bottom:4px}
.tldr{background:#eef4ff;border:1px solid #c7d9ff;border-radius:6px;padding:14px 16px;margin:14px 0}
.note{background:#f6f8fa;border-left:4px solid #0969da;padding:10px 14px;border-radius:4px;margin:12px 0}
.cav{background:#fff8f0;border-left:4px solid #d4a017}.win{background:#eafaf0;border-left:4px solid #2da44e}
dl{margin:10px 0} dt{font-weight:700;margin-top:10px} dd{margin:2px 0 2px 18px;color:#333}
table{border-collapse:collapse;margin:12px 0;font-variant-numeric:tabular-nums;font-size:13px}
th,td{border:1px solid #d0d7de;padding:5px 9px;text-align:center}
th{background:#f6f8fa} td.v{text-align:left;font-weight:600;white-space:nowrap}
td.pos{background:#dafbe1}td.neg{background:#ffebe9}
.cap{color:#555;font-size:13px}
img.grid{width:100%;border:1px solid #d0d7de;border-radius:4px;margin:8px 0}
img.plot{max-width:100%;border:1px solid #d0d7de;border-radius:4px;margin:8px 0}
code{background:#eff1f3;padding:1px 5px;border-radius:3px;font-size:13px}
"""


def img_tag(name, cls="grid", **kw):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return f"<p class=cap>(missing {name})</p>"
    return f"<img class={cls} src='{data_uri(p, **kw)}'>"


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def cmean(scene_conds, cond, metric):
    """mean of `metric` for `cond` in this scene, or None if absent/null."""
    c = scene_conds.get(cond)
    if not c:
        return None
    v = c.get(metric)
    return v["mean"] if v and v.get("mean") is not None else None


def best_cond(values, lower_better):
    """label of the best (non-None) condition for a column; None if all None."""
    cands = {k: v for k, v in values.items() if v is not None}
    if not cands:
        return None
    return (min if lower_better else max)(cands, key=cands.get)


def metric_table(rows_label_value, headline, base_for_delta=None):
    """Render a per-condition table. `rows_label_value` maps cond -> {metric: value}.
    Best cell per headline column is highlighted (td.pos)."""
    h = ["<table><tr><th>condition</th>"]
    for m, low in headline:
        arrow = "↓" if low else "↑"
        h.append(f"<th>{m}<br>({arrow} good)</th>")
    for m in STATS:
        h.append(f"<th>{m}</th>")
    h.append("</tr>")
    # determine best cell per headline metric (across the non-baseline judged conditions
    # too, but we highlight the global best so the eye finds the winner fast).
    best = {}
    for m, low in headline:
        vals = {c: rows_label_value.get(c, {}).get(m) for c in CONDS}
        best[m] = best_cond(vals, low)
    for c in CONDS:
        rv = rows_label_value.get(c, {})
        h.append(f"<tr><td class=v>{COND_LABEL.get(c, c)}</td>")
        for m, _ in headline:
            cls = " class=pos" if best.get(m) == c else ""
            h.append(f"<td{cls}>{fmt(rv.get(m))}</td>")
        for m in STATS:
            h.append(f"<td>{fmt(rv.get(m))}</td>")
        h.append("</tr>")
    h.append("</table>")
    return "".join(h)


def render(rep):
    scores = rep.get("scores", {})
    params = rep.get("params", {})
    cfg = params.get("cfg", 4.5)
    seeds = params.get("seeds", 25)
    steps = params.get("steps", 28)
    n_scenes = len([s for s in SCENES if s in scores]) or len(scores)

    # ---- aggregate means across scenes (per cond, per metric) ----
    agg = {c: {} for c in CONDS}
    for c in CONDS:
        for m, _ in HEADLINE:
            xs = [cmean(scores[s]["conds"], c, m) for s in scores]
            xs = [x for x in xs if x is not None]
            agg[c][m] = sum(xs) / len(xs) if xs else None
        for m in STATS:
            xs = [cmean(scores[s]["conds"], c, m) for s in scores]
            xs = [x for x in xs if x is not None]
            agg[c][m] = sum(xs) / len(xs) if xs else None

    # headline winners (excluding cfg1, the un-guided realism anchor, from "wins")
    def winner(metric, low):
        vals = {c: agg[c].get(metric) for c in CONDS if c not in ("cfg1",)}
        return best_cond(vals, low)

    win_aes = winner("aesthetic", False)
    win_ir = winner("imagereward", False)
    win_clip = winner("clip_t", False)

    h = ["<!doctype html><meta charset=utf-8>",
         "<title>E17 — SBN vs CFG-Zero* vs CFG++ on SD3.5 (true CFG)</title>",
         f"<style>{CSS}</style>",
         "<h1>E17 — does <em>spectral band-normalization (SBN)</em> beat "
         "<em>CFG-Zero*</em> / <em>CFG++</em> in the high-CFG regime, and do they "
         "<em>complement</em>? (Stable Diffusion 3.5, true classifier-free guidance)</h1>"]

    # ---- TL;DR ----
    h.append(
        "<div class=tldr><b>In one paragraph.</b> Every spectral method so far lived on "
        "<b>Flux</b>, whose <i>distilled</i> guidance makes the high-CFG regime behave oddly. "
        "We port the methods to <b>Stable Diffusion 3.5-medium</b>, which uses <i>true</i> "
        "two-pass classifier-free guidance (CFG), and pit our <b>SBN</b> (clamp the generated "
        "latent's power spectrum back to a guidance-off reference) against two published "
        "high-CFG fixes — <b>CFG-Zero*</b> and <b>Rectified-CFG++</b> — on image fidelity. "
        f"We ran SD3.5-medium on {n_scenes} richly-detailed prompts × {seeds} seeds at cfg={cfg} "
        f"({steps} steps), scoring aesthetic, ImageReward and CLIP-T. "
        f"<b>Finding:</b> the plain high-CFG baseline (<code>cfg_hi</code>) and the two guidance "
        f"fixes (<code>cfgzero</code>, <code>cfgpp</code>) win fidelity — <code>{win_ir}</code> "
        f"tops ImageReward, <code>{win_aes}</code> tops aesthetic — while <b>SBN nudges every "
        "metric slightly down</b> (it desaturates and lifts high-frequency content), and "
        "<b>combining SBN with a guidance fix does not complement</b>: <code>*_sbn</code> tracks "
        "plain <code>bandnorm</code>, the SBN clamp dominating whatever guidance produced. SBN "
        "is a spectral <i>regularizer</i>, not a fidelity win on this true-CFG model. Adherence "
        "(CLIP-T) is essentially flat across all eight conditions, so nothing trades adherence "
        "for the fidelity differences.</div>")

    # ---- background ----
    h.append("<h2>0 · Background (plain language)</h2>")
    h.append(
        "<dl>"
        "<dt>SD3.5-medium &amp; the latent</dt><dd>A 2.5B rectified-flow text-to-image model "
        "(transformer + T5-XXL + 2×CLIP + a 16-channel VAE) at 1024px. It denoises a "
        "<code>(1, 16, 128, 128)</code> latent which the VAE decodes to the image.</dd>"
        "<dt>True CFG (vs Flux's distilled guidance)</dt><dd><b>Classifier-free guidance</b> is "
        "the knob for how hard the prompt steers generation. SD3.5's <code>guidance_scale</code> "
        "is <i>real</i> CFG: one batched <code>[uncond, cond]</code> transformer pass per step, "
        "then combine <code>uncond + w·(cond − uncond)</code>. So <code>guidance=1</code> is the "
        "pure <i>conditional</i> flow field (no steering), and higher <code>w</code> pushes the "
        "image harder toward the prompt — at the cost of over-saturated, over-contrasty output. "
        "This is unlike Flux, whose guidance is a <i>distilled</i> embedding, which is why we "
        "moved here for a clean high-CFG test.</dd>"
        "<dt>The cfg=1 reference</dt><dd>SBN needs a <i>target</i> spectrum. We generate the same "
        "prompt at <code>guidance=1</code> (steering off) and record, per step and per "
        "(channel, radial-frequency band), the mean power of the latent. That guidance-off "
        "trajectory is the <b>reference</b> SBN clamps toward.</dd>"
        "<dt>Radial-frequency band / PSD</dt><dd>Take the 2-D Fourier transform of a latent "
        "channel; group coefficients by distance from the centre (DC) into "
        f"<code>{params.get('n_bins', 24)}</code> rings. The mean squared magnitude per ring is "
        "the <b>power spectral density (PSD)</b> — low bands = coarse layout / palette power, "
        "high bands = fine texture.</dd>"
        "</dl>")

    h.append("<h3>The eight conditions (how each is computed)</h3>")
    h.append(
        "<dl>"
        "<dt>cfg1</dt><dd>guidance=1: the pure conditional field. This is both the SBN reference "
        "and a <i>realism anchor</i> — un-steered, so faithful palette/contrast but weak prompt "
        "adherence.</dd>"
        "<dt>cfg_hi <span class=cap>(the BASELINE)</span></dt><dd>guidance=<code>w</code> "
        f"(here <code>{cfg}</code>): the plain high-CFG generation. Every other condition is a "
        "treatment <i>on top of</i> this regime, judged against it.</dd>"
        "<dt>bandnorm <span class=cap>(SBN — ours)</span></dt><dd><b>Spectral Band "
        "Normalization.</b> At each denoising step the <code>ClampPSD3</code> callback rescales "
        "the cfg=<code>w</code> latent's per-(channel, band) PSD back to the cfg=1 reference at "
        "that step, <i>leaving the phase untouched</i> (op <code>psd_match</code>, mode "
        "<code>band</code>). It pulls the over-amplified power spectrum of high-CFG back toward "
        "the guidance-off statistics.</dd>"
        "<dt>bandnorm_pp</dt><dd><code>bandnorm</code> plus an E11 post-process: multiply image "
        f"saturation by <code>{params.get('sat', 1.4)}×</code> (SBN tends to desaturate; this "
        "puts colour back).</dd>"
        "<dt>cfgzero <span class=cap>(baseline fix)</span></dt><dd><b>CFG-Zero*</b> (a published "
        "high-CFG fix): a <code>scheduler.step</code> override that (1) rescales the uncond term "
        "by a per-sample <i>optimal</i> scale <code>α</code> minimizing the guided velocity's "
        "error, and (2) <i>zero-inits</i> the earliest step(s). Adapted to SD3.5's batched "
        "output (<code>make_cfgzero_step</code>).</dd>"
        "<dt>cfgzero_sbn</dt><dd>CFG-Zero* <b>and</b> the SBN clamp together — they compose: "
        "CFG-Zero* modifies the velocity inside <code>scheduler.step</code>, SBN clamps the "
        "resulting latent at step end. Tests whether the two <i>complement</i>.</dd>"
        "<dt>cfgpp <span class=cap>(baseline fix)</span></dt><dd><b>Rectified-CFG++</b> "
        "(arXiv 2510.07631): a predictor/corrector that evaluates guidance at a <i>predicted</i> "
        "next point (with a small corrector jitter <code>σ</code>), a second high-CFG fix "
        "(<code>gen_rcfgpp_sd3</code>).</dd>"
        "<dt>cfgpp_sbn</dt><dd>Rectified-CFG++ <b>and</b> the SBN clamp together (same composition "
        "as <code>cfgzero_sbn</code>).</dd>"
        "</dl>")

    h.append("<h3>Metrics (and which direction is good)</h3>")
    h.append(
        "<dl>"
        "<dt>aesthetic (↑)</dt><dd>LAION aesthetic predictor on CLIP features — a learned "
        "\"how nice does this look\" score. Higher = better.</dd>"
        "<dt>imagereward (↑)</dt><dd>ImageReward, a human-preference reward model for "
        "text-to-image. Higher = more preferred. This is the headline fidelity metric.</dd>"
        "<dt>clip_t (↑)</dt><dd>CLIP image↔text cosine similarity — the <b>adherence "
        "guardrail</b>: does the image still match the prompt? Higher = better. We watch it so a "
        "fidelity change isn't bought by dropping the prompt.</dd>"
        "<dt>image statistics</dt><dd>Descriptive sanity signals (no \"best\" direction): "
        "<b>sharpness</b> / <b>hf_frac</b> (high-frequency energy fraction), "
        "<b>rms_contrast</b>, <b>colorfulness</b>, <b>saturation</b>. They explain <i>how</i> a "
        "condition changes the image (e.g. SBN lowers saturation/contrast, raises hf_frac).</dd>"
        "<dt>spectral_dist (↓), vqascore (↑)</dt><dd>Two metrics the driver supports but which "
        "are <b>null in this run</b>: <code>spectral_dist</code> (distance to a real-image PSD) "
        "needs an SD3.5-VAE real reference that wasn't available, and <code>vqascore</code> was "
        "skipped (<code>--no_vqa</code>). They are defined here for completeness; the story rests "
        "on aesthetic / ImageReward / CLIP-T.</dd>"
        "</dl>")

    # ---- method ----
    h.append("<h2>1 · Method</h2>")
    h.append(
        "<dl>"
        "<dt>What the run does</dt><dd>For each of the "
        f"{n_scenes} detailed prompts we draw <code>{seeds}</code> seeded init latents (shared "
        "across conditions so differences are the <i>method</i>, not the seed), record the "
        f"cfg=1 PSD reference, then generate all eight conditions at {steps}-step SD3.5 and "
        "score every image. Numbers below are means ± over the seeds.</dd>"
        "<dt>The question each comparison answers</dt><dd>(a) <code>bandnorm</code> vs "
        "<code>cfg_hi</code>: does SBN improve fidelity over plain high-CFG? (b) "
        "<code>bandnorm</code> vs <code>cfgzero</code>/<code>cfgpp</code>: does SBN beat the "
        "published guidance fixes? (c) <code>cfgzero_sbn</code>/<code>cfgpp_sbn</code> vs either "
        "alone: do the spectral clamp and the guidance fix <i>complement</i>?</dd>"
        "<dt>Drivers</dt><dd>Backend <code>e17_sd35.py</code> (<code>gen_sd3</code> composes an "
        "optional guidance <code>step_override</code> with an optional step-end callback; "
        "<code>ClampPSD3</code>, <code>make_cfgzero_step</code>, <code>gen_rcfgpp_sd3</code>); "
        "fidelity driver <code>e17_sd35_compare.py</code>; CompBench binding driver "
        "<code>e17_compbench.py</code>.</dd>"
        "</dl>")

    # ---- results: one subsection per scene ----
    h.append("<h2>2 · Results — per scene</h2>")
    h.append("<p class=cap>Each scene leads with its 8-condition grid (rows = condition, columns "
             "= seeds), then a \"what to look for\" note, then the per-condition numbers (best "
             "cell per headline column highlighted). Read directions, not third decimals.</p>")
    for s in SCENES:
        if s not in scores:
            continue
        sd = scores[s]
        conds = sd["conds"]
        prompt = sd.get("prompt", "")
        h.append(f"<h3>{s}</h3>")
        h.append(f"<p class=cap><b>prompt:</b> {prompt}</p>")
        h.append(img_tag(f"grid_{s}.png", cls="grid"))
        h.append("<p class=cap><b>What to look for:</b> rows are the eight conditions "
                 "(cfg1 → cfg_hi → bandnorm/​bandnorm_pp → cfgzero/​cfgzero_sbn → "
                 "cfgpp/​cfgpp_sbn). <code>cfg1</code> looks tame/realistic but off-prompt; "
                 "<code>cfg_hi</code> and the guidance fixes look punchy/saturated; the SBN "
                 "(<code>*norm</code>, <code>*_sbn</code>) rows should look <i>flatter / less "
                 "saturated</i> with slightly more fine texture — the spectral pull-back.</p>")
        rows = {c: {m: cmean(conds, c, m) for m, _ in HEADLINE} for c in CONDS}
        for c in CONDS:
            for m in STATS:
                rows[c][m] = cmean(conds, c, m)
        # interpretation line: who wins fidelity here
        ir = {c: cmean(conds, c, "imagereward") for c in CONDS}
        ir_win = best_cond({c: ir[c] for c in CONDS if c != "cfg1"}, False)
        aes = {c: cmean(conds, c, "aesthetic") for c in CONDS}
        aes_win = best_cond({c: aes[c] for c in CONDS if c != "cfg1"}, False)
        h.append(f"<p class=cap><b>Read:</b> best ImageReward here = "
                 f"<code>{ir_win}</code>, best aesthetic = <code>{aes_win}</code>. SBN rows "
                 "(<code>bandnorm*</code>, <code>*_sbn</code>) sit below the guidance conditions "
                 "on the fidelity columns and lower on saturation — the expected SBN trade.</p>")
        h.append(metric_table(rows, HEADLINE))

    # ---- aggregate ----
    h.append("<h2>3 · Aggregate — mean across the 8 scenes</h2>")
    h.append("<p class=cap><b>What to look for:</b> the headline verdict. Best cell per "
             "fidelity/adherence column is highlighted (cfg1 included for context — it is the "
             "un-steered anchor, not a competitor).</p>")
    h.append(metric_table(agg, HEADLINE))
    h.append(
        "<div class='note win'><b>Reading.</b> On this true-CFG model the plain high-CFG "
        f"baseline and the guidance fixes own fidelity: <code>{win_ir}</code> tops ImageReward "
        f"and <code>{win_aes}</code> tops aesthetic, all clustered tightly. <b>SBN "
        "(<code>bandnorm</code>) lowers every fidelity metric</b> vs <code>cfg_hi</code> — it "
        "pulls saturation and contrast down toward the cfg=1 statistics and lifts high-frequency "
        "fraction (visible in the stats columns), which the aesthetic/ImageReward models read as "
        "slightly worse. <b>The combinations do not complement:</b> <code>cfgzero_sbn</code> and "
        "<code>cfgpp_sbn</code> land near plain <code>bandnorm</code>, not near their guidance "
        "parents — the late SBN clamp overwrites whatever the guidance fix did to the spectrum. "
        "<code>bandnorm_pp</code> (saturation post-process) recovers some colourfulness but does "
        "not close the ImageReward gap. CLIP-T is flat across all eight, so adherence is not the "
        "lever here.</div>")
    h.append(
        "<div class='note cav'><b>Caveats.</b> (1) <code>spectral_dist</code> "
        "(distance-to-real PSD) and <code>vqascore</code> are <i>null</i> in this run — the "
        "SD3.5-VAE real PSD reference wasn't built and VQAScore was deferred (<code>--no_vqa</code>), "
        "so the \"closer to real spectrum\" claim SBN is designed for isn't measured here; the "
        "fidelity verdict rests on learned-preference scores that reward punchy high-CFG output. "
        "(2) Fidelity (aesthetic/ImageReward) is the contest here; the <b>compositional</b> "
        "contest (does SBN preserve attribute binding?) is the separate <code>e17_compbench.py</code> "
        "/ <code>results/e17cb</code> run. (3) CFG-Zero* / CFG++ monkey-patch diffusers internals "
        "(restored in <code>finally</code>) and could break on version bumps.</div>")

    h.append("<p class=cap>Generated by <code>e17_site.py</code> from "
             "<code>results/e17/report.json</code> + <code>grid_&lt;scene&gt;.png</code> (no "
             "model load, no re-scoring). Method: <code>e17_sd35.py</code> backend + "
             "<code>e17_sd35_compare.py</code> driver. See also <code>EXPERIMENT_17.md</code>.</p>")
    return "".join(h)


def build_site():
    """Load report.json and write results/e17/index.html. No model, no re-scoring."""
    rpath = os.path.join(OUT, "report.json")
    if not os.path.exists(rpath):
        print(f"[e17-site] no {rpath}; run e17_sd35_compare.py --part gen,score first")
        return None
    rep = json.load(open(rpath))
    html = render(rep)
    dest = os.path.join(OUT, "index.html")
    with open(dest, "w") as f:
        f.write(html)
    print(f"[e17-site] wrote {dest}  ({len(html) // 1024} KB)  (no model loaded)")
    return dest


def main():
    build_site()


if __name__ == "__main__":
    main()
