"""Build a self-contained HTML explainer for E23 (real-image spectral gap + real-SBN).

Reads results/e23/{scores.json, report.json, examples.json} and references the
plot/example PNGs by RELATIVE path, so the page is light -- open
results/e23/index.html in place (images live under results/e23/plots and
results/e23/examples). Honors CN_RESULTS.

The page is written to STAND ALONE: it defines every term (real-SBN, offline,
psd_match, the metrics) and explains the pipeline end-to-end, so a reader needs
no prior project context. (See memory: experiment-documentation-standard.)

    python e23_site.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

OUT = os.path.join(RESULTS, "e23")

LABELS = {
    "cfg1.0": "cfg 1.0 — weak-guidance baseline",
    "sbn_cfg1": "SBN→cfg1 — the OLD method (clamp toward cfg=1)",
    "sbn_real_last": "real-SBN, during generation (last step)",
    "sbn_real_init": "real-SBN, initial-noise shaping (exploratory)",
}
METRICS = [("spectral_dist", "spec-dist→real ↓"), ("aesthetic", "aesthetic ↑"),
           ("imagereward", "ImageReward ↑"), ("clip_t", "CLIP-T ↑")]
LOWER_BETTER = {"spectral_dist"}

CSS = """
body{font:15px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px auto;max-width:980px;color:#1a1a1a;padding:0 16px}
h1{font-size:25px;line-height:1.25} h2{font-size:19px;margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:5px}
h3{font-size:15px;margin-bottom:4px}
.tldr{background:#eef4ff;border:1px solid #c7d9ff;border-radius:6px;padding:14px 16px;margin:14px 0}
.note{background:#f6f8fa;border-left:4px solid #0969da;padding:10px 14px;border-radius:4px;margin:12px 0}
.cav{background:#fff8f0;border-left:4px solid #d4a017}
.win{background:#eafaf0;border-left:4px solid #2da44e}
dl{margin:10px 0} dt{font-weight:700;margin-top:10px} dd{margin:2px 0 2px 18px;color:#333}
table{border-collapse:collapse;margin:12px 0;font-variant-numeric:tabular-nums;width:100%}
th,td{border:1px solid #d0d7de;padding:6px 10px;text-align:center}
th{background:#f6f8fa} td.v{text-align:left;font-weight:600;white-space:nowrap}
td.best{font-weight:700;background:#dafbe1} tr.ours td.v{background:#fff8c5}
.cap{color:#555;font-size:13px}
img.plot{max-width:460px;width:100%;border:1px solid #d0d7de;border-radius:4px;margin:8px 8px 0 0;vertical-align:top}
img.ex{width:100%;max-width:900px;border:1px solid #d0d7de;border-radius:4px;margin:6px 0}
code{background:#eff1f3;padding:1px 5px;border-radius:3px;font-size:13px}
.row{display:flex;flex-wrap:wrap;gap:6px;align-items:flex-start}
ol.steps>li{margin:6px 0}
"""


# Self-contained pipeline schematic (inline SVG, no external asset).
SCHEMATIC = """
<svg viewBox="0 0 960 280" width="100%" style="max-width:960px;border:1px solid #d0d7de;border-radius:6px;background:#fff;margin:10px 0" font-family="-apple-system,Segoe UI,Roboto,sans-serif" font-size="13">
<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
<path d="M0,0 L7,3 L0,6 Z" fill="#555"/></marker></defs>
<text x="480" y="16" text-anchor="middle" font-weight="700" font-size="13">E23 — build a real-photo spectral TARGET, then re-level a cfg-3.5 generation toward it</text>
<!-- real-target branch -->
<rect x="150" y="28" width="185" height="44" rx="6" fill="#f6f8fa" stroke="#999"/>
<text x="242" y="55" text-anchor="middle">500 real photos (COCO)</text>
<rect x="360" y="28" width="110" height="44" rx="6" fill="#f6f8fa" stroke="#999"/>
<text x="415" y="55" text-anchor="middle">VAE encode</text>
<rect x="500" y="28" width="210" height="44" rx="6" fill="#eafaf0" stroke="#2da44e"/>
<text x="605" y="48" text-anchor="middle">real per-band power</text>
<text x="605" y="65" text-anchor="middle" font-size="11" fill="#555">(TARGET, per channel)</text>
<line x1="335" y1="50" x2="356" y2="50" stroke="#555" marker-end="url(#ah)"/>
<line x1="470" y1="50" x2="496" y2="50" stroke="#555" marker-end="url(#ah)"/>
<line x1="605" y1="72" x2="605" y2="147" stroke="#555" marker-end="url(#ah)"/>
<!-- generation branch -->
<rect x="20" y="158" width="90" height="44" rx="6" fill="#f6f8fa" stroke="#999"/>
<text x="65" y="185" text-anchor="middle">prompt</text>
<rect x="150" y="158" width="120" height="44" rx="6" fill="#f6f8fa" stroke="#999"/>
<text x="210" y="185" text-anchor="middle">Flux (cfg 3.5)</text>
<rect x="310" y="158" width="130" height="44" rx="6" fill="#f6f8fa" stroke="#999"/>
<text x="375" y="185" text-anchor="middle">generated latent</text>
<rect x="500" y="150" width="210" height="64" rx="6" fill="#fff8c5" stroke="#d4a017"/>
<text x="605" y="176" text-anchor="middle" font-weight="700">real-SBN = psd_match(s)</text>
<text x="605" y="197" text-anchor="middle" font-size="11" fill="#555">mag ← √(target/cur) , phase kept</text>
<rect x="750" y="158" width="95" height="44" rx="6" fill="#f6f8fa" stroke="#999"/>
<text x="797" y="185" text-anchor="middle">VAE decode</text>
<rect x="875" y="158" width="80" height="44" rx="6" fill="#f6f8fa" stroke="#999"/>
<text x="915" y="185" text-anchor="middle">output</text>
<line x1="110" y1="180" x2="148" y2="180" stroke="#555" marker-end="url(#ah)"/>
<line x1="270" y1="180" x2="308" y2="180" stroke="#555" marker-end="url(#ah)"/>
<line x1="440" y1="180" x2="497" y2="181" stroke="#555" marker-end="url(#ah)"/>
<line x1="710" y1="181" x2="748" y2="180" stroke="#555" marker-end="url(#ah)"/>
<line x1="845" y1="180" x2="873" y2="180" stroke="#555" marker-end="url(#ah)"/>
<!-- captions -->
<text x="605" y="236" text-anchor="middle" font-size="11" fill="#555">applied: offline (final latent) · last-step (in loop) · init-noise ✗</text>
<text x="480" y="262" text-anchor="middle" font-size="12" fill="#333">phase = layout (kept)  ·  magnitude = texture (re-leveled toward real); strength s dials how far (≈0.25 best)</text>
</svg>
"""


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def cond_order(scores, strengths):
    base = f"cfg{scores.get('_cfg', '3.5')}"
    order = ["cfg1.0", base, "sbn_cfg1"]
    order += [f"sbn_real_off{s}" for s in strengths]
    order += ["sbn_real_last", "sbn_real_init"]
    present = {c for k in scores if not k.startswith("_") for c in scores[k]}
    return [c for c in order if c in present], base


def cond_means(scores, conds):
    prompts = [k for k in scores if not k.startswith("_")]
    out = {}
    for c in conds:
        out[c] = {}
        for mk, _ in METRICS:
            vs = [scores[k].get(c, {}).get(mk, {}).get("mean")
                  for k in prompts if scores[k].get(c, {}).get(mk)]
            vs = [v for v in vs if v is not None]
            out[c][mk] = sum(vs) / len(vs) if vs else None
    return out


def label_for(c, base):
    if c == base:
        return f"{c} — standard-guidance baseline"
    if c.startswith("sbn_real_off"):
        return f"real-SBN, offline, strength={c.replace('sbn_real_off', '')} (ours)"
    return LABELS.get(c, c)


def render(scores, strengths, examples, adherence):
    conds, base = cond_order(scores, strengths)
    means = cond_means(scores, conds)
    nprompts = len([k for k in scores if not k.startswith("_")])
    pool = [c for c in conds if c != "sbn_real_init"]
    best = {}
    for mk, _ in METRICS:
        vals = [(c, means[c][mk]) for c in pool if means[c][mk] is not None]
        if vals:
            best[mk] = (min if mk in LOWER_BETTER else max)(
                vals, key=lambda t: t[1])[0]

    h = ["<!doctype html><meta charset=utf-8><title>E23 — real-image spectral gap</title>",
         f"<style>{CSS}</style>",
         "<h1>E23 — making generated images match the <em>frequency content</em> of "
         "real photos (“real-SBN”)</h1>"]

    # ---- TL;DR ----
    h.append(
        "<div class=tldr><b>In one paragraph.</b> A diffusion model's images aren't quite "
        "like real photos — in particular their <b>frequency spectrum</b> differs (they're a "
        "bit under-textured). We <b>measured</b> that generated-vs-real difference across the "
        "frequency bands, then <b>corrected</b> each generated image's spectrum toward the "
        "real-photo spectrum (a step we call <b>real-SBN</b>). Doing so makes images look "
        "better (higher aesthetic score) with no real loss of prompt-faithfulness, and it "
        "works better than the previous trick of matching a weak-guidance reference.</div>")

    # ---- Schematic ----
    h.append("<h2>Pipeline at a glance</h2>")
    h.append(SCHEMATIC)

    # ---- Section 0: background / vocabulary ----
    h.append("<h2>0 · Background you need (plain language)</h2>")
    h.append(
        "<p>We use <b>Flux</b>, a text-to-image diffusion model. It doesn't paint pixels "
        "directly — it works in a compressed <b>latent</b> (a 16-channel, 128×128 array that a "
        "VAE turns into the final 1024×1024 image). Everything below is measured on these "
        "latents.</p>")
    h.append("<dl>"
             "<dt>Fourier / frequency spectrum</dt><dd>Any image (or latent) can be written as a "
             "sum of waves of different spatial frequencies. <b>Low</b> frequencies = coarse "
             "structure / big shapes; <b>high</b> frequencies = fine detail / texture.</dd>"
             "<dt>Radial band</dt><dd>We group those frequencies into ~24 rings from low to high "
             "(a “band” = all waves at roughly one scale). “Per-(channel, band) power” = how much "
             "energy a latent has at that scale, in that channel.</dd>"
             "<dt>PSD (power spectral density)</dt><dd>The curve of power vs frequency band. The "
             "“fingerprint” of how much coarse vs fine content an image has.</dd>"
             "<dt>CFG / cfg scale</dt><dd>Classifier-free guidance — the knob that controls how "
             "strongly the image obeys the prompt. <code>cfg≈3.5</code> is the normal setting; "
             "<code>cfg=1</code> is weak guidance (soft, low-contrast images). A known side "
             "effect: higher cfg <b>inflates</b> low-frequency power above the natural level "
             "(prior experiment E10).</dd>"
             "<dt>phase vs magnitude</dt><dd>Each frequency has a <b>magnitude</b> (how strong) "
             "and a <b>phase</b> (where it sits). In these latents, <b>phase carries the layout / "
             "content</b>; per-band <b>magnitude carries texture + palette</b>. This split is why "
             "we can change texture without moving the content.</dd>"
             "</dl>")

    # ---- Section 1: what SBN is, and the proxy problem ----
    h.append("<h2>1 · What “SBN” is, and why we changed its target</h2>")
    h.append(
        "<p><b>SBN = Spectral Band Normalization.</b> It re-levels a generated latent's "
        "per-(channel, band) power so the image's spectrum <b>matches a reference</b> spectrum. "
        "The operator is <code>psd_match</code>: for each (channel, band), multiply the FFT "
        "magnitude by <code>sqrt(reference / current)</code> and leave the phase untouched. "
        "Because only magnitude changes, the layout/content is preserved and only texture/palette "
        "shifts. (This is “AdaIN, but in Fourier space.”)</p>")
    h.append(
        "<div class=note><b>The problem this experiment fixes.</b> The old SBN used the "
        "<b>cfg=1 spectrum as its reference</b> — a stand-in for “calmer / less inflated.” But "
        "cfg=1 isn't <i>real</i>; it's just a softer generation. So the model was being matched "
        "to another model output, not to reality. <b>real-SBN</b> swaps the reference: it uses "
        "the spectrum of <b>500 real photographs</b> (MS-COCO images, VAE-encoded into Flux's "
        "latent space) as the target.</p>")

    # ---- Section 2: the three ways we apply real-SBN ----
    h.append("<h2>2 · Three ways to apply real-SBN</h2>")
    h.append("<dl>"
             "<dt>offline (the main one)</dt><dd>Generate the image normally, then apply "
             "<code>psd_match</code> to the <b>final</b> latent and decode. Post-hoc, one shot, "
             "cheap (no extra diffusion).</dd>"
             "<dt>strength (the dial)</dt><dd>An exponent on the per-band correction: each band's "
             "magnitude is multiplied by <code>(real/own)^(strength/2)</code>. 0 = no change, "
             "1 = full match to the real spectrum. <b>Recommended ≈ 0.25</b>: the high bands have "
             "little real signal, so full strength over-amplifies them into visible grain/artifacts; "
             "a gentle nudge restores texture cleanly. (Note: the LAION aesthetic predictor "
             "<i>rewards</i> that over-sharpening, so its score keeps rising past the point a human "
             "would — trust the eye, use ~0.25.)</dd>"
             "<dt>during generation, last step</dt><dd>Apply the same correction <b>inside</b> the "
             "denoising loop, but only on the final step. Stays “on-manifold” (the model still "
             "produced it) and needs no separate decode. In practice it matches the offline result.</dd>"
             "<dt>initial-noise shaping (exploratory)</dt><dd>Shape the <b>starting</b> noise toward "
             "the real spectrum, then denoise normally. This <b>fails</b> — see results.</dd>"
             "</dl>"
             "<p class=cap>Why not correct at every denoising step? The real target is a "
             "<i>clean</i>-image spectrum, but mid-denoising latents are still mostly noise, so "
             "matching them every step compares apples to oranges. The correct place to apply a "
             "clean-image target is the near-finished latent — hence offline / last-step.</p>")

    # ---- Section 3: the measured gap ----
    h.append("<h2>3 · The measured gap (generated vs 500 real photos)</h2>")
    h.append("<p class=cap>Left: channel-mean power per band, real vs generated, with p10–p90 "
             "spread. Middle: the correction ratio <code>real / generated</code> per band "
             "(grey = individual channels, red = mean; the dashed line at 1.0 = no change "
             "needed). Right: the same ratio as a (channel × band) heatmap.</p>")
    h.append("<div class=row>"
             "<img class=plot src='plots/psd_real_vs_gen.png'>"
             "<img class=plot src='plots/correction_curve.png'>"
             "<img class=plot src='plots/correction_heatmap.png'></div>")
    h.append(
        "<div class=note><b>The gap is bimodal — and it's not just the CFG story.</b> "
        "At the <b>lowest bands</b> (ratio &lt; 1) the model has <i>more</i> power than real "
        "photos — the CFG low-frequency inflation. But across the <b>mid/high bands</b> "
        "(ratio ≈ 1.2–1.3) the model has <i>less</i> power than real — a broad "
        "<b>high-frequency deficit</b>: generated images are systematically under-textured "
        "vs real photographs. So the honest correction does two things at once: trim the "
        "lowest bands a little and lift the mid/high bands ~1.25×. The heatmap shows this is "
        "channel-specific, which is why the target is stored per channel rather than averaged.</div>")

    # ---- Section 4: results ----
    h.append("<h2>4 · Does it help? (numbers)</h2>")
    h.append("<p>Metrics, each averaged over " + str(nprompts) + " prompt classes:</p>")
    h.append("<dl>"
             "<dt>spec-dist→real ↓</dt><dd>How far the image's spectrum is from the real-photo "
             "spectrum (RMS in log-power). Lower = closer to real. <i>This is the thing real-SBN "
             "directly minimizes, so treat its value cautiously — see caveats.</i></dd>"
             "<dt>aesthetic ↑</dt><dd>LAION aesthetic predictor — a model trained to rate visual "
             "quality (~1–10).</dd>"
             "<dt>ImageReward ↑</dt><dd>A model trained on human preferences (quality + prompt "
             "match).</dd>"
             "<dt>CLIP-T ↑</dt><dd>How well the image matches the prompt text (our "
             "<b>adherence guardrail</b> — we don't want the fix to break prompt-faithfulness).</dd>"
             "</dl>")
    h.append("<table><tr><th>condition (what it is)</th>"
             + "".join(f"<th>{lab}</th>" for _, lab in METRICS) + "</tr>")
    for c in conds:
        ours = " class='ours'" if c.startswith("sbn_real_off") else ""
        h.append(f"<tr{ours}><td class=v>{label_for(c, base)}</td>")
        for mk, _ in METRICS:
            cls = "best" if best.get(mk) == c else ""
            h.append(f"<td class='{cls}'>{fmt(means[c][mk])}</td>")
        h.append("</tr>")
    h.append("</table>")
    h.append("<p class=cap>Green = best (excluding the broken init-noise row); yellow rows are "
             "real-SBN (ours).</p>")
    h.append(
        "<div class='note win'><b>Verdict.</b> Correcting toward the real spectrum (offline at "
        "full strength, or the during-gen last-step clamp) gives the <b>biggest aesthetic gain "
        "of any condition</b> at essentially <b>zero adherence cost</b> (CLIP-T within ~0.7% of "
        "baseline). It also <b>beats the old cfg-1 SBN</b>, which actually moves the spectrum "
        "<i>away</i> from real (cfg=1 is softer, not realer). The strength knob is a dial: lower "
        "strengths keep ImageReward higher while still cutting the gap.</div>")
    h.append("<div class=row>"
             "<img class=plot src='plots/spectral_dist.png'>"
             "<img class=plot src='plots/aesthetic_vs_spectral.png'>"
             "<img class=plot src='plots/clip_t.png'></div>")

    # ---- Section 5: qualitative examples ----
    h.append("<h2>5 · Where the gain is biggest (look for yourself)</h2>")
    if examples:
        h.append("<p class=cap>Each panel is one prompt+seed (so the content is identical). "
                 "<b>Left = cfg baseline, right = real-SBN.</b> These are the pairs with the "
                 "largest fidelity gain; captions show the aesthetic / ImageReward scores and the "
                 "strength used.</p>")
        for tag, human in (("aesthetic", "aesthetic score"),
                           ("imagereward", "ImageReward (human-preference)")):
            items = [m for m in examples if m["rank_by"] == tag]
            if not items:
                continue
            h.append(f"<h3>Biggest gains by {human}</h3>")
            for m in items:
                cap = (f"<b>{m['key']}</b> · seed {m['seed']} · strength {m['strength']} · "
                       f"Δaesthetic {m['d_aesthetic']:+.2f}, ΔImageReward {m['d_imagereward']:+.2f}")
                h.append(f"<div class=cap>{cap}</div>")
                h.append(f"<img class=ex src='{m['panel']}'>")
    else:
        h.append("<p class=cap>(no <code>examples.json</code> yet — run "
                 "<code>e23_real_sbn.py --part examples</code>, then rebuild this page)</p>")

    # ---- Section 5b: cfg1 vs cfg adherence on complex prompts ----
    if adherence:
        ad, conds = adherence["scores"], adherence["conds"]
        rec = adherence["rec_strength"]
        h.append("<h2>5b · Why not just use cfg = 1? (prompt adherence)</h2>")
        h.append(
            "<p>cfg = 1 often looks the most <i>natural</i>, but weak guidance <b>drops prompt "
            "elements</b> — worst on long, compositional prompts. cfg = 3.5 keeps them but "
            "over-bakes; <b>real-SBN keeps cfg = 3.5's adherence while looking more natural</b>. "
            "Below, adherence is scored by "
            + ("<b>B-VQA</b> (BLIP-VQA attribute binding — the T2I-CompBench metric: "
               "P(yes) per noun phrase, multiplied; catches compositional drops CLIP-T misses)"
               if any((ad[k].get(c) or {}).get("bvqa") for k in ad for c in conds)
               else "<b>CLIP-T</b> (B-VQA was off)")
            + ", higher = more faithful.</p>")
        has_vqa = any((ad[k].get(c) or {}).get("bvqa") for k in ad for c in conds)
        mk = "bvqa" if has_vqa else "clip_t"
        clabel = {"cfg1.0": "cfg 1.0", "_rec": f"real-SBN s={rec}"}
        h.append("<table><tr><th>prompt</th>"
                 + "".join(f"<th>{clabel.get(c, c)}</th>" for c in conds) + "</tr>")
        for k in ad:
            h.append(f"<tr><td class=v>{k}</td>")
            row = [(c, (ad[k].get(c) or {}).get(mk, {})) for c in conds]
            vals = [(c, (e or {}).get("mean")) for c, e in row]
            bestc = max((cv for cv in vals if cv[1] is not None),
                        key=lambda t: t[1], default=(None, None))[0]
            for c, v in vals:
                cls = "best" if c == bestc else ""
                h.append(f"<td class='{cls}'>{fmt(v)}</td>")
            h.append("</tr>")
        h.append("</table>")
        h.append("<p class=cap>Panels: <b>left cfg 1.0 · middle cfg 3.5 · right real-SBN "
                 f"(s={rec})</b>. Look for objects/attributes/counts the cfg 1.0 image misses "
                 "that the others keep.</p>")
        for m in adherence["panels"]:
            h.append(f"<div class=cap>{m['key']} · seed {m['seed']}</div>")
            h.append(f"<img class=ex src='{m['panel']}'>")

    # ---- Section 6: caveats ----
    h.append("<h2>6 · Caveats &amp; what's next</h2>")
    h.append(
        "<div class='note cav'>"
        "<b>(1) The gap metric is partly circular.</b> <code>spec-dist→real</code> measures the "
        "very spectrum that full-strength real-SBN forces to match, so its drop to ~0 is by "
        "construction. The <b>independent</b> wins are the aesthetic gain and the near-zero "
        "adherence cost — not the gap number itself.<br>"
        "<b>(2) Initial-noise shaping fails.</b> Coloring the starting noise toward a clean-image "
        "spectrum collapses generation (the model expects white noise to start from); it's in the "
        "table only as a documented negative result.<br>"
        "<b>(3) Isotropic only.</b> Radial bands carry texture-energy + palette, not oriented "
        "structure (brush strokes etc.).<br>"
        "<b>Next:</b> test a single <b>fixed</b> per-channel correction curve (apply the average "
        "ratio to every image, no per-image matching) to see whether one universal gain vector "
        "generalizes — that would make real-SBN a free, deterministic post-step.</div>")

    h.append("<h2>Glossary</h2><dl>"
             "<dt>real-SBN</dt><dd>This experiment's method: re-level a generated image's "
             "per-band power toward the <b>real-photo</b> spectrum (vs the old SBN that targeted "
             "cfg=1).</dd>"
             "<dt>SBN</dt><dd>Spectral Band Normalization — match per-(channel, band) power to a "
             "reference via <code>psd_match</code>.</dd>"
             "<dt>psd_match / AdaIN-in-Fourier</dt><dd>Per (channel, band): scale FFT magnitude by "
             "<code>sqrt(target/current)</code>, keep phase. Changes texture/palette, preserves "
             "layout.</dd>"
             "<dt>offline</dt><dd>Apply the correction after generation, to the final latent, then "
             "decode (vs “during generation”).</dd>"
             "<dt>strength</dt><dd>0→1 dial: how far to push from the image's own spectrum toward "
             "the real one (1 = full match).</dd>"
             "<dt>latent</dt><dd>The 16×128×128 compressed array the model works in; a VAE decodes "
             "it to the final image.</dd>"
             "</dl>")
    h.append("<p class=cap>Generated by <code>e23_site.py</code> from "
             "<code>results/e23/{scores,examples}.json</code>. Method/operators live in "
             "<code>real_spectral.py</code>, <code>spectral_ops.py</code> (psd_match), "
             "<code>style_ops.py</code> (restyle_latent), driver <code>e23_real_sbn.py</code>.</p>")
    return "".join(h)


def main():
    spath = os.path.join(OUT, "scores.json")
    if not os.path.exists(spath):
        print(f"[e23-site] no {spath}; run e23_real_sbn.py --part score first")
        return
    scores = json.load(open(spath))
    strengths, cfg = [0.5, 1.0], 3.5
    rpath = os.path.join(OUT, "report.json")
    if os.path.exists(rpath):
        params = json.load(open(rpath)).get("params", {})
        strengths = params.get("strength_sweep", strengths)
        cfg = params.get("cfg", cfg)
    scores["_cfg"] = f"{cfg:g}" if isinstance(cfg, float) else cfg
    epath = os.path.join(OUT, "examples.json")
    examples = json.load(open(epath)) if os.path.exists(epath) else None
    apath = os.path.join(OUT, "adherence", "adherence.json")
    adherence = json.load(open(apath)) if os.path.exists(apath) else None

    html = render(scores, strengths, examples, adherence)
    dest = os.path.join(OUT, "index.html")
    with open(dest, "w") as f:
        f.write(html)
    print(f"[e23-site] wrote {dest}  ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
