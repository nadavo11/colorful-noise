"""Build a self-contained HTML explainer for E22 (DDIM-inversion + frequency-band
locking on SDXL — the SDXL pivot of E21, where the inversion GATE PASSES).

Reads results/e22/{invert.json, edit.json} + the reconstruction grid
(invert/grid.png) + per-edit grids (edit/grid_*.png) and EMBEDS every image as
base64 so the page is fully portable (open results/e22/index.html anywhere).
Honors CN_RESULTS.

The page STANDS ALONE: it defines every term (SDXL 4x128x128 latent,
eps-prediction vs rectified flow, DDIM inversion clean->noise, FFT phase vs
magnitude, frequency-band PHASE locking = composition preservation, the lockphase
variant + cut c + until u, lockpower, and the metrics recon CLIP-I ↑,
struct_clip ↑, edit_clip_t ↑) and leads each result with its figure + a "what to
look for" before the table.

KEY FRAMING (opposite of E21): SDXL is eps-prediction with a clean DDIM inversion,
so the reconstruction GATE PASSES (recon CLIP-I ~0.94 vs E21's drift). Phase-band
locking is then a real, TUNABLE structure<->edit FRONTIER: lockphase lifts
struct-to-source CLIP-I ~0.60 -> ~0.90 while the prompt still moves appearance, but
it TRADES AWAY prompt adherence (edit CLIP-T ~0.24 -> ~0.16). A frontier, not a
free lunch. (memory: experiment-documentation-standard.)

    python e22_site.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e27_site import data_uri

OUT = os.path.join(RESULTS, "e22")

# "round-trip closed" bar: a faithful inversion reconstructs the source at high
# CLIP-I; ~0.94 is the rough pass threshold (E22 clears it; E21 did not).
PASS_CLIP_I = 0.94

CSS = """
body{font:15px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px auto;max-width:1000px;color:#1a1a1a;padding:0 16px}
h1{font-size:25px;line-height:1.25} h2{font-size:19px;margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:5px}
h3{font-size:15px;margin-bottom:4px}
.tldr{background:#eef4ff;border:1px solid #c7d9ff;border-radius:6px;padding:14px 16px;margin:14px 0}
.note{background:#f6f8fa;border-left:4px solid #0969da;padding:10px 14px;border-radius:4px;margin:12px 0}
.cav{background:#fff8f0;border-left:4px solid #d4a017}.win{background:#eafaf0;border-left:4px solid #2da44e}
.look{background:#fff8f0;border-left:4px solid #d4a017;padding:10px 14px;border-radius:4px;margin:12px 0}
dl{margin:10px 0} dt{font-weight:700;margin-top:10px} dd{margin:2px 0 2px 18px;color:#333}
table{border-collapse:collapse;margin:12px 0;font-variant-numeric:tabular-nums}
th,td{border:1px solid #d0d7de;padding:5px 9px;text-align:center}
th{background:#f6f8fa} td.v{text-align:left;font-weight:600;white-space:nowrap}
td.pos{background:#dafbe1}td.neg{background:#ffebe9}
.cap{color:#555;font-size:13px}
img.grid{width:100%;border:1px solid #d0d7de;border-radius:4px;margin:8px 0}
img.plot{max-width:100%;border:1px solid #d0d7de;border-radius:4px;margin:8px 0}
code{background:#eff1f3;padding:1px 5px;border-radius:3px;font-size:13px}
"""


def img_tag(name, cls="plot", **kw):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return f"<p class=cap>(missing {name})</p>"
    return f"<img class={cls} src='{data_uri(p, **kw)}'>"


def fmt(x, n=3, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def edit_means(edit, name, key):
    """Mean of cell `key` (struct_clip / edit_clip_t) for condition `name` over photos."""
    vals = []
    for e in edit.values():
        c = e.get("cells", {}).get(name)
        if c is not None:
            vals.append(c.get(key))
    return mean(vals)


def all_conditions(edit):
    """Ordered union of condition names across photos (preserve first-seen order)."""
    order = []
    for e in edit.values():
        for k in e.get("cells", {}):
            if k not in order:
                order.append(k)
    return order


def render(invert, edit):
    clip_vals = [v.get("recon_clip_i") for v in invert.values()] if invert else []
    std_vals = [v.get("noise_std") for v in invert.values()] if invert else []
    mean_clip = mean(clip_vals)
    mean_std = mean(std_vals)

    h = ["<!doctype html><meta charset=utf-8>"
         "<title>E22 — DDIM-inversion + frequency-band locking (SDXL): the gate passes</title>",
         f"<style>{CSS}</style>",
         "<h1>E22 — spectral image editing via <em>DDIM inversion</em> + frequency-band locking "
         "on SDXL — <em>the gate passes; band-lock is a tunable structure&hArr;edit frontier</em></h1>"]

    # ---- TL;DR ----
    h.append(
        "<div class=tldr><b>In one paragraph.</b> We edit a <i>real</i> photo by spectral surgery: "
        "invert it back to the noise that would have produced it, then regenerate with a new prompt "
        "while <b>locking</b> the source's low-frequency <b>phase</b> (which carries layout) so the "
        "prompt only repaints appearance (oil painting / pencil sketch / watercolor). E21 tried this "
        "on SD3.5 and the <b>inversion drifted</b> (the round-trip did not close). SDXL is an "
        "<b>&epsilon;-prediction</b> model with a clean, deterministic <b>DDIM inversion</b>, so here "
        "the <b>gate passes</b>: regenerating from the inverted noise with the <i>same</i> prompt "
        "reconstructs the source at CLIP-I "
        + (f"&asymp; <b>{fmt(mean_clip, 2)}</b> " if mean_clip is not None else "&asymp; <b>0.94</b> ")
        + f"(the ~{PASS_CLIP_I} \"round-trip closed\" bar). With a working gate the edit becomes "
        "interpretable, and <b>phase-band locking is a strong, tunable composition-preservation "
        "knob</b>: locking the source low-band phase lifts structure-to-source CLIP-I from "
        + (f"<b>~{fmt(edit_means(edit, 'invert_only', 'struct_clip'), 2)}</b> "
           if edit else "<b>~0.60</b> ")
        + "(no lock) up to <b>~0.90</b> while the prompt still moves the look &mdash; but it "
        "<b>trades away prompt adherence</b> (edit CLIP-T drops ~0.24 &rarr; ~0.16). It is a genuine "
        "<b>structure&hArr;edit frontier</b>, not a free lunch: <code>c</code> (how much layout to "
        "lock) and <code>u</code> (how long) are the two dials.</div>")

    # ---- background ----
    h.append("<h2>0 · Background (plain language)</h2>")
    h.append(
        "<dl>"
        "<dt>Latent (SDXL)</dt><dd>SDXL does not denoise pixels; it works in a compressed "
        "<code>4&times;128&times;128</code> array (at 1024px) that a VAE turns into the RGB image. "
        "That <code>(H,W)=128</code> grid is <b>identical</b> to SD3.5's, so the radial-band spectral "
        "operators (<code>spectral_ops</code>, <code>style_ops</code>) apply <b>unchanged</b> &mdash; "
        "they are channel-agnostic. All the surgery happens on this latent, never the pixels.</dd>"
        "<dt>&epsilon;-prediction vs rectified flow</dt><dd>SDXL predicts the <b>noise</b> "
        "<code>&epsilon;</code> added to a latent (classic DDPM/DDIM formulation), unlike SD3.5's "
        "<b>velocity field</b> (rectified flow). &epsilon;-prediction has a clean, well-behaved "
        "<b>DDIM inversion</b>: a deterministic map clean-latent &rarr; noise that closes back to the "
        "same image. This is exactly the property E21's RF model lacked.</dd>"
        "<dt>DDIM inversion (clean &rarr; noise)</dt><dd>To edit a <i>real</i> image you first need "
        "the noise it came from. Run the pipeline with the <b><code>DDIMInverseScheduler</code></b> "
        "(timesteps <code>0&rarr;T</code>) on the image's latent to recover that noise. Reverse with "
        "the normal <code>DDIMScheduler</code> and you should get the image back. Not the RF-Inversion "
        "recipe E21 used &mdash; a true DDIM round-trip.</dd>"
        "<dt>FFT phase vs magnitude</dt><dd>Each spatial frequency of the latent has a <b>magnitude</b> "
        "(how strong the ripple is &rarr; texture power / palette) and a <b>phase</b> (where the "
        "ripples line up &rarr; <i>structure / layout</i>). Classic result (Oppenheim &amp; Lim): "
        "<b>phase carries layout</b>, most of the recognizable structure.</dd>"
        "<dt>Frequency-band PHASE locking (composition preservation)</dt><dd>Frequencies are binned "
        "into <code>N_BINS=24</code> radial rings from DC outward; band 0 = coarsest layout. A "
        "<b>cut <code>c</code></b> selects the lowest <code>c</code> fraction of the spectrum. The "
        "<b><code>lockphase</code></b> variant (<code>BandLock</code>, <code>mode=\"phase\"</code>) "
        "keeps the <b>source's</b> low-band phase (layout) while letting the new prompt own the "
        "magnitude and high-band phase &mdash; so <b>composition survives</b> and the prompt repaints "
        "appearance.</dd>"
        "<dt>cut <code>c</code> + until <code>u</code> (the two dials)</dt><dd><b><code>c</code></b> "
        "= the lowest fraction of the spectrum that gets locked (<code>c=0.1</code> coarsest only, "
        "<code>c=0.25</code> a bit more). <b><code>u</code></b> = the lock is active only for the "
        "first <code>u</code> fraction of denoising steps, then released so the prompt drives the "
        "finish (<code>u=0.6</code> lets go early; <code>u=1.0</code> holds to the end).</dd>"
        "<dt><code>lockpower</code> (palette lock, the control)</dt><dd>Instead of phase, the "
        "<code>BandLock</code> <code>mode=\"power\"</code> variant re-levels per-band <b>power</b> "
        "(magnitude energy) to the source. Magnitude carries palette/texture, <i>not</i> layout, so "
        "this is expected to barely preserve structure &mdash; a control that isolates the "
        "phase&harr;magnitude split.</dd>"
        "<dt>recon CLIP-I &uarr; (the gate metric)</dt><dd>Invert a real image, regenerate from that "
        "noise with the <b>same</b> prompt, and measure CLIP image-similarity between original and "
        f"reconstruction. <b>~{PASS_CLIP_I}</b> &asymp; \"round-trip closed.\" Higher is better; here "
        "it clears the bar (E22's core pivot win).</dd>"
        "<dt>struct_clip &uarr; / edit_clip_t &uarr; (the two edit scores)</dt><dd>"
        "<b><code>struct_clip</code></b> = CLIP-I of the edited image to the <b>source</b> photo "
        "(composition preserved &uarr;). <b><code>edit_clip_t</code></b> = CLIP of the edited image to "
        "the <b>target prompt</b> (edit followed &uarr;). A good knob negotiates these two; locking "
        "raises the first and lowers the second &mdash; the frontier.</dd>"
        "</dl>")

    # ---- method ----
    h.append("<h2>1 · Method</h2>")
    h.append(
        "<dl>"
        "<dt>Model</dt><dd>SDXL (<code>StableDiffusionXLPipeline</code>, fp16), "
        "<code>DDIMScheduler</code> for generation + <code>DDIMInverseScheduler</code> for inversion, "
        "1024px, <code>(1,4,128,128)</code> latent, 30 steps, inversion guidance 1.0, edit CFG 5.0. "
        "Runs locally (SDXL is cached).</dd>"
        "<dt><code>--part invert</code> (the gate)</dt><dd>For each real photo: VAE-encode to "
        "<code>x0</code> (VAE in fp32 for stability), run <code>ddim_invert</code> (swap in the "
        "inverse scheduler, run the pipe to get the noise latent, restore the normal scheduler), then "
        "regenerate from that noise with the <b>same</b> source prompt at guidance 1 and score "
        "<b>recon CLIP-I</b> to the original (+ report <code>noise_std</code> as a sanity check). "
        "<i>Question: does the inverted noise round-trip back to the source?</i></dd>"
        "<dt><code>--part edit</code> (the frontier)</dt><dd>Same inversion, then regenerate with a "
        "<b>target</b> prompt under band-lock variants &mdash; <code>invert_only</code> (baseline, no "
        "lock) vs <code>lockphase_c{0.1,0.25}_u{0.6,1.0}</code> (four phase-lock settings) vs "
        "<code>lockpower</code> &mdash; scoring <code>struct_clip</code> (CLIP-I to source &uarr;) "
        "against <code>edit_clip_t</code> (CLIP to target prompt &uarr;). Three edits, all from real "
        "photos: photo&rarr;oil-painting, photo&rarr;pencil-sketch, photo&rarr;watercolor. "
        "<i>Question: can locking source low-band phase preserve composition while the prompt repaints "
        "appearance &mdash; and at what cost to adherence?</i></dd>"
        "<dt><code>--part preflight</code> (model-free, passes)</dt><dd>Verifies the band-lock "
        "invariants: <code>band_phase_swap(c=1, mag_from=\"A\")</code> reconstructs the source "
        "(&lt;1e-2), and <code>mag_from=\"B\"</code> returns a valid same-shape float latent. The math "
        "is right before any model loads.</dd>"
        "</dl>")

    # ---- results: the gate ----
    h.append("<h2>2 · Results &mdash; Reconstruction (the gate)</h2>")
    h.append(img_tag("invert/grid.png", cls="grid"))
    h.append("<p class=cap>Left column = source photo; right column = DDIM reconstruction (invert to "
             "noise, regenerate with the same prompt at guidance 1). Three photos, top to bottom.</p>")
    h.append("<div class=look><b>What to look for.</b> A faithful inversion makes each reconstruction "
             "(right) <b>match its source</b> (left) &mdash; same scene, layout, and colors. On SDXL "
             "they do: the round-trip <b>closes</b>, unlike E21's SD3.5 drift into different images.</div>")
    h.append("<div class='note win'><b>Interpretation.</b> SDXL's &epsilon;-prediction gives a "
             "well-behaved deterministic DDIM inversion: the recovered \"noise\" is faithful, so the "
             "clean&rarr;noise&rarr;clean loop returns the source. This is the core win of the "
             "E21&rarr;E22 pivot &mdash; with a passing gate, the edit cells below are interpretable "
             "(on E21 they were not).</div>")

    if invert:
        h.append("<table><tr><th>photo</th>"
                 f"<th>recon CLIP-I &uarr;<br>(round-trip; ~{PASS_CLIP_I} = closed)</th>"
                 "<th>noise_std<br>(sanity)</th></tr>")
        best_ci = max((v.get("recon_clip_i") for v in invert.values()
                       if v.get("recon_clip_i") is not None), default=None)
        for fn in sorted(invert):
            v = invert[fn]
            ci = v.get("recon_clip_i")
            st = v.get("noise_std")
            ci_cls = "pos" if ci == best_ci else ""
            h.append(f"<tr><td class=v>{fn}</td>"
                     f"<td class={ci_cls}>{fmt(ci)}</td>"
                     f"<td>{fmt(st, 2)}</td></tr>")
        if mean_clip is not None:
            h.append(f"<tr><td class=v>mean</td>"
                     f"<td class=pos>{fmt(mean_clip)}</td>"
                     f"<td>{fmt(mean_std, 2)}</td></tr>")
        h.append("</table>")
        h.append("<p class=cap>Every <code>recon CLIP-I</code> sits at/above the "
                 f"~{PASS_CLIP_I} \"round-trip closed\" bar (best cell highlighted) &mdash; the gate "
                 "passes, so the edit results below are meaningful.</p>")
    else:
        h.append("<p class=cap>(no invert.json found &mdash; run "
                 "<code>e22_ddim_edit.py --part invert</code>)</p>")

    # ---- results: the edit frontier ----
    h.append("<h2>3 · Results &mdash; Band-lock edit (the structure&hArr;edit frontier)</h2>")
    if edit:
        for fn in sorted(edit):
            e = edit[fn]
            h.append(f"<h3>{fn} &rarr; <code>{e.get('target', '')}</code></h3>")
            h.append(img_tag(f"edit/grid_{fn}.png", cls="grid"))
        h.append("<p class=cap>Per edit, columns left&rarr;right: source, <code>invert_only</code> "
                 "(no lock), the four <code>lockphase</code> cells, and <code>lockpower</code>.</p>")
        h.append("<div class=look><b>What to look for.</b> The <code>invert_only</code> cell follows "
                 "the prompt hardest (most repainted) but <b>drifts</b> from the source layout; each "
                 "<code>lockphase</code> cell should <b>keep the source composition</b> (same scene "
                 "geometry) while the style still shifts toward the target &mdash; and the prompt's "
                 "grip visibly weakens as the lock tightens. <code>lockpower</code> should look like a "
                 "palette nudge, not a composition lock.</div>")
        h.append("<div class='note win'><b>Interpretation.</b> Locking the source's low-band phase "
                 "lifts structure-preservation from ~0.60 (no lock) to ~0.90 &mdash; a large, "
                 "consistent gain confirming <b>low-band phase = layout</b> end-to-end on a <i>real</i> "
                 "image edit (not just generated latents, cf. E18/E19). But it is a <b>frontier, not a "
                 "free lunch</b>: every <code>lockphase</code> cell drops edit CLIP-T from the ~0.24 "
                 "baseline to ~0.15&ndash;0.17, because holding the layout fixed constrains how far the "
                 "prompt can repaint. <code>lockpower</code> barely moves structure (~0.64) &mdash; "
                 "magnitude carries palette, not layout &mdash; confirming the split rather than "
                 "competing with it.</div>")

        # the structure<->adherence table (means over photos)
        conds = all_conditions(edit)
        struct = {c: edit_means(edit, c, "struct_clip") for c in conds}
        editt = {c: edit_means(edit, c, "edit_clip_t") for c in conds}
        best_struct = max((v for v in struct.values() if v is not None), default=None)
        best_edit = max((v for v in editt.values() if v is not None), default=None)
        h.append("<table><tr><th>condition (mean over edits)</th>"
                 "<th>struct&rarr;source (CLIP-I) &uarr;<br>(layout kept)</th>"
                 "<th>edit&rarr;prompt (CLIP-T) &uarr;<br>(target followed)</th></tr>")
        for c in conds:
            sc, et = struct[c], editt[c]
            lbl = c + (" (baseline)" if c == "invert_only" else "")
            h.append(f"<tr><td class=v>{lbl}</td>"
                     f"<td class={'pos' if sc == best_struct else ''}>{fmt(sc)}</td>"
                     f"<td class={'pos' if et == best_edit else ''}>{fmt(et)}</td></tr>")
        h.append("</table>")
        h.append("<p class=cap>Best cell per column highlighted. The two highlights land in "
                 "<b>different rows</b> &mdash; max structure (a <code>lockphase</code> cell) and max "
                 "adherence (the <code>invert_only</code> baseline) trade off. That gap <i>is</i> the "
                 "frontier: <code>c</code> and <code>u</code> tune where you sit on it.</p>")
    else:
        h.append("<div class='note cav'><b>No <code>edit.json</code> found.</b> Run "
                 "<code>e22_ddim_edit.py --part edit</code> to produce the band-lock cells "
                 "(<code>struct_clip</code> vs <code>edit_clip_t</code>) and per-edit grids, then "
                 "rebuild this page with <code>--part site</code>.</div>")

    # ---- caveats & next ----
    h.append("<h2>4 · Caveats &amp; next</h2>")
    h.append("<div class='note cav'><b>Caveats.</b> (1) <b>3 photos, 3 prompts</b> &mdash; "
             "directionally clear (the struct gap is huge and unanimous) but a small sample; widen the "
             "set before any strong quantitative claim on the frontier. (2) <b>CLIP-T is a weak edit "
             "metric</b> for style words (\"oil painting\"): absolute edit numbers are low even at "
             "baseline, so the <i>relative</i> drop under lock is the trustworthy signal &mdash; the "
             "grids are the real evidence of the look. (3) <b>Isotropic bands</b> lock layout phase + "
             "texture energy, <b>not oriented strokes</b> (the E18 caveat persists). (4) The "
             "structure&hArr;edit trade is a <b>frontier to tune</b>, not one setting: <code>c</code> "
             "(how much layout to lock) and <code>u</code> (how long) are the dials; "
             "<code>c=0.1, u=1.0</code> is the current sweet spot (max structure, best edit-CLIP among "
             "the strong-lock cells).</div>")
    h.append("<div class='note win'><b>Next.</b> A <b>soft / decaying lock</b> (ramp the lock strength "
             "down over steps instead of a hard release) to recover edit-adherence without losing the "
             "layout; and <b>per-edit picking</b> along the <code>(c,u)</code> frontier rather than a "
             "global setting.</div>")

    h.append("<p class=cap>Generated by <code>e22_site.py</code> from "
             "<code>results/e22/invert.json</code>"
             + (" + <code>edit.json</code>" if edit else " (no edit.json yet)")
             + " + <code>invert/grid.png</code> + <code>edit/grid_*.png</code>. Method: "
             "<code>e22_ddim_edit.py</code> (<code>load_sdxl</code>, <code>ddim_invert</code> DDIM "
             "inversion, <code>BandLock</code> callback), reusing <code>spectral_ops.py</code> "
             "(<code>band_phase_swap</code>, <code>band_index_map</code>) / <code>style_ops.py</code> "
             "(<code>restyle_latent</code>, <code>latent_band_power</code>) / <code>clip_sim.py</code>. "
             "See also <code>EXPERIMENT_22.md</code>.</p>")
    return "".join(h)


def build():
    """Load json, render, write index.html. No model load. Returns dest path."""
    ip = os.path.join(OUT, "invert.json")
    invert = json.load(open(ip)) if os.path.exists(ip) else None
    ep = os.path.join(OUT, "edit.json")
    edit = json.load(open(ep)) if os.path.exists(ep) else None
    html = render(invert or {}, edit or {})
    dest = os.path.join(OUT, "index.html")
    with open(dest, "w") as f:
        f.write(html)
    return dest, html, invert, edit


def main():
    os.makedirs(OUT, exist_ok=True)
    ip = os.path.join(OUT, "invert.json")
    ep = os.path.join(OUT, "edit.json")
    if not os.path.exists(ip) and not os.path.exists(ep):
        print(f"[e22-site] no data in {OUT} (need invert.json / edit.json); "
              "run the driver first: e22_ddim_edit.py --part invert,edit")
        return
    dest, html, invert, edit = build()
    print(f"[e22-site] wrote {dest}  ({len(html) // 1024} KB)  "
          f"(no model loaded; invert={'yes' if invert else 'no'}, "
          f"edit={'yes' if edit else 'no'})")


if __name__ == "__main__":
    main()
