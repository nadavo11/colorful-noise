"""Build a self-contained HTML explainer for E21 (RF-inversion + frequency-band
locking on SD3.5 — a GATED / pending-negative result).

Reads results/e21/invert.json (and edit.json IF it exists) + the reconstruction
grid (invert/grid.png) and EMBEDS every image as base64 so the page is fully
portable (open results/e21/index.html anywhere). Honors CN_RESULTS.

The page STANDS ALONE: it defines every term (SD3.5 latent, rectified-flow /
velocity field, RF-inversion clean->noise, naive forward-Euler vs implicit
fixed-point Euler, frequency-band locking of the source low-band phase, the
metrics recon_clip_i ↑ and noise_std ~1.0, and the gate logic) and leads the
results with the reconstruction grid + a "what to look for" before the table.

KEY FRAMING: the edit is GATED on the inversion round-tripping the source. On
SD3.5 it DRIFTS (recon CLIP-I ~0.63-0.74, far below the ~0.94 "round-trip closed"
bar; noise_std ~1.11 instead of ~1.0). The failed gate IS the finding, and it
motivates the pivot to E22 (SDXL + DDIM inversion). There is no edit.json — the
edit was gated and not run — so the page must handle missing edit data gracefully.
(memory: experiment-documentation-standard.)

    python e21_site.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e27_site import data_uri

OUT = os.path.join(RESULTS, "e21")

# "round-trip closed" bar: a faithful inversion reconstructs the source at high
# CLIP-I; ~0.94 is the rough pass threshold, ~1.0 the ideal noise std.
PASS_CLIP_I = 0.94
TARGET_STD = 1.0

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


def render(invert, edit):
    clip_vals = [v.get("recon_clip_i") for v in invert.values()] if invert else []
    std_vals = [v.get("noise_std") for v in invert.values()] if invert else []
    mean_clip = mean(clip_vals)
    mean_std = mean(std_vals)

    h = ["<!doctype html><meta charset=utf-8>"
         "<title>E21 — RF-inversion + frequency-band locking (SD3.5): the gate fails</title>",
         f"<style>{CSS}</style>",
         "<h1>E21 — spectral image editing via <em>RF-inversion</em> + frequency-band locking "
         "on SD3.5 — <em>the gate fails (and that is the finding)</em></h1>"]

    # ---- TL;DR ----
    h.append(
        "<div class=tldr><b>In one paragraph.</b> We wanted to edit a <i>real</i> photo by spectral "
        "surgery: invert it back to the noise that would have produced it, then regenerate with a new "
        "prompt while <b>locking</b> the source's low-frequency <b>phase</b> (which carries layout) so "
        "the prompt only repaints appearance (oil painting / pencil sketch / watercolor). SD3.5 is a "
        "<b>rectified-flow</b> model, so \"inversion\" means integrating its velocity field backwards "
        "(clean&nbsp;&rarr;&nbsp;noise). The whole edit is <b>gated</b>: it only means anything if the "
        "inversion <i>round-trips</i> &mdash; regenerate from the inverted noise with the <i>same</i> "
        "prompt and you should get the source back. <b>It does not.</b> On SD3.5 the reverse-flow ODE "
        "<b>drifts</b>: reconstruction CLIP-I to the source is only "
        + (f"<b>~{fmt(mean_clip, 2)}</b> " if mean_clip is not None else "")
        + f"(a closed round-trip is ~{PASS_CLIP_I}), and the recovered \"noise\" has std "
        + (f"~<b>{fmt(mean_std, 2)}</b> " if mean_std is not None else "")
        + f"instead of the ~{TARGET_STD:.0f} a valid Gaussian seed should have. With a broken gate, "
        "the band-locked edit is uninterpretable, so it was <b>not run</b>. <b>The failed gate is the "
        "result</b>: RF-inversion on SD3.5 is the weak link, which is exactly why the thread pivots to "
        "<b>E22</b> (SDXL + DDIM inversion, where the round-trip closes) carrying the <i>identical</i> "
        "band-lock operators over unchanged.</div>")

    # ---- background ----
    h.append("<h2>0 · Background (plain language)</h2>")
    h.append(
        "<dl>"
        "<dt>Latent (SD3.5)</dt><dd>SD3.5 does not denoise pixels; it works in a compressed "
        "<code>16&times;128&times;128</code> array (at 1024px) that a VAE turns into the RGB image. "
        "All the spectral surgery happens on this latent, never the pixels.</dd>"
        "<dt>Rectified flow / velocity field</dt><dd>SD3.5 is <b>not</b> a noise-predictor. It learns "
        "a <b>velocity field</b> <code>v(x, &sigma;)</code> that flows a clean latent "
        "(<code>&sigma;=0</code>) along a near-straight path to pure noise (<code>&sigma;=1</code>) and "
        "back. Generation is the Euler step <code>x += (&sigma;_next &minus; &sigma;_cur)&middot;"
        "v(x, &sigma;)</code> walking <code>&sigma;: 1&rarr;0</code>.</dd>"
        "<dt>RF-inversion (clean &rarr; noise)</dt><dd>To edit a <i>real</i> image you first need the "
        "noise it came from. RF-inversion integrates the <i>same</i> velocity field in the "
        "<b>opposite</b> direction, <code>&sigma;: 0&rarr;1</code>. This is <i>not</i> DDIM inversion; "
        "it is the RF-Inversion / FlowEdit recipe.</dd>"
        "<dt>Naive forward-Euler vs implicit fixed-point Euler</dt><dd><b>Forward (explicit) Euler</b> "
        "evaluates the velocity at the <i>current</i> &sigma;; it is only exact when <code>v</code> "
        "does not depend on <code>x</code>. The <b>fixed-point (implicit) Euler</b> instead solves "
        "<code>x_hi = x_lo + (&sigma;_hi &minus; &sigma;_lo)&middot;v(x_hi, &sigma;_hi)</code> by "
        "iterating a few times (<code>fp_iters=4</code>) &mdash; it evaluates the velocity at the "
        "<i>next</i> &sigma; and is far more accurate for a state-dependent field. E21 uses the "
        "fixed-point variant; on SD3.5 even it drifts.</dd>"
        "<dt>FFT phase vs magnitude</dt><dd>Each spatial frequency of the latent has a "
        "<b>magnitude</b> (how strong the ripple is &rarr; texture power / palette) and a <b>phase</b> "
        "(where the ripples line up &rarr; structure / layout). Classic result (Oppenheim &amp; Lim): "
        "phase carries most recognizable structure.</dd>"
        "<dt>Frequency-band locking (the edit knob)</dt><dd>Frequencies are binned into "
        "<code>N_BINS=24</code> radial rings from DC outward; band 0 = coarsest layout. A <b>cut "
        "<code>c</code></b> selects the lowest <code>c</code> fraction of the spectrum. The "
        "<code>BandLock</code> callback keeps the <b>source's</b> low-band phase (layout) while letting "
        "the new prompt own the magnitude and high-band phase &mdash; so composition survives and the "
        "prompt repaints appearance. (A <code>power</code> mode instead re-levels per-band power to the "
        "source, a palette lock.) The lock runs for the first <code>until</code> fraction of steps, "
        "then releases.</dd>"
        "<dt>recon_clip_i &uarr; (the gate metric)</dt><dd>Invert a real image, regenerate from that "
        "noise with the <b>same</b> prompt, and measure CLIP image-similarity (CLIP-I) between original "
        f"and reconstruction. <b>~{PASS_CLIP_I}</b> &asymp; \"round-trip closed\"; the values here "
        "(~0.63&ndash;0.74) mean the inversion <b>drifts</b> and the reconstruction is a different "
        "image. Higher is better.</dd>"
        "<dt>noise_std (a drift symptom)</dt><dd>The std of the recovered \"noise\" latent. A valid "
        f"Gaussian seed has std &asymp; <b>{TARGET_STD:.0f}</b>; the values here (~1.11) are "
        "<b>inflated</b> &mdash; the integration overshoots, another sign the inversion is off the "
        "manifold of true seeds.</dd>"
        "<dt>The gate logic</dt><dd>Reconstruction is the prerequisite for <i>every</i> downstream "
        "edit cell. The edit is <b>only valid if the round-trip closes</b>. Here it does not, so the "
        "edit is gated and was not run; the failure invalidates any edit numbers rather than merely "
        "weakening them.</dd>"
        "</dl>")

    # ---- method ----
    h.append("<h2>1 · Method</h2>")
    h.append(
        "<dl>"
        "<dt><code>--part invert</code> (the gate)</dt><dd>For each of three real photos: VAE-encode "
        "to <code>x0</code>, run <code>invert_sd3</code> (28-step fixed-point reverse-Euler, "
        "<code>&sigma;:&nbsp;0&rarr;1</code>, guidance 1) to recover the noise, then regenerate from "
        "that noise with the <b>same</b> source prompt at guidance 1 and score "
        "<code>recon_clip_i</code> + report <code>noise_std</code>. <i>Question: does the inverted "
        "noise actually round-trip back to the source?</i></dd>"
        "<dt><code>--part edit</code> (gated &mdash; not run)</dt><dd>Same inversion, then regenerate "
        "with a <b>target</b> prompt under band-lock variants &mdash; <code>invert_only</code> "
        "(baseline, no lock) vs <code>lockphase_c{0.1,0.25}_u{0.6,1.0}</code> vs <code>lockpower</code> "
        "&mdash; scoring <code>struct_clip</code> (CLIP-I to the source, layout preserved &uarr;) "
        "against <code>edit_clip_t</code> (CLIP-T to the target prompt, edit followed &uarr;). "
        "<i>Question: can locking source low-band phase preserve composition while the prompt repaints "
        "appearance?</i> &mdash; only answerable once the gate passes, so it awaits a working "
        "inversion.</dd>"
        "<dt><code>--part preflight</code> (model-free, passes)</dt><dd>Verifies (1) reverse-Euler is "
        "<b>exact on a state-independent</b> velocity field (round-trip error &lt; 1e-3 &mdash; the "
        "<i>math</i> is right, so the real-model drift is the <i>field's</i> state-dependence, not a "
        "bug), and (2) the band-lock invariants (<code>c=1, mag_from=\"A\"</code> reconstructs the "
        "source exactly; <code>mag_from=\"B\"</code> keeps the generation's magnitude).</dd>"
        "</dl>")

    # ---- results: the gate ----
    h.append("<h2>2 · Results &mdash; Reconstruction (the gate)</h2>")
    h.append(img_tag("invert/grid.png", cls="grid"))
    h.append("<p class=cap>Left column = source photo; right column = reconstruction (invert to "
             "noise, regenerate with the same prompt at guidance 1). Three photos, top to bottom.</p>")
    h.append("<div class=look><b>What to look for.</b> A faithful inversion would make each "
             "reconstruction (right) <b>match its source</b> (left) &mdash; same scene, layout, and "
             "colors. Instead the reconstructions <b>drift</b> into different images: the round-trip "
             "does not close, so the recovered \"noise\" is not the seed that made the photo.</div>")
    h.append("<div class='note cav'><b>Interpretation.</b> Both the naive forward-Euler and the "
             "<code>fp_iters=4</code> fixed-point inversion fail to round-trip a real SD3.5 image. "
             "The preflight proves the integrator is exact for a state-independent field, so the "
             "drift comes from the <i>trained</i> velocity field being strongly state-dependent: small "
             "per-step errors compound over 28 steps and the recovered latent lands off the manifold "
             "of valid seeds (hence the inflated std). A broken gate makes every downstream edit cell "
             "uninterpretable &mdash; so we report the gate, not edits.</div>")

    if invert:
        h.append("<table><tr><th>photo</th>"
                 f"<th>recon_clip_i &uarr;<br>(round-trip; ~{PASS_CLIP_I} = closed)</th>"
                 f"<th>noise_std<br>(target ~{TARGET_STD:.0f})</th></tr>")
        for fn in sorted(invert):
            v = invert[fn]
            ci = v.get("recon_clip_i")
            st = v.get("noise_std")
            # honest highlight: ALL recon below the pass bar -> mark `neg`.
            ci_cls = "neg" if (ci is not None and ci < PASS_CLIP_I) else ""
            st_cls = "neg" if (st is not None and abs(st - TARGET_STD) > 0.05) else ""
            h.append(f"<tr><td class=v>{fn}</td>"
                     f"<td class={ci_cls}>{fmt(ci)}</td>"
                     f"<td class={st_cls}>{fmt(st)}</td></tr>")
        if mean_clip is not None:
            h.append(f"<tr><td class=v>mean</td>"
                     f"<td class=neg>{fmt(mean_clip)}</td>"
                     f"<td class=neg>{fmt(mean_std)}</td></tr>")
        h.append("</table>")
        h.append("<p class=cap>Every <code>recon_clip_i</code> sits well below the "
                 f"~{PASS_CLIP_I} \"round-trip closed\" bar (cells marked red), and every "
                 f"<code>noise_std</code> is inflated above ~{TARGET_STD:.0f} &mdash; both confirm the "
                 "gate fails. (Highlighting a <i>failure</i> is awkward: rather than pick a \"best\" "
                 "cell we mark every cell that misses its target, which is the honest read.)</p>")
    else:
        h.append("<p class=cap>(no invert.json found)</p>")

    # ---- edit: gated, not run ----
    h.append("<h2>3 · Edit (gated &mdash; not run)</h2>")
    if edit:
        # If edit data ever appears, surface it (struct_clip vs edit_clip_t).
        h.append("<p class=cap>An <code>edit.json</code> was found; per-photo band-lock cells "
                 "(<code>struct_clip</code> = CLIP-I to source &uarr;, <code>edit_clip_t</code> = "
                 "CLIP-T to target &uarr;):</p>")
        for fn in sorted(edit):
            e = edit[fn]
            h.append(f"<h3>{fn} &rarr; <code>{e.get('target', '')}</code></h3>")
            h.append(img_tag(f"edit/grid_{fn}.png", cls="grid"))
            cells = e.get("cells", {})
            if cells:
                struct = {k: c.get("struct_clip") for k, c in cells.items()}
                editt = {k: c.get("edit_clip_t") for k, c in cells.items()}
                best_struct = max((v for v in struct.values() if v is not None), default=None)
                best_edit = max((v for v in editt.values() if v is not None), default=None)
                h.append("<table><tr><th>variant</th><th>struct_clip &uarr;<br>(layout kept)</th>"
                         "<th>edit_clip_t &uarr;<br>(target followed)</th></tr>")
                for k in cells:
                    sc, et = struct[k], editt[k]
                    h.append(f"<tr><td class=v>{k}</td>"
                             f"<td class={'pos' if sc == best_struct else ''}>{fmt(sc)}</td>"
                             f"<td class={'pos' if et == best_edit else ''}>{fmt(et)}</td></tr>")
                h.append("</table>")
        h.append("<div class='note cav'><b>Caveat.</b> These edit numbers sit on top of an inversion "
                 "that does not round-trip (Section 2), so the source-preservation column does not mean "
                 "what it claims &mdash; the \"source\" the edit started from is already a drifted "
                 "image. Read them only as smoke output, not a result.</div>")
    else:
        h.append("<div class='note cav'><b>The edit was gated and not run.</b> There is no "
                 "<code>edit.json</code>: the band-locked edit only makes sense once the inversion "
                 "round-trips, and on SD3.5 it does not (Section 2). Running the edit on top of a "
                 "broken reconstruction would produce numbers that <i>look</i> like a "
                 "structure-vs-edit frontier but are actually measuring agreement with a drifted "
                 "image, not the source. So the edit <b>awaits a working inversion</b> &mdash; which is "
                 "the pivot to <b>E22</b> (SDXL + <code>DDIMInverseScheduler</code>). SDXL's "
                 "<code>4&times;128&times;128</code> latent shares SD3.5's <code>(H,W)=128</code> grid, "
                 "so the band-lock operators carry over with <b>no changes</b>; only the inversion "
                 "backbone is swapped to one that round-trips.</div>")

    # ---- reading ----
    h.append("<h2>4 · Caveats &amp; next</h2>")
    h.append("<div class='note cav'><b>Caveats.</b> (1) <b>RF inversion is the weak link</b>, not the "
             "band-lock idea &mdash; the spectral operators are model-agnostic and unit-checked in "
             "preflight. (2) The drift is intrinsic to integrating a state-dependent flow field over "
             "many steps; RF-Inversion-style controllers can help but are finicky. (3) Locking "
             "<i>low-band phase</i> preserves layout but cannot, by construction, transfer "
             "<i>oriented</i> brushstrokes &mdash; radial bands are isotropic (the E18 caveat carries "
             "over). (4) Single guidance / 28 steps; not swept, because the gate fails first.</div>")
    h.append("<div class='note win'><b>Next &mdash; E22.</b> Swap the backbone to <b>SDXL + "
             "<code>DDIMInverseScheduler</code></b>, an &epsilon;-prediction model whose DDIM "
             "round-trip is reliable, and carry the <i>identical</i> band-lock editing operators over "
             "unchanged. E21 is the documented negative that motivates that pivot.</div>")

    h.append("<p class=cap>Generated by <code>e21_site.py</code> from "
             "<code>results/e21/invert.json</code>"
             + (" + <code>edit.json</code>" if edit else " (no edit.json &mdash; gated)")
             + " + <code>invert/grid.png</code>. Method: <code>e21_spectral_edit.py</code> "
             "(<code>invert_sd3</code> RF inversion, <code>BandLock</code> callback), reusing "
             "<code>spectral_ops.py</code> / <code>style_ops.py</code> / <code>e17_sd35.py</code>. "
             "See also <code>EXPERIMENT_21.md</code>.</p>")
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
    if not os.path.exists(ip):
        print(f"[e21-site] no {ip}; run e21_spectral_edit.py --part invert first")
        return
    dest, html, invert, edit = build()
    print(f"[e21-site] wrote {dest}  ({len(html) // 1024} KB)  "
          f"(no model loaded; invert={'yes' if invert else 'no'}, "
          f"edit={'yes' if edit else 'no/gated'})")


if __name__ == "__main__":
    main()
