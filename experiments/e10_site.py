"""Build a self-contained HTML explainer for E10 (CFG inflates spectral power).

Reads results/e10/cfg_spectral.json (the analyze output) + plots/cfg_power.png /
plots/cfg_psd.png and EMBEDS every image as base64 so the page is fully portable
(open results/e10/index.html anywhere). Honors CN_RESULTS via common.RESULTS.

The page STANDS ALONE: it defines every term (Flux flow / velocity field, true-CFG
scale w, the latent, radial PSD / per-band power, the real-photo reference, and each
spectral + image metric) and shows the power-vs-CFG and radial-PSD figures with the
numbers pulled from the JSON. (memory: experiment-documentation-standard.)

This is the SBN-motivation experiment: spectral power rises monotonically with CFG
(~3x over w=1->5), and real photographs sit at standard guidance (w~3) -- the
unguided field is spectrally WEAKER than real, full guidance OVERSHOOTS.

    python e10_site.py        # rebuilds index.html, NO model load
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e27_site import data_uri  # base64 embed -> portable single file

OUT = os.path.join(RESULTS, "e10")

# CSS reused verbatim from e29_site.py / e30 (.tldr/.note/.cav/.win, glossary dl, td.pos).
CSS = """
body{font:15px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px auto;max-width:1000px;color:#1a1a1a;padding:0 16px}
h1{font-size:25px;line-height:1.25} h2{font-size:19px;margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:5px}
h3{font-size:15px;margin-bottom:4px}
.tldr{background:#eef4ff;border:1px solid #c7d9ff;border-radius:6px;padding:14px 16px;margin:14px 0}
.note{background:#f6f8fa;border-left:4px solid #0969da;padding:10px 14px;border-radius:4px;margin:12px 0}
.look{background:#fff8f0;border-left:4px solid #d4a017;padding:10px 14px;border-radius:4px;margin:12px 0}
.read{background:#f0f4ff;border-left:4px solid #0969da;padding:10px 14px;border-radius:4px;margin:12px 0}
.cav{background:#fff8f0;border-left:4px solid #d4a017}.win{background:#eafaf0;border-left:4px solid #2da44e}
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
    """Embed an image as base64; print a graceful "(missing X)" if it isn't there."""
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return f"<p class=cap>(missing {name})</p>"
    return f"<img class={cls} src='{data_uri(p, **kw)}'>"


def fmt(x, n=3, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def _g(cell, k):
    """Pull a mean from a report cell that stores agg dicts {'mean':..}; tolerate
    a bare float or a missing key."""
    if cell is None:
        return None
    v = cell.get(k)
    if isinstance(v, dict):
        return v.get("mean")
    return v


def _near(a, b, rel=0.06):
    """True if a is within `rel` (relative) of b -- used to highlight the cfg row
    whose spectral scale best matches the real-photo reference."""
    if a is None or b is None or b == 0:
        return False
    return abs(a - b) / abs(b) <= rel


def render(rep):
    cfgs = rep.get("cfgs", [])                 # list of "1", "1.5", ... strings
    pc = rep.get("per_cfg", {})
    real = rep.get("real")
    guidance = rep.get("guidance_scale")
    steps = rep.get("steps")
    nclass = rep.get("num_classes")
    seeds = rep.get("seeds")

    # headline numbers: power ratio across the sweep
    pow_first = _g(pc.get(cfgs[0]) if cfgs else None, "power")
    pow_last = _g(pc.get(cfgs[-1]) if cfgs else None, "power")
    ratio = (pow_last / pow_first) if (pow_first and pow_last) else None
    real_pow = _g(real, "power")
    # which cfg row best matches the real power (the "crossing" row)
    cross = None
    if real_pow is not None:
        best_d = None
        for w in cfgs:
            p = _g(pc.get(w), "power")
            if p is None:
                continue
            d = abs(p - real_pow)
            if best_d is None or d < best_d:
                best_d, cross = d, w

    h = ["<!doctype html><meta charset=utf-8><title>E10 — CFG inflates spectral power "
         "(the SBN motivation)</title>",
         f"<style>{CSS}</style>",
         "<h1>E10 — classifier-free guidance <em>inflates</em> the latent's spectral power; "
         "real photographs sit at <em>standard</em> guidance</h1>"]

    # ---- TL;DR ----
    headline = ""
    if ratio is not None:
        headline = (f"<b>Finding:</b> the latent's mean Fourier power rises <b>monotonically</b> "
                    f"with the guidance scale — about <b>{ratio:.2f}×</b> across the sweep "
                    f"(w={cfgs[0]}→{cfgs[-1]}). ")
    if real_pow is not None and cross is not None:
        headline += (f"<b>Real photographs sit near standard guidance</b> (power "
                     f"{real_pow:.3f} ≈ the <code>w={cross}</code> row): the unguided field "
                     f"(<code>w=1</code>) is spectrally <i>weaker</i> than real data, and full "
                     f"guidance <i>overshoots</i>.")
    h.append(
        "<div class=tldr><b>In one paragraph.</b> Flux is a <b>flow model</b>: its transformer "
        "predicts a velocity field that transports noise to data, and only <b>true-CFG scale "
        "<code>w=1</code></b> integrates the trained field — <code>w&gt;1</code> is an "
        "inference-time <i>extrapolation</i> that pushes the trajectory off the data manifold to "
        "obey the prompt harder. We sweep <code>w</code> over a fixed set of classes &amp; seeds "
        "and measure the <b>spectral power</b> of the generated <b>latent</b> (its per-frequency "
        "energy), then compare against real photographs encoded through the <b>same VAE</b>. " +
        (headline or
         "(Numbers load from <code>cfg_spectral.json</code> once the analyze part is re-run.)") +
        " This is the foundational fact <b>Spectral Band Normalization (SBN)</b> clamps back "
        "(E9), and the seed for E23's switch to a measured <b>real-photo</b> spectral target."
        "</div>")

    # ---- background ----
    h.append("<h2>0 · Background (plain language)</h2>")
    h.append("<dl>"
             "<dt>Flow model / velocity field</dt><dd>Flux is trained by <b>flow-matching</b>: the "
             "transformer predicts a velocity <code>v_θ(x, t)</code> that transports a Gaussian "
             "sample to a data sample. <b>Integrating the trained field</b> is the <code>w=1</code> "
             "case — the only one the loss actually fit.</dd>"
             "<dt>True-CFG scale w (classifier-free guidance)</dt><dd>The knob for how hard the "
             "prompt steers generation. At inference you extrapolate between the conditional and "
             "unconditional velocities: <code>ṽ = v_u + w·(v_c − v_u)</code>. <code>w=1</code> ≈ no "
             "steering (the trained field); higher <code>w</code> overrides more of the seed but "
             "leaves the data manifold. We use diffusers' <b>real two-pass CFG</b> "
             "(<code>true_cfg_scale</code> + an empty <code>negative_prompt</code>) and hold Flux's "
             "<i>distilled</i> <code>guidance_scale</code> fixed at a neutral "
             f"<code>{fmt(guidance, 1)}</code>, so the sweep isolates the cfg-equation effect, not "
             "Flux's distilled-guidance embedding.</dd>"
             "<dt>Latent</dt><dd>Flux denoises a compressed <code>16×128×128</code> array; the VAE "
             "turns it into the image. Every spectral metric below is computed on the (unpacked) "
             "latent.</dd>"
             "<dt>Radial PSD / per-band power</dt><dd>Any latent is a sum of spatial-frequency waves "
             "(<b>low</b> = coarse structure, <b>high</b> = fine texture). Binning the 2-D Fourier "
             "power into radial rings and averaging gives the <b>radial PSD</b> — the curve of how "
             "much coarse-vs-fine content the latent carries. The split <code>LOW_CUT=0.25</code> of "
             "the radial frequency separates the <b>low band</b> from the high.</dd>"
             "<dt>Real-photo reference</dt><dd>Natural photographs (seeded picsum + optional MS-COCO) "
             "encoded through the <i>same</i> Flux VAE into the generation latent space, so generated "
             "and real latents are directly comparable. This is the band the generated spectral scale "
             "is judged against (and the pool E23 later builds its real-PSD target from).</dd>"
             "<dt>Spectral metrics (the intensity axis)</dt><dd>"
             "<b>power</b> = mean |X|² (Parseval: equals mean-squared latent value) — the headline "
             "intensity; ↑ with CFG. <b>lat_std</b> = latent standard deviation; tracks power. "
             "<b>spec_norm</b> = the literal top singular value σ_max of each channel (matrix "
             "2-norm), averaged. <b>low_power</b> = mean power in the low radial band (the steepest "
             "riser — inflation is low-frequency-heavy). <b>low_frac</b> = fraction of total power in "
             "the low band (CFG tilts the spectrum toward coarse structure). There is no single "
             "\"good\" direction here — the point is that <i>all</i> rise above the real level once "
             "<code>w</code> is large.</dd>"
             "<dt>Image metrics (the decoded-pixel correlate)</dt><dd>From the saved PNGs: "
             "<b>rms_contrast</b> = grayscale standard deviation (↑ = more contrasty), "
             "<b>saturation</b> = mean per-pixel (max−min)/max over channels (↑ = more saturated), "
             "<b>hf_frac</b> = fraction of image FFT power above the 0.25 spatial-frequency ring "
             "(↑ = more fine detail). These show the \"over-cooked high-CFG look\" as the pixel-space "
             "shadow of the low-band power inflation.</dd>"
             "</dl>")

    # ---- method ----
    h.append("<h2>1 · Method</h2>")
    h.append("<dl>"
             "<dt><code>--part download</code> / <code>--part coco</code></dt><dd>Build the real pool: "
             "<code>download</code> fetches <code>--n_real</code> seeded <code>picsum.photos</code> "
             "images (reproducible); <code>coco</code> grows it with the first <code>--n_coco</code> "
             "MS-COCO val2017 photos (cached zip, lazy extract). <i>Question: what does a natural "
             "spectrum look like in this latent space?</i></dd>"
             "<dt><code>--part gen</code></dt><dd>For each class × cfg × seed, run a true-CFG "
             f"generation ({fmt(steps, 0) if steps else 'N'}-step), cache the image + the unpacked "
             "fp32 latent. Text encoders are pre-encoded then dropped to avoid OOM; killed runs "
             "resume for free. <i>Question: how does the latent change as w rises?</i></dd>"
             "<dt><code>--part real</code></dt><dd>Center-crop-square each photo (so a non-square "
             "aspect doesn't warp the radial PSD), VAE-encode, invert the decode convention "
             "(<code>lat = (z − shift)·sf</code>) into the generation latent space; stack to "
             "<code>real_latents.pt</code>. <i>Question: where does real data land on the same "
             "axes?</i></dd>"
             "<dt><code>--part analyze</code></dt><dd>Per-latent spectral metrics + per-image metrics "
             "from the PNGs, aggregated per cfg and for the real set; writes "
             "<code>cfg_spectral.json</code> and the <code>cfg_power.png</code> / "
             "<code>cfg_psd.png</code> figures. <i>Question: does power rise monotonically, and where "
             "does real sit?</i></dd>"
             "<dt><code>--part site</code></dt><dd>Model-free: re-template this page from "
             "<code>cfg_spectral.json</code> + the cached figures, no model load.</dd>"
             f"<dt>This run</dt><dd>true-CFG sweep <code>w∈{{{', '.join(cfgs)}}}</code>, "
             f"{fmt(nclass, 0) if nclass else '?'} classes × {fmt(seeds, 0) if seeds else '?'} seeds "
             f"= {(nclass*seeds) if (nclass and seeds) else '?'} latents/cfg; "
             f"real = {_g(real,'lat_std') is not None and (real.get('n')) or '?'} photos.</dd>"
             "</dl>")

    # ---- results ----
    h.append("<h2>2 · Results</h2>")

    # --- subsection A: power vs CFG ---
    h.append("<h3>Power vs CFG — does the spectral scale inflate, and where does real sit?</h3>")
    h.append(img_tag("plots/cfg_power.png"))
    h.append("<div class=look><b>What to look for.</b> Four intensity metrics (Fourier power, "
             "latent std, spectral norm, low-band power) vs the true-CFG scale <code>w</code>. The "
             "blue curve should climb <b>monotonically</b>; the green dashed line + band is the real "
             "photographs. The real line should cross the blue curve near <b>standard guidance</b> "
             "(<code>w≈3</code>), with <code>w=1</code> below it and high <code>w</code> above.</div>")
    h.append("<div class='note win'><b>Interpretation.</b> CFG <b>inflates</b> the latent's "
             "per-frequency power — every intensity metric rises with <code>w</code>, the steepest "
             "being <b>low_power</b> (the inflation is low-frequency-heavy, not a flat gain). Real "
             "data lands near <code>w≈3</code>: the trained field at <code>w=1</code> is spectrally "
             "<i>weaker</i> than real, normal guidance is roughly where the generated scale "
             "<b>crosses</b> real, and beyond that CFG overshoots. A flat re-scaling can't fix a "
             "band-shaped inflation — which is exactly the failure a <b>per-band</b> power clamp "
             "(SBN) is built for.</div>")

    # power-vs-cfg table
    cols = [("power", "power"), ("lat_std", "lat_std"),
            ("spec_norm", "spec_norm"), ("low_power", "low_power"),
            ("low_frac", "low_frac")]
    h.append("<table><tr><th>cfg w</th>" +
             "".join(f"<th>{lbl}</th>" for _, lbl in cols) + "</tr>")
    for w in cfgs:
        cell = pc.get(w)
        tds = []
        for k, _ in cols:
            v = _g(cell, k)
            hot = (k == "power" and real_pow is not None and w == cross)
            n = 4 if k == "low_frac" else 3
            tds.append(f"<td class=pos>{fmt(v, n)}</td>" if hot else f"<td>{fmt(v, n)}</td>")
        h.append(f"<tr><td class=v>w={w}</td>{''.join(tds)}</tr>")
    if real is not None:
        tds = []
        for k, _ in cols:
            n = 4 if k == "low_frac" else 3
            tds.append(f"<td class=pos>{fmt(_g(real, k), n)}</td>")
        h.append(f"<tr><td class=v>real ({real.get('n', '?')})</td>{''.join(tds)}</tr>")
    h.append("</table>")
    h.append("<p class=cap>Highlighted: the cfg row whose Fourier power best matches the real-photo "
             "reference (the spectral \"crossing\" point), and the full real row.</p>")

    # --- image metrics table (decoded-pixel correlate) ---
    img_cols = [("rms_contrast", "rms_contrast ↑"), ("saturation", "saturation ↑"),
                ("hf_frac", "hf_frac ↑")]
    if any(_g(pc.get(w), "rms_contrast") is not None for w in cfgs):
        h.append("<h4>Image-space correlate — the over-cooked high-CFG look</h4>")
        h.append("<div class=read><b>Reading.</b> As <code>w</code> rises the decoded images get more "
                 "contrasty and saturated — the familiar over-cooked high-CFG appearance is the "
                 "pixel-space shadow of the low-band power inflation. (Real-set image metrics are "
                 "absent: only latents are stored for the real pool, no decoded PNGs to score.)</div>")
        h.append("<table><tr><th>cfg w</th>" +
                 "".join(f"<th>{lbl}</th>" for _, lbl in img_cols) + "</tr>")
        for w in cfgs:
            cell = pc.get(w)
            tds = "".join(f"<td>{fmt(_g(cell, k))}</td>" for k, _ in img_cols)
            h.append(f"<tr><td class=v>w={w}</td>{tds}</tr>")
        h.append("</table>")

    # --- subsection B: radial PSD per CFG ---
    h.append("<h3>Radial PSD per CFG — the shape of the inflation</h3>")
    h.append(img_tag("plots/cfg_psd.png"))
    h.append("<div class=look><b>What to look for.</b> Log-log power-per-radial-ring for each "
             "<code>w</code> (viridis, dark = low w) with the real photographs as the black dashed "
             "curve. Watch how the whole curve lifts as <code>w</code> grows, and whether the lift is "
             "<b>uniform</b> or concentrated at the <b>low-frequency (left)</b> end.</div>")
    h.append("<div class='note win'><b>Interpretation.</b> The PSD lifts with CFG, most at low "
             "frequency, so the generated curve straddles the real (black dashed) one: at high "
             "<code>w</code> the low end overshoots real while the high end can stay below it — a "
             "<b>bimodal residual</b> (too much low, too little high). Clamping to the <code>w=1</code> "
             "output (E10/E9's first SBN reference) addresses the inflation but is itself <i>below</i> "
             "real; that mismatch is why E23 retargets SBN to the measured <b>real</b> PSD.</div>")

    # ---- caveats & next ----
    h.append("<h2>3 · Caveats &amp; next</h2>")
    h.append("<div class='note cav'><b>Caveats.</b> "
             "(1) <code>cfg=1</code> is a <i>proxy</i> for \"natural\", not natural itself — real "
             "data sits near <code>w≈3</code>, so clamping to <code>cfg=1</code> moves <i>away</i> "
             "from real at the low bands (the measured motivation for E23's real-target SBN). "
             "(2) Isotropic, radial-only: bands carry texture-energy + palette, not oriented "
             "structure. (3) Small/seeded sample (per-cfg latents + ~20 picsum photos) — enough for "
             "the monotone trend and the crossing point, not tight per-class claims; E23 rebuilds the "
             "real target from 500 MS-COCO photos. (4) VAE-space comparison: generated vs real are "
             "only comparable because both are encoded by the <i>same</i> VAE; the numbers are not "
             "transferable across models (cf. E17, which needs an SD3.5-VAE real reference). "
             "<b>Next:</b> package the band clamp as a method (E9/SBN), then replace the cfg=1 proxy "
             "with a measured real-photo target (E23).</div>")

    h.append("<p class=cap>Generated by <code>e10_site.py</code> from "
             "<code>results/e10/cfg_spectral.json</code> + <code>plots/{cfg_power,cfg_psd}.png</code>. "
             "Method: <code>e10_cfg_spectral.py</code> (reuses <code>spectral_ops.py</code>, "
             "<code>e7_flux_phase.py</code>, <code>e9_bandnorm_classes.py</code>). See also "
             "<code>EXPERIMENT_10.md</code>.</p>")
    return "".join(h)


def build_site():
    """Load cfg_spectral.json and write index.html; no model load. Returns the
    output path, or None (with a clear message) if the report is absent."""
    rpath = os.path.join(OUT, "cfg_spectral.json")
    if not os.path.exists(rpath):
        print(f"[e10-site] no {rpath}; run --part analyze first")
        return None
    with open(rpath) as f:
        rep = json.load(f)
    html = render(rep)
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "index.html")
    with open(dest, "w") as f:
        f.write(html)
    print(f"[e10-site] wrote {dest}  ({len(html) // 1024} KB)")
    return dest


def main():
    build_site()


if __name__ == "__main__":
    main()
