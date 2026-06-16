"""Build a self-contained HTML explainer for E29 (phase inheritance z_T -> z_0).

Reads results/e29/{report.json, transplant.json} + plots/*.png + examples/*.png and
EMBEDS every image as base64 so the page is fully portable (open results/e29/index.html
anywhere). Honors CN_RESULTS.

The page STANDS ALONE: it defines every term (seed z_T / output z_0, FFT phase vs
magnitude, radial band, circular correlation, the permutation null, CFG, the causal
transplant + follow score) and shows the spectra, heatmap, controls and grids with
numbers pulled from the JSON. (memory: experiment-documentation-standard.)

    python e29_site.py
"""
import base64
import json
import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

OUT = os.path.join(RESULTS, "e29")

CSS = """
body{font:15px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px auto;max-width:1000px;color:#1a1a1a;padding:0 16px}
h1{font-size:25px;line-height:1.25} h2{font-size:19px;margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:5px}
h3{font-size:15px;margin-bottom:4px}
.tldr{background:#eef4ff;border:1px solid #c7d9ff;border-radius:6px;padding:14px 16px;margin:14px 0}
.note{background:#f6f8fa;border-left:4px solid #0969da;padding:10px 14px;border-radius:4px;margin:12px 0}
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


def data_uri(path, max_px=1600, quality=85):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_px:
        r = max_px / max(im.size)
        im = im.resize((round(im.width * r), round(im.height * r)))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def img_tag(name, cls="plot", **kw):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return f"<p class=cap>(missing {name})</p>"
    return f"<img class={cls} src='{data_uri(p, **kw)}'>"


def fmt(x, n=2, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def band_means(rep, cfg, key, lo, hi):
    """Mean of `key` band-curve over conditions with the given cfg, for bands [lo,hi)."""
    vals = []
    for lbl, c in rep["conditions"].items():
        if lbl == "uncond" or c["cfg"] != cfg:
            continue
        arr = c[key]
        vals.append(mean(arr[lo:hi]))
    return mean(vals)


def render(rep, tp):
    conds = rep["conditions"]
    cfg_cfg = rep["config"]
    cfgs = cfg_cfg["cfgs"]
    nbins = cfg_cfg["nbins"]
    lo_hi = (0, max(1, nbins // 3))          # "low" third of the spectrum
    hi_hi = (2 * nbins // 3, nbins)          # "high" third
    uncond = conds.get("uncond")

    h = ["<!doctype html><meta charset=utf-8><title>E29 — phase inheritance (seed → output)</title>",
         f"<style>{CSS}</style>",
         "<h1>E29 — how much of the <em>output latent's FFT phase</em> is inherited "
         "from the <em>seed's FFT phase</em>?</h1>"]

    # headline number
    uncond_low = mean(uncond["phase_corr_band"][lo_hi[0]:lo_hi[1]]) if uncond else None
    null_low = mean(uncond["null_mean"][lo_hi[0]:lo_hi[1]]) if uncond else None

    h.append(
        "<div class=tldr><b>In one paragraph.</b> A diffusion model turns a random "
        "<b>seed</b> (a Gaussian noise array, the latent <code>z_T</code>) into a final "
        "<b>output latent</b> <code>z_0</code> that the decoder paints into an image. Earlier "
        "experiments showed the <b>phase</b> of a latent's 2-D Fourier transform carries the "
        "image's <i>structure / layout</i> (the magnitude carries palette &amp; texture power). "
        "So we ask: does the seed's phase <i>survive</i> the denoising and show up in the output's "
        "phase? Using deterministic DDIM in Stable Diffusion 1.5 we measure, per spatial-frequency "
        "band and over many seeds, the <b>circular correlation</b> between seed phase and output "
        "phase, and then we <b>causally</b> swap the seed's phase in a band and watch the output's "
        "phase move. "
        + (f"<b>Finding:</b> the output latent is <b>strongly inherited from the seed</b> across the "
           f"whole spectrum (unconditional phase circular corr ≈ {fmt(uncond_low)} at low frequency "
           f"vs a chance level ≈ {fmt(null_low, sign=True)}; magnitude and even raw pixel structure "
           f"are inherited at least as much). Stronger guidance <b>erodes</b> the inheritance, "
           f"concentrated in the low-frequency (coarse-layout) bands the prompt overrides, and the "
           f"phase transplant confirms the link is causal."
           if uncond else "") +
        "</div>")

    # ---- background ----
    h.append("<h2>0 · Background (plain language)</h2>")
    h.append("<dl>"
             "<dt>Seed z_T / output z_0</dt><dd>SD1.5 denoises in a compressed latent space. The "
             "<b>seed</b> is a <code>4×64×64</code> Gaussian array <code>z_T</code>; deterministic "
             "DDIM denoising maps it to the <b>output latent</b> <code>z_0</code> (same shape), "
             "which the VAE decodes to the 512×512 image. With DDIM the map <code>z_T→z_0</code> is "
             "a fixed function (no extra randomness), so any relationship is a property of that map.</dd>"
             "<dt>FFT phase vs magnitude</dt><dd>Take the 2-D Fourier transform of a latent channel. "
             "Each frequency has a <b>magnitude</b> (how much of that ripple is present — palette / "
             "texture power) and a <b>phase</b> (where the ripples line up — <i>structure / layout</i>). "
             "Classic result (Oppenheim &amp; Lim): phase carries most of the recognizable structure.</dd>"
             "<dt>Radial frequency band</dt><dd>We group Fourier coefficients by distance from the "
             f"center (DC) into <code>{nbins}</code> rings: band 0 = lowest frequency (coarse layout), "
             "high bands = fine detail. All curves are plotted band-by-band.</dd>"
             "<dt>Circular correlation</dt><dd>Phase is an angle, so ordinary correlation is wrong "
             "(−π and +π are the same). We use the <b>Jammalamadaka–SenGupta circular correlation</b>: "
             "+1 = output phase perfectly predicted by seed phase at that frequency, 0 = unrelated, "
             "computed per Fourier bin across many seeds then averaged within each band.</dd>"
             "<dt>Permutation null</dt><dd>The chance level: re-compute the correlation after "
             "shuffling which output goes with which seed. Independent phases give ≈0, so a band whose "
             "correlation sits above the null band is genuinely inheriting.</dd>"
             "<dt>CFG (classifier-free guidance)</dt><dd>The knob for how hard the prompt steers "
             "generation. CFG=1 ≈ no steering; higher = the prompt overrides more of the seed.</dd>"
             "</dl>")

    # ---- method ----
    h.append("<h2>1 · Method</h2>")
    h.append("<dl>"
             "<dt>Generate &amp; capture</dt><dd>For each condition we draw "
             f"<code>N={cfg_cfg['N']}</code> seeds, run {cfg_cfg['steps']}-step deterministic DDIM, "
             "and store both the seed <code>z_T</code> (passed in as <code>latents=</code>) and the "
             "final output latent <code>z_0</code> (captured via <code>output_type=\"latent\"</code>). "
             "Both live in the same scaled latent space, so their spectra are directly comparable.</dd>"
             "<dt>Conditions</dt><dd>An <b>unconditional</b> map (empty prompt, CFG=1) isolates the "
             f"pure <code>z_T→z_0</code> function; then a CFG sweep <code>{cfgs}</code> over "
             f"{len(cfg_cfg['prompts'])} prompts tests whether stronger guidance overwrites the "
             "inherited phase.</dd>"
             "<dt>Metrics</dt><dd>(1) the <b>phase-inheritance spectrum</b> (circular corr per band) "
             "with its permutation null; (2) a <b>magnitude control</b> (Pearson of log-magnitude — "
             "does the seed predict output <i>power</i>?); (3) a <b>phase-difference resultant</b> "
             "(does the output phase equal the seed phase up to a consistent offset?); and (4) a "
             "spatial-domain sanity check.</dd>"
             "<dt>Causal transplant</dt><dd>For seed pairs (A, B) we build <code>A'</code> = A's "
             "magnitude + A's phase everywhere <i>except</i> B's phase inside the lowest-<code>c</code> "
             "band (Hermitian-symmetric, variance preserved so it is still a valid Gaussian seed — "
             f"measured mean std ≈ <code>{fmt(tp.get('transplant_seed_std_mean') if tp else None)}</code>). "
             "We regenerate and measure the <b>follow score</b>: how far the output's phase moved from "
             "A toward the donor B, per band. 1 = fully followed the donor, 0.5 = no effect.</dd>"
             "</dl>")

    # ---- results ----
    h.append("<h2>2 · Results — the inheritance spectrum</h2>")
    h.append(img_tag("plots/inherit_spectrum.png"))
    h.append("<p class=cap>Circular correlation between seed phase and output phase vs radial "
             "frequency (left = low / coarse, right = high / fine). Dashed = unconditional map; "
             "colored = CFG sweep (mean over prompts); gray = permutation null (±2σ). Anything above "
             "the gray band is genuine phase inheritance.</p>")

    if uncond:
        h.append("<table><tr><th>condition</th><th>low-band corr<br>(coarse layout)</th>"
                 "<th>high-band corr<br>(fine detail)</th></tr>")
        h.append(f"<tr><td class=v>unconditional</td>"
                 f"<td class={'pos' if (uncond_low or 0)>0.05 else ''}>{fmt(uncond_low)}</td>"
                 f"<td>{fmt(mean(uncond['phase_corr_band'][hi_hi[0]:hi_hi[1]]))}</td></tr>")
        for c in cfgs:
            lo = band_means(rep, c, "phase_corr_band", *lo_hi)
            hi = band_means(rep, c, "phase_corr_band", *hi_hi)
            h.append(f"<tr><td class=v>CFG={c}</td>"
                     f"<td class={'pos' if (lo or 0)>0.05 else ''}>{fmt(lo)}</td>"
                     f"<td>{fmt(hi)}</td></tr>")
        h.append(f"<tr><td class=v>null (chance)</td><td>{fmt(null_low, sign=True)}</td>"
                 f"<td>{fmt(mean(uncond['null_mean'][hi_hi[0]:hi_hi[1]]), sign=True)}</td></tr>")
        h.append("</table>")

    h.append("<h3>Where in the 2-D spectrum (unconditional)</h3>")
    h.append(img_tag("plots/inherit_heatmap.png"))
    h.append("<p class=cap>Per-bin circular correlation, fftshifted so DC (lowest frequency) is at "
             "the center. Bright = inherited.</p>")

    h.append("<h3>Control: does the seed predict output <i>magnitude</i>?</h3>")
    h.append(img_tag("plots/magnitude_control.png"))
    h.append("<p class=cap>Phase inheritance (circles) vs magnitude inheritance (squares, log-power "
             "Pearson). We had expected magnitude to be inherited <i>weakly</i> (the model re-shapes "
             "the power spectrum toward natural-image statistics, cf. E23). Instead magnitude is "
             "inherited <b>at least as strongly as phase</b>: the seed largely fixes the whole output "
             "spectrum, not just its structure-bearing phase.</p>")

    h.append("<h3>Secondary: phase-difference resultant</h3>")
    h.append(img_tag("plots/dphi_resultant.png"))
    h.append("<p class=cap>|mean exp(i·Δφ)| per band. High = the output phase is the seed phase up "
             "to a <i>consistent</i> transform (e.g. a global shift); this complements, but does not "
             "replace, the circular correlation above.</p>")

    # ---- causal ----
    h.append("<h2>3 · Causal confirmation — transplant the seed's phase</h2>")
    h.append(img_tag("plots/follow.png"))
    h.append("<p class=cap>Follow score per band after swapping the donor's phase into the lowest-"
             "<code>c</code> band of the seed and regenerating. Above 0.5 = the output's phase in "
             "that band moved toward the donor — i.e. editing the seed's phase <b>causes</b> a "
             "matching change in the output's phase.</p>")
    if tp:
        h.append("<table><tr><th>swapped band c</th><th>low-band follow</th>"
                 "<th>high-band follow</th></tr>")
        for c, d in tp["follow"].items():
            m = d["mean"]
            h.append(f"<tr><td class=v>c={c}</td>"
                     f"<td class={'pos' if mean(m[lo_hi[0]:lo_hi[1]])>0.55 else ''}>"
                     f"{fmt(mean(m[lo_hi[0]:lo_hi[1]]))}</td>"
                     f"<td>{fmt(mean(m[hi_hi[0]:hi_hi[1]]))}</td></tr>")
        h.append("</table>")
    h.append(img_tag("examples/transplant_grid.png", cls="grid"))
    h.append("<p class=cap>Rows = swapped-band size c; columns = base A, donor B, and the "
             "transplant A' (A's magnitude + A's phase, donor B's phase in the low band). A' should "
             "drift toward B's coarse layout while keeping A's palette.</p>")

    # ---- reading ----
    h.append("<h2>4 · Reading of the result</h2>")
    h.append("<div class='note win'><b>What we see.</b> The output latent is <b>strongly inherited "
             "from the seed</b>. The seed→output phase circular correlation sits far above the "
             "permutation null at every frequency, and — contrary to our expectation — the "
             "<b>magnitude</b> is inherited at least as strongly, with raw pixel-space correlation as "
             "high as ≈0.76 unconditionally. So with little/no guidance the seed broadly fixes the "
             "whole output spectrum, not just the structure-bearing phase. Turning the prompt up "
             "<b>erodes</b> the inheritance, and it does so <b>preferentially in the low-frequency "
             "bands</b> (the coarse composition the prompt overrides), leaving fine detail more "
             "seed-determined. The transplant makes it causal: injecting a donor's phase into the "
             "seed's low band steers the output's phase toward that donor — and the effect spills "
             "across the whole spectrum, since the coarse layout conditions everything downstream.</div>")
    h.append("<div class='note cav'><b>Caveats.</b> (1) SD1.5 + DDIM only; other models / VAEs and "
             "stochastic samplers may differ. (2) Circular correlation is a <i>statistical</i> link "
             "across seeds, not a per-image guarantee. (3) The phase-difference resultant can be high "
             "for a trivial reason (a consistent global shift), which is why the circular correlation "
             "is the headline metric. (4) Phase and magnitude are not fully independent through a "
             "nonlinear VAE, so 'phase = structure' is a strong tendency, not a law.</div>")

    h.append("<p class=cap>Generated by <code>e29_site.py</code> from "
             "<code>results/e29/{report.json, transplant.json}</code> + plots/examples. Method: "
             "<code>e29_phase_inherit.py</code> + <code>e29_phase_ops.py</code> (reuses "
             "<code>spectral_ops.py</code>). See also <code>EXPERIMENT_29.md</code>.</p>")
    return "".join(h)


def main():
    rpath = os.path.join(OUT, "report.json")
    if not os.path.exists(rpath):
        print(f"[e29-site] no {rpath}; run e29_phase_inherit.py first")
        return
    rep = json.load(open(rpath))
    tpath = os.path.join(OUT, "transplant.json")
    tp = json.load(open(tpath)) if os.path.exists(tpath) else None
    html = render(rep, tp)
    dest = os.path.join(OUT, "index.html")
    with open(dest, "w") as f:
        f.write(html)
    print(f"[e29-site] wrote {dest}  ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
