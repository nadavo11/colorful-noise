"""Build a self-contained HTML explainer for E18 (offline two-image spectral
recombination, "AdaIN-in-Fourier").

Reads results/e18/report_<vae>.json (written by e18_spectral_recombine.py --part
analyze) + grids/recombine_<vae>.png and EMBEDS every image as base64 so the page
is fully portable (open results/e18/index.html anywhere). Honors CN_RESULTS.

The page STANDS ALONE: it defines every term (latent FFT phase vs per-band
magnitude/power; content = low-band phase, style = radial power envelope; psd_match
/ AdaIN-in-Fourier; the isotropy caveat = radial bands move tone, not strokes), every
variant the driver decodes, and every metric with its ↑/↓ direction, then leads each
result with the figure before the numbers. (memory: experiment-documentation-standard.)

    python e18_site.py            # builds from whichever report_<vae>.json exist
    python e18_spectral_recombine.py --part site
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e27_site import data_uri  # base64 embed -> one portable file

OUT = os.path.join(RESULTS, "e18")

# Which VAE runs to look for, and how to label them on the page.
VAES = ["sd35", "flux"]
VAE_LABEL = {"sd35": "SD3.5 VAE (the E19 generation model)",
             "flux": "Flux VAE (cached smoke test)"}

# Display order + human labels for the decoded variants.
ORDER = ["baseA", "baseB", "styleA_s0.5", "styleA_s1", "phaseA_magB",
         "hybrid_c0.1", "hybrid_c0.25", "hybrid_c0.5", "phaseonlyA", "magonlyA"]
LABELS = {
    "baseA": "A — content (orig)", "baseB": "B — style (orig)",
    "styleA_s0.5": "restyle A→B, s=0.5 (ours)",
    "styleA_s1": "restyle A→B, s=1.0 (ours)",
    "phaseA_magB": "phase A + magnitude B (wholesale)",
    "hybrid_c0.1": "hybrid low-A / high-B, c=0.1",
    "hybrid_c0.25": "hybrid low-A / high-B, c=0.25",
    "hybrid_c0.5": "hybrid low-A / high-B, c=0.5",
    "phaseonlyA": "phase-only(A) — control",
    "magonlyA": "magnitude-only(A) — control",
}
# what each variant is meant to demonstrate (shown inline)
NOTES = {
    "styleA_s0.5": "half-strength restyle — gentler tone shift",
    "styleA_s1": "keep A's phase + within-band texture, re-level per-band power → B "
                 "(isotropic spectral style = AdaIN-in-Fourier)",
    "phaseA_magB": "stronger style but drags B's structure in through magnitude",
    "hybrid_c0.25": "coarse structure from A, fine detail from B (Oliva 2006)",
    "magonlyA": "textured palette swatch, no layout (Oppenheim–Lim)",
    "phaseonlyA": "recognizable layout, flat / desaturated (Oppenheim–Lim)",
}
METRICS = ["clip_to_A", "clip_to_B", "psd_to_A", "psd_to_B",
           "colorfulness", "saturation"]
MLAB = {"clip_to_A": "CLIP→A ↑", "clip_to_B": "CLIP→B",
        "psd_to_A": "PSD→A ↓", "psd_to_B": "PSD→B ↓",
        "colorfulness": "colorful", "saturation": "satur."}
LOWER_BETTER = {"psd_to_A", "psd_to_B"}


CSS = """
body{font:15px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px auto;max-width:1100px;color:#1a1a1a;padding:0 16px}
h1{font-size:25px;line-height:1.25} h2{font-size:19px;margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:5px}
h3{font-size:16px;margin-top:24px;margin-bottom:4px} h4{font-size:14px;margin-bottom:4px;color:#333}
.tldr{background:#eef4ff;border:1px solid #c7d9ff;border-radius:6px;padding:14px 16px;margin:14px 0}
.note{background:#f6f8fa;border-left:4px solid #0969da;padding:10px 14px;border-radius:4px;margin:12px 0}
.look{background:#f3f0ff;border-left:4px solid #8957e5;padding:10px 14px;border-radius:4px;margin:10px 0}
.read{background:#f6f8fa;border-left:4px solid #57606a;padding:10px 14px;border-radius:4px;margin:10px 0}
.cav{background:#fff8f0;border-left:4px solid #d4a017}.win{background:#eafaf0;border-left:4px solid #2da44e}
dl{margin:10px 0} dt{font-weight:700;margin-top:10px} dd{margin:2px 0 2px 18px;color:#333}
table{border-collapse:collapse;margin:12px 0;font-variant-numeric:tabular-nums}
th,td{border:1px solid #d0d7de;padding:5px 9px;text-align:center}
th{background:#f6f8fa} td.v{text-align:left;font-weight:600;white-space:nowrap}
td.pos{background:#dafbe1}td.neg{background:#ffebe9} tr.ours td.v{background:#fff8c5}
td.desc{text-align:left;color:#666;font-size:12px}
.cap{color:#555;font-size:13px}
img.grid{width:100%;border:1px solid #d0d7de;border-radius:4px;margin:8px 0}
code{background:#eff1f3;padding:1px 5px;border-radius:3px;font-size:13px}
"""


def img_tag(name, cls="grid", **kw):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return f"<p class=cap>(missing {name})</p>"
    return f"<img class={cls} src='{data_uri(p, **kw)}'>"


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def variant_means(report):
    """Per-variant mean of each metric across all (A,B) pairs in one VAE report."""
    agg = {}
    for pair in report.values():
        for v, m in pair.items():
            agg.setdefault(v, {k: [] for k in METRICS})
            for k in METRICS:
                if k in m:
                    agg[v][k].append(m[k])
    return {v: {k: (sum(xs) / len(xs) if xs else None) for k, xs in d.items()}
            for v, d in agg.items()}


def means_table(means):
    """Per-variant means table; best recombination cell per column highlighted,
    the isotropic-style rows shaded (ours)."""
    conds = [v for v in ORDER if v in means] + [v for v in means if v not in ORDER]
    recomb = [v for v in conds if v not in ("baseA", "baseB")]
    best = {}
    for k in METRICS:
        vals = [(v, means[v][k]) for v in recomb if means[v].get(k) is not None]
        if vals:
            best[k] = (min if k in LOWER_BETTER else max)(vals, key=lambda t: t[1])[0]

    h = ["<table><tr><th>variant</th>"
         + "".join(f"<th>{MLAB[k]}</th>" for k in METRICS)
         + "<th class=desc>what it shows</th></tr>"]
    for v in conds:
        ours = " class=ours" if v.startswith("styleA") else ""
        h.append(f"<tr{ours}><td class=v>{LABELS.get(v, v)}</td>")
        for k in METRICS:
            cls = "pos" if best.get(k) == v else ""
            h.append(f"<td class={cls}>{fmt(means[v].get(k))}</td>" if cls
                     else f"<td>{fmt(means[v].get(k))}</td>")
        h.append(f"<td class=desc>{NOTES.get(v, '')}</td></tr>")
    h.append("</table>")
    return "".join(h)


def render(reports):
    """reports = {vae: report_dict} for every VAE run that exists on disk."""
    h = ["<!doctype html><meta charset=utf-8><title>E18 — spectral style transfer "
         "(AdaIN-in-Fourier)</title>",
         f"<style>{CSS}</style>",
         "<h1>E18 — offline two-image spectral recombination "
         "<span class=cap>(“AdaIN-in-Fourier”)</span></h1>"]

    # ---- TL;DR ----
    h.append(
        "<div class=tldr><b>In one paragraph.</b> A diffusion model works in a compressed "
        "<b>latent</b> space. Take the 2-D Fourier transform of a latent and split it: the "
        "<b>phase</b> (especially the low/coarse bands) carries the image's <b>content / layout</b>, "
        "while the per-(channel, radial-band) <b>magnitude / power</b> carries its <b>“style”</b> "
        "— the radial texture-energy envelope plus palette/contrast. That is the Gatys/AdaIN "
        "content–style split moved into frequency space: re-leveling per-band power is "
        "<b>AdaIN on the radial power spectrum</b> (the <code>psd_match</code> operator). E18 tests "
        "this <b>offline, with no diffusion</b>: VAE-encode pairs of real images (A = content, "
        "B = style), recombine their spectra in latent space, VAE-decode, and score. "
        "<b>Finding:</b> the premise holds — restyling A toward B keeps A’s layout (CLIP→A "
        "≈ 0.90–0.97) while shifting palette/tone toward B. But the effect is "
        "<b>VAE-dependent</b> (nearly inert in the Flux latent, <b>real in the SD3.5 latent</b> — "
        "restyling a photo toward a painting roughly <b>halves</b> the spectral distance) and it is "
        "<b>isotropic</b>: radial bands move <b>tone / palette, not oriented brush-strokes</b> "
        "(Gram matrices would). So the win is honest <b>spectral tone transfer</b>, content-safe, "
        "and it gives the generation-time methods (E19–E22) solid ground to stand on.</div>")

    # ---- background ----
    h.append("<h2>0 · Background (plain language)</h2><dl>"
             "<dt>Latent FFT: phase vs. per-band magnitude/power</dt><dd>Encode an image to a "
             "<code>(C, H, W)</code> latent and take its 2-D Fourier transform per channel. Each "
             "frequency has a <b>magnitude</b> (how much of that ripple is present) and a <b>phase</b> "
             "(where the ripples line up). Classic result (Oppenheim &amp; Lim): <b>phase carries the "
             "recognizable structure</b>, magnitude carries the energy/texture.</dd>"
             "<dt>Content = low-band phase; style = radial power envelope</dt><dd>From E7–E14 in this "
             "project: in the latent, the <b>low-band phase</b> fixes <b>content / layout / identity</b>, "
             "and the per-(channel, <b>radial band</b>) <b>power</b> is the <b>“style”</b> — the radial "
             "texture-energy envelope (slope) plus palette/contrast (which live in the DC + low bands). "
             "A <b>radial band</b> groups Fourier coefficients by distance from the center (DC): "
             "band 0 = coarse/low frequency, high bands = fine detail.</dd>"
             "<dt>psd_match / AdaIN-in-Fourier</dt><dd><code>psd_match</code> re-scales a latent so its "
             "per-band power matches a target envelope <b>while leaving the phase untouched</b>. That is "
             "exactly <b>AdaIN on the radial power spectrum</b>: keep the content latent’s phase + "
             "within-band texture, re-level its band power toward the style latent. The "
             "<code>strength</code> knob interpolates the target in log space between A’s own power "
             "(0 = no change) and B’s power (1 = full style envelope).</dd>"
             "<dt>Isotropy caveat (the stated ceiling)</dt><dd>Every operator here works on "
             "<b>radial</b> bands — concentric rings, no orientation. So it can transfer "
             "<b>texture-energy + palette/tone</b> but <b>not oriented brush-strokes</b> "
             "(those need anisotropic / Gram-matrix statistics). E18 measures where that ceiling bites; "
             "an anisotropic-band variant is a later extension.</dd>"
             "<dt>The variants (one decoded latent per (A, B) pair)</dt><dd>"
             "<code>baseA</code> / <code>baseB</code> = the two original images (content / style). "
             "<code>styleA_s{p}</code> = <b>restyle_latent</b>: A’s phase + A’s within-band texture, "
             "per-band power driven toward B at strength <code>p</code> — the <b>isotropic pure-style</b> "
             "op (AdaIN-in-Fourier). <code>phaseA_magB</code> = <b>band_phase_swap</b>: A’s phase + B’s "
             "<b>full</b> magnitude (a stronger style lever, but it drags B’s structure in through the "
             "magnitude). <code>hybrid_c{c}</code> = <b>band_spectrum_split</b>: the full complex "
             "spectrum (phase <i>and</i> magnitude) from A inside the lowest-<code>c</code> radial "
             "fraction and from B outside it — a latent <b>hybrid image</b> (Oliva 2006: coarse "
             "structure of A + fine detail of B). <code>phaseonlyA</code> / <code>magonlyA</code> = "
             "Oppenheim–Lim controls (phase-only should be recognizable-but-flat; magnitude-only should "
             "be a textured palette swatch with no layout).</dd>"
             "<dt>The metrics</dt><dd>"
             "<b>CLIP→A</b> (0–1, <b>↑</b> = content kept): CLIP-image cosine of the decoded result to "
             "the original A — high means A’s subject/layout survived. "
             "<b>CLIP→B</b>: cosine to B (how far the result pulled toward the style image; on natural "
             "photos this is content-dominated, so it moves little). "
             "<b>PSD→A</b> / <b>PSD→B</b> (<b>↓</b> = closer): distance between the result’s "
             "luminance log-radial power spectrum and A’s / B’s — a style/texture-energy distance. "
             "<b>colorful</b> / <b>satur.</b>: image colorfulness and saturation (palette readout — "
             "should move from A’s toward B’s under restyle).</dd>"
             "</dl>")

    # ---- method ----
    h.append("<h2>1 · Method</h2><dl>"
             "<dt><code>preflight</code> — the recombination math (no model)</dt><dd>Numeric asserts on "
             "synthetic 1/f latents: <code>restyle(strength=1)</code> re-levels A’s bands to B’s power "
             "<i>without</i> moving phase (band-power rel-err &lt;1e-3, phase drift &lt;1e-2); the hybrid "
             "endpoints are exact (<code>c=1 → A</code>, <code>c=0 → B</code>) and every recombination "
             "stays real (imag residue ~1e-6). <i>Is the operator algebra correct?</i></dd>"
             "<dt><code>analyze</code> — encode, recombine, decode, score</dt><dd>VAE-encode the image "
             "bank, build all variants for each (A, B) pair, VAE-decode, score with CLIP-I (→A, →B) and "
             "the luminance log-radial-PSD distance (→A, →B) plus colorfulness/saturation, save the "
             "decoded grid and <code>report_&lt;vae&gt;.json</code>. <code>--vae flux</code> uses the "
             "cached Flux VAE (instant smoke test); <code>--vae sd35</code> matches the E19+ generation "
             "model. <code>--styles</code> swaps in a painting set as the B (style) source for the "
             "cross-domain photo→painting test. <i>Does the premise hold on real images, and how much "
             "does it depend on the VAE?</i></dd>"
             "</dl>")

    # ---- results ----
    h.append("<h2>2 · Results</h2>")
    if not reports:
        h.append("<p class=cap>(no <code>report_&lt;vae&gt;.json</code> on disk — run "
                 "<code>e18_spectral_recombine.py --part analyze</code> first)</p>")
    for vae in VAES:
        if vae not in reports:
            continue
        report = reports[vae]
        means = variant_means(report)
        npairs = len(report)
        h.append(f"<h3>{vae.upper()} latent — {VAE_LABEL[vae]}</h3>")
        h.append(f"<h4>Decoded variants ({npairs} (A, B) pair"
                 f"{'s' if npairs != 1 else ''})</h4>")
        h.append(img_tag(f"grids/recombine_{vae}.png"))
        h.append("<div class=look><b>What to look for.</b> Each <b>row</b> is one (A, B) pair; "
                 "columns are the variants left→right (A · B · restyle s0.5 · restyle s1 · "
                 "phase A+mag B · hybrid c0.1/0.25/0.5 · phase-only A · mag-only A). Read across a "
                 "row: under <b>restyle</b> A’s scene should persist while palette/tone bends toward B; "
                 "<b>mag-only</b> should be a textured palette swatch with no layout; <b>phase-only</b> "
                 "recognizable but flat.</div>")
        if vae == "sd35":
            h.append("<div class='note read'><b>Reading (SD3.5).</b> Isotropic band-power restyle is "
                     "<b>real</b> here: restyling a photo toward a painting roughly <b>halves</b> the "
                     "spectral distance (PSD→B ≈1.92→1.03) and moves colorfulness ~30% toward the "
                     "painting, while keeping content (CLIP→A ≈0.90). But CLIP→B barely moves "
                     "(0.604→0.609): the <b>layout/subject stays photographic</b> — what transfers is "
                     "global <b>palette/tone</b>, not painterly brushwork. The isotropy ceiling holds; "
                     "the win is honest spectral tone transfer. Wholesale-magnitude and hybrid are the "
                     "heavier (but content-costlier / structural) levers.</div>")
        else:
            h.append("<div class='note read'><b>Reading (Flux).</b> In the Flux latent isotropic "
                     "band-power is nearly <b>inert</b> across domains (PSD→B 1.92→1.80, colorful "
                     "+0.006) — the original “weak style” read was partly a Flux-VAE artifact. The "
                     "effect being real on SD3.5 but not Flux is the key <b>VAE-dependence</b> "
                     "finding.</div>")
        h.append(f"<h4>Per-variant means (over {npairs} pair"
                 f"{'s' if npairs != 1 else ''})</h4>")
        h.append(means_table(means))
        h.append("<p class=cap>Best recombination cell per column highlighted (green); the "
                 "isotropic spectral-style rows (ours) are shaded yellow. <b>CLIP→A</b> ↑ = content "
                 "kept; <b>PSD→·</b> ↓ = spectrally closer.</p>")

    # ---- overall reading ----
    h.append("<h2>3 · Reading of the result</h2>")
    h.append("<div class='note win'><b>What we see.</b> The phase=content / power=style "
             "decomposition <b>recombines two real images in latent space</b>. Re-leveling A’s "
             "per-band power to B (restyle) keeps A’s layout (CLIP→A 0.90–0.97) while colorfulness "
             "shifts A→B with strength — the palette lives in the DC/low bands, exactly as predicted. "
             "Wholesale magnitude (<code>phase A + mag B</code>) transfers more style but degrades "
             "content (CLIP→A drops to ~0.78–0.84) by dragging B’s structure in through the magnitude. "
             "The hybrid’s CLIP→A rises monotonically with the cutoff <code>c</code> (more "
             "low-band-from-A = more A identity), and the Oppenheim–Lim controls behave (mag-only = "
             "textured palette swatch, phase-only = recognizable-but-flat). Crucially the win is "
             "<b>VAE-dependent</b>: real on SD3.5 (the E19 model), nearly inert on Flux.</div>")
    h.append("<div class='note cav'><b>Caveats &amp; next.</b> (1) The within-photo smoke run uses "
             "the E10 photo bank — all natural scenes (A↔B baseline CLIP ≈0.73), so style contrast is "
             "mild; the photo→painting set (<code>--styles</code>, baseline CLIP ≈0.60) shows it "
             "dramatically. (2) <b>Isotropy ceiling:</b> radial bands transfer texture-energy + palette, "
             "<i>not</i> oriented strokes (Gram matrices would) — the stated scope limit, probed in "
             "E19. (3) The luminance-PSD metric doesn’t cleanly capture a <i>latent</i> power-matching "
             "op; CLIP-I + colorfulness + the visual grid are the reliable readouts here, and E19 "
             "measures style match in latent band-power space. <b>Next:</b> E19 moves this into "
             "generation (content prompt + style-image envelope via the per-step <code>ClampPSD3</code> "
             "reference); frame it honestly as <b>spectral tone/palette transfer</b>, with the hybrid "
             "split as the stronger structural blend.</div>")

    h.append("<p class=cap>Generated by <code>e18_site.py</code> from "
             "<code>results/e18/report_&lt;vae&gt;.json</code> + <code>grids/recombine_&lt;vae&gt;.png</code>. "
             "Method: <code>e18_spectral_recombine.py</code> + <code>style_ops.py</code> (built on "
             "<code>spectral_ops.py</code>). See also <code>EXPERIMENT_18.md</code>.</p>")
    return "".join(h)


def build():
    """Load whatever report_<vae>.json exist and write index.html. Returns the dest
    path, or None if no report was found (caller prints the run-driver message)."""
    reports = {}
    for vae in VAES:
        rpath = os.path.join(OUT, f"report_{vae}.json")
        if os.path.exists(rpath):
            reports[vae] = json.load(open(rpath))
    if not reports:
        return None
    os.makedirs(OUT, exist_ok=True)
    html = render(reports)
    dest = os.path.join(OUT, "index.html")
    with open(dest, "w") as f:
        f.write(html)
    print(f"[e18-site] wrote {dest}  ({len(html) // 1024} KB) "
          f"from {sorted(reports)} report(s)")
    return dest


def main():
    if build() is None:
        print(f"[e18-site] no results/e18/report_<vae>.json; run "
              f"e18_spectral_recombine.py --part analyze --vae sd35 first")


if __name__ == "__main__":
    main()
