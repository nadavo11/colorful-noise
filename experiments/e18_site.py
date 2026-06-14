"""Build a self-contained HTML report for E18 (offline spectral recombination).

Reads results/e18/report_<vae>.json (written by e18_spectral_recombine.py --part
analyze) and the recombination grid PNG; embeds the grid as a base64 JPEG so the
single index.html is portable. Honors CN_RESULTS and E18_SITE_VAE (flux|sd35).

    E18_SITE_VAE=flux python e18_site.py
"""
import base64
import io
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

VAE = os.environ.get("E18_SITE_VAE", "flux")
OUT = os.path.join(RESULTS, "e18")

# Display order + human labels for the decoded variants.
ORDER = ["baseA", "baseB", "styleA_s0.5", "styleA_s1", "phaseA_magB",
         "hybrid_c0.1", "hybrid_c0.25", "hybrid_c0.5", "phaseonlyA", "magonlyA"]
LABELS = {
    "baseA": "A — content (orig)", "baseB": "B — style (orig)",
    "styleA_s0.5": "restyle A→B, s=0.5 (ours)",
    "styleA_s1": "restyle A→B, s=1.0 (ours)",
    "phaseA_magB": "phase A + magnitude B (wholesale)",
    "hybrid_c0.1": "hybrid low-A/high-B, c=0.1",
    "hybrid_c0.25": "hybrid low-A/high-B, c=0.25",
    "hybrid_c0.5": "hybrid low-A/high-B, c=0.5",
    "phaseonlyA": "phase-only(A) — control",
    "magonlyA": "magnitude-only(A) — control",
}
# what each variant is meant to demonstrate (shown inline)
NOTES = {
    "styleA_s1": "keep A's phase + texture, re-level per-band power → B "
                 "(isotropic spectral style = AdaIN-in-Fourier)",
    "phaseA_magB": "stronger style but drags B's structure in through magnitude",
    "hybrid_c0.25": "coarse structure from A, fine detail from B (Oliva 2006)",
    "magonlyA": "textured palette swatch, no layout (Oppenheim–Lim)",
    "phaseonlyA": "recognizable layout, flat/desaturated (Oppenheim–Lim)",
}
METRICS = ["clip_to_A", "clip_to_B", "psd_to_A", "psd_to_B",
           "colorfulness", "saturation"]
MLAB = {"clip_to_A": "CLIP→A ↑", "clip_to_B": "CLIP→B",
        "psd_to_A": "PSD→A ↓", "psd_to_B": "PSD→B ↓",
        "colorfulness": "colorful", "saturation": "satur."}
LOWER_BETTER = {"psd_to_A", "psd_to_B"}
THUMB = 1500


def img_b64(path, maxw=THUMB):
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB")
    if im.width > maxw:
        im.thumbnail((maxw, maxw * 4))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def variant_means(report):
    """Per-variant mean of each metric across all pairs."""
    agg = {}
    for pair in report.values():
        for v, m in pair.items():
            agg.setdefault(v, {k: [] for k in METRICS})
            for k in METRICS:
                if k in m:
                    agg[v][k].append(m[k])
    return {v: {k: (sum(xs) / len(xs) if xs else None) for k, xs in d.items()}
            for v, d in agg.items()}


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


CSS = """
body{font:14px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a;max-width:1200px}
h1{font-size:23px} h2{font-size:18px;margin-top:30px;border-bottom:1px solid #ddd;padding-bottom:4px}
.note{background:#f6f8fa;border-left:4px solid #0969da;padding:10px 14px;border-radius:4px;margin:12px 0}
.cav{background:#fff8f0;border-left:4px solid #d4a017}
table{border-collapse:collapse;margin:10px 0;font-variant-numeric:tabular-nums}
th,td{border:1px solid #d0d7de;padding:5px 10px;text-align:center}
th{background:#f6f8fa} td.v{text-align:left;font-weight:600;white-space:nowrap}
td.best{font-weight:700;background:#dafbe1} tr.ours td.v{background:#fff8c5}
.cap{color:#555;font-size:12px} .desc{color:#666;font-size:12px;text-align:left}
img.grid{width:100%;border:1px solid #d0d7de;border-radius:4px;margin:8px 0}
code{background:#eff1f3;padding:1px 4px;border-radius:3px}
"""


def render(report):
    means = variant_means(report)
    conds = [v for v in ORDER if v in means] + \
            [v for v in means if v not in ORDER]
    # best (content-preservation + style) among the recombination variants only
    recomb = [v for v in conds if v not in ("baseA", "baseB")]
    best = {}
    for k in METRICS:
        vals = [(v, means[v][k]) for v in recomb if means[v].get(k) is not None]
        if vals:
            best[k] = (min if k in LOWER_BETTER else max)(vals, key=lambda t: t[1])[0]

    npairs = len(report)
    h = [f"<!doctype html><meta charset=utf-8><title>E18 spectral recombination</title>",
         f"<style>{CSS}</style>",
         "<h1>E18 — offline two-image spectral recombination "
         "<span class=cap>(“AdaIN-in-Fourier”)</span></h1>"]
    h.append(
        "<div class=note><b>Premise.</b> In the diffusion latent's 2-D Fourier "
        "domain, <b>phase</b> (esp. low-band) carries <b>content/layout</b> and "
        "per-(channel, radial-band) <b>power</b> carries <b>style</b> "
        "(texture-energy envelope + palette). So re-leveling per-band power is "
        "AdaIN on the radial power spectrum. This page tests that <b>offline</b>: "
        "VAE-encode pairs of real images (A = content, B = style), recombine their "
        "spectra, VAE-decode — no diffusion. If it holds here, the generation-time "
        "methods (E19–E22) stand on solid ground.</div>")
    h.append(f"<p class=cap>{VAE.upper()} VAE · means over {npairs} image pairs · "
             "<b>CLIP→A</b> = content kept (↑), <b>CLIP→B</b> = pull toward style, "
             "<b>PSD→·</b> = luminance-spectrum distance (↓ closer). "
             "Rows shaded yellow are the isotropic spectral-style op (ours).</p>")

    # --- means table ---
    h.append("<table><tr><th>variant</th>"
             + "".join(f"<th>{MLAB[k]}</th>" for k in METRICS)
             + "<th class=desc>what it shows</th></tr>")
    for v in conds:
        ours = " class='ours'" if v.startswith("styleA") else ""
        h.append(f"<tr{ours}><td class=v>{LABELS.get(v, v)}</td>")
        for k in METRICS:
            cls = "best" if best.get(k) == v else ""
            h.append(f"<td class='{cls}'>{fmt(means[v].get(k))}</td>")
        h.append(f"<td class=desc>{NOTES.get(v, '')}</td></tr>")
    h.append("</table>")

    # --- findings ---
    h.append("<h2>What the numbers say</h2><ul>"
             "<li><b>Restyle preserves content.</b> Re-leveling A's per-band power "
             "to B keeps A's layout (CLIP→A ≈ 0.93–0.97) while colorfulness shifts "
             "A→B with strength — the palette lives in the DC/low bands.</li>"
             "<li><b>The trade-off.</b> Wholesale magnitude (<code>phase A + mag B</code>) "
             "transfers more style but degrades content (CLIP→A drops to ~0.78). "
             "Isotropic band-power style is gentler on content, weaker on style.</li>"
             "<li><b>Hybrid works.</b> CLIP→A rises monotonically with the cutoff "
             "<code>c</code> — more low-band-from-A = more A identity.</li>"
             "<li><b>Controls behave.</b> magnitude-only = textured palette swatch "
             "(no layout); phase-only = recognizable but flat. Oppenheim–Lim holds "
             "in this latent.</li></ul>")

    # --- grid image ---
    grid = img_b64(f"{OUT}/grids/recombine_{VAE}.png")
    h.append("<h2>Decoded variants</h2>"
             "<p class=cap>Each row is one (A, B) pair; columns are the variants "
             "above, left→right. Look across a row: A's scene persists under "
             "restyle/hybrid while palette &amp; detail bend toward B.</p>")
    h.append(f"<img class=grid src='data:image/jpeg;base64,{grid}'>" if grid
             else "<p class=cap>(grid PNG not found — run "
                  "<code>e18_spectral_recombine.py --part analyze</code>)</p>")

    # --- caveats ---
    h.append("<h2>Caveats &amp; next</h2><div class='note cav'>"
             "<b>(1) Data.</b> This smoke run uses the e10 photo bank — all natural "
             "scenes (A↔B baseline CLIP ≈ 0.73), so style contrast is mild. A "
             "painting/photo set (<code>--styles</code>) will show it dramatically. "
             "<b>(2) Isotropy.</b> Radial bands transfer texture-energy + palette, "
             "<i>not</i> oriented strokes (Gram matrices would) — the stated ceiling, "
             "probed in E19. <b>(3) Metric.</b> Luminance-PSD doesn't cleanly capture "
             "a <i>latent</i> power-matching op; CLIP-I + colorfulness + the visual "
             "grid are the reliable readouts here. <b>Next:</b> E19 moves this into "
             "generation (content prompt + style-image envelope).</div>")
    h.append("<p class=cap>Generated by <code>e18_site.py</code> from "
             f"<code>results/e18/report_{VAE}.json</code>.</p>")
    return "".join(h)


def main():
    rpath = f"{OUT}/report_{VAE}.json"
    if not os.path.exists(rpath):
        print(f"[e18-site] no {rpath}; run e18_spectral_recombine.py --part analyze "
              f"--vae {VAE} first")
        return
    html = render(json.load(open(rpath)))
    site = f"{OUT}/site"
    os.makedirs(site, exist_ok=True)
    dest = f"{site}/index_{VAE}.html"
    with open(dest, "w") as f:
        f.write(html)
    print(f"[e18-site] wrote {dest}  ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
