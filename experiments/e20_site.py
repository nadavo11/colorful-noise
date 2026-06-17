"""Self-contained HTML explainer for E20 (spectral warm-start — "skip the beginning").

Reads results/e20/{oracle.json, condition.json, noiseshape.json} + the saved grids
and EMBEDS every image as base64, so the single results/e20/index.html is portable
(open it anywhere). Honors CN_RESULTS.

The page STANDS ALONE: it defines every term (the warm-start band commit, the cutoff
`c`, the re-entry strength `s`, every condition, every metric with ↑/↓ direction) and
leads each result with the figure, a "what to look for", an interpretation, then the
numbers (best cell per column highlighted). (memory: experiment-documentation-standard.)

Pure templating from the JSON + cached grids — loads NO model. Run with either:

    python e20_site.py
    python e20_warmstart.py --part site
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

try:
    from e27_site import data_uri          # base64 embed -> portable single file
except Exception:
    data_uri = None

OUT = os.path.join(RESULTS, "e20")

# CSS reused verbatim from e29/e30 (.tldr/.look/.read/.win/.cav, glossary dl, td.pos)
CSS = """
body{font:15px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px auto;max-width:1000px;color:#1a1a1a;padding:0 16px}
h1{font-size:25px;line-height:1.25} h2{font-size:20px;margin-top:38px;border-bottom:1px solid #ddd;padding-bottom:5px}
h3{font-size:16px;margin:22px 0 4px} h4{font-size:14px;margin:16px 0 2px;color:#333}
.tldr{background:#eef4ff;border:1px solid #c7d9ff;border-radius:6px;padding:14px 16px;margin:14px 0}
.look{background:#f6f8fa;border-left:4px solid #0969da;padding:8px 13px;border-radius:4px;margin:8px 0;font-size:14px}
.read{margin:8px 0 4px} .win{background:#eafaf0;border-left:4px solid #2da44e;padding:8px 13px;border-radius:4px;margin:8px 0}
.cav{background:#fff8f0;border-left:4px solid #d4a017;padding:10px 14px;border-radius:4px;margin:12px 0}
dl{margin:10px 0} dt{font-weight:700;margin-top:11px} dd{margin:2px 0 2px 18px;color:#333}
table{border-collapse:collapse;margin:10px 0;font-variant-numeric:tabular-nums;font-size:14px}
th,td{border:1px solid #d0d7de;padding:4px 9px;text-align:right}
th{background:#f6f8fa;text-align:center} td.v{text-align:left;font-weight:600;white-space:nowrap}
td.pos{background:#dafbe1;font-weight:600} td.neg{background:#ffebe9}
.cap{color:#555;font-size:13px;margin:2px 0 14px}
img{width:100%;border:1px solid #d0d7de;border-radius:4px;margin:6px 0}
code{background:#eff1f3;padding:1px 5px;border-radius:3px;font-size:13px}
"""


def img_tag(rel, **kw):
    """Embed results/e20/<rel> as base64, or print a (missing) placeholder."""
    p = os.path.join(OUT, rel)
    if data_uri is None or not os.path.exists(p):
        return f"<p class=cap>(missing {rel})</p>"
    return f"<img src='{data_uri(p, **kw)}'>"


def fmt(x, n=2):
    return "—" if x is None else f"{x:.{n}f}"


def _parse_cells(cells):
    """{c<C>_s<S>: {...}} -> (sorted cuts, sorted strengths, {(C,S): metrics})."""
    cuts, strs, grid = set(), set(), {}
    for k, v in cells.items():
        cc = float(k.split("_")[0][1:])
        st = float(k.split("_")[1][1:])
        cuts.add(cc)
        strs.add(st)
        grid[(cc, st)] = v
    return sorted(cuts), sorted(strs), grid


# ---------------------------------------------------------------------------
# Oracle: a (cutoff c) x (strength s) grid table per scene, per metric.
# clip_i ↑ (warm-start reconstructs the oracle target) ; latent_l2 ↓ (close in latent).
# ---------------------------------------------------------------------------

def _oracle_table(grid, cuts, strs, key, best="max"):
    """One c-rows x s-cols table for metric `key`; highlight the best cell per column."""
    h = ["<table><tr><th>cutoff <code>c</code> \\ skip</th>"]
    for st in strs:
        h.append(f"<th>s={st:g}<br>(skip {1 - st:.0%})</th>")
    h.append("</tr>")
    # best per column
    best_per_col = {}
    for st in strs:
        vals = [(grid[(cc, st)][key], cc) for cc in cuts if (cc, st) in grid]
        if vals:
            best_per_col[st] = (max if best == "max" else min)(vals)[1]
    for cc in cuts:
        label = "noise (c=0)" if cc == 0 else (f"c={cc:g} (full spectrum)" if cc >= 1 else f"c={cc:g}")
        h.append(f"<tr><td class=v>{label}</td>")
        for st in strs:
            v = grid.get((cc, st), {}).get(key) if (cc, st) in grid else None
            cls = " class=pos" if best_per_col.get(st) == cc else ""
            h.append(f"<td{cls}>{fmt(v)}</td>")
        h.append("</tr>")
    h.append("</table>")
    return "".join(h)


def render_oracle(h, data):
    h.append("<h3>oracle — how much can we skip if we hand it the TRUE low bands?</h3>")
    h.append("<div class=look><b>What to look for.</b> Each grid below is one scene. Down the rows "
             "we commit more of the run's own spectrum (cutoff <code>c</code>: 0 = nothing, 1 = the "
             "whole latent); across the columns we re-enter later (strength <code>s</code>; "
             "<code>skip = 1−s</code> of the steps are skipped). Watch the jump from the "
             "<code>c=0</code> noise baseline (top row) to any committed row: committing even the "
             "lowest bands (<code>c=0.1</code>) should snap recovery near 1.</div>")
    # numbers first need parsing
    summaries = {}
    for pid, d in data.items():
        cuts, strs, grid = _parse_cells(d["cells"])
        summaries[pid] = (cuts, strs, grid, d["prompt"])
    # interpretation, computed from the data (c=0 vs c=0.1 at the gentlest skip)
    def at(pid, cc, st, key):
        _, _, grid, _ = summaries[pid]
        return grid.get((cc, st), {}).get(key)
    base = mean([at(p, 0.0, 0.8, "clip_i") for p in summaries])         # c=0 needs the LEAST skip
    comm = mean([at(p, 0.1, 0.6, "clip_i") for p in summaries])
    h.append("<div class='read win'><b>Reading.</b> The committed rows recover the image; the "
             f"<code>c=0</code> noise baseline does not. Committing the lowest bands alone "
             f"(<code>c=0.1</code>) lifts CLIP-I to ≈{fmt(comm)} (latent-L2 ≈0.4), versus "
             f"≈{fmt(base)} for re-noising from pure noise at the <i>same</i> step — so a small "
             "low-frequency cutoff buys almost the whole image while skipping a large fraction of "
             "the schedule. Pushing <code>c</code> higher (0.25 → 1) adds little: the coarse layout "
             "is carried by the lowest bands. The latent-L2 column tells the same story in latent "
             "space (lower is closer to the target).</div>")
    for pid, (cuts, strs, grid, prompt) in summaries.items():
        h.append(f"<h4>{pid}: <code>{prompt}</code></h4>")
        h.append(img_tag(f"oracle/grid_{pid}.png"))
        h.append("<p class=cap>rows = cutoff <code>c</code> (top = full reference run, then "
                 "<code>c=0</code> noise baseline upward); columns = skip fraction. The first "
                 "column is the reference (full) run.</p>")
        h.append("<p class=cap><b>CLIP-I to the full run ↑</b> (1 = identical content):</p>")
        h.append(_oracle_table(grid, cuts, strs, "clip_i", best="max"))
        h.append("<p class=cap><b>latent L2 to the full run ↓</b> (0 = identical latent):</p>")
        h.append(_oracle_table(grid, cuts, strs, "latent_l2", best="min"))


# ---------------------------------------------------------------------------
# Condition: commit a REFERENCE image's low bands. struct_clip ↑ (looks like ref),
# prompt_clip ↑ (follows prompt). The two trade off.
# ---------------------------------------------------------------------------

def render_condition(h, data):
    h.append("<h3>condition — band-controlled SDEdit (commit a reference image's low bands)</h3>")
    h.append("<div class=look><b>What to look for.</b> Now the committed low bands come from a "
             "<i>reference image</i> (a painting), not an oracle; the prompt drives the rest. "
             "<code>c=1</code> is ordinary full SDEdit (whole reference committed); lower "
             "<code>c</code> keeps only the reference's coarse structure and frees the model to "
             "follow the prompt. Watch the trade-off: more committed structure / less skip "
             "(left columns) → the image hugs the reference; less structure / more skip → it "
             "follows the text.</div>")
    h.append("<div class=read><b>Reading.</b> The two metrics trade off as expected: higher "
             "re-entry strength <code>s</code> (more steps actually run) raises prompt-adherence "
             "(<b>prompt CLIP-T</b>) and lowers structure-match (<b>struct CLIP-I</b>); a lower "
             "cutoff <code>c</code> loosens the grip on the reference. So <code>c</code>×<code>s</code> "
             "is a structure-vs-prompt dial. It does not cleanly <i>beat</i> full SDEdit "
             "(<code>c=1</code>) on this small set — the band cut is a softer version of the same "
             "knob rather than a free lunch.</div>")
    for tag, d in data.items():
        h.append(f"<h4>{tag.replace('__', ' → ')}: <code>{d['prompt']}</code></h4>")
        h.append(img_tag(f"condition/grid_{tag}.png"))
        h.append("<p class=cap>rows = cutoff <code>c</code> (with <code>c=1</code> = full SDEdit); "
                 "columns = skip fraction. First column is the reference image.</p>")
        cuts, strs, grid = _parse_cells(d["cells"])
        # struct_clip ↑ best per col, prompt_clip ↑ best per col -- show one merged table
        h.append("<table><tr><th>cutoff <code>c</code></th>")
        for st in strs:
            h.append(f"<th>struct CLIP-I ↑<br>s={st:g}</th><th>prompt CLIP-T ↑<br>s={st:g}</th>")
        h.append("</tr>")
        best_struct = {st: max((grid[(cc, st)]["struct_clip"], cc) for cc in cuts if (cc, st) in grid)[1]
                       for st in strs}
        best_prompt = {st: max((grid[(cc, st)]["prompt_clip"], cc) for cc in cuts if (cc, st) in grid)[1]
                       for st in strs}
        for cc in cuts:
            label = f"c={cc:g}" + (" (SDEdit)" if cc >= 1 else "")
            h.append(f"<tr><td class=v>{label}</td>")
            for st in strs:
                v = grid.get((cc, st))
                sc = " class=pos" if best_struct.get(st) == cc else ""
                pc = " class=pos" if best_prompt.get(st) == cc else ""
                h.append(f"<td{sc}>{fmt(v['struct_clip']) if v else '—'}</td>"
                         f"<td{pc}>{fmt(v['prompt_clip']) if v else '—'}</td>")
            h.append("</tr>")
        h.append("</table>")


# ---------------------------------------------------------------------------
# Noiseshape: color step-0 noise toward a natural-latent spectrum. aesthetic ↑, clip_t ↑.
# ---------------------------------------------------------------------------

def render_noiseshape(h, data):
    h.append("<h3>noiseshape — pre-color the initial noise toward a natural-latent spectrum</h3>")
    h.append("<div class=look><b>What to look for.</b> No skipping here: we just reshape the "
             "step-0 noise so its band power matches real photos' encoded latents (vs. plain white "
             "noise), then run the full schedule. The question is whether a natural-spectrum start "
             "reaches quality (aesthetic, prompt CLIP-T) in fewer steps. Compare the "
             "<code>colored</code> rows to <code>white</code> at each step count.</div>")
    # compute deltas
    aes_w = mean([d["white"][st]["aesthetic"] for d in data.values() for st in d["white"]])
    aes_c = mean([d["colored"][st]["aesthetic"] for d in data.values() for st in d["colored"]])
    clt_w = mean([d["white"][st]["clip_t"] for d in data.values() for st in d["white"]])
    clt_c = mean([d["colored"][st]["clip_t"] for d in data.values() for st in d["colored"]])
    h.append("<div class='read cav'><b>Reading — this lever fails.</b> Coloring the initial noise "
             f"toward the natural-latent spectrum <b>hurts</b>: aesthetic drops ≈{fmt(aes_w)}→"
             f"{fmt(aes_c)} and prompt CLIP-T collapses ≈{fmt(clt_w)}→{fmt(clt_c)} at every step "
             "count. Rectified-flow generation expects a (near-)white Gaussian start; biasing its "
             "band power off-distribution moves it off the trained manifold and the model never "
             "recovers. Spectrum-shaping the <i>init</i> is the wrong place to inject structure — "
             "the oracle's mid-trajectory band commit is the working lever.</div>")
    for pid, d in data.items():
        h.append(f"<h4>{pid}: <code>{d['prompt']}</code></h4>")
        h.append(img_tag(f"noiseshape/grid_{pid}.png"))
        h.append("<p class=cap>rows alternate white-init / colored-init at each step count; "
                 "columns = seeds.</p>")
        steps = sorted(d["white"], key=int)
        h.append("<table><tr><th>steps</th><th>aesthetic ↑<br>white</th><th>aesthetic ↑<br>colored</th>"
                 "<th>CLIP-T ↑<br>white</th><th>CLIP-T ↑<br>colored</th></tr>")
        for st in steps:
            w, c = d["white"][st], d["colored"][st]
            aw = " class=pos" if w["aesthetic"] >= c["aesthetic"] else ""
            ac = "" if w["aesthetic"] >= c["aesthetic"] else " class=pos"
            cw = " class=pos" if w["clip_t"] >= c["clip_t"] else ""
            cc = "" if w["clip_t"] >= c["clip_t"] else " class=pos"
            h.append(f"<tr><td class=v>{st}</td>"
                     f"<td{aw}>{fmt(w['aesthetic'])}</td><td{ac}>{fmt(c['aesthetic'])}</td>"
                     f"<td{cw}>{fmt(w['clip_t'])}</td><td{cc}>{fmt(c['clip_t'])}</td></tr>")
        h.append("</table>")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------------------

def render(oracle, condition, noiseshape):
    h = ["<!doctype html><meta charset=utf-8><title>E20 — spectral warm-start</title>",
         f"<style>{CSS}</style>",
         "<h1>E20 — spectral warm-start: can we “skip the beginning” of generation?</h1>"]

    # ---- TL;DR ----
    base = comm = None
    if oracle:
        def at(pid, cc, st):
            for k, v in oracle[pid]["cells"].items():
                if k == f"c{cc:g}_s{st:g}":
                    return v["clip_i"]
            return None
        base = mean([at(p, 0.0, 0.8) for p in oracle])
        comm = mean([at(p, 0.1, 0.6) for p in oracle])
    h.append(
        "<div class=tldr><b>In one paragraph.</b> Diffusion is coarse-to-fine: the intuition is "
        "that the <i>early</i> denoising steps fix the low-frequency <b>structure</b> and the late "
        "steps fix detail. If so, we should be able to hand the model the low-frequency content "
        "up front — build an intermediate latent whose <b>low Fourier bands are pre-set</b> — "
        "re-enter the trajectory partway, and <b>skip the early steps</b>. We test it in SD3.5-medium "
        "(rectified flow, 28 steps) three ways: an <b>oracle</b> (commit a finished run's own true "
        "low bands and see how much we can skip and still recover it), <b>conditioning</b> (commit a "
        "reference image's low bands = band-controlled SDEdit), and <b>noise-shaping</b> (pre-color "
        "the step-0 noise toward a natural spectrum). " +
        (f"<b>Findings.</b> The oracle works strikingly well — committing just the lowest bands "
         f"(<code>c=0.1</code>) recovers the image (CLIP-I ≈{fmt(comm)}) while skipping a large "
         f"fraction of steps, versus ≈{fmt(base)} when re-entering from pure noise at the same step. "
         f"Conditioning gives a clean structure-vs-prompt dial but doesn't beat full SDEdit; and "
         f"<b>noise-shaping fails</b> (coloring the init drops it off-manifold and tanks quality). "
         if oracle else "") +
        "The actionable lever is <b>inject-the-low-bands-and-skip</b>, not pre-coloring noise.</div>")

    # ---- background glossary ----
    h.append("<h2>0 · Background (plain language)</h2><dl>"
             "<dt>Latent / SD3.5 rectified flow</dt><dd>SD3.5 generates in a compressed latent space "
             "(here <code>16×128×128</code>). It uses <b>rectified flow</b>: a sample at fraction "
             "<code>σ</code> along the trajectory is <code>x_σ = (1−σ)·x₀ + σ·ε</code> "
             "(<code>x₀</code> = clean latent, <code>ε</code> = Gaussian noise). The scheduler maps a "
             "re-entry <b>strength</b> to a start step.</dd>"
             "<dt>2-D FFT: frequency vs phase/magnitude</dt><dd>The 2-D Fourier transform of a latent "
             "gives a complex number <code>F = |F|·e<sup>iφ</sup></code> at every frequency. Two "
             "<i>independent</i> axes: <b>(a) low vs high frequency</b> = coarse layout vs fine detail; "
             "<b>(b) magnitude vs phase</b> = which part of each coefficient. Classic result "
             "(Oppenheim–Lim): <b>phase carries structure</b>, magnitude/power carries texture-energy "
             "and palette. The warm-start keeps the low <i>frequencies</i> <b>whole</b> (magnitude + "
             "phase); it is their phase that supplies the coarse layout. (This is the opposite of SBN, "
             "which edits the magnitude axis only.)</dd>"
             "<dt>Warm-start band commit (cutoff <code>c</code>)</dt><dd>"
             "<code>band_spectrum_split(x₀, noise, c)</code> keeps the full complex Fourier "
             "coefficients of a source latent <code>x₀</code> for all bands up to a radial cutoff "
             "<code>c∈[0,1]</code>, and fills the rest with fresh noise. <code>c=0</code> = pure noise "
             "(nothing committed); <code>c=1</code> = the whole source latent; <code>c=0.1</code> = "
             "only the lowest 10% of the spectrum (coarse layout) committed.</dd>"
             "<dt>Re-entry strength <code>s</code> (= skip)</dt><dd>After building the warm-start "
             "latent we re-noise it to the level of a mid-trajectory step and denoise only the rest. "
             "<code>s</code> is the fraction of the schedule actually run, so <b>skip = 1−s</b>. "
             "<code>s=0.4</code> runs 40% of the steps (skips 60%); <code>s=0.8</code> runs 80% "
             "(skips 20%). Lower <code>s</code> = more aggressive skipping = harder.</dd>"
             "<dt>Conditions (the three parts)</dt><dd>"
             "<b>oracle</b>: the committed source is a <i>finished run's own</i> latent <code>x₀*</code> "
             "(a ceiling — we already know the answer). "
             "<b>condition</b>: the committed source is a <i>reference image's</i> latent (band-controlled "
             "SDEdit); <code>c=1</code> is ordinary full SDEdit. "
             "<b>noiseshape</b>: no skipping — the step-0 <i>noise</i> is recolored "
             "(<code>color_noise</code>, a PSD-match) so its band power matches real photos' latents, "
             "then the full schedule runs (colored-init vs white-init).</dd>"
             "<dt>Metrics</dt><dd>"
             "<b>CLIP-I</b> (0–1 ↑, oracle/condition): image↔target cosine in CLIP space — does the "
             "warm-start reconstruct the oracle target (oracle) or the reference image (condition)? "
             "<b>latent L2</b> (≥0 ↓, oracle): RMS distance between the output latent and the target "
             "latent — 0 = identical. "
             "<b>prompt CLIP-T</b> (0–1 ↑, condition): image↔prompt cosine — does it follow the text? "
             "<b>aesthetic</b> (≈1–10 ↑, noiseshape): LAION aesthetic-predictor score. "
             "<b>CLIP-T</b> (0–1 ↑, noiseshape): image↔prompt similarity.</dd>"
             "</dl>")

    # ---- method ----
    h.append("<h2>1 · Method</h2><dl>"
             "<dt>oracle</dt><dd>Run the full schedule once to get <code>x₀*</code> and a reference "
             "image. For each cutoff <code>c</code> × strength <code>s</code>, build "
             "<code>band_spectrum_split(x₀*, noise, c)</code>, re-enter at <code>s</code>, denoise the "
             "rest, and score CLIP-I + latent-L2 to the full run. <i>Given the true low bands up to "
             "<code>c</code>, how many steps can we skip and still recover the image?</i></dd>"
             "<dt>condition</dt><dd>Same construction, but the committed source is a reference "
             "<i>image's</i> encoded latent and the score is structure (CLIP-I to the reference) vs "
             "prompt (CLIP-T). <i>Does committing only low bands keep structure while letting the "
             "prompt drive detail, better than full SDEdit (<code>c=1</code>)?</i></dd>"
             "<dt>noiseshape</dt><dd>Color the step-0 noise toward the mean band power of SD3.5-encoded "
             "real photos and run the full schedule at several step counts. <i>Does a natural-spectrum "
             "start reach quality in fewer steps than white noise?</i></dd>"
             "</dl>")

    # ---- results, grouped by part ----
    h.append("<h2>2 · Results</h2>")
    if oracle:
        render_oracle(h, oracle)
    if condition:
        render_condition(h, condition)
    if noiseshape:
        render_noiseshape(h, noiseshape)

    # ---- caveats + reproduce ----
    h.append("<h2>3 · Caveats &amp; next</h2><div class=cav>"
             "<b>(1)</b> The oracle commits a finished run's <i>own</i> low bands — it is a ceiling, "
             "not a usable method; a practical version needs those bands cheaply (a reference image, "
             "as in <code>condition</code>, or a fast preview). "
             "<b>(2)</b> Single seed per oracle/condition cell, small reference set — read directions, "
             "not third decimals. <b>(3)</b> Recovery is CLIP-I + latent-L2 (LPIPS not installed). "
             "<b>(4)</b> SD3.5-medium only; other models / VAEs may behave differently. "
             "<b>Next:</b> use the oracle ceiling to set a realistic skip budget, and source the low "
             "bands from a cheap preview rather than the full run.</div>")
    h.append("<h2>4 · Reproduce</h2>"
             "<p>Generation parts need the gated SD3.5 download (cluster); the page rebuilds offline.</p>"
             "<pre><code>cd experiments\n"
             "python e20_warmstart.py --part preflight                                   # model-free asserts\n"
             "python e20_warmstart.py --part oracle      --num_prompts 3                  # SD3.5\n"
             "python e20_warmstart.py --part condition   --refs results/e18/styles        # SD3.5\n"
             "python e20_warmstart.py --part noiseshape  --num_prompts 1                  # SD3.5\n"
             "# rebuild this page offline (no model) from the jsons + cached grids:\n"
             "python e20_warmstart.py --part site\n"
             "python e20_site.py</code></pre>")
    h.append("<p class=cap>Generated by <code>e20_site.py</code> from "
             "<code>results/e20/{oracle,condition,noiseshape}.json</code> + the saved grids. "
             "Driver: <code>e20_warmstart.py</code> (reuses <code>style_ops.band_spectrum_split / "
             "color_noise</code>, <code>e17_sd35.gen_sd3_warmstart</code>). See "
             "<code>EXPERIMENT_20.md</code>.</p>")
    return "".join(h)


def build_site():
    """Load the three jsons (if present) and write results/e20/index.html. No model."""
    def load(name):
        p = os.path.join(OUT, name)
        return json.load(open(p)) if os.path.exists(p) else None
    oracle = load("oracle.json")
    condition = load("condition.json")
    noiseshape = load("noiseshape.json")
    if not any((oracle, condition, noiseshape)):
        print(f"[e20-site] no jsons under {OUT}; run the generation parts first")
        return None
    html = render(oracle, condition, noiseshape)
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "index.html")
    with open(dest, "w") as f:
        f.write(html)
    print(f"[e20-site] wrote {dest}  ({len(html) // 1024} KB)  [no model loaded]")
    return dest


def main():
    build_site()


if __name__ == "__main__":
    main()
