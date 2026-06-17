"""Build a self-contained HTML explainer for E27 (concept directions in seed space).

Reads results/e27/{report.json, grid_direction.png, grid_anchors.png, grid_heavy.png,
deltaclip.png} and EMBEDS the images as base64 so the page is fully portable (open
results/e27/index.html anywhere). Honors CN_RESULTS.

The page STANDS ALONE to the project explainer standard (experiments/EXPLAINER_STANDARD.md):
TL;DR -> a glossary that DEFINES EVERY TERM (seed, the ||z||=sqrt(d) sphere, CLIP, the
two-stage CLIP->latent direction + its chain-rule equivalence, each anchor
chain/noise/mean/fit/nofit, Arm A additive strength s, Arm B heavy optimization N,
deltaCLIP) -> Method (each part + the question it answers) -> Results, ONE subsection per
arm/figure, each as figure-first -> "what to look for" -> interpretation -> a numbers table
with the best cell highlighted. (memory: experiment-documentation-standard.)

This module ALSO exports `data_uri`, imported by the other eNN_site pages; its name and
signature are part of the project API and must stay intact.

    python e27_site.py                          # rebuild results/e27/index.html
    python experiments/e27_seeddir.py --part site   # same, model-free, the canonical path
"""
import base64
import json
import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

OUT = os.path.join(RESULTS, "e27")

# Reuse the E29/E30 explainer CSS verbatim (.tldr/.look/.read/.win/.cav, glossary dl, td.pos).
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


def data_uri(path, max_px=1600, quality=85):
    """Read an image and return a base64 data: URI (downscaled JPEG). Imported by other
    eNN_site pages too -- keep this name/signature stable."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_px:
        r = max_px / max(im.size)
        im = im.resize((round(im.width * r), round(im.height * r)))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def img_tag(name, **kw):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return f"<p class=cap>(missing <code>{name}</code>)</p>"
    return f"<img src='{data_uri(p, **kw)}' alt='{name}'>"


def fmt(x, n=3, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(rep):
    """Compute mean ΔCLIP-T tables + the mean pairwise v-cos matrix from the per-prompt
    records. Δ = aligned − baseline, averaged over all (prompt, seed) cells."""
    cfg = rep["config"]
    prompts = rep["prompts"]
    seeds = [(p, s) for p in prompts for s in prompts[p]["seeds"]]

    def srec(p, s):
        return prompts[p]["seeds"][s]

    # Arm A: ΔCLIP-T by strength s (relative to the s=0 baseline column)
    s_keys = [f"{s:+.2f}" for s in cfg["s_sweep"]]
    base_key = "+0.00"
    armA = {}
    for sk in s_keys:
        armA[sk] = mean([srec(p, s)["armA_clip_by_s"][sk]
                         - srec(p, s)["armA_clip_by_s"][base_key] for p, s in seeds])
    # Arm B: ΔCLIP-T by N optimization steps (relative to N=0)
    armB = {}
    for n in cfg["heavy_n"]:
        armB[str(n)] = mean([srec(p, s)["armB_clip_by_N"][str(n)]
                             - srec(p, s)["armB_clip_by_N"]["0"] for p, s in seeds])
    # anchors: ΔCLIP-T vs baseline at the fixed anchor strength
    anchors = cfg["anchors"]
    anc = {}
    for a in anchors:
        anc[a] = mean([srec(p, s)["anchor_clip"][a] - srec(p, s)["anchor_clip"]["baseline"]
                       for p, s in seeds])
    # mean pairwise cosine between the latent directions v (across prompts)
    allkeys = list(dict.fromkeys(["chain"] + anchors))
    cosm = {a: {b: mean([prompts[p]["v_cos_matrix"][a][b] for p in prompts])
                for b in allkeys} for a in allkeys}
    return cfg, s_keys, base_key, armA, armB, anchors, anc, allkeys, cosm


def render(rep):
    cfg, s_keys, base_key, armA, armB, anchors, anc, allkeys, cosm = summarize(rep)
    sd = cfg["sqrt_d"]
    anchor_s = cfg.get("anchor_s", 0.25)
    nprompts = len(rep["prompts"])
    nseeds = len(next(iter(rep["prompts"].values()))["seeds"]) if rep["prompts"] else 0
    best_s = max(((k, armA[k]) for k in s_keys if k != base_key), key=lambda t: (t[1] or -9))
    best_N = max(((str(n), armB[str(n)]) for n in cfg["heavy_n"] if n != 0),
                 key=lambda t: (t[1] or -9))

    h = ["<!doctype html><meta charset=utf-8><title>E27 — concept directions in seed space</title>",
         f"<style>{CSS}</style>",
         "<h1>E27 — a single <em>concept direction</em> in the diffusion seed, via a "
         "CLIP→latent pullback</h1>"]

    # ---- TL;DR ----
    h.append(
        "<div class=tldr><b>In one paragraph.</b> A diffusion model starts from a random "
        "<b>seed</b> (a Gaussian noise array) and the seed leaves traces in the final image. So "
        "we ask: can we compute <b>one direction per prompt</b> in seed space that means “more of "
        "this prompt”, and just <b>add</b> it to any seed? We build the direction in two stages — "
        "find a direction in CLIP image space that points toward the text, then <b>pull it back "
        "through the image decoder</b> into latent (seed) space — and keep the seed Gaussian by "
        "re-standardizing (so it stays on the <code>‖z‖=√d</code> sphere a real seed lives on). "
        "We then (A) add the direction at various strengths <code>s</code> and (B) compare against "
        "running the per-seed optimization hard. <b>Findings:</b> the two stages collapse to a "
        "<b>single chain-rule backward pass</b>, and the <b>anchor barely matters</b> (every anchor "
        "gives essentially the same latent direction). The single additive direction is <b>too "
        "blunt</b>: a gentle push is break-even (no measurable CLIP-T change), a large one "
        f"(<code>s≈1</code>, the added vector ≈ the seed's own size) destroys the image. The heavy "
        "iterative version (Arm B) is well-behaved but gives <b>palette/appearance, not "
        "composition</b> — CLIP-T stays flat. Consistent with E25/E26: the seed's trace is an "
        "appearance signal, not a composition lever.</div>")

    # ---- background / glossary ----
    h.append("<h2>0 · Background (plain language)</h2><dl>"
             "<dt>Seed / latent</dt><dd>Image diffusion works in a compressed <b>latent</b> space. "
             f"For SDXL a <b>seed</b> is a <code>4×128×128</code> Gaussian noise array (dimension "
             f"<code>d={cfg['d']}</code>); the model denoises it into the final "
             f"<code>{cfg['size']}×{cfg['size']}</code> image. The same seed produces the same "
             "image, and its statistics leave traces in the output.</dd>"
             "<dt>The ‖z‖=√d sphere</dt><dd>A standard Gaussian vector in <code>d</code> dimensions "
             f"has norm ≈ <code>√d={sd:.0f}</code> (since <code>‖z‖²=d·(var+mean²)</code> and a "
             "real seed has var≈1, mean≈0). We keep every seed we make exactly on that sphere by "
             "<b>re-standardizing</b> to zero-mean/unit-variance after any edit "
             "(<code>renorm(z)=(z−mean)/std</code>). So our edits are moves <i>along the sphere "
             "real seeds live on</i>, not off into low-probability noise — this is what makes the "
             "edit safe (cf. E25/E26).</dd>"
             "<dt>CLIP</dt><dd>A model with an <b>image encoder</b> and a <b>text encoder</b> that "
             "map into one shared space; <b>cosine similarity</b> there measures how well an image "
             "matches a text. We use it both to build the direction and to score results.</dd>"
             "<dt>Decoder Jacobian <code>J</code></dt><dd><code>decode</code> turns a latent into "
             "pixels; <code>CLIP_image∘decode</code> turns a latent into a CLIP vector. Its "
             "Jacobian <code>J</code> says how a small latent change moves the CLIP vector; "
             "<code>Jᵀ</code> (one backward pass) turns a desired CLIP direction into the latent "
             "direction that best produces it.</dd>"
             "<dt>The two-stage direction</dt><dd><b>Stage 1 (CLIP space)</b>: a unit direction "
             "<code>g</code> that raises <code>cosine(image-embedding, CLIP_text(c))</code>, taken "
             "as the one-step cosine gradient at a base image's embedding <code>e₀</code>: "
             "<code>g = normalize(e_text − ⟨e_text,e₀⟩·e₀)</code> (the part of the text direction "
             "not already in the base image). <b>Stage 2 (decoder pullback)</b>: the latent "
             "direction whose decoded image moves along <code>g</code>: "
             "<code>v = normalize(∇_z⟨CLIP_image(decode(z_base)),g⟩) = normalize(Jᵀg)</code>. Both "
             "are unit vectors.</dd>"
             "<dt>Chain-rule equivalence</dt><dd>Composing the two stages is a <b>single backward "
             "pass</b>: <code>v_chain = normalize(∇_z cosine(CLIP_image(decode(z)), text))</code>. "
             "The intermediate normalization of <code>g</code> is irrelevant because we normalize "
             "<code>v</code> at the end. So the <b>only substantive choice is where <code>g</code> "
             "is anchored</b> — the base image <code>e₀</code> — which we sweep:</dd>"
             "<dt>chain</dt><dd><code>g = e_text</code> anchored at the base latent's <i>own</i> "
             "decoded image — the pure chain-rule gradient.</dd>"
             "<dt>noise</dt><dd><code>e₀</code> from random-pixel images.</dd>"
             "<dt>mean</dt><dd><code>e₀</code> from the <i>mean</i> of a small image pool (a "
             "gray-ish prior).</dd>"
             "<dt>fit</dt><dd><code>e₀</code> from an image that <b>matches</b> the prompt.</dd>"
             "<dt>nofit</dt><dd><code>e₀</code> from an image that <b>does not</b> match the "
             "prompt.</dd>"
             "<dt>Arm A — additive direction (strength s)</dt><dd>Apply the direction once: "
             "<code>z' = renorm(z₀ + s·√d·v)</code>. Here <b><code>s</code> is the ratio of the "
             "added vector's norm to the seed's own norm</b>, so <code>s=1</code> is a ~45° tilt of "
             "the seed (expected to be destructive); the useful regime is small <code>s</code>. The "
             f"sweep is <code>{cfg['s_sweep']}</code> (note the one negative −v column).</dd>"
             "<dt>Arm B — heavy optimization (N)</dt><dd>For contrast, instead of one fixed jump we "
             "<b>iterate the seed itself</b> for <code>N</code> steps on the same "
             "decode→CLIP→cosine objective, re-standardizing each step (E25/E26 latent-mode taken "
             f"hard). Sweep <code>N={cfg['heavy_n']}</code>. Each step is a small, re-projected, "
             "on-manifold move — which is why it doesn't blow up like Arm A.</dd>"
             "<dt>ΔCLIP-T (↑)</dt><dd>The metric. <b>CLIP-T</b> = cosine(generated image, prompt "
             "text) in CLIP space (image↔prompt match). <b>Δ</b> = aligned − baseline (the edited "
             "seed minus the unedited seed), so <b>Δ&gt;0 means the edit moved the image toward the "
             "prompt</b> (higher is better). Means are over all prompt×seed cells.</dd>"
             "</dl>")

    # ---- method ----
    h.append("<h2>1 · Method</h2><dl>"
             f"<dt>Setup</dt><dd>SDXL at <code>{cfg['size']}px</code>, "
             f"{nprompts} medium scenes × {nseeds} seeds, "
             f"{cfg.get('gen_steps', '?')}-step generation at guidance "
             f"<code>{cfg.get('guidance', '?')}</code>. The pullback direction is averaged over "
             f"<code>B={cfg.get('bases', '?')}</code> random base latents (the Jacobian is "
             "point-dependent, so averaging makes the direction more transferable). All heavy GPU "
             "ops have OOM-retry; the seed is re-standardized after every edit "
             "(<code>‖z‖≡√d</code> verified).</dd>"
             "<dt>Arm A — additive direction, strength sweep</dt><dd>Add the pure chain-rule "
             "direction <code>v_chain</code> at each <code>s</code> and re-score. <i>Is there a "
             "small <code>s</code> that reliably raises CLIP-T? Where does it collapse?</i></dd>"
             "<dt>Anchor comparison (fixed s)</dt><dd>At the fixed small "
             f"<code>s={anchor_s}</code>, build the direction from each anchor "
             "(chain/noise/mean/fit/nofit) and compare ΔCLIP-T <i>and</i> the pairwise cosines of "
             "the resulting latent directions. <i>Does the anchor choice change the "
             "direction?</i></dd>"
             "<dt>Arm B — heavy per-seed optimization</dt><dd>Iterate the seed for "
             "<code>N</code> steps and re-score. <i>What does “pushing on-manifold” do — does it "
             "raise CLIP-T, and does it change composition or just appearance?</i></dd>"
             "</dl>")

    h.append("<h2>2 · Results</h2>")
    h.append(f"<p class=cap>Means over {nprompts} prompts × {nseeds} seeds. Metric = CLIP-T "
             "(image vs prompt text), Δ = aligned − baseline (↑ better).</p>")

    # --- anchor comparison FIRST (it justifies using v_chain for the rest) ---
    h.append(f"<h3>Anchor comparison — does the anchor matter? (fixed s={anchor_s})</h3>")
    h.append(img_tag("grid_anchors.png"))
    h.append("<p class=cap>columns: baseline · chain · noise · mean · fit · nofit (each direction "
             f"added at the fixed small s={anchor_s}).</p>")
    h.append("<div class=look><b>What to look for.</b> If the columns all look nearly identical to "
             "each other (and the cosine matrix below is near 1), the <b>anchor choice barely "
             "changes the direction</b> — so “the prompt direction in seed space” is essentially "
             "anchor-independent and we can just use the pure chain-rule one.</div>")
    h.append("<div class='read'><b>Reading.</b> Every anchor yields nearly the same latent "
             "direction (cosines 0.89–0.97), all break-even on CLIP-T, with <code>chain</code> "
             "marginally best. The <code>fit</code> anchor (an image already of the prompt) "
             "deviates most and helps least — subtracting the already-on-prompt component changes "
             "<code>g</code> the most. <b>Conclusion: the anchor is almost irrelevant</b>; the rest "
             "of the experiment uses the pure chain-rule <code>v_chain</code>.</div>")
    h.append("<table><tr><th>anchor</th>" + "".join(f"<th>{a}</th>" for a in anchors)
             + "</tr><tr><td class=v>mean ΔCLIP-T ↑</td>")
    best_anc = max(anc.values()) if anc else None
    for a in anchors:
        v = anc[a]
        hot = best_anc is not None and v is not None and abs(v - best_anc) < 1e-9
        cls = "pos" if hot else ("neg" if (v or 0) < 0 else "")
        h.append(f"<td class={cls}>{fmt(v, sign=True)}</td>")
    h.append("</tr></table>")
    h.append("<p class=cap>Mean pairwise cosine between the latent directions <code>v</code> "
             "(across prompts) — high = anchor-independent:</p>")
    h.append("<table><tr><th></th>" + "".join(f"<th>{b}</th>" for b in allkeys) + "</tr>")
    for a in allkeys:
        h.append(f"<tr><td class=v>{a}</td>"
                 + "".join(f"<td>{fmt(cosm[a][b], 2)}</td>" for b in allkeys) + "</tr>")
    h.append("</table>")

    # --- Arm A ---
    h.append("<h3>Arm A — add the direction once, sweep strength s</h3>")
    h.append(img_tag("grid_direction.png"))
    h.append("<p class=cap>rows = prompt×seed; columns = the strength sweep "
             f"<code>{', '.join(s_keys)}</code> (incl. one −v column).</p>")
    h.append("<div class=look><b>What to look for.</b> Left→right is increasing push. A useful "
             "additive direction would make the image more on-prompt at some small <code>s</code> "
             "<i>without</i> wrecking it. Watch where structure survives and where it collapses to "
             "a washed-out / noisy frame.</div>")
    h.append("<div class='read'><b>Reading.</b> The single additive direction is <b>too "
             "blunt</b>: at gentle strength (<code>s≤0.25</code>) it is <b>break-even</b> — no "
             "measurable CLIP-T change, image visually unchanged; at <code>s=0.5</code> it washes "
             "the image out; at <code>s=1</code> (added vector ≈ the seed's own size, a ~45° tilt) "
             "it collapses to noise. There is no “free lunch” additive <code>s</code> that raises "
             "CLIP-T.</div>")
    h.append("<table><tr><th>s (·√d)</th>" + "".join(f"<th>{k}</th>" for k in s_keys) + "</tr><tr>"
             "<td class=v>mean ΔCLIP-T ↑</td>")
    best_a = max((armA[k] for k in s_keys if k != base_key
                  and armA[k] is not None), default=None)
    for k in s_keys:
        v = 0.0 if k == base_key else armA[k]
        if k == base_key:
            cls = ""
        elif best_a is not None and v is not None and abs(v - best_a) < 1e-9 and best_a > 0:
            cls = "pos"
        else:
            cls = "neg" if (v or 0) < 0 else ""
        h.append(f"<td class={cls}>{fmt(v, sign=(k != base_key))}</td>")
    h.append("</tr></table>")

    # --- Arm B ---
    h.append("<h3>Arm B — heavy per-seed optimization, sweep N steps</h3>")
    h.append(img_tag("grid_heavy.png"))
    h.append("<p class=cap>rows = prompt×seed; columns = number of optimization steps "
             f"<code>N={cfg['heavy_n']}</code> (N=0 = baseline).</p>")
    h.append("<div class=look><b>What to look for.</b> Does iterating the seed enhance the concept "
             "smoothly, and does it change <i>which objects appear</i> (composition) or only "
             "<i>palette / saturation / detail</i> (appearance)? Watch for over-optimization / "
             "CLIP-adversarial artifacts at large N.</div>")
    h.append("<div class='read'><b>Reading.</b> Iterating <i>on the sphere</i> is far "
             "better-behaved than the single additive jump: it <b>progressively intensifies "
             "palette / saturation / detail toward the concept while preserving composition</b>, "
             "even at the largest N (not destroyed). But CLIP-T stays ≈flat — it enhances "
             "<b>appearance, not object presence</b>. The reason it doesn't blow up like Arm A is "
             "the <b>per-step re-standardization</b> (each step is a small, re-projected, "
             "on-manifold move). This is the real positive use of seed-biasing: palette/appearance "
             "steering, not adherence.</div>")
    h.append("<table><tr><th>N steps</th>" + "".join(f"<th>{n}</th>" for n in cfg["heavy_n"])
             + "</tr><tr><td class=v>mean ΔCLIP-T ↑</td>")
    best_b = max((armB[str(n)] for n in cfg["heavy_n"] if n != 0
                  and armB[str(n)] is not None), default=None)
    for n in cfg["heavy_n"]:
        v = armB[str(n)]
        if n == 0:
            cls = ""
        elif best_b is not None and v is not None and abs(v - best_b) < 1e-9 and best_b > 0:
            cls = "pos"
        else:
            cls = "neg" if (v or 0) < 0 else ""
        h.append(f"<td class={cls}>{fmt(v, sign=(n != 0))}</td>")
    h.append("</tr></table>")

    # --- trends ---
    h.append("<h3>Trends (ΔCLIP-T summary)</h3>")
    h.append(img_tag("deltaclip.png"))
    h.append("<p class=cap>left: Arm A vs strength s (collapses past ~0.25); middle: Arm B vs N "
             "(≈flat); right: anchor bars vs baseline (all break-even).</p>")

    # ---- verdict / caveats ----
    h.append("<h2>3 · Reading of the result</h2>")
    h.append(
        "<div class='win'><b>What we see.</b> (1) The two stages are <b>one chain-rule backward "
        "pass</b>, confirmed, and the <b>anchor is almost irrelevant</b> (directions 0.89–0.97 "
        "correlated, <code>chain</code> marginally best). (2) A single additive concept-direction "
        "is <b>too blunt for adherence</b>: gentle = no effect, strong = destruction; the best "
        f"small-strength point here is <code>s={best_s[0]}</code> "
        f"(mean ΔCLIP-T {fmt(best_s[1], sign=True)}), within noise. (3) On-manifold iterative "
        "steering (Arm B) is the well-behaved version — it gracefully intensifies the concept's "
        f"palette/appearance up to many steps without wrecking structure (best "
        f"<code>N={best_N[0]}</code>, {fmt(best_N[1], sign=True)}) — but it shifts <b>appearance, "
        "not composition</b>, so CLIP-T stays flat. Consistent with E25/E26: <b>the seed's trace "
        "is a palette / global-appearance signal, not a composition one</b>.</div>")
    h.append("<h2>4 · Caveats &amp; next</h2>"
             "<div class=cav>(1) The direction <code>v</code> is computed at a few random base "
             "latents but applied to a different seed; the decoder Jacobian is point-dependent, so "
             "transfer is imperfect (averaging over bases mitigates). (2) CLIP-T is a coarse metric "
             "blind to dropped elements — the visual grids are the real evidence; the small "
             "ΔCLIP-T signs are within noise. (3) This is a linear edit in a highly nonlinear "
             "pipeline — expect it to shift palette/global-appearance more than composition. "
             "<b>Next:</b> E28 takes this to hard compositional prompts with a metric (B-VQA) that "
             "<i>does</i> see dropped elements, to ask whether biasing the seed can rescue a "
             "missing object.</div>")

    h.append("<h2>5 · Reproduce</h2>"
             "<pre><code>python experiments/e27_seeddir.py quick   # 1-prompt smoke\n"
             "python experiments/e27_seeddir.py          # full -> results/e27/{grids, report.json}\n"
             "python experiments/e27_seeddir.py --part site  # model-free rebuild of index.html\n"
             "</code></pre>")
    h.append("<p class=cap>Generated by <code>e27_site.py</code> from "
             "<code>results/e27/report.json</code> + grids. Method: <code>e27_seeddir.py</code> "
             "(reuses <code>e26_seedalign_sdxl.py</code>, <code>clip_sim.py</code>). See also "
             "<code>EXPERIMENT_27.md</code>, and E25/E26 for the lineage.</p>")
    return "".join(h)


def build():
    """Load report.json, render, write index.html. Returns the dest path or None if no data."""
    rpath = os.path.join(OUT, "report.json")
    if not os.path.exists(rpath):
        print(f"[e27-site] no {rpath}; run e27_seeddir.py (the generator) first — "
              "nothing to template, leaving index.html untouched")
        return None
    rep = json.load(open(rpath))
    html = render(rep)
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "index.html")
    with open(dest, "w") as f:
        f.write(html)
    print(f"[e27-site] wrote {dest}  ({len(html) // 1024} KB, no model loaded)")
    return dest


def main():
    build()


if __name__ == "__main__":
    main()
