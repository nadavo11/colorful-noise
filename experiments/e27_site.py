"""Build a self-contained HTML explainer for E27 (concept directions in seed space).

Reads results/e27/{report.json, grid_direction.png, grid_anchors.png, grid_heavy.png,
deltaclip.png} and EMBEDS the images as base64 so the page is fully portable (open
results/e27/index.html anywhere). Honors CN_RESULTS.

The page STANDS ALONE: it defines every term (seed, latent, the sqrt(d) sphere, CLIP,
the two-stage direction, the chain-rule equivalence, the anchors, both arms) and shows
the grids + a results table computed from report.json. (memory: experiment-documentation-standard.)

    python e27_site.py
"""
import base64
import json
import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

OUT = os.path.join(RESULTS, "e27")

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


def img_tag(name, cls="grid", **kw):
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


def summarize(rep):
    """Compute mean ΔCLIP-T tables + mean v-cos matrix from the per-prompt records."""
    cfg = rep["config"]
    prompts = rep["prompts"]
    seeds = [(p, s) for p in prompts for s in prompts[p]["seeds"]]

    def srec(p, s):
        return prompts[p]["seeds"][s]

    # Arm A: ΔCLIP-T by s
    s_keys = [f"{s:+.2f}" for s in cfg["s_sweep"]]
    base_key = "+0.00"
    armA = {}
    for sk in s_keys:
        armA[sk] = mean([srec(p, s)["armA_clip_by_s"][sk]
                         - srec(p, s)["armA_clip_by_s"][base_key] for p, s in seeds])
    # Arm B: ΔCLIP-T by N
    armB = {}
    for n in cfg["heavy_n"]:
        armB[str(n)] = mean([srec(p, s)["armB_clip_by_N"][str(n)]
                             - srec(p, s)["armB_clip_by_N"]["0"] for p, s in seeds])
    # anchors: ΔCLIP-T vs baseline at fixed s
    anchors = cfg["anchors"]
    anc = {}
    for a in anchors:
        anc[a] = mean([srec(p, s)["anchor_clip"][a] - srec(p, s)["anchor_clip"]["baseline"]
                       for p, s in seeds])
    # mean pairwise v-cos matrix
    keys = ["chain"] + anchors[1:] if anchors and anchors[0] == "chain" else anchors
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

    h = ["<!doctype html><meta charset=utf-8><title>E27 — concept directions in seed space</title>",
         f"<style>{CSS}</style>",
         "<h1>E27 — a single <em>concept direction</em> in the diffusion seed, via a "
         "CLIP→latent pullback</h1>"]

    h.append(
        "<div class=tldr><b>In one paragraph.</b> A diffusion model starts from a random "
        "<b>seed</b> (a Gaussian noise array). The seed leaves traces in the final image, so we "
        "ask: can we compute <b>one direction per prompt</b> in seed space that means “more of "
        "this prompt”, and just <b>add</b> it to any seed? We build it in two stages — find a "
        "direction in CLIP image space that points toward the text, then <b>pull it back through "
        "the image decoder</b> into latent (seed) space — and keep the seed Gaussian by "
        "re-standardizing (so it stays on the <code>‖z‖=√d</code> sphere). We then (A) add the "
        "direction at various strengths and (B) compare it against running the optimization hard. "
        "Finding: the direction is gentle and break-even; a tiny push is fine, a large one "
        "(strength ≈ the seed's own size) destroys the image; the choice of where we anchor the "
        "CLIP direction barely matters (all anchors give nearly the same latent direction).</div>")

    # ---- background ----
    h.append("<h2>0 · Background (plain language)</h2>")
    h.append("<dl>"
             "<dt>Seed / latent</dt><dd>Image diffusion works in a compressed <b>latent</b> space. "
             f"For SDXL a seed is a <code>4×128×128</code> Gaussian array (dimension "
             f"<code>d={cfg['d']}</code>). The model denoises it into the final 1024×1024 image.</dd>"
             "<dt>The √d sphere</dt><dd>A standard Gaussian vector in d dimensions has norm ≈ "
             f"<code>√d={sd:.0f}</code> (since <code>‖z‖²=d·(var+mean²)</code>). We keep every seed "
             "we make exactly on that sphere by re-standardizing to zero-mean/unit-variance after "
             "any edit — so our edits are moves <i>along the sphere a real seed lives on</i>, not "
             "off into low-probability noise.</dd>"
             "<dt>CLIP</dt><dd>A model with an <b>image encoder</b> and a <b>text encoder</b> that "
             "map into one shared space; <b>cosine similarity</b> there measures image↔text match.</dd>"
             "<dt>Decoder Jacobian</dt><dd><code>decode</code> turns a latent into pixels; "
             "<code>CLIP_image∘decode</code> turns a latent into a CLIP vector. Its Jacobian "
             "<code>J</code> tells us how a small latent change moves the CLIP vector; "
             "<code>Jᵀ</code> (one backward pass) turns a desired CLIP direction into the latent "
             "direction that best produces it.</dd></dl>")

    # ---- method ----
    h.append("<h2>1 · The two-stage direction (and why it's one backward pass)</h2>")
    h.append("<dl>"
             "<dt>Stage 1 — CLIP-space direction <code>g</code></dt><dd>Take one gradient step that "
             "increases <code>cosine(image-embedding, CLIP_text(c))</code> from a base image's "
             "embedding <code>e₀</code>: <code>g = normalize(e_text − ⟨e_text,e₀⟩·e₀)</code> — the "
             "part of the text direction not already in the base image. Unit vector in CLIP space.</dd>"
             "<dt>Stage 2 — pull back to latent space <code>v</code></dt><dd>One gradient step so the "
             "decoded image moves along <code>g</code>: <code>v = normalize(∇_z ⟨CLIP_image(decode(z)), "
             "g⟩) = normalize(Jᵀg)</code>. Unit vector in latent (seed) space.</dd></dl>")
    h.append("<div class=note><b>Chain rule.</b> Composing the two stages is a single backward pass: "
             "<code>v = normalize(∇_z cosine(CLIP_image(decode(z)), text))</code>. The intermediate "
             "normalization of <code>g</code> is irrelevant because we normalize <code>v</code> at the "
             "end. The only real choice is <b>where g is anchored</b> (the base image <code>e₀</code>) "
             "— which we sweep below.</div>")
    h.append("<dl>"
             "<dt>chain</dt><dd><code>g = e_text</code> anchored at the base latent's own decoded "
             "image (the pure chain-rule gradient).</dd>"
             "<dt>noise</dt><dd><code>e₀</code> from random-pixel images.</dd>"
             "<dt>mean</dt><dd><code>e₀</code> from the mean of a small image pool (a gray-ish prior).</dd>"
             "<dt>fit</dt><dd><code>e₀</code> from an image that <b>matches</b> the prompt.</dd>"
             "<dt>nofit</dt><dd><code>e₀</code> from an image that <b>does not</b> match the prompt.</dd>"
             "</dl>")
    h.append("<dl><dt>Applying it (Arm A)</dt><dd><code>z' = renorm(z₀ + s·√d·v)</code>. Here "
             "<b>s is the ratio of the added vector's norm to the seed's own norm</b>, so "
             "<code>s=1</code> is a ~45° tilt of the seed (expected to be destructive); the useful "
             "regime is small s.</dd>"
             "<dt>Heavy optimization (Arm B)</dt><dd>For contrast, instead of one fixed direction we "
             "iterate the seed itself for N steps on the same CLIP objective (re-standardizing each "
             "step) — to see what “pushing hard” does.</dd></dl>")

    # ---- results: numbers ----
    h.append("<h2>2 · Results</h2>")
    h.append(f"<p class=cap>Means over {nprompts} prompts × {nseeds} seeds (SDXL "
             f"{cfg['size']}px, metric = CLIP-T, image vs prompt text). Δ = aligned − baseline.</p>")

    h.append("<h3>Arm A — add the direction, strength sweep</h3>")
    h.append("<table><tr><th>s (·√d)</th>" + "".join(f"<th>{k}</th>" for k in s_keys) + "</tr><tr>"
             "<td class=v>mean ΔCLIP-T</td>")
    for k in s_keys:
        v = 0.0 if k == base_key else armA[k]
        cls = "" if k == base_key else ("pos" if (v or 0) > 0 else "neg")
        h.append(f"<td class={cls}>{fmt(v, sign=(k!=base_key))}</td>")
    h.append("</tr></table>")
    h.append(img_tag("grid_direction.png"))
    h.append("<p class=cap>Rows = prompt×seed; columns = the strength sweep (incl. one −v column). "
             "Small s = gentle nudge; large s wrecks structure.</p>")

    h.append("<h3>Arm B — heavy per-seed optimization</h3>")
    h.append("<table><tr><th>N steps</th>" + "".join(f"<th>{n}</th>" for n in cfg["heavy_n"])
             + "</tr><tr><td class=v>mean ΔCLIP-T</td>")
    for n in cfg["heavy_n"]:
        v = armB[str(n)]
        cls = "" if n == 0 else ("pos" if (v or 0) > 0 else "neg")
        h.append(f"<td class={cls}>{fmt(v, sign=(n!=0))}</td>")
    h.append("</tr></table>")
    h.append(img_tag("grid_heavy.png"))
    h.append("<p class=cap>Columns = number of optimization steps N (0 = baseline). Watch for "
             "over-optimization / CLIP-adversarial artifacts at large N.</p>")

    h.append(f"<h3>Anchor comparison (fixed s = {anchor_s})</h3>")
    h.append("<table><tr><th>anchor</th>" + "".join(f"<th>{a}</th>" for a in anchors)
             + "</tr><tr><td class=v>mean ΔCLIP-T</td>")
    for a in anchors:
        v = anc[a]
        h.append(f"<td class={'pos' if (v or 0)>0 else 'neg'}>{fmt(v, sign=True)}</td>")
    h.append("</tr></table>")
    h.append("<p><b>How similar are the resulting latent directions?</b> Mean pairwise cosine "
             "between the <code>v</code>'s across prompts:</p>")
    h.append("<table><tr><th></th>" + "".join(f"<th>{b}</th>" for b in allkeys) + "</tr>")
    for a in allkeys:
        h.append(f"<tr><td class=v>{a}</td>"
                 + "".join(f"<td>{fmt(cosm[a][b], 2)}</td>" for b in allkeys) + "</tr>")
    h.append("</table>")
    h.append(img_tag("grid_anchors.png"))
    h.append("<p class=cap>Columns = baseline, then the direction from each anchor at the fixed "
             "small s. If the cosines are high, the anchor choice barely changes the direction.</p>")

    h.append("<h3>Trends</h3>")
    h.append(img_tag("deltaclip.png", cls="plot"))

    # ---- verdict / caveats ----
    best_s = max(((k, armA[k]) for k in s_keys if k != base_key),
                 key=lambda t: (t[1] or -9))
    h.append("<h2>3 · Reading of the result</h2>")
    h.append(
        "<div class='note win'><b>What we see.</b> A single per-prompt direction added to the seed "
        "is a <b>gentle, mostly do-no-harm</b> edit at small strength and a <b>destructive</b> one "
        "once the added vector approaches the seed's own size (s→1, a ~45° tilt). The Stage-1 "
        f"anchor barely matters — the latent directions are highly correlated (see the cosine "
        "matrix), so “the prompt direction in seed space” is essentially anchor-independent. Best "
        f"small-strength point here: <code>s={best_s[0]}</code> (mean ΔCLIP-T {fmt(best_s[1], sign=True)}).</div>")
    h.append("<div class='note cav'><b>Caveats.</b> (1) The direction <code>v</code> is computed at "
             "a few random base latents but applied to a different seed; the decoder Jacobian is "
             "point-dependent, so transfer is imperfect (averaging over bases mitigates). "
             "(2) CLIP-T is a coarse metric; the visual grids are the real evidence. "
             "(3) This is a linear edit in a highly nonlinear pipeline — expect it to shift "
             "palette/global-appearance more than composition (consistent with E25/E26).</div>")

    h.append("<p class=cap>Generated by <code>e27_site.py</code> from "
             "<code>results/e27/report.json</code> + grids. Method: <code>e27_seeddir.py</code> "
             "(reuses <code>e26_seedalign_sdxl.py</code>, <code>clip_sim.py</code>). See also "
             "<code>EXPERIMENT_27.md</code>, and E25/E26 for the lineage.</p>")
    return "".join(h)


def main():
    rpath = os.path.join(OUT, "report.json")
    if not os.path.exists(rpath):
        print(f"[e27-site] no {rpath}; run e27_seeddir.py first")
        return
    rep = json.load(open(rpath))
    html = render(rep)
    dest = os.path.join(OUT, "index.html")
    with open(dest, "w") as f:
        f.write(html)
    print(f"[e27-site] wrote {dest}  ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
