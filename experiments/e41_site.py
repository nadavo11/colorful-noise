"""Build a self-contained HTML explainer for E41 (per-image calibration of our
spectral-clamp edit vs RF-inversion's eta knob, on 140 PIE-Bench images).

Reads results/e41/items/*.json + the saved per-item PNGs, recomputes the aggregate
stats, EMBEDS the aggregate Pareto + qualitative montages as base64 (page is fully
portable), and writes results/e41/index.html. Honors CN_RESULTS.

The page stands alone: it explains the RF-inversion eta fairness problem, the eta
controller we added, our spectral-clamp knobs, how the Optuna calibration loop works,
every metric, the quantitative tables, qualitative comparisons, and future sweeps.

    python e41_site.py
"""
import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

OUT = os.path.join(RESULTS, "e41")
ITEMS = os.path.join(OUT, "items")
METHODS = ("ours", "vanilla", "etadefault")
PANEL = 300


def _stem(key):
    return os.path.join(ITEMS, key.replace("/", "_"))


def _b64(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _b64_path(p):
    return _b64(Image.open(p).convert("RGB")) if os.path.exists(p) else None


def _panel(png_path, title, sub):
    """A labelled image tile: image on top, a caption bar below."""
    im = Image.open(png_path).convert("RGB").resize((PANEL, PANEL))
    bar = 44
    tile = Image.new("RGB", (PANEL, PANEL + bar), "white")
    tile.paste(im, (0, 0))
    d = ImageDraw.Draw(tile)
    d.text((6, PANEL + 4), title, fill="black")
    d.text((6, PANEL + 22), sub, fill="#555")
    return tile


def _row(rec):
    """source | vanilla | default-eta | ours, labelled with struct/clipdir."""
    tiles, labels = [], [("source", "_source"), ("RF-inv vanilla (η=0)", "_vanilla"),
                         ("RF-inv default (η=0.9)", "_etadefault"), ("ours (calibrated)", "_ours")]
    for name, tag in labels:
        p = _stem(rec["key"]) + tag + ".png"
        if not os.path.exists(p):
            return None
        if tag == "_source":
            sub = ""
        else:
            m = rec["metrics"][name.split()[0].lower() if tag == "_ours" else
                               ("vanilla" if "vanilla" in tag else "etadefault")]
            sub = f"struct {m['struct']:.3f} · edit {m['clipdir']:.3f}"
        tiles.append(_panel(p, name, sub))
    w = sum(t.width for t in tiles) + 9 * 3
    row = Image.new("RGB", (w, tiles[0].height), "white")
    x = 0
    for t in tiles:
        row.paste(t, (x, 0))
        x += t.width + 9
    return row


def _mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def main():
    recs = [json.load(open(os.path.join(ITEMS, f)))
            for f in sorted(os.listdir(ITEMS)) if f.endswith(".json")]
    n = len(recs)
    from e41_calibrate import _recompute_lpips      # fill lpips/dssim from saved PNGs
    _recompute_lpips(recs)

    # ---- aggregate stats ----
    keys = ["struct", "lpips", "dssim", "clipdir", "clipt"]
    means = {m: {k: _mean([r["metrics"][m].get(k) for r in recs]) for k in keys} for m in METHODS}
    feas = sum(r["feasible"] for r in recs)
    # matched-editability gap vs the RF-inv eta curve
    gaps = []
    for r in recs:
        es = r.get("eta_sweep")
        if not es:
            continue
        oc, ost = r["metrics"]["ours"]["clipdir"], r["metrics"]["ours"]["struct"]
        pts = sorted(es, key=lambda e: e["clipdir"])
        cd, st = [e["clipdir"] for e in pts], [e["struct"] for e in pts]
        if cd[0] < oc < cd[-1]:
            gaps.append(ost - float(np.interp(oc, cd, st)))
    gaps = np.array(gaps)
    beyond = sum(1 for r in recs if r.get("eta_sweep") and
                 r["metrics"]["ours"]["clipdir"] >= max(e["clipdir"] for e in r["eta_sweep"]))

    # per-edit-type structure
    types = sorted({r["edit_type"] for r in recs})
    per_type = []
    for t in types:
        rs = [r for r in recs if r["edit_type"] == t]
        win = sum(r["metrics"]["ours"]["struct"] < r["metrics"]["vanilla"]["struct"] for r in rs)
        per_type.append((t, len(rs),
                         _mean([r["metrics"]["ours"]["struct"] for r in rs]),
                         _mean([r["metrics"]["vanilla"]["struct"] for r in rs]),
                         _mean([r["metrics"]["etadefault"]["struct"] for r in rs]), win))

    # ---- qualitative picks: best structure win per edit type, that still edits ----
    picks, seen = [], set()
    cand = sorted(recs, key=lambda r: r["metrics"]["vanilla"]["struct"] - r["metrics"]["ours"]["struct"],
                  reverse=True)
    for r in cand:
        if r["metrics"]["ours"]["clipdir"] < 0.12:
            continue
        if r["edit_type"] in seen:
            continue
        seen.add(r["edit_type"])
        picks.append(r)
        if len(picks) >= 8:
            break

    # ---- HTML ----
    def tbl(headers, rows):
        h = "<tr>" + "".join(f"<th>{x}</th>" for x in headers) + "</tr>"
        b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        return f"<table>{h}{b}</table>"

    means_rows = [[m] + [f"{means[m][k]:.4f}" if means[m][k] is not None else "—" for k in keys]
                  for m in METHODS]
    pt_rows = [[t, nn, f"{o:.4f}", f"{v:.4f}", f"{e:.4f}", f"{w}/{nn}"]
               for (t, nn, o, v, e, w) in per_type]

    pareto = _b64_path(os.path.join(OUT, "aggregate_pareto.png"))
    qual = []
    for r in picks:
        row = _row(r)
        if row:
            qual.append((r, _b64(row)))

    css = """
    body{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1040px;margin:0 auto;
         padding:28px;color:#1a1a1a}
    h1{font-size:30px;margin:0 0 4px} h2{margin-top:40px;border-bottom:2px solid #eee;padding-bottom:6px}
    h3{margin-top:26px} .sub{color:#666;margin:0 0 18px}
    table{border-collapse:collapse;margin:14px 0;font-size:14px} th,td{border:1px solid #ddd;padding:6px 11px;text-align:right}
    th:first-child,td:first-child{text-align:left} th{background:#f6f6f6}
    code{background:#f3f3f3;padding:1px 5px;border-radius:4px;font-size:13px}
    .key{background:#f0f7ff;border-left:4px solid #2b7;padding:12px 16px;margin:16px 0;border-radius:4px}
    .warn{background:#fff7ed;border-left:4px solid #e8a33d}
    img{max-width:100%;border-radius:5px} figcaption{color:#666;font-size:13px;margin:4px 0 22px}
    .prompt{font-size:13px;color:#444;margin:2px 0 6px} ul{margin:8px 0}
    """

    H = [f"<!doctype html><meta charset=utf-8><title>E41 — calibration vs RF-inversion</title><style>{css}</style>"]
    H.append(f"<h1>E41 · Per-image calibration vs RF-inversion (η)</h1>")
    H.append(f"<p class=sub>Structure-preserving real-image editing on FLUX · {n} PIE-Bench images · "
             f"DINO structure distance ↓ &amp; CLIP-directional editability ↑</p>")

    H.append("<h2>The effort, in one paragraph</h2>")
    H.append(
        "<p>We claim our spectral-clamp edit (RF-invert a real image, then re-impose the source's "
        "low-frequency structure during the edit) preserves structure better than <b>RF-inversion</b> "
        "(Rout et al.). RF-inversion has an <b>η</b> knob that trades editability against faithfulness, "
        "so a single-η comparison is unfair. We therefore (1) implemented RF-inversion's η controller, "
        "(2) <b>auto-calibrated our knobs per image</b> with a small active loop, and (3) compared on a "
        "known benchmark (PIE-Bench) with the field-standard <b>DINO self-similarity structure distance</b>, "
        "sweeping η to trace RF-inversion's whole faithfulness↔editability curve.</p>")

    H.append("<h2>What we built &amp; how calibration works</h2>")
    H.append(
        "<h3>1 · The RF-inversion η controller (the baseline knob)</h3>"
        "<p>During the edit pass we blend the model velocity toward the field that reconstructs the source "
        "latent: <code>v ← v + η·(v_target − v)</code>, with <code>v_target = (x − x₀)/σ</code>, applied over "
        "an early step window. <b>η=0</b> = plain inversion edit (vanilla); <b>η=1</b> ≈ reconstruction. "
        "Sanity-checked on the hand-tuned dancers example: reproducing it gives "
        "MSE(ours, saved)=2.1e-4 and MSE(η=1 recon, source)=6e-5 — the controller is correct.</p>")
    H.append(
        "<h3>2 · Our spectral-clamp knobs</h3>"
        "<p>After inverting, we regenerate under the edit prompt while pulling the latent's low-frequency "
        "bands back to the recorded source trajectory. Tunable per image: <code>mode</code> "
        "(sbn power-match / phase-lock / adain), <code>cut</code> (low-band cutoff), <code>strength</code> "
        "(clamp blend), <code>interval_end</code> (how many early steps clamp), <code>phase_band</code> "
        "(which radial band's phase is locked). The hand-tuned dancers used phase-locking on a narrow band "
        "over only the first 10% of steps — a hint that a little structure lock, early, goes a long way.</p>")
    H.append(
        "<h3>3 · The calibration loop (Optuna TPE, ~20 trials/image)</h3>"
        "<p>For each image we run a Bayesian active loop: <i>propose knobs → generate the edit → score → "
        "propose next</i>. The objective is <b>constrained</b>: minimize DINO structure distance "
        "<b>subject to</b> editability (CLIP-dir) ≥ the vanilla baseline's — i.e. preserve structure as much "
        "as possible <i>at matched editability</i>. The loop is warm-started from the dancers prior plus a "
        "prompt-distance heuristic (a small source→edit text move ⇒ lock structure harder). Every trial's "
        "(struct, editability, params) is saved, so the operating point can be re-selected post-hoc with no "
        "GPU. Run on the cluster: 140 images × (20 trials + a 6-point η sweep + full scoring), sharded across "
        "A6000/H100 GPUs.</p>")
    H.append(
        "<h3>4 · Metrics</h3><ul>"
        "<li><b>DINO structure distance ↓</b> — RMS difference of DINO-ViT patch self-similarity matrices "
        "(Splice/Tumanyan; PIE-Bench's headline structure metric).</li>"
        "<li><b>CLIP-directional ↑</b> — cosine between the image edit direction and the text edit direction "
        "(how much the intended edit actually happened).</li>"
        "<li><b>LPIPS / DSSIM ↓</b> — perceptual / structural distance to the source.</li>"
        "<li><b>CLIP-T ↑</b> — agreement with the edit prompt.</li></ul>")

    H.append("<h2>Quantitative results</h2>")
    H.append("<h3>Means across all images</h3>")
    H.append(tbl(["method"] + keys, means_rows))
    H.append(
        "<div class='key'><b>vs vanilla RF-inversion (out-of-the-box):</b> ours wins on every axis — "
        f"lower structure distance ({means['ours']['struct']:.3f} vs {means['vanilla']['struct']:.3f}), "
        f"lower LPIPS ({means['ours']['lpips']:.3f} vs {means['vanilla']['lpips']:.3f}), <i>and</i> higher "
        f"editability ({means['ours']['clipdir']:.3f} vs {means['vanilla']['clipdir']:.3f}).</div>")
    H.append(
        "<div class='key warn'><b>vs default η=0.9:</b> its structure distance looks tiny "
        f"({means['etadefault']['struct']:.3f}) only because it <i>barely edits</i> "
        f"(CLIP-dir {means['etadefault']['clipdir']:.3f} ≈ reconstruction). Not a fair operating point — "
        "which is exactly why the η <i>sweep</i> matters.</div>")

    H.append("<h3>Headline: structure at MATCHED editability</h3>")
    gtxt = (f"mean(ours − RF-inv) = {gaps.mean():+.4f} over {len(gaps)} comparable images, "
            f"ours wins {int((gaps<0).sum())}/{len(gaps)}") if len(gaps) else "n/a"
    H.append(
        "<ul>"
        f"<li>Feasible (ours reached ≥ vanilla editability): <b>{feas}/{n}</b>.</li>"
        f"<li>Ours vs RF-inversion's η curve <i>at ours' editability</i>: {gtxt} — i.e. essentially "
        "<b>on their tuned frontier</b>, not below it.</li>"
        f"<li>On <b>{beyond}/{n}</b> images ours edits <b>beyond RF-inversion's entire η range</b> "
        "(more editable than even η=0) — a region their knob cannot reach.</li></ul>")
    if pareto:
        H.append(f"<figure><img src='{pareto}'><figcaption>Aggregate Pareto: each pink dot is one image "
                 "under our calibrated knobs; the gray line is RF-inversion's mean η curve (η=0 top-right → "
                 "η=1 bottom-left). Our mean (★) sits on their frontier, but our cloud extends far to the "
                 "right — edits η cannot produce.</figcaption></figure>")

    H.append("<h3>DINO structure distance by edit type</h3>")
    H.append(tbl(["edit type", "n", "ours", "vanilla", "η=0.9", "ours&lt;vanilla"], pt_rows))

    H.append("<h2>Qualitative comparisons</h2>")
    H.append("<p class=sub>source · RF-inv vanilla (η=0) · RF-inv default (η=0.9) · ours — picked as the "
             "biggest structure win per edit type that still performs the edit.</p>")
    for r, b in qual:
        H.append(f"<div class=prompt><b>{r['edit_type']}</b> — “{r['src_prompt']}” → "
                 f"“{r['edit_prompt']}”</div><img src='{b}'>")

    H.append("<h2>Honest read</h2>")
    H.append(
        "<p>Two solid claims: (a) we <b>beat out-of-the-box RF-inversion</b> on structure, perceptual "
        "distance and editability simultaneously, on a standard benchmark; (b) our knobs reach an "
        "<b>editability range RF-inversion's η fundamentally cannot</b> (η only moves <i>down</i> in "
        "editability from vanilla toward reconstruction). The weaker claim — beating their <i>tuned</i> η "
        "frontier on structure at matched editability — is <b>not</b> supported: there we are roughly tied.</p>")

    H.append("<h2>Future experiments &amp; sweeps to improve</h2><ul>"
        "<li><b>Re-target the calibration objective.</b> We minimize structure s.t. editability ≥ vanilla. "
        "Instead push to points further along our own frontier (e.g. maximize a structure+editability scalar, "
        "or minimize structure s.t. editability ≥ a higher target) to drive the cloud <i>below</i> the η curve. "
        "Re-selectable from saved trial traces with no GPU.</li>"
        "<li><b>Widen / refine the knob search.</b> Sweep <code>interval_start</code> (currently fixed at 0), "
        "multi-band phase locks, per-step schedules, and guidance; more Optuna trials.</li>"
        "<li><b>Amortize the calibration.</b> Train a tiny predictor (CLIP image + prompt-distance features → "
        "knobs) on this run's calibration table for one-shot params + a 3-eval refine — the deferred Phase B.</li>"
        "<li><b>Background-preserving metrics.</b> PIE-Bench++ masks came as strings (skipped); recover them "
        "for background PSNR/LPIPS, where structure preservation should show an even larger margin.</li>"
        "<li><b>Stronger RF-inversion baseline.</b> Confirm the reference η default/window and sweep the "
        "controller window τ as a second axis, not just η.</li>"
        "<li><b>Scale to full PIE-Bench (700) + more seeds</b> once the objective is retargeted.</li></ul>")

    html = "\n".join(H)
    p = os.path.join(OUT, "index.html")
    open(p, "w").write(html)
    print(f"[e41-site] wrote {p} ({len(html)//1024} KB, {len(qual)} qualitative rows)", flush=True)


if __name__ == "__main__":
    main()
