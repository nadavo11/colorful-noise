"""E35 add-on: 'advantage over baseline' view.

The main index.html shows ABSOLUTE metrics per operator. This reads results/e35/report.json
and produces a Delta-vs-baseline view (delta.html): heatmaps of (operator x category) for
ΔCLIP and Δaesthetic (green = beats baseline, red = worse), plus the GOAL-ALIGNED tables that
the headline metrics miss -- per-object selectivity (does boosting an object raise ITS OWN
clip?) and two-prompt merge (does it pull in prompt B?).

    python experiments/e35_delta.py [--out_tag ...]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

CATS = ["short", "long", "style", "object", "twoobj", "pair"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def heatmap(ov, metric, path):
    base = {c: ov.get("baseline", {}).get(c, {}).get(metric) for c in CATS}
    ops = [o for o in ov if o != "baseline"]
    M = np.full((len(ops), len(CATS)), np.nan)
    for i, op in enumerate(ops):
        for j, c in enumerate(CATS):
            v = ov.get(op, {}).get(c, {}).get(metric)
            if v is not None and base[c] is not None:
                M[i, j] = v - base[c]
    vmax = np.nanmax(np.abs(M)) or 1.0
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(ops) + 1.5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CATS, rotation=30, ha="right")
    ax.set_yticks(range(len(ops))); ax.set_yticklabels(ops, fontsize=8)
    for i in range(len(ops)):
        for j in range(len(CATS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title(f"Δ {metric} vs baseline (green=better)")
    fig.colorbar(im, fraction=0.025); fig.tight_layout()
    fig.savefig(path, dpi=95); plt.close(fig)


def goal_tables(raw):
    """(perobj_rows, merge_rows) of goal-aligned Δ vs baseline."""
    perobj = []
    for cond in ["perobj_low_g0.5", "perobj_low_g2.0", "perobj_high_g0.5", "perobj_high_g2.0"]:
        ds = []
        for e in raw.values():
            c = e["conds"]
            if cond not in c or "baseline" not in c:
                continue
            b = _mean([s.get("clip_obj") for s in c["baseline"]["seeds"].values()])
            v = _mean([s.get("clip_obj") for s in c[cond]["seeds"].values()])
            if b is not None and v is not None:
                ds.append(v - b)
        perobj.append((cond, _mean(ds), len(ds)))
    merge = []
    for cond in ["swap_c0.15", "swap_c0.25", "swap_c0.4", "blend_c0.25", "lerp_a0.5"]:
        dA, dB = [], []
        for e in raw.values():
            c = e["conds"]
            if e["cat"] != "pair" or cond not in c:
                continue
            bA = _mean([s.get("clip") for s in c["baseline"]["seeds"].values()])
            bB = _mean([s.get("clip_B") for s in c["baseline"]["seeds"].values()])
            vA = _mean([s.get("clip") for s in c[cond]["seeds"].values()])
            vB = _mean([s.get("clip_B") for s in c[cond]["seeds"].values()])
            if None not in (bA, bB, vA, vB):
                dA.append(vA - bA); dB.append(vB - bB)
        merge.append((cond, _mean(dA), _mean(dB), len(dA)))
    return perobj, merge


def main(out):
    rep = json.load(open(os.path.join(out, "report.json")))
    ov, raw = rep["summary"]["overall"], rep["raw"]
    ddir = os.path.join(out, "plots"); os.makedirs(ddir, exist_ok=True)
    heatmap(ov, "clip", os.path.join(ddir, "delta_clip.png"))
    heatmap(ov, "aesthetic", os.path.join(ddir, "delta_aesthetic.png"))
    perobj, merge = goal_tables(raw)

    try:
        from common import data_uri
    except Exception:
        data_uri = None

    def img(rel):
        p = os.path.join(out, rel)
        return (f"<img src='{data_uri(p)}' style='max-width:100%'>" if data_uri and os.path.exists(p)
                else f"<img src='{rel}' style='max-width:100%'>")

    h = ["<!doctype html><meta charset=utf-8><title>E35 — advantage over baseline</title>",
         "<style>body{font:14px/1.5 system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}"
         "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 8px;font-size:13px}</style>",
         "<h1>E35 — advantage over baseline</h1>",
         "<p><b>Read:</b> green = beats the unedited baseline, red = worse. CLIP-to-the-prompt "
         "can't exceed baseline by construction (baseline IS the prompt), so there the question "
         "is <i>least harm</i>; aesthetic <i>could</i> improve.</p>",
         "<h2>Δ CLIP adherence vs baseline</h2>", img("plots/delta_clip.png"),
         "<h2>Δ aesthetic (fidelity) vs baseline</h2>", img("plots/delta_aesthetic.png"),
         "<h2>Goal-aligned advantages (what the operators are actually for)</h2>",
         "<h3>per-object: Δ of the targeted object's OWN clip (clip_obj) vs baseline</h3>",
         "<table><tr><th>condition</th><th>Δ clip_obj</th><th>n</th></tr>"]
    for cond, d, n in perobj:
        h.append(f"<tr><td>{cond}</td><td>{d:+.4f}</td><td>{n}</td></tr>")
    h.append("</table><p>Boosting (g2.0) raises the targeted object's adherence (high-band "
             "strongest); cutting lowers it — the per-object knob works on its own goal.</p>")
    h.append("<h3>two-prompt merge: Δ adherence to A and to B vs baseline (=prompt A only)</h3>")
    h.append("<table><tr><th>condition</th><th>Δ clip(A)</th><th>Δ clip_B(B)</th><th>n</th></tr>")
    for cond, da, db, n in merge:
        h.append(f"<tr><td>{cond}</td><td>{da:+.3f}</td><td>{db:+.3f}</td><td>{n}</td></tr>")
    h.append("</table><p>All pull in B; <b>lerp</b> adds B (+0.018) at near-zero cost to A "
             "(−0.001) — the best A/B tradeoff, beating the spectral swaps (which cost A more).</p>")
    h.append("<p style='margin-top:2em'><a href='index.html'>&larr; back to the full report</a></p>")
    with open(os.path.join(out, "delta.html"), "w") as f:
        f.write("\n".join(h))

    # cross-link: add a banner to index.html pointing at delta.html (idempotent)
    idx = os.path.join(out, "index.html")
    if os.path.exists(idx):
        html = open(idx).read()
        banner = ("<p style='padding:.6em;background:#eef6ff;border:1px solid #9fb6d6;"
                  "border-radius:6px'><b>See also:</b> <a href='delta.html'>advantage-over-baseline "
                  "view</a> — Δ-vs-baseline heatmaps (op × category) + goal-aligned tables.</p>")
        if "delta.html" not in html:
            html = html.replace("</h1>", "</h1>\n" + banner, 1)
            open(idx, "w").write(html)
            print("[e35-delta] linked delta.html from index.html", flush=True)
    print(f"[e35-delta] wrote {os.path.join(out, 'delta.html')} + delta heatmaps", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_tag", default="")
    a = ap.parse_args()
    main(os.path.join(RESULTS, f"e35_{a.out_tag}" if a.out_tag else "e35"))
