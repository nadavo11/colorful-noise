"""Build a simple self-contained HTML report for E16: metrics tables + image
galleries. Reads results/e16/scores.json (written by e16_prompt_adherence.py
--part score) and the saved PNGs; embeds thumbnails as base64 so the single
index.html is portable. Honors CN_RESULTS (so it can point at the cluster output
on shared storage).

    CN_RESULTS=/storage/.../results python e16_site.py
"""
import base64
import io
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

# EXP selects which experiment dir + condition ordering (E16 Flux / E17 SD3.5).
EXP = os.environ.get("E16_SITE_EXP", "e16")
OUT = os.path.join(RESULTS, EXP)
# master ordering + labels; the actual conds shown are whatever scores.json has.
ORDER = ["cfg1.0", "cfg1", "cfg3.5", "cfg_hi", "bandnorm", "bandnorm_pp",
         "cfgzero", "cfgzero_sbn", "cfgpp", "cfgpp_sbn", "negprompt", "seg"]
LABELS = {"cfg1.0": "cfg=1 (anchor)", "cfg1": "cfg=1 (anchor)",
          "cfg3.5": "cfg=3.5 (baseline)", "cfg_hi": "high-cfg (baseline)",
          "bandnorm": "SBN", "bandnorm_pp": "SBN+postproc (ours)",
          "cfgzero": "CFG-Zero*", "cfgzero_sbn": "CFG-Zero*+SBN (ours)",
          "cfgpp": "CFG++", "cfgpp_sbn": "CFG+++SBN (ours)",
          "negprompt": "neg-prompt (NAG proxy)", "seg": "SEG"}
BASE = "cfg3.5" if EXP == "e16" else "cfg_hi"
FID = ["aesthetic", "imagereward", "spectral_dist"]
ADH = ["clip_t", "vqascore"]
EXTRA = ["rms_contrast", "colorfulness", "sharpness"]
LOWER_BETTER = {"spectral_dist"}
GALLERY_SEEDS = [0, 1, 2]
THUMB = 240


def thumb_b64(path):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    im.thumbnail((THUMB, THUMB))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def m(entry, k):
    """mean of metric k from a per-cond score entry, or None."""
    v = entry.get(k)
    return v["mean"] if v else None


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def present_conds(scores):
    """Conditions actually present in the data, in master ORDER."""
    seen = set()
    for pdata in scores.values():
        seen |= set(pdata["conds"])
    return [c for c in ORDER if c in seen]


def is_ours(c):
    return "bandnorm" in c or c.endswith("_sbn")


def aggregate(scores, CONDS):
    """Mean-over-prompts of each cond×metric, and paired Δ vs BASE."""
    allm = FID + ADH + EXTRA
    absv = {c: {k: [] for k in allm} for c in CONDS}
    for pid, pdata in scores.items():
        pc = pdata["conds"]
        for c in CONDS:
            if c in pc:
                for k in allm:
                    val = m(pc[c], k)
                    if val is not None:
                        absv[c][k].append(val)
    abs_mean = {c: {k: (sum(v) / len(v) if v else None) for k, v in d.items()}
                for c, d in absv.items()}
    # paired Δ vs BASE per prompt then averaged
    delt = {c: {k: [] for k in FID + ADH} for c in CONDS}
    for pid, pdata in scores.items():
        pc = pdata["conds"]
        if BASE not in pc:
            continue
        for c in CONDS:  # noqa: PLR1704
            if c not in pc:
                continue
            for k in FID + ADH:
                a, b = m(pc[c], k), m(pc[BASE], k)
                if a is not None and b is not None:
                    delt[c][k].append(a - b)
    delt_mean = {c: {k: (sum(v) / len(v) if v else None) for k, v in d.items()}
                 for c, d in delt.items()}
    return abs_mean, delt_mean


def best_cells(abs_mean, CONDS):
    """Which cond is best per metric (for bolding)."""
    best = {}
    for k in FID + ADH + EXTRA:
        vals = [(c, abs_mean[c][k]) for c in CONDS if abs_mean[c][k] is not None]
        if not vals:
            continue
        best[k] = (min if k in LOWER_BETTER else max)(vals, key=lambda t: t[1])[0]
    return best


def delta_color(k, x):
    if x is None or k == "" or abs(x) < 1e-9:
        return ""
    good = (x < 0) if k in LOWER_BETTER else (x > 0)
    return "pos" if good else "neg"


CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a;max-width:1400px}
h1{font-size:22px} h2{font-size:18px;margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}
.note{background:#f6f8fa;border-left:4px solid #0969da;padding:10px 14px;border-radius:4px;margin:12px 0}
table{border-collapse:collapse;margin:10px 0;font-variant-numeric:tabular-nums}
th,td{border:1px solid #d0d7de;padding:5px 10px;text-align:center}
th{background:#f6f8fa} td.cond{text-align:left;font-weight:600;white-space:nowrap}
td.best{font-weight:700;background:#dafbe1}
td.pos{background:#dafbe1;color:#0a6b2b} td.neg{background:#ffebe9;color:#b3261e}
.ours{background:#fff8c5}
.gal{display:grid;grid-template-columns:130px repeat(3,1fr);gap:4px;align-items:center;margin:6px 0 18px}
.gal img{width:100%;border-radius:3px;display:block}
.gal .rl{font-weight:600;font-size:12px;text-align:right;padding-right:8px}
.gal .hd{font-size:12px;color:#555;text-align:center}
.cap{color:#555;font-size:12px}
"""


def render(scores):
    CONDS = present_conds(scores)
    abs_mean, delt = aggregate(scores, CONDS)
    best = best_cells(abs_mean, CONDS)
    has_vqa = any(abs_mean[c].get("vqascore") is not None for c in CONDS)
    adh = ADH if has_vqa else ["clip_t"]
    nprompt = len(scores)
    title = "E16 — fidelity at high CFG (Flux-dev)" if EXP == "e16" \
        else "E17 — SBN vs CFG-Zero* (SD3.5-medium, true CFG)"
    has_spec = any(abs_mean[c].get("spectral_dist") is not None for c in CONDS)
    cols = [m for m in FID if has_spec or m != "spectral_dist"]
    h = [f"<!doctype html><meta charset=utf-8><title>{EXP} fidelity</title><style>{CSS}</style>"]
    h.append(f"<h1>{title}</h1>")
    h.append(f"<div class=note><b>Contest = fidelity</b> (aesthetic, ImageReward"
             + (", spectral-distance-to-real" if has_spec else "") + "); "
             f"<b>adherence is a guardrail</b> (CLIP-T) — it should not drop vs "
             f"<b>{LABELS.get(BASE, BASE)}</b>. Means over <b>{nprompt} prompts × up to 25 "
             f"seeds</b>. Higher is better"
             + (", except <i>spectral_dist</i> (lower = closer to real)." if has_spec else ".")
             + " Rows marked <span class=ours>ours</span> use SBN.</div>")

    # --- absolute means table ---
    acols = cols + adh + EXTRA
    h.append("<h2>Absolute means</h2><table><tr><th>condition</th>"
             + "".join(f"<th>{c}{' ↓' if c in LOWER_BETTER else ''}</th>" for c in acols) + "</tr>")
    for c in CONDS:
        ours = " ours" if is_ours(c) else ""
        row = [f"<td class='cond{ours}'>{LABELS.get(c, c)}</td>"]
        for k in acols:
            cls = "best" if best.get(k) == c else ""
            row.append(f"<td class='{cls}'>{fmt(abs_mean[c][k])}</td>")
        h.append("<tr>" + "".join(row) + "</tr>")
    h.append("</table>")

    # --- paired delta table ---
    h.append(f"<h2>Paired Δ vs {LABELS.get(BASE, BASE)}</h2><p class=cap>green = better, "
             "red = worse.</p><table><tr><th>condition</th>"
             + "".join(f"<th>Δ {c}{' ↓' if c in LOWER_BETTER else ''}</th>" for c in cols + adh)
             + "</tr>")
    for c in CONDS:
        if c == BASE:
            continue
        ours = " ours" if is_ours(c) else ""
        row = [f"<td class='cond{ours}'>{LABELS.get(c, c)}</td>"]
        for k in cols + adh:
            x = delt[c][k]
            sign = "" if x is None or x < 0 else "+"
            row.append(f"<td class='{delta_color(k, x)}'>{'' if x is None else sign}{fmt(x)}</td>")
        h.append("<tr>" + "".join(row) + "</tr>")
    h.append("</table>")

    # --- galleries ---
    h.append("<h2>Sample images</h2><p class=cap>Same seed across a row → compare conditions "
             "directly (anchor vs high-cfg baseline vs SBN variants vs guidance baselines).</p>")
    for pid, pdata in scores.items():
        h.append(f"<h3>{pid}</h3><div class=cap>{pdata.get('prompt','')}</div>")
        h.append("<div class=gal><div class=rl></div>"
                 + "".join(f"<div class=hd>seed {s}</div>" for s in GALLERY_SEEDS))
        for c in CONDS:
            h.append(f"<div class=rl>{LABELS.get(c, c)}</div>")
            for s in GALLERY_SEEDS:
                p = f"{OUT}/{pid}/images/{c}_s{s}.png"
                b = thumb_b64(p) if os.path.exists(p) else None
                h.append(f"<img src='data:image/jpeg;base64,{b}'>" if b else "<div></div>")
        h.append("</div>")
    return "".join(h)


def main():
    scores = json.load(open(f"{OUT}/scores.json"))
    html = render(scores)
    site = f"{OUT}/site"
    os.makedirs(site, exist_ok=True)
    with open(f"{site}/index.html", "w") as f:
        f.write(html)
    print(f"[e16-site] wrote {site}/index.html  ({len(html)//1024} KB, "
          f"{len(scores)} prompts)")


if __name__ == "__main__":
    main()
