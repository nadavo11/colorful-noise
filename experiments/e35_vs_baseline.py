"""E35 add-on: edit-vs-BASELINE-GENERATION view (the right reference).

The main report scores the edited image against the *prompt* (CLIP-T), where the unedited
baseline is the ceiling by construction -- so the Delta-vs-baseline heatmaps can only ever be
red. That measures "how far did the edit fall from the prompt", not "is the edited image a
better/different image than the model would have made". This script re-references everything to
the BASELINE MODEL GENERATION (the same-seed unedited image) and asks two distinct questions:

  DIRECTIONAL (better or worse than baseline; green is ACHIEVABLE here):
    d_aesthetic    LAION aesthetic(edit) - aesthetic(baseline)            [free from report.json]
    d_imagereward  ImageReward(edit, prompt) - ImageReward(baseline,prompt) [needs images + pkg]
                   ImageReward is prompt-aware AND learned, so unlike CLIP-T it is NOT capped at
                   baseline -- an edit can genuinely score higher. This is the "improvement" axis.
    d_sharpness / d_hf_frac / d_colorfulness                              [free from report.json]

  DISTANCE (how much / in what way it moved from baseline; magnitude, not better/worse):
    clip_i2i_dist  1 - cosine(CLIP_img(edit), CLIP_img(baseline)) == drift [free from report.json]
    lpips          perceptual distance                                    [needs images + lpips]
    dssim          1 - SSIM structural similarity                         [needs images + skimage]
    img_psd_l2     L2 of (log radial PSD of edit - of baseline), image-domain spectral change
                   -- this is the axis that captures what low-pass / phase-only do by eye  [numpy]
    color_l2       L2 of per-channel mean-color difference                [numpy]

Same-seed pairing is exact: for prompt dir <pid>/, condition <name>, seed s, the reference is
<pid>/baseline_s{s}.png. Tier 1 (the *free* metrics) needs only report.json; Tier 2 loads the
saved PNGs (pass --with_images; degrades to None per missing dep, like fidelity_metrics).

    python experiments/e35_vs_baseline.py                 # tier 1 only (report.json)
    python experiments/e35_vs_baseline.py --with_images   # + LPIPS/SSIM/PSD/color/ImageReward
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
# +better-is-greener for directional; for distance metrics bigger=more change (sequential cmap)
DIRECTIONAL = ["d_aesthetic", "d_imagereward", "d_imagereward_B",
               "d_sharpness", "d_hf_frac", "d_colorfulness"]
DISTANCE = ["clip_i2i_dist", "lpips", "dssim", "img_psd_l2", "color_l2"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------------------
# Tier 2: image-based paired metrics (loaded lazily; each degrades to None)
# ---------------------------------------------------------------------------
def _load_lpips():
    try:
        import lpips, torch
        net = lpips.LPIPS(net="alex").to("cuda" if torch.cuda.is_available() else "cpu").eval()
        return net
    except Exception as e:
        print(f"[vsbase] lpips unavailable ({e}); skipping perceptual distance", flush=True)
        return None


def _load_ssim():
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim
    except Exception as e:
        print(f"[vsbase] skimage unavailable ({e}); skipping SSIM", flush=True)
        return None


def _radial_log_psd(gray, n_bins=32):
    """Radially-averaged log power spectrum of a 2-D grayscale array (numpy)."""
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    p = np.abs(F) ** 2
    h, w = gray.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = r.max()
    bins = (r / rmax * (n_bins - 1)).astype(int)
    out = np.array([p[bins == b].mean() if np.any(bins == b) else 0.0 for b in range(n_bins)])
    return np.log(out + 1e-12)


def _img_metrics(edit_img, base_img, lpips_net, ssim_fn):
    """Distance metrics between an edit PIL image and its same-seed baseline PIL image."""
    e = np.asarray(edit_img.convert("RGB"), dtype=np.float32)
    b = np.asarray(base_img.convert("RGB"), dtype=np.float32)
    out = {}
    # image-domain spectral change (grayscale radial log-PSD L2) -- the low-pass/phase-only axis
    ge = e.mean(2) / 255.0
    gb = b.mean(2) / 255.0
    out["img_psd_l2"] = float(np.sqrt(((_radial_log_psd(ge) - _radial_log_psd(gb)) ** 2).mean()))
    # mean-color L2 (per-channel mean difference)
    out["color_l2"] = float(np.sqrt(((e.mean((0, 1)) - b.mean((0, 1))) ** 2).sum()) / 255.0)
    if ssim_fn is not None:
        try:
            out["dssim"] = 1.0 - float(ssim_fn(e, b, channel_axis=2, data_range=255.0))
        except Exception:
            out["dssim"] = None
    if lpips_net is not None:
        try:
            import torch
            dev = next(lpips_net.parameters()).device
            def t(a):  # HWC[0,255] -> 1CHW[-1,1]
                return (torch.from_numpy(a).permute(2, 0, 1)[None] / 127.5 - 1.0).to(dev)
            with torch.no_grad():
                out["lpips"] = float(lpips_net(t(e), t(b)).item())
        except Exception:
            out["lpips"] = None
    return out


def _imagereward_pairs(out_dir, raw, pair_map):
    """{(pid, name, s): {d_imagereward[, _B]}} via ImageReward, or {} if unavailable."""
    from fidelity_metrics import load_imagereward, imagereward_scores
    model = load_imagereward()
    if model is None:
        return {}
    res = {}
    for pid, e in raw.items():
        d = os.path.join(out_dir, pid)
        promptA = e["main"]
        promptB = pair_map.get(pid)  # B for pairs, else None
        # baseline reward per seed (against A, and B if a pair)
        baseA, baseB = {}, {}
        for s in _seeds_of(e):
            bp = os.path.join(d, f"baseline_s{s}.png")
            if os.path.exists(bp):
                baseA[s] = imagereward_scores(model, promptA, [bp])[0]
                if promptB:
                    baseB[s] = imagereward_scores(model, promptB, [bp])[0]
        for name, c in e["conds"].items():
            if name == "baseline":
                continue
            for s in _seeds_of(e):
                p = os.path.join(d, f"{name}_s{s}.png")
                if not os.path.exists(p) or baseA.get(s) is None:
                    continue
                rec = {"d_imagereward": imagereward_scores(model, promptA, [p])[0] - baseA[s]}
                if promptB and baseB.get(s) is not None:
                    rec["d_imagereward_B"] = imagereward_scores(model, promptB, [p])[0] - baseB[s]
                res[(pid, name, str(s))] = rec
        print(f"[vsbase] ImageReward scored {pid}", flush=True)
    return res


def _seeds_of(entry):
    seeds = set()
    for c in entry["conds"].values():
        seeds.update(c["seeds"].keys())
    return sorted(seeds, key=int)


# ---------------------------------------------------------------------------
# build per-(pid, cond, seed) paired records, then aggregate to op x category
# ---------------------------------------------------------------------------
def build_records(out_dir, raw, pair_map, with_images):
    lpips_net = ssim_fn = None
    ir_pairs = {}
    if with_images:
        from PIL import Image
        lpips_net = _load_lpips()
        ssim_fn = _load_ssim()
        ir_pairs = _imagereward_pairs(out_dir, raw, pair_map)

    records = []  # (op, cat, name, metrics)
    for pid, e in raw.items():
        cat = e["cat"]
        d = os.path.join(out_dir, pid)
        base = e["conds"].get("baseline", {}).get("seeds", {})
        for name, c in e["conds"].items():
            if name == "baseline":
                continue
            op = c["op"]
            for s, sd in c["seeds"].items():
                bsd = base.get(s)
                m = {}
                # --- tier 1: free directional deltas from report.json ---
                if sd.get("drift") is not None:
                    m["clip_i2i_dist"] = sd["drift"]              # == 1 - cos(edit, baseline)
                if bsd is not None:
                    for key, dk in (("aesthetic", "d_aesthetic"), ("sharpness", "d_sharpness"),
                                    ("hf_frac", "d_hf_frac"), ("colorfulness", "d_colorfulness")):
                        if sd.get(key) is not None and bsd.get(key) is not None:
                            m[dk] = sd[key] - bsd[key]
                # --- tier 2: image-based paired metrics ---
                if with_images:
                    ep = os.path.join(d, f"{name}_s{s}.png")
                    bp = os.path.join(d, f"baseline_s{s}.png")
                    if os.path.exists(ep) and os.path.exists(bp):
                        m.update(_img_metrics(Image.open(ep), Image.open(bp), lpips_net, ssim_fn))
                    m.update(ir_pairs.get((pid, name, s), {}))
                records.append((op, cat, name, m))
    return records


def aggregate(records):
    """op -> {category -> {metric -> mean}}  (+ '_all' pseudo-category over all cats)."""
    ov = {}
    for op, cat, _name, m in records:
        for c in (cat, "_all"):
            node = ov.setdefault(op, {}).setdefault(c, {})
            for mk, v in m.items():
                if v is not None:
                    node.setdefault(mk, []).append(v)
    for op in ov:
        for c in ov[op]:
            for mk in list(ov[op][c]):
                ov[op][c][mk] = round(_mean(ov[op][c][mk]), 4)
    return ov


# ---------------------------------------------------------------------------
# heatmaps
# ---------------------------------------------------------------------------
def heatmap(ov, metric, path, directional):
    ops = sorted(o for o in ov if any(metric in ov[o].get(c, {}) for c in CATS))
    if not ops:
        return False
    M = np.full((len(ops), len(CATS)), np.nan)
    for i, op in enumerate(ops):
        for j, c in enumerate(CATS):
            v = ov[op].get(c, {}).get(metric)
            if v is not None:
                M[i, j] = v
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(ops) + 1.6))
    if directional:
        vmax = np.nanmax(np.abs(M)) or 1.0
        im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        sub = "green = edit BEATS baseline generation"
    else:
        im = ax.imshow(M, cmap="magma", aspect="auto")
        sub = "brighter = moved further from baseline generation"
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CATS, rotation=30, ha="right")
    ax.set_yticks(range(len(ops))); ax.set_yticklabels(ops, fontsize=8)
    for i in range(len(ops)):
        for j in range(len(CATS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if not directional else "black")
    ax.set_title(f"{metric}  ({sub})")
    fig.colorbar(im, fraction=0.025); fig.tight_layout()
    fig.savefig(path, dpi=95); plt.close(fig)
    return True


def main(out_dir, with_images):
    rep = json.load(open(os.path.join(out_dir, "report.json")))
    raw = rep["raw"]
    # pair B prompts (for ImageReward-vs-B) from the gen script's PAIRS table
    try:
        from e35_op_sweep import PAIRS
        pair_map = {pid: B for pid, _A, B in PAIRS}
    except Exception:
        pair_map = {}

    records = build_records(out_dir, raw, pair_map, with_images)
    ov = aggregate(records)
    pdir = os.path.join(out_dir, "plots"); os.makedirs(pdir, exist_ok=True)

    made = []
    for mk in DIRECTIONAL:
        if heatmap(ov, mk, os.path.join(pdir, f"vsbase_{mk}.png"), directional=True):
            made.append((mk, True))
    for mk in DISTANCE:
        if heatmap(ov, mk, os.path.join(pdir, f"vsbase_{mk}.png"), directional=False):
            made.append((mk, False))

    # ranked tables (over _all): best operators on each directional axis
    def ranked(metric):
        rows = [(op, ov[op]["_all"][metric]) for op in ov if metric in ov[op].get("_all", {})]
        return sorted(rows, key=lambda r: r[1], reverse=True)

    try:
        from e27_site import data_uri
    except Exception:
        data_uri = None

    def img(rel):
        p = os.path.join(out_dir, rel)
        return (f"<img src='{data_uri(p)}' style='max-width:100%'>" if data_uri and os.path.exists(p)
                else f"<img src='{rel}' style='max-width:100%'>")

    h = ["<!doctype html><meta charset=utf-8><title>E35 — vs baseline generation</title>",
         "<style>body{font:14px/1.5 system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}"
         "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 8px;font-size:13px}"
         "h2{margin-top:1.6em}</style>",
         "<h1>E35 — edit vs the baseline model generation</h1>",
         "<p><b>Reference = the same-seed unedited image</b>, not the prompt. The prompt-adherence "
         "heatmaps (see <a href='delta.html'>delta.html</a>) are red by construction because the "
         "baseline IS the prompt. Here the question is different and <b>green is achievable</b>: "
         "is the edited image a <i>better</i> image (directional), and how far did it move "
         "(distance)?</p>",
         "<h2>Directional — does the operator IMPROVE on the baseline generation?</h2>",
         "<p>ImageReward & aesthetic are not capped at baseline, so an operator can win here even "
         "though it 'loses' on CLIP-to-prompt. This is the axis that matches what looked good by "
         "eye.</p>"]
    for mk, _d in [m for m in made if m[1]]:
        h.append(f"<h3>{mk}</h3>{img('plots/vsbase_' + mk + '.png')}")
    h.append("<h2>Distance — how much / in what way it moved</h2>")
    h.append("<p><code>img_psd_l2</code> is the image-domain spectral change — the axis that "
             "captures what low-pass / phase-only do to texture that CLIP-img similarity misses.</p>")
    for mk, _d in [m for m in made if not m[1]]:
        h.append(f"<h3>{mk}</h3>{img('plots/vsbase_' + mk + '.png')}")
    # ranked directional tables
    h.append("<h2>Operators ranked by improvement over baseline (all categories)</h2>")
    for mk in ("d_imagereward", "d_aesthetic"):
        rows = ranked(mk)
        if not rows:
            continue
        h.append(f"<h3>{mk} (positive = beats baseline)</h3>")
        h.append("<table><tr><th>operator</th><th>" + mk + "</th></tr>")
        for op, v in rows:
            h.append(f"<tr><td>{op}</td><td>{v:+.4f}</td></tr>")
        h.append("</table>")
    h.append("<p style='margin-top:2em'><a href='index.html'>&larr; full report</a> · "
             "<a href='delta.html'>vs-prompt delta view</a></p>")
    with open(os.path.join(out_dir, "vs_baseline.html"), "w") as f:
        f.write("\n".join(h))

    # persist the aggregated numbers + cross-link from index.html (idempotent)
    with open(os.path.join(out_dir, "vs_baseline.json"), "w") as f:
        json.dump({"overall": ov, "with_images": with_images}, f, indent=2)
    idx = os.path.join(out_dir, "index.html")
    if os.path.exists(idx):
        html = open(idx).read()
        banner = ("<p style='padding:.6em;background:#eefbf0;border:1px solid #9fd6b6;"
                  "border-radius:6px'><b>See also:</b> <a href='vs_baseline.html'>vs-baseline-"
                  "generation view</a> — does each operator IMPROVE on the unedited image "
                  "(ImageReward/aesthetic, green achievable) + how far it moved.</p>")
        if "vs_baseline.html" not in html:
            open(idx, "w").write(html.replace("</h1>", "</h1>\n" + banner, 1))
    print(f"[vsbase] wrote vs_baseline.html + {len([m for m in made])} heatmaps "
          f"(with_images={with_images})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_tag", default="")
    ap.add_argument("--with_images", action="store_true",
                    help="also compute LPIPS/SSIM/image-PSD/color/ImageReward from saved PNGs")
    a = ap.parse_args()
    main(os.path.join(RESULTS, f"e35_{a.out_tag}" if a.out_tag else "e35"), a.with_images)
