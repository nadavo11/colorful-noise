"""E51 figures (anaconda env, matplotlib + PIL). All saved to outputs/.../figures/.
Every figure here is referenced and explained in the HTML report."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import config as C

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.titlesize": 11})

CV = C.CACHE_VARIANTS
COL = C.VARIANT_COLOR
SH = C.VARIANT_SHORT


def _df():
    import pandas as pd
    return pd.read_csv(C.OUT / "per_example_metrics.csv")


def _summary():
    return json.loads((C.OUT / "summary.json").read_text())


def _traj(eid):
    return json.loads((C.DIAG / f"trajectory_{eid}.json").read_text())


def _all_traj():
    return [json.loads(p.read_text()) for p in sorted(C.DIAG.glob("trajectory_*.json"))]


def _save(fig, name):
    p = C.FIG / name
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ----------------------------------------------------------------- 1. exec summary
def exec_summary():
    s = _summary()
    prim = {r["variant"]: r for r in s["primary_by_variant"]}
    order = ["full_compute_reference"] + CV
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    names = [SH.get(v, v) for v in order]
    dino = [prim.get(v, {}).get("dino_to_ref", np.nan) for v in order]
    lpips = [prim.get(v, {}).get("lpips_to_ref", np.nan) for v in order]
    cols = [COL.get(v, "#888") for v in order]
    ax[0].bar(names, dino, color=cols); ax[0].set_title("Fidelity to full-compute reference (DINOv2 ↑)")
    ax[0].set_ylim(min(0.7, np.nanmin(dino) - 0.02), 1.005); ax[0].axhline(1.0, ls=":", c="k", lw=.8)
    ax[1].bar(names, lpips, color=cols); ax[1].set_title("Reconstruction error vs reference (LPIPS ↓)")
    for a in ax:
        a.tick_params(axis="x", rotation=25)
    fig.suptitle(f"Caching @ ~{int(s['primary_skip']*100)}% step-skip — verdict: {s['verdict']}", weight="bold")
    return _save(fig, "fig01_exec_summary.png")


# ----------------------------------------------------------------- 2. qualitative grids
def _font(sz=14):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def _label(img, text, h=22):
    w = img.width
    strip = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(strip)
    d.text((4, 3), text, fill="black", font=_font(13))
    out = Image.new("RGB", (w, img.height + h), "white")
    out.paste(strip, (0, 0)); out.paste(img, (0, h))
    return out


def qualitative_grids(ids=None, per_fig=6, sz=240):
    man = json.loads((C.DIAG / "examples.json").read_text())
    ex = {e["id"]: e for e in man["examples"]}
    ids = ids or [e["id"] for e in man["examples"]]
    cols = ["input", "full_compute_reference", "raw_full_prediction_cache",
            "spectral_full_prediction_cache", "raw_edit_delta_cache", "spectral_edit_delta_cache"]
    head = ["input", "full (ref)", "raw full$", "spec full$", "raw delta$", "spec delta$"]
    paths = []
    chunks = [ids[i:i + per_fig] for i in range(0, len(ids), per_fig)]
    for ci, chunk in enumerate(chunks):
        rows = []
        for eid in chunk:
            sdir = C.SAMP / eid
            cells = []
            for c, h in zip(cols, head):
                fp = sdir / f"{c}.png"
                im = Image.open(fp).convert("RGB").resize((sz, sz)) if fp.exists() else Image.new("RGB", (sz, sz), "#ddd")
                cells.append(_label(im, h))
            cap = ex[eid]
            band = Image.new("RGB", (sz, sz), "white")
            d = ImageDraw.Draw(band)
            txt = f"{eid}\n[{cap['task_type']}]\n\nsrc: {cap['source_prompt']}\n\ntgt: {cap['target_prompt']}"
            d.multiline_text((6, 6), txt, fill="black", font=_font(12), spacing=3)
            rowimg = Image.new("RGB", (sz * (len(cells) + 1), cells[0].height), "white")
            rowimg.paste(_label(band, ""), (0, 0))
            for j, cc in enumerate(cells):
                rowimg.paste(cc, (sz * (j + 1), 0))
            rows.append(rowimg)
        H = sum(r.height for r in rows)
        grid = Image.new("RGB", (rows[0].width, H), "white")
        y = 0
        for r in rows:
            grid.paste(r, (0, y)); y += r.height
        p = C.FIG / f"fig02_qualitative_{ci}.png"
        grid.save(p); paths.append(p)
    return paths


# ----------------------------------------------------------------- 3. pareto
def pareto_plots():
    s = _summary(); par = s["pareto"]
    metrics = [("dino_to_ref", "DINOv2 fidelity to ref ↑", False),
               ("lpips_to_ref", "LPIPS to ref ↓", True),
               ("clipT_gain", "CLIP-T edit gain ↑", False)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, (mk, title, low_good) in enumerate(metrics):
        for v in CV:
            pts = par[v]
            x = [p["realized_skip"] for p in pts]; y = [p[mk] for p in pts]
            ax[k].plot(x, y, "-o", color=COL[v], label=SH[v], lw=2, ms=5)
        ax[k].set_xlabel("realized step-skip ratio"); ax[k].set_title(title)
    ax[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Speed–quality frontier (Pareto subset, 8 examples × 5 operating points)", weight="bold")
    return _save(fig, "fig03_pareto.png")


def pareto_speedup():
    s = _summary(); par = s["pareto"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for v in CV:
        pts = par[v]
        sp = [p["speedup_cfg"] for p in pts]
        ax[0].plot(sp, [p["dino_to_ref"] for p in pts], "-o", color=COL[v], label=SH[v], lw=2)
        ax[1].plot(sp, [p["lpips_to_ref"] for p in pts], "-o", color=COL[v], label=SH[v], lw=2)
    ax[0].set_xlabel("speedup × (true-CFG accounting)"); ax[0].set_ylabel("DINOv2 fidelity ↑"); ax[0].set_title("Quality vs speedup")
    ax[1].set_xlabel("speedup × (true-CFG accounting)"); ax[1].set_ylabel("LPIPS ↓"); ax[1].set_title("Error vs speedup")
    ax[0].legend(fontsize=8)
    return _save(fig, "fig04_pareto_speedup.png")


# ----------------------------------------------------------------- 4. trajectory smoothness
def smoothness_traj(ids=None):
    trajs = _all_traj()
    if ids:
        trajs = [t for t in trajs if t["id"] in ids]
    trajs = trajs[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, d in zip(axes.ravel(), trajs):
        steps = np.arange(d["n"])
        ax.plot(steps[1:], d["abs_edit"][1:], "-o", c="#c0392b", label="v_edit", ms=3)
        ax.plot(steps[1:], d["abs_delta"][1:], "-o", c="#1e8449", label="Δ_edit", ms=3)
        ax.set_title(f"{d['id']} [{d['task_type']}]", fontsize=9)
        ax.set_xlabel("denoising step"); ax.set_ylabel("absolute adjacent change ‖Δ‖")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Temporal smoothness: adjacent-step change of v_edit vs Δ_edit (lower = more cacheable)", weight="bold")
    return _save(fig, "fig05_smoothness_traj.png")


def smoothness_aggregate():
    trajs = _all_traj()
    n = max(t["n"] for t in trajs)
    re_, rd = np.full((len(trajs), n), np.nan), np.full((len(trajs), n), np.nan)
    for i, d in enumerate(trajs):
        # normalise BOTH curves by this example's mean v_edit change, so absolute scale differences
        # across examples don't dominate; v_edit then sits ~1 and Δ_edit shows its relative smoothness.
        scale = np.nanmean(np.array(d["abs_edit"][1:])) + 1e-12
        re_[i, :d["n"]] = np.array(d["abs_edit"]) / scale
        rd[i, :d["n"]] = np.array(d["abs_delta"]) / scale
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(n)
    me, se = np.nanmean(re_, 0), np.nanstd(re_, 0)
    md, sd = np.nanmean(rd, 0), np.nanstd(rd, 0)
    ax.plot(x, me, c="#c0392b", lw=2, label="v_edit (full prediction)")
    ax.fill_between(x, me - se, me + se, color="#c0392b", alpha=0.15)
    ax.plot(x, md, c="#1e8449", lw=2, label="Δ_edit (edit direction)")
    ax.fill_between(x, md - sd, md + sd, color="#1e8449", alpha=0.15)
    ax.axhline(1.0, ls=":", c="#888", lw=.8)
    ax.set_xlabel("denoising step"); ax.set_ylabel("adjacent change (÷ mean v_edit change)")
    ax.set_title("Cacheability over denoising — mean ± std across examples (lower = more reusable)")
    ax.legend()
    return _save(fig, "fig06_smoothness_aggregate.png")


# ----------------------------------------------------------------- 5. frequency bands
def freq_bands():
    trajs = _all_traj()
    n = max(t["n"] for t in trajs)
    be = np.full((len(trajs), n, 3), np.nan); bd = np.full((len(trajs), n, 3), np.nan)
    for i, d in enumerate(trajs):
        be[i, :d["n"]] = np.array(d["band_edit"]); bd[i, :d["n"]] = np.array(d["band_delta"])
    x = np.arange(n); labels = ["low", "mid", "high"]; cols = ["#2471a3", "#b9770e", "#922b21"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for b in range(3):
        ax[0].plot(x, np.nanmean(be[:, :, b], 0), "-o", c=cols[b], label=labels[b], ms=3)
        ax[1].plot(x, np.nanmean(bd[:, :, b], 0), "-o", c=cols[b], label=labels[b], ms=3)
    ax[0].set_title("v_edit — fractional band energy"); ax[1].set_title("Δ_edit — fractional band energy")
    for a in ax:
        a.set_xlabel("denoising step")
    ax[0].set_ylabel("fraction of spectral power"); ax[0].legend()
    fig.suptitle("Spectral evolution: where energy lives over denoising (mean across examples)", weight="bold")
    return _save(fig, "fig07_freq_bands.png")


# ----------------------------------------------------------------- 6. cacheability heatmaps
def cacheability_heatmaps():
    trajs = sorted(_all_traj(), key=lambda d: d["id"])
    n = max(t["n"] for t in trajs)
    Me = np.full((len(trajs), n), np.nan); Md = np.full((len(trajs), n), np.nan)
    for i, d in enumerate(trajs):
        # each row ÷ its own mean v_edit change so the two panels share one scale (1 = typical v_edit step)
        scale = np.nanmean(np.array(d["abs_edit"][1:])) + 1e-12
        e = np.array(d["abs_edit"]) / scale; dd = np.array(d["abs_delta"]) / scale
        Me[i, :len(e)] = e; Md[i, :len(dd)] = dd
    vmax = np.nanpercentile(Me, 95)
    fig, ax = plt.subplots(1, 2, figsize=(14, max(4, len(trajs) * 0.35)))
    for a, M, t in zip(ax, [Me, Md], ["v_edit adjacent change", "Δ_edit adjacent change"]):
        im = a.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        a.set_yticks(range(len(trajs))); a.set_yticklabels([d["id"] for d in trajs], fontsize=7)
        a.set_xlabel("denoising step"); a.set_title(t)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("Cacheability heatmaps — dark = stable = safe to reuse (Δ_edit is darker/flatter)", weight="bold")
    return _save(fig, "fig08_cacheability_heatmap.png")


# ----------------------------------------------------------------- 7. category breakdown
def category_breakdown():
    s = _summary()
    import pandas as pd
    cb = pd.DataFrame(s["category_breakdown"])
    cats = sorted(cb["category"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    width = 0.18
    for k, (mk, title, lo) in enumerate([("dino_to_ref", "DINOv2 fidelity ↑", False),
                                          ("lpips_to_ref", "LPIPS ↓", True)]):
        for vi, v in enumerate(CV):
            vals = [cb[(cb["category"] == c) & (cb["variant"] == v)][mk].mean() for c in cats]
            ax[k].bar(np.arange(len(cats)) + vi * width, vals, width, color=COL[v], label=SH[v])
        ax[k].set_xticks(np.arange(len(cats)) + 1.5 * width); ax[k].set_xticklabels(cats, rotation=20, fontsize=8)
        ax[k].set_title(title)
    ax[0].legend(fontsize=7)
    fig.suptitle("Per-category behaviour at primary skip ratio", weight="bold")
    return _save(fig, "fig09_category.png")


# ----------------------------------------------------------------- 8. failure gallery
def failure_gallery(k=4):
    df = _df()
    prim = df[(df["is_primary"] == True) & (df["variant"] != "full_compute_reference")]
    worst = prim.sort_values("lpips_to_ref", ascending=False).head(k)
    sz = 200
    cols = ["input", "full_compute_reference", "spectral_full_prediction_cache", "spectral_edit_delta_cache"]
    head = ["input", "full (ref)", "spec full-cache", "spec delta-cache"]
    rows = []
    for _, r in worst.iterrows():
        sdir = C.SAMP / r["id"]; cells = []
        for c, h in zip(cols, head):
            fp = sdir / f"{c}.png"
            im = Image.open(fp).convert("RGB").resize((sz, sz)) if fp.exists() else Image.new("RGB", (sz, sz), "#ddd")
            cells.append(_label(im, h))
        cap = Image.new("RGB", (sz, sz + 22), "white")
        ImageDraw.Draw(cap).multiline_text((5, 5),
            f"{r['id']}\n[{r['variant'].replace('_cache','')}]\nLPIPS {r['lpips_to_ref']:.2f}\nfailing variant", fill="black", font=_font(11), spacing=3)
        row = Image.new("RGB", (sz * (len(cells) + 1), cells[0].height), "white")
        row.paste(cap, (0, 0))
        for j, cc in enumerate(cells):
            row.paste(cc, (sz * (j + 1), 0))
        rows.append(row)
    grid = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        grid.paste(r, (0, y)); y += r.height
    p = C.FIG / "fig10_failures.png"; grid.save(p); return p


# ----------------------------------------------------------------- 9. representation (FFT) visuals
def representation_visuals():
    npzs = sorted(C.DIAG.glob("spectra_*.npz"))
    if not npzs:
        return None
    d = np.load(npzs[0]); eid = npzs[0].stem.replace("spectra_", "")
    ae, ad = d["amp_edit"], d["amp_delta"]; n = ae.shape[0]
    idx = np.linspace(0, n - 1, min(6, n)).astype(int)
    fig, ax = plt.subplots(2, len(idx), figsize=(2.1 * len(idx), 4.6))
    for j, t in enumerate(idx):
        ax[0, j].imshow(ae[t], cmap="magma"); ax[0, j].set_title(f"t={t}", fontsize=8)
        ax[1, j].imshow(ad[t], cmap="magma")
        for a in (ax[0, j], ax[1, j]):
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
    ax[0, 0].set_ylabel("v_edit\nlog|FFT|", fontsize=9); ax[1, 0].set_ylabel("Δ_edit\nlog|FFT|", fontsize=9)
    fig.suptitle(f"What the cache 'sees' — FFT amplitude of v_edit vs Δ_edit over denoising ({eid})", weight="bold")
    return _save(fig, "fig11_representation_fft.png")


def all_figs():
    out = {}
    out["exec"] = exec_summary()
    out["pareto"] = pareto_plots()
    out["pareto_speedup"] = pareto_speedup()
    out["smooth_traj"] = smoothness_traj()
    out["smooth_agg"] = smoothness_aggregate()
    out["freq"] = freq_bands()
    out["heatmap"] = cacheability_heatmaps()
    out["category"] = category_breakdown()
    out["failures"] = failure_gallery()
    out["repr"] = representation_visuals()
    out["quals"] = qualitative_grids()
    print("[viz] figures:", {k: (str(v) if not isinstance(v, list) else f"{len(v)} files") for k, v in out.items()})
    return out


if __name__ == "__main__":
    all_figs()
