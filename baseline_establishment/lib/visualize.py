"""Build all visuals: image grids, representation plots, and the MP4 walkthrough.
Run in the anaconda env (matplotlib + PIL + imageio).
  python visualize.py --phase pilot
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

THUMB = 256
EDIT_MODELS = ["flux_img2img", "flux_kontext"]
STYLE_MODELS = ["flux_redux", "flux_ipadapter", "styleid", "flux_kontext"]
PRETTY = {k: v["name"] for k, v in C.MODELS.items()}


# ----------------------------------------------------------------- helpers
def _font(sz=15):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _img(path, t=THUMB):
    try:
        return Image.open(C.ROOT / path if not str(path).startswith("/") else path).convert("RGB").resize((t, t))
    except Exception:
        return Image.new("RGB", (t, t), (40, 40, 40))


def labeled_grid(cells, col_labels, row_labels=None, t=THUMB, title=None, pad=6, wrap=28):
    """cells: 2D list of PIL images (rows x cols)."""
    f, ftitle = _font(14), _font(17)
    rows, cols = len(cells), len(cells[0])
    head = 26
    left = 150 if row_labels else 0
    top = (34 if title else 0) + head
    W = left + cols * (t + pad) + pad
    H = top + rows * (t + pad) + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((pad, 6), title, fill=(10, 10, 10), font=ftitle)
    for c, lab in enumerate(col_labels):
        x = left + c * (t + pad) + pad
        d.text((x, top - head + 4), lab[:wrap], fill=(20, 20, 90), font=f)
    for r in range(rows):
        if row_labels:
            d.text((6, top + r * (t + pad) + t // 2 - 8), row_labels[r][:22], fill=(20, 20, 20), font=f)
        for c in range(cols):
            x = left + c * (t + pad) + pad
            y = top + r * (t + pad) + pad
            canvas.paste(cells[r][c], (x, y))
    return canvas


def load_metrics(phase):
    p = C.METRICS / "baseline_establishment_metrics.csv"
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    pass
            rows.append(r)
    return rows


def index_outputs(phase):
    """job_id -> {model: output_path}; plus job meta."""
    by_job = defaultdict(dict)
    meta = {}
    for mid in C.MODELS:
        man = C.OUT / mid / f"manifest_{phase}.jsonl"
        if not man.exists():
            continue
        for line in man.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("ok") and r.get("output"):
                by_job[r["job_id"]][mid] = r["output"]
                meta[r["job_id"]] = r
    return by_job, meta


# ----------------------------------------------------------------- image grids
def edit_grids(by_job, meta, n=8):
    jobs = [j for j, m in meta.items() if m["kind"] == "edit" and
            any(mm in by_job[j] for mm in EDIT_MODELS)]
    jobs = sorted(jobs)[:n]
    cells, rows = [], []
    cols = ["content"] + [PRETTY[m] for m in EDIT_MODELS]
    for j in jobs:
        m = meta[j]
        row = [_img(m["content"])]
        for mm in EDIT_MODELS:
            row.append(_img(by_job[j][mm]) if mm in by_job[j] else Image.new("RGB", (THUMB, THUMB), (60, 60, 60)))
        cells.append(row)
        rows.append(f"{m.get('task_type','')[:10]}")
    g = labeled_grid(cells, cols, rows, title="Instruction editing — content vs FLUX editors")
    g.save(C.FIG / "grids" / "edit_comparison.png")
    return g


def style_grids(by_job, meta, n=8):
    jobs = [j for j, m in meta.items() if m["kind"] == "style"]
    jobs = sorted(jobs)[:n]
    cols = ["content", "style ref"] + [PRETTY[m] for m in STYLE_MODELS]
    cells, rows = [], []
    for j in jobs:
        m = meta[j]
        row = [_img(m["content"]), _img(m["style"])]
        for mm in STYLE_MODELS:
            row.append(_img(by_job[j][mm]) if mm in by_job[j] else Image.new("RGB", (THUMB, THUMB), (60, 60, 60)))
        cells.append(row)
        rows.append(f"{m.get('pair_type','')[:9]}/{m.get('style_category','')[:7]}")
    g = labeled_grid(cells, cols, rows, title="Style transfer — content + reference vs style baselines")
    g.save(C.FIG / "grids" / "style_comparison.png")
    return g


def best_worst_grids(rows, by_job, meta):
    # editing: best/worst by clipT_gain among edit rows that have it
    er = [r for r in rows if r.get("kind") == "edit" and isinstance(r.get("clipT_gain"), float)]
    er.sort(key=lambda r: r["clipT_gain"])
    def edit_panel(sel, title, fname):
        cells, cols, rowl = [], ["content", "output"], []
        for r in sel:
            m = meta.get(r["job_id"], {})
            cells.append([_img(m.get("content", "")), _img(r["output"])])
            rowl.append(f"{r['model'][:8]} g={r['clipT_gain']:.2f}")
        if cells:
            labeled_grid(cells, cols, rowl, title=title).save(C.FIG / fname)
    edit_panel(er[-6:][::-1], "Best edits (highest CLIP-T gain)", "best_cases/best_edits.png")
    edit_panel(er[:6], "Worst edits (lowest CLIP-T gain)", "worst_cases/worst_edits.png")
    # leakage: style rows ranked by dino_style (high = copies reference semantics)
    sr = [r for r in rows if r.get("kind") == "style" and isinstance(r.get("dino_style"), float)]
    sr.sort(key=lambda r: r["dino_style"])
    def leak_panel(sel, title, fname):
        cells, cols, rowl = [], ["content", "style ref", "output"], []
        for r in sel:
            m = meta.get(r["job_id"], {})
            cells.append([_img(m.get("content", "")), _img(m.get("style", "")), _img(r["output"])])
            rowl.append(f"{r['model'][:8]} L={r['dino_style']:.2f}")
        if cells:
            labeled_grid(cells, cols, rowl, title=title).save(C.FIG / fname)
    leak_panel(sr[-6:][::-1], "Highest reference leakage (DINO->style ref)", "leakage_cases/high_leakage.png")
    leak_panel(sr[:6], "Lowest reference leakage (content preserved)", "leakage_cases/low_leakage.png")


# ----------------------------------------------------------------- representation visuals
def rep_visuals(rows):
    fig_dir = C.FIG / "representation_visuals"
    models = [m for m in C.MODELS if any(r["model"] == m for r in rows)]

    def mean(model, key, pred=lambda r: True):
        v = [r[key] for r in rows if r["model"] == model and isinstance(r.get(key), float) and pred(r)]
        return float(np.mean(v)) if v else np.nan

    # 1) model x metric heatmap
    metric_keys = ["clipI_content", "dino_content", "siglip_content", "lpips_content",
                   "clipT_target", "clipT_gain", "clipI_style", "dino_style",
                   "colorhist_style", "fourier_style"]
    Mtx = np.array([[mean(m, k) for k in metric_keys] for m in models])
    # column-normalize for color
    Z = (Mtx - np.nanmean(Mtx, 0)) / (np.nanstd(Mtx, 0) + 1e-9)
    plt.figure(figsize=(11, 4 + 0.3 * len(models)))
    plt.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    plt.xticks(range(len(metric_keys)), metric_keys, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(models)), [PRETTY[m] for m in models], fontsize=9)
    for i in range(len(models)):
        for j in range(len(metric_keys)):
            if not np.isnan(Mtx[i, j]):
                plt.text(j, i, f"{Mtx[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("Model × metric (z-scored color, raw labels)")
    plt.colorbar(label="z"); plt.tight_layout()
    plt.savefig(fig_dir / "heatmap_model_metric.png", dpi=120); plt.close()

    # 2) task-type x model heatmap (CLIP-T gain for edits)
    tasks = sorted({r["task_type"] for r in rows if r.get("kind") == "edit"})
    em = [m for m in models if m in EDIT_MODELS]
    if tasks and em:
        H = np.array([[mean(m, "clipT_gain", lambda r, t=t: r["task_type"] == t) for m in em] for t in tasks])
        plt.figure(figsize=(2 + 1.6 * len(em), 1 + 0.5 * len(tasks)))
        plt.imshow(H, aspect="auto", cmap="viridis")
        plt.xticks(range(len(em)), [PRETTY[m] for m in em], fontsize=9)
        plt.yticks(range(len(tasks)), tasks, fontsize=9)
        for i in range(len(tasks)):
            for j in range(len(em)):
                if not np.isnan(H[i, j]):
                    plt.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        plt.title("Edit task × model: CLIP-T gain"); plt.colorbar(); plt.tight_layout()
        plt.savefig(fig_dir / "heatmap_task_model.png", dpi=120); plt.close()

    # 3) scatter: content preservation vs edit strength (edits)
    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for m, col in zip(models, colors):
        xs = [r["clipT_gain"] for r in rows if r["model"] == m and r.get("kind") == "edit" and isinstance(r.get("clipT_gain"), float)]
        ys = [r["clipI_content"] for r in rows if r["model"] == m and r.get("kind") == "edit" and isinstance(r.get("clipI_content"), float)]
        n = min(len(xs), len(ys))
        if n:
            plt.scatter(xs[:n], ys[:n], label=PRETTY[m], color=col, alpha=0.7, s=40)
    plt.xlabel("edit strength  (CLIP-T gain →)"); plt.ylabel("content preservation  (CLIP-I to content →)")
    plt.title("Editing: preservation vs edit strength"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir / "scatter_preservation_vs_edit.png", dpi=120); plt.close()

    # 4) scatter: style transfer vs leakage (style jobs)
    plt.figure(figsize=(7, 6))
    for m, col in zip(models, colors):
        xs = [r["clipI_style"] for r in rows if r["model"] == m and r.get("kind") == "style" and isinstance(r.get("clipI_style"), float)]
        ys = [r["dino_content"] for r in rows if r["model"] == m and r.get("kind") == "style" and isinstance(r.get("dino_content"), float)]
        n = min(len(xs), len(ys))
        if n:
            plt.scatter(xs[:n], ys[:n], label=PRETTY[m], color=col, alpha=0.7, s=40)
    plt.xlabel("style adherence  (CLIP-I to style ref →)"); plt.ylabel("content preservation  (DINO to content →)")
    plt.title("Style transfer: adherence vs content preservation"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir / "scatter_style_vs_leakage.png", dpi=120); plt.close()

    # 5) Pareto bar: per-model leakage-resistance score (style jobs)
    #    score = content preserved (DINO content) - leakage (DINO style)
    sm = [m for m in models if any(r["model"] == m and r.get("kind") == "style" for r in rows)]
    score = [mean(m, "dino_content", lambda r: r["kind"] == "style") - mean(m, "dino_style", lambda r: r["kind"] == "style") for m in sm]
    plt.figure(figsize=(7, 4))
    order = np.argsort(score)[::-1]
    plt.bar([PRETTY[sm[i]] for i in order], [score[i] for i in order], color="#2b8a5c")
    plt.ylabel("leakage-resistance  (DINO_content − DINO_style)")
    plt.title("Reference stylization: content kept minus reference leaked")
    plt.xticks(rotation=20, ha="right", fontsize=8); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir / "pareto_leakage_resistance.png", dpi=120); plt.close()

    # 6) model similarity matrix (correlation of per-job metric vectors across shared jobs)
    keys2 = ["clipI_content", "dino_content", "clipT_target", "clipI_style", "colorhist_content"]
    vecs = {}
    for m in models:
        v = [mean(m, k) for k in keys2]
        vecs[m] = np.array(v)
    S = np.zeros((len(models), len(models)))
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            va, vb = vecs[a], vecs[b]
            mask = ~(np.isnan(va) | np.isnan(vb))
            if mask.sum() > 1:
                S[i, j] = np.corrcoef(va[mask], vb[mask])[0, 1]
    plt.figure(figsize=(6, 5))
    plt.imshow(S, cmap="magma", vmin=-1, vmax=1)
    plt.xticks(range(len(models)), [PRETTY[m] for m in models], rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(models)), [PRETTY[m] for m in models], fontsize=8)
    plt.title("Model similarity (metric-profile correlation)"); plt.colorbar()
    plt.tight_layout(); plt.savefig(fig_dir / "similarity_matrix.png", dpi=120); plt.close()

    # 7) runtime bar
    plt.figure(figsize=(7, 4))
    secs = [mean(m, "seconds") for m in models]
    plt.bar([PRETTY[m] for m in models], secs, color="#39608f")
    plt.ylabel("seconds / image"); plt.title("Inference cost")
    plt.xticks(rotation=20, ha="right", fontsize=8); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir / "runtime.png", dpi=120); plt.close()
    print("representation visuals written")


# ----------------------------------------------------------------- video
def make_video(by_job, meta, phase):
    import imageio
    frames = []
    f = _font(16)
    pad, t, head = 8, 320, 60
    MAXP = 6                               # fixed panels per frame -> constant canvas size
    W = pad + MAXP * (t + pad)
    H = t + head + pad
    for j in sorted(meta):
        m = meta[j]
        if m["kind"] == "edit":
            imgs = [("content", _img(m["content"], t))]
            for mm in EDIT_MODELS:
                if mm in by_job[j]:
                    imgs.append((PRETTY[mm], _img(by_job[j][mm], t)))
        else:
            imgs = [("content", _img(m["content"], t)), ("style", _img(m["style"], t))]
            for mm in STYLE_MODELS:
                if mm in by_job[j]:
                    imgs.append((PRETTY[mm], _img(by_job[j][mm], t)))
        if len(imgs) < 2:
            continue
        imgs = imgs[:MAXP]
        canvas = Image.new("RGB", (W, H), (18, 18, 24))    # constant size every frame
        d = ImageDraw.Draw(canvas)
        d.text((pad, 8), f"{m['kind'].upper()} · {j} · {m.get('task_type','')}{m.get('style_category','')}",
               fill=(240, 240, 240), font=f)
        for i, (lab, im) in enumerate(imgs):
            x = pad + i * (t + pad)
            canvas.paste(im, (x, head))
            d.text((x + 2, head - 20), lab[:24], fill=(180, 200, 255), font=f)
        arr = np.asarray(canvas)
        for _ in range(18):
            frames.append(arr)
    if frames:
        C.VIDEO.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(C.VIDEO / "baseline_walkthrough.mp4", frames, fps=12, quality=8)
        print(f"video: {len(frames)} frames -> baseline_walkthrough.mp4")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="pilot")
    a = ap.parse_args()
    rows = load_metrics(a.phase)
    by_job, meta = index_outputs(a.phase)
    print(f"viz: {len(rows)} metric rows, {len(meta)} jobs")
    edit_grids(by_job, meta)
    style_grids(by_job, meta)
    best_worst_grids(rows, by_job, meta)
    rep_visuals(rows)
    make_video(by_job, meta, a.phase)
    print("VISUALS DONE")


if __name__ == "__main__":
    main()
