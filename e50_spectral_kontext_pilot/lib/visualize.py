"""E50 visuals: grids, Fourier visuals, representation heatmaps, and a walkthrough video.

Run in anaconda env (matplotlib + PIL + imageio). Visuals-first, per the brief: grids and Fourier
panels carry the argument; the heatmaps/scatters are diagnostic summaries.
"""
from __future__ import annotations
import sys, json, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import spectral as S

REPO = C.REPO
F = C.FIG


def P(rel):
    return REPO / rel if rel else None


def load_rows():
    return [r for r in csv.DictReader(open(C.METRICS / "e50_metrics.csv"))]


def load_results():
    return [json.loads(l) for l in open(C.MANIFESTS / "e50_results.jsonl") if l.strip()]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _img(path, size=256):
    return Image.open(P(path)).convert("RGB").resize((size, size)) if path and P(path).exists() else \
        Image.new("RGB", (size, size), (40, 40, 40))


def _label(img, text, sub=""):
    im = img.copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    d.text((3, 4), text[:34], fill=(255, 255, 255))
    if sub:
        d.rectangle([0, im.height - 16, im.width, im.height], fill=(0, 0, 0))
        d.text((3, im.height - 14), sub[:40], fill=(180, 220, 255))
    return im


def _row(images, labels, subs=None, size=256):
    subs = subs or [""] * len(images)
    tiles = [_label(im.resize((size, size)), lb, sb) for im, lb, sb in zip(images, labels, subs)]
    w = size * len(tiles)
    canvas = Image.new("RGB", (w, size), (20, 20, 20))
    for i, t in enumerate(tiles):
        canvas.paste(t, (i * size, 0))
    return canvas


def _stack(rows):
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows)
    canvas = Image.new("RGB", (w, h), (20, 20, 20))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y)); y += r.height
    return canvas


def _diff(a, b, size=256):
    aa = np.asarray(a.resize((size, size)), np.float32)
    bb = np.asarray(b.resize((size, size)), np.float32)
    d = np.abs(aa - bb).mean(-1)
    d = (d / (d.max() + 1e-6) * 255).astype(np.uint8)
    return Image.fromarray(d).convert("RGB")


# ---------------------------------------------------------------- 1. source-spectral grids
def source_grids(rows, results):
    res = {r["job_id"]: r for r in results}
    by_src = {}
    for r in rows:
        if r["experiment"] != "spectral_source":
            continue
        by_src.setdefault(r["source_id"], {})[r["spectral_op"]] = r
    panel_rows = []
    for sid, ops in by_src.items():
        base = ops.get("raw")
        imgs, labs, subs = [], [], []
        # source
        cref = base["output"] if base else None
        src_path = res.get(f"src_{sid}_raw", {}).get("content_ref")
        imgs.append(_img(src_path)); labs.append(f"source {sid}"); subs.append(res.get(f"src_{sid}_raw", {}).get("task_type", ""))
        for op in C.SOURCE_OPS:
            r = ops.get(op)
            if not r:
                continue
            imgs.append(_img(r["output"]))
            labs.append(op)
            subs.append(f"dino {_f(r['dino_content']):.2f} gain {_f(r.get('clipT_gain', 'nan')):+.2f}")
        panel_rows.append(_row(imgs, labs, subs))
    if panel_rows:
        _stack(panel_rows).save(F / "grids" / "source_spectral_grid.png")
    print("  source_spectral_grid.png")


# ---------------------------------------------------------------- 2. reference-spectral grids
def reference_grids(rows, results):
    res = {r["job_id"]: r for r in results}
    by_pair = {}
    for r in rows:
        if r["experiment"] != "spectral_reference":
            continue
        by_pair.setdefault(r["source_id"] + "|" + str(r["style_id"]), {})[r["spectral_op"]] = r
    panel_rows = []
    for key, ops in by_pair.items():
        any_r = next(iter(ops.values()))
        cref = any_r and res.get(any_r["job_id"], {}).get("content_ref")
        sref = any_r and res.get(any_r["job_id"], {}).get("style_ref")
        imgs = [_img(cref), _img(sref)]
        labs = ["content", "style ref"]
        subs = ["", any_r.get("style_category", "")]
        for op in C.REF_OPS:
            r = ops.get(op)
            if not r:
                continue
            # show the spectral composite input + the kontext output
            inp = res.get(r["job_id"], {}).get("input_image")
            imgs.append(_img(r["output"]))
            labs.append(op)
            subs.append(f"leakgap {_f(r.get('leak_gap','nan')):+.2f} sty {_f(r.get('clipI_style','nan')):.2f}")
        panel_rows.append(_row(imgs, labs, subs))
    if panel_rows:
        _stack(panel_rows).save(F / "grids" / "reference_spectral_grid.png")
    # also a leakage-focused grid: composite INPUT vs OUTPUT for the two phase/amp swaps
    print("  reference_spectral_grid.png")


# ---------------------------------------------------------------- 3. prompt-variant grid
def prompt_grid(rows, results):
    res = {r["job_id"]: r for r in results}
    by_pair = {}
    for r in rows:
        if r["experiment"] != "prompt_variants":
            continue
        by_pair.setdefault(r["source_id"] + "|" + str(r["style_id"]), {})[r["prompt_key"]] = r
    panel_rows = []
    for key, pk in by_pair.items():
        any_r = next(iter(pk.values()))
        cref = res.get(any_r["job_id"], {}).get("content_ref")
        sref = res.get(any_r["job_id"], {}).get("style_ref")
        imgs = [_img(cref), _img(sref)]
        labs = ["content", "style ref"]
        subs = ["", any_r.get("style_category", "")]
        for k in ["neutral", "content_preserving", "anti_leakage"]:
            r = pk.get(k)
            if not r:
                continue
            imgs.append(_img(r["output"]))
            labs.append(k)
            subs.append(f"leakgap {_f(r.get('leak_gap','nan')):+.2f}")
        panel_rows.append(_row(imgs, labs, subs))
    if panel_rows:
        _stack(panel_rows).save(F / "grids" / "prompt_variant_grid.png")
    print("  prompt_variant_grid.png")


# ---------------------------------------------------------------- 4. Fourier visuals
def fourier_panels(results):
    # pick a couple of representative spectral inputs (one source, one reference composite)
    picks = []
    for r in results:
        if r["job_id"] in ("src_pie_7_0_raw", "ref_leak_3_adversarial_content_phase_style_amp"):
            picks.append(r)
    for r in picks:
        inp = P(r["input_image"])
        if not inp or not inp.exists():
            continue
        base = Image.open(inp).convert("RGB").resize((256, 256))
        fig, ax = plt.subplots(2, 3, figsize=(11, 7.5))
        ax[0, 0].imshow(base); ax[0, 0].set_title("input image")
        ax[0, 1].imshow(S.amp_spectrum_img(base), cmap="magma"); ax[0, 1].set_title("log amplitude spectrum")
        ax[0, 2].imshow(S.phase_img(base), cmap="twilight"); ax[0, 2].set_title("phase")
        ax[1, 0].imshow(S.op_low_band(base)); ax[1, 0].set_title("low band (<0.15)")
        ax[1, 1].imshow(S.op_mid_band(base)); ax[1, 1].set_title("mid band (0.15-0.45)")
        ax[1, 2].imshow(S.op_high_band(base)); ax[1, 2].set_title("high band (>0.45)")
        for a in ax.ravel():
            a.axis("off")
        fig.suptitle(f"Fourier decomposition — {r['job_id']}", fontsize=12)
        fig.tight_layout()
        fig.savefig(F / "fourier" / f"fourier_{r['job_id']}.png", dpi=95)
        plt.close(fig)
    # radial power spectrum: source raw vs phase_only vs amplitude_only vs high_band
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rmap = {r["job_id"]: r for r in results}
    for op, col in [("raw", "k"), ("phase_only", "tab:blue"), ("amplitude_only", "tab:red"),
                    ("low_band", "tab:green"), ("high_band", "tab:orange")]:
        jid = f"src_pie_7_0_{op}"
        r = rmap.get(jid)
        if r and P(r["output"]).exists():
            x, y = S.radial_power(Image.open(P(r["output"])).convert("RGB"))
            ax.plot(x, y, col, label=op, lw=1.5)
    ax.set_xlabel("radial frequency"); ax.set_ylabel("log amplitude")
    ax.set_title("Radial power spectrum of Kontext outputs by source spectral op (pie_7_0)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "fourier" / "radial_power_by_op.png", dpi=95); plt.close(fig)
    print("  fourier panels + radial_power_by_op.png")


# ---------------------------------------------------------------- 5. representation heatmaps + scatters
def _heat(matrix, rlabels, clabels, title, path, fmt="{:.2f}", cmap="viridis"):
    M = np.array(matrix, float)
    if M.size == 0 or not len(rlabels) or not len(clabels):
        print(f"  [skip empty heatmap] {Path(path).name}")
        return
    fig, ax = plt.subplots(figsize=(1.1 * len(clabels) + 3, 0.5 * len(rlabels) + 2))
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(clabels))); ax.set_xticklabels(clabels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(rlabels))); ax.set_yticklabels(rlabels, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if im.norm(M[i, j]) < 0.6 else "black")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, fraction=0.025); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def representation(rows):
    import statistics as stt
    # spectral_op x metric (source experiment)
    src = [r for r in rows if r["experiment"] == "spectral_source"]
    ops = [o for o in C.SOURCE_OPS if any(r["spectral_op"] == o for r in src)]
    mets = ["dino_content", "clipI_content", "lpips_content", "clipT_target", "clipT_gain"]
    M = [[stt.mean([_f(r[m]) for r in src if r["spectral_op"] == o and r.get(m) not in (None, "")] or [float("nan")])
          for m in mets] for o in ops]
    _heat(M, ops, mets, "Source spectral op x metric (instruction edits)",
          F / "representation_visuals" / "heat_source_op_metric.png", cmap="viridis")

    # op x task (dino_content)
    tasks = sorted({r["task_type"] for r in src})
    Mt = [[stt.mean([_f(r["dino_content"]) for r in src if r["spectral_op"] == o and r["task_type"] == t] or [float("nan")])
           for t in tasks] for o in ops]
    _heat(Mt, ops, tasks, "DINO content preservation: source op x task",
          F / "representation_visuals" / "heat_source_op_task.png", cmap="cividis")

    # reference op x style metric
    ref = [r for r in rows if r["bucket"] in ("spectral_reference", "kontext_baseline_replay")]
    rops = [o for o in C.REF_OPS if any(r["spectral_op"] == o for r in ref)]
    smets = ["dino_content", "clipI_style", "dino_style", "leak_gap", "fourier_style"]
    Mr = [[stt.mean([_f(r[m]) for r in ref if r["spectral_op"] == o and r.get(m) not in (None, "")] or [float("nan")])
           for m in smets] for o in rops]
    _heat(Mr, rops, smets, "Reference spectral op x style/leakage metric",
          F / "representation_visuals" / "heat_reference_op_metric.png", cmap="magma")

    # preservation vs edit scatter (source)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cmap = {o: c for o, c in zip(C.SOURCE_OPS, ["k", "tab:blue", "tab:red", "tab:green", "tab:orange"])}
    for r in src:
        ax.scatter(_f(r["dino_content"]), _f(r.get("clipT_gain", "nan")),
                   c=cmap.get(r["spectral_op"], "gray"), s=55, edgecolor="white", lw=0.5)
    for o in ops:
        ax.scatter([], [], c=cmap.get(o, "gray"), label=o, s=55)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("DINO content preservation ->"); ax.set_ylabel("CLIP-T gain (edit) ->")
    ax.set_title("Source spectral ops: preservation vs edit strength")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(F / "representation_visuals" / "scatter_preservation_vs_edit.png", dpi=110); plt.close(fig)

    # style adherence vs leakage scatter (reference)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    rcmap = {o: c for o, c in zip(C.REF_OPS, ["k", "tab:green", "tab:red", "tab:purple"])}
    for r in ref:
        ax.scatter(_f(r.get("clipI_style", "nan")), _f(r.get("leak_gap", "nan")),
                   c=rcmap.get(r["spectral_op"], "gray"), s=60, edgecolor="white", lw=0.5)
    for o in rops:
        ax.scatter([], [], c=rcmap.get(o, "gray"), label=o, s=60)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("CLIP-I style adherence ->"); ax.set_ylabel("leak gap (content - style) ->")
    ax.set_title("Reference spectral ops: style adherence vs leakage resistance")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(F / "representation_visuals" / "scatter_style_vs_leakage.png", dpi=110); plt.close(fig)

    # prompt-variant bar (leak_gap)
    pv = [r for r in rows if r["experiment"] == "prompt_variants"]
    if pv:
        keys = ["neutral", "content_preserving", "anti_leakage"]
        vals = [stt.mean([_f(r["leak_gap"]) for r in pv if r["prompt_key"] == k and r.get("leak_gap") not in (None, "")] or [float("nan")]) for k in keys]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(keys, vals, color=["tab:gray", "tab:blue", "tab:green"])
        ax.set_ylabel("mean leak gap (content - style)")
        ax.set_title("Prompt wording vs reference-leakage resistance")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:+.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout(); fig.savefig(F / "representation_visuals" / "bar_prompt_leakage.png", dpi=110); plt.close(fig)
    print("  representation visuals")


# ---------------------------------------------------------------- 6. best / worst / leakage cases
def case_grids(rows, results):
    res = {r["job_id"]: r for r in results}
    src = [r for r in rows if r["experiment"] == "spectral_source" and r["spectral_op"] != "raw"]
    # best = highest (dino_content + clipT_gain); worst = lowest
    def score(r):
        return _f(r["dino_content"]) + _f(r.get("clipT_gain", "nan"))
    src_ok = [r for r in src if not np.isnan(score(r))]
    src_ok.sort(key=score, reverse=True)
    def case_row(r):
        cref = res.get(r["job_id"], {}).get("content_ref")
        base = res.get(f"src_{r['source_id']}_raw", {})
        return _row([_img(cref), _img(base.get("output")), _img(r["output"]),
                     _diff(_img(base.get("output")), _img(r["output"]))],
                    ["source", "raw->kontext", f"{r['spectral_op']}", "diff"],
                    ["", f"gain {_f(base.get('clipT_gain') if False else 'nan'):.2f}",
                     f"dino {_f(r['dino_content']):.2f} gain {_f(r.get('clipT_gain','nan')):+.2f}", ""])
    if src_ok:
        _stack([case_row(r) for r in src_ok[:4]]).save(F / "best_cases" / "best_source_edits.png")
        _stack([case_row(r) for r in src_ok[-4:]]).save(F / "worst_cases" / "worst_source_edits.png")
    # leakage: reference ops sorted by leak_gap (low = catastrophic leak)
    ref = [r for r in rows if r["experiment"] == "spectral_reference" and r.get("leak_gap") not in (None, "")]
    ref.sort(key=lambda r: _f(r["leak_gap"]))
    def leak_row(r):
        cref = res.get(r["job_id"], {}).get("content_ref")
        sref = res.get(r["job_id"], {}).get("style_ref")
        return _row([_img(cref), _img(sref), _img(r["output"])],
                    ["content", "style ref", f"{r['spectral_op']}"],
                    ["", r.get("style_category", ""), f"leakgap {_f(r['leak_gap']):+.2f}"])
    if ref:
        _stack([leak_row(r) for r in ref[:4]]).save(F / "leakage_cases" / "worst_leakage.png")
        _stack([leak_row(r) for r in ref[-4:]]).save(F / "leakage_cases" / "best_leakage_resistant.png")
    print("  best/worst/leakage case grids")


# ---------------------------------------------------------------- 7. walkthrough video
def video(rows, results):
    import imageio.v2 as imageio
    res = {r["job_id"]: r for r in results}
    SZ = 320
    frames = []
    def titled(im, title, sub):
        canvas = Image.new("RGB", (SZ * 3, SZ + 40), (12, 12, 18))
        d = ImageDraw.Draw(canvas)
        d.text((8, 8), title[:70], fill=(255, 255, 255))
        d.text((8, 24), sub[:90], fill=(150, 200, 255))
        return canvas
    # source experiment montage: per source, raw then each op
    by_src = {}
    for r in rows:
        if r["experiment"] == "spectral_source":
            by_src.setdefault(r["source_id"], {})[r["spectral_op"]] = r
    for sid, ops in by_src.items():
        cref = res.get(f"src_{sid}_raw", {}).get("content_ref")
        for op in C.SOURCE_OPS:
            r = ops.get(op)
            if not r:
                continue
            canvas = titled(None, f"SOURCE SPECTRAL — {sid} ({r['task_type']})",
                            f"op={op}  dino_content={_f(r['dino_content']):.2f}  clipT_gain={_f(r.get('clipT_gain','nan')):+.2f}")
            canvas.paste(_label(_img(cref, SZ), "source"), (0, 40))
            canvas.paste(_label(_img(r['input_image'] if False else res[r['job_id']]['input_image'], SZ), f"input ({op})"), (SZ, 40))
            canvas.paste(_label(_img(r["output"], SZ), "Kontext output"), (SZ * 2, 40))
            for _ in range(18):
                frames.append(np.asarray(canvas))
    # reference experiment montage
    by_pair = {}
    for r in rows:
        if r["experiment"] == "spectral_reference":
            by_pair.setdefault(r["source_id"] + "|" + str(r["style_id"]), {})[r["spectral_op"]] = r
    for key, ops in by_pair.items():
        for op in C.REF_OPS:
            r = ops.get(op)
            if not r:
                continue
            cref = res.get(r["job_id"], {}).get("content_ref")
            sref = res.get(r["job_id"], {}).get("style_ref")
            canvas = titled(None, f"SPECTRAL REFERENCE — {r['source_id']} x {r.get('style_category','')}",
                            f"op={op}  clipI_style={_f(r.get('clipI_style','nan')):.2f}  leak_gap={_f(r.get('leak_gap','nan')):+.2f}")
            canvas.paste(_label(_img(cref, SZ), "content"), (0, 40))
            canvas.paste(_label(_img(sref, SZ), "style ref"), (SZ, 40))
            canvas.paste(_label(_img(r["output"], SZ), f"Kontext ({op})"), (SZ * 2, 40))
            for _ in range(18):
                frames.append(np.asarray(canvas))
    if frames:
        C.VIDEO.mkdir(parents=True, exist_ok=True)
        out = C.VIDEO / "e50_kontext_spectral_walkthrough.mp4"
        imageio.mimsave(out, frames, fps=12, macro_block_size=1)
        print(f"  video -> {out} ({len(frames)} frames)")


def main():
    for d in [F / "grids", F / "fourier", F / "representation_visuals",
              F / "best_cases", F / "worst_cases", F / "leakage_cases"]:
        d.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    results = load_results()
    print("visuals:")
    source_grids(rows, results)
    reference_grids(rows, results)
    prompt_grid(rows, results)
    fourier_panels(results)
    representation(rows)
    case_grids(rows, results)
    video(rows, results)
    print("done.")


if __name__ == "__main__":
    main()
