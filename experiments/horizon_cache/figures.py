"""Figure generation for the HorizonCache report — matplotlib → PNG files (also usable as
data URIs). Dark aesthetic matching the project deck."""
from __future__ import annotations

import base64
import collections
import io
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .scheduler import ACTIONS

BG = "#0e1116"; PANEL = "#161b22"; INK = "#e6edf3"; MUT = "#9aa7b4"
ACC = "#6ea8fe"; ACC2 = "#7ee787"; WARN = "#f0b429"; BAD = "#ff7b72"; LINE = "#283039"
ACTION_COLOR = {"fresh": BAD, "cache": ACC, "jump_1.25": ACC2, "jump_1.5": WARN, "jump_2.0": "#d2a8ff"}


def _style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(True, color=LINE, alpha=0.4, linewidth=0.6)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


def data_uri(path: Path) -> str:
    b = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


# ---------------------------------------------------------------- frontier
def frontier_plot(rows: list[dict], tau_grid, out: Path, ycol="psnr", ylabel="PSNR vs full (dB) ↑", higher=True):
    by_method = collections.defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    def mean(rs, k):
        vals = [r[k] for r in rs if k in r and r[k] is not None]
        return float(np.mean(vals)) if vals else None

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _style(ax)
    # SeaCache frontier (sweep tau)
    sea = sorted([(mean(by_method[f"seacache_t{t:g}"], "compute_speedup"),
                   mean(by_method[f"seacache_t{t:g}"], ycol)) for t in tau_grid
                  if by_method.get(f"seacache_t{t:g}")])
    hor = sorted([(mean(by_method[f"horizon_v0_t{t:g}"], "compute_speedup"),
                   mean(by_method[f"horizon_v0_t{t:g}"], ycol)) for t in tau_grid
                  if by_method.get(f"horizon_v0_t{t:g}")])
    if sea:
        xs, ys = zip(*sea)
        ax.plot(xs, ys, "-o", color=ACC, label="SeaCache", linewidth=2, markersize=7)
    if hor:
        xs, ys = zip(*hor)
        ax.plot(xs, ys, "-o", color=ACC2, label="HorizonCache-v0", linewidth=2, markersize=7)
    # baselines as points
    for name, col, lab in [("uniform_k2", MUT, "uniform k2"), ("uniform_k3", "#6b7684", "uniform k3"),
                           ("random_k", BAD, "random-k")]:
        if by_method.get(name):
            ax.scatter([mean(by_method[name], "compute_speedup")], [mean(by_method[name], ycol)],
                       color=col, s=55, marker="s", label=lab, zorder=5)
    # jump2 ablation
    ab = [m for m in by_method if m.startswith("horizon_v0_jump2")]
    if ab:
        ax.scatter([mean(by_method[ab[0]], "compute_speedup")], [mean(by_method[ab[0]], ycol)],
                   color="#d2a8ff", s=70, marker="*", label="v0 +jump2 (ablation)", zorder=6)
    # v1
    if by_method.get("horizon_v1"):
        ax.scatter([mean(by_method["horizon_v1"], "compute_speedup")], [mean(by_method["horizon_v1"], ycol)],
                   color=WARN, s=90, marker="D", label="HorizonCache-v1", zorder=7)
    ax.set_xlabel("achieved speedup (block-stack-equivalent) →")
    ax.set_ylabel(ylabel)
    ax.set_title("Frontier: quality vs achieved speedup (FLUX)")
    leg = ax.legend(fontsize=8, facecolor=PANEL, edgecolor=LINE, labelcolor=INK, loc="best")
    return _save(fig, out)


# ---------------------------------------------------------------- per-image delta
def delta_plot(rows, tau_grid, out: Path):
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["method"]][r["key"]] = r
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _style(ax)
    xs, ys, es = [], [], []
    for t in tau_grid:
        sea = by.get(f"seacache_t{t:g}", {}); hor = by.get(f"horizon_v0_t{t:g}", {})
        keys = set(sea) & set(hor)
        if not keys:
            continue
        d = [hor[k]["psnr"] - sea[k]["psnr"] for k in keys]
        sp = np.mean([hor[k]["compute_speedup"] for k in keys])
        xs.append(sp); ys.append(np.mean(d)); es.append(np.std(d))
    if xs:
        ax.errorbar(xs, ys, yerr=es, fmt="-o", color=ACC2, ecolor=MUT, capsize=4, linewidth=2)
    ax.axhline(0, color=BAD, linewidth=1, linestyle="--")
    ax.set_xlabel("HorizonCache achieved speedup →")
    ax.set_ylabel("Δ PSNR vs SeaCache (same τ) dB")
    ax.set_title("Per-image ΔPSNR (HorizonCache-v0 − SeaCache), same τ")
    return _save(fig, out)


# ---------------------------------------------------------------- action timeline
def action_timeline(trace_path: Path, out: Path, title: str):
    d = json.loads(Path(trace_path).read_text())
    tr = d["traces"]
    steps = [t["step_index"] for t in tr]
    sigma = [t["sigma"] for t in tr]
    acc = [t["acc_rel_l1"] for t in tr]
    raw = [t["raw_rel_l1"] for t in tr]
    actions = [t["action"] for t in tr]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2]})
    _style(ax1); _style(ax2)
    # top: action bars along sigma
    for s, a in zip(steps, actions):
        ax1.axvspan(s - 0.5, s + 0.5, color=ACTION_COLOR.get(a, MUT), alpha=0.85)
    ax1.set_yticks([]); ax1.set_ylabel("action")
    ax1.set_title(title)
    handles = [plt.Line2D([0], [0], color=ACTION_COLOR[a], lw=8) for a in ACTIONS]
    ax1.legend(handles, ACTIONS, ncol=5, fontsize=7, facecolor=PANEL, edgecolor=LINE,
               labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, 2.0))
    # bottom: score traces
    ax2.plot(steps, acc, "-o", color=ACC, label="accumulated relL1", markersize=3)
    ax2.plot(steps, raw, "-", color=ACC2, label="raw relL1", alpha=0.8)
    ax2b = ax2.twinx()
    ax2b.plot(steps, sigma, "--", color=MUT, label="sigma", alpha=0.7)
    ax2b.set_ylabel("sigma", color=MUT); ax2b.tick_params(colors=MUT)
    for s in ax2b.spines.values():
        s.set_color(LINE)
    ax2.set_xlabel("step index"); ax2.set_ylabel("SeaCache score")
    ax2.legend(fontsize=8, facecolor=PANEL, edgecolor=LINE, labelcolor=INK, loc="upper right")
    return _save(fig, out)


# ---------------------------------------------------------------- safe-horizon scatter
def safe_horizon_scatter(csv_path: Path, out: Path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    from .scheduler import action_to_horizon
    df["horizon"] = df["label_action"].map(action_to_horizon)
    feats = [("acc_rel_l1", "accumulated relL1"), ("sigma", "sigma"),
             ("raw_rel_l1", "raw relL1"), ("h_norm_drift", "h-norm drift")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
    for ax, (f, lab) in zip(axes.flat, feats):
        _style(ax)
        jitter = (np.random.rand(len(df)) - 0.5) * 0.08
        ax.scatter(df[f], df["horizon"] + jitter, c=df["horizon"], cmap="viridis", s=22, alpha=0.7)
        ax.set_xlabel(lab); ax.set_ylabel("safe jump horizon")
    fig.suptitle("Safe-horizon labels vs causal features (rollout)", color=INK)
    return _save(fig, out)


# ---------------------------------------------------------------- confusion matrix
def confusion_fig(cm: dict, out: Path):
    labels = [a for a in ACTIONS if any(cm.get(a, {}).values()) or any(cm.get(t, {}).get(a, 0) for t in ACTIONS)]
    labels = labels or ACTIONS
    mat = np.array([[cm.get(t, {}).get(p, 0) for p in labels] for t in labels], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.set_facecolor(PANEL)
    im = ax.imshow(mat, cmap="magma")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", color=MUT, fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, color=MUT, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] < mat.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("predicted", color=INK); ax.set_ylabel("true (safe-horizon label)", color=INK)
    ax.set_title("HorizonCache-v1 confusion", color=INK)
    return _save(fig, out)
