"""Mechanism-visual package for the HorizonCache consolidation report (E56).

One job: make the method *obvious* — why a conservative headroom-adaptive stride extension
shifts the FLUX SeaCache frontier upward in the safe band, and why aggressive jumps overshoot.

All figures read a finished generation run (metrics.csv + traces/*.json) and write PNGs. The
trace schema is stable across the local A5000 sig run and the cluster H100 scale-up, so the same
code serves both. Run:

    python -m horizon_cache.mechanism_figures --gen-dir <run> --out <assets_dir>
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figures import BG, PANEL, INK, MUT, ACC, ACC2, WARN, BAD, LINE, ACTION_COLOR, _style, _save

# adaptive jumps log the action "jump_adaptive"; color them by stride so the timeline reads
# green(small/safe) → amber → red(large) instead of falling back to gray.
JUMP_PURPLE = "#d2a8ff"


# purple gradient for adaptive jumps by stride (kept distinct from fresh=red / cache=blue)
JUMP_SMALL = "#e2c6ff"; JUMP_MID = "#c08cf5"; JUMP_BIG = "#8b5cf6"


def _span_color(action: str, jf: float | None = None) -> str:
    if action.startswith("jump"):
        if jf is None:
            return JUMP_MID
        return JUMP_SMALL if jf < 1.15 else (JUMP_MID if jf < 1.4 else JUMP_BIG)
    if action in ACTION_COLOR:
        return ACTION_COLOR[action]
    return MUT


def _draw_actions(ax, tr, xmax):
    jf_by_step = {j["step"]: j["jump_factor"] for j in tr.get("jumps", [])}
    for t in tr["traces"]:
        s = t["step_index"]
        ax.axvspan(s - 0.5, s + 0.5, color=_span_color(t["action"], jf_by_step.get(s)), alpha=0.9)
    ax.set_yticks([]); ax.set_xlim(-0.5, xmax + 0.5)


# ----------------------------------------------------------------------- helpers
def _mean_speedup(df: pd.DataFrame) -> dict[str, float]:
    return df.groupby("method")["compute_speedup"].mean().to_dict()


def _closest_method(speedups: dict[str, float], prefix: str, target: float) -> str | None:
    cands = {m: s for m, s in speedups.items() if m.startswith(prefix)}
    if not cands:
        return None
    return min(cands, key=lambda m: abs(cands[m] - target))


def _load_trace(gen: Path, prompt_key: str, method: str) -> dict | None:
    p = gen / "traces" / f"{prompt_key}__{method}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _pick_prompt(gen: Path, method: str) -> str | None:
    """A representative prompt key that has a trace for `method`."""
    for p in sorted((gen / "traces").glob(f"*__{method}.json")):
        return p.name[: -len(f"__{method}.json")]
    return None


def _tau_of(method: str) -> float | None:
    if "_t" in method:
        try:
            return float(method.rsplit("_t", 1)[1])
        except ValueError:
            return None
    return None


# ------------------------------------------------------- 3.1 action-timeline pair
def mechanism_pair(gen: Path, out: Path, target_speed: float = 2.5) -> Path:
    """THE figure: SeaCache vs HorizonCache adaptive at matched achieved speedup.

    Two action rows on a shared step axis + the SeaCache score / headroom trace, so the
    reader sees *rare-refresh-long-cache* (SeaCache) against *refresh-often-small-jumps*
    (HorizonCache) at the same speed."""
    df = pd.read_csv(gen / "metrics.csv")
    sp = _mean_speedup(df)
    sea_m = _closest_method(sp, "seacache_t", target_speed)
    hor_m = (_closest_method(sp, "horizon_adaptive_1.25_t", target_speed)
             or _closest_method(sp, "horizon_", target_speed))
    if not sea_m or not hor_m:
        raise SystemExit(f"need seacache+horizon methods near {target_speed}x; have {list(sp)[:8]}")
    # a prompt with both traces
    key = None
    for cand in sorted((gen / "traces").glob(f"*__{hor_m}.json")):
        k = cand.name[: -len(f"__{hor_m}.json")]
        if (gen / "traces" / f"{k}__{sea_m}.json").exists():
            key = k
            break
    if key is None:
        raise SystemExit("no shared-prompt trace pair found")
    sea = _load_trace(gen, key, sea_m)
    hor = _load_trace(gen, key, hor_m)
    tau_h = _tau_of(hor_m)

    fig = plt.figure(figsize=(9.6, 6.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2.2], hspace=0.32)
    ax_s = fig.add_subplot(gs[0]); ax_h = fig.add_subplot(gs[1]); ax_t = fig.add_subplot(gs[2])
    for ax in (ax_s, ax_h, ax_t):
        _style(ax)

    xmax = max(len(sea["traces"]), len(hor["traces"]))  # shared axis: HorizonCache visibly ends earlier

    def _row(ax, tr, label, speed):
        _draw_actions(ax, tr, xmax)
        ax.set_ylabel(label, fontsize=10)
        nf = tr["ledger"]["num_full_forwards"]; nj = tr["ledger"]["num_jump_actions"]
        ne = tr["ledger"]["num_executed_nodes"]
        ax.text(0.005, 0.5, f"{speed:.2f}×  ·  {nf} fresh · {nj} jumps · {ne} nodes",
                transform=ax.transAxes, ha="left", va="center", color=INK, fontsize=8.5,
                bbox=dict(facecolor=BG, edgecolor=LINE, boxstyle="round,pad=0.25", alpha=0.75))

    _row(ax_s, sea, "SeaCache", sp[sea_m])
    _row(ax_h, hor, "HorizonCache\nadaptive→1.25", sp[hor_m])
    fig.suptitle(f"Matched achieved speedup (~{target_speed:g}×) — same prompt & seed ({key})",
                 fontsize=11.5, color=INK, y=1.0)
    # action legend — matches what is drawn (adaptive jumps colored by stride)
    leg = [("fresh", ACTION_COLOR["fresh"]), ("cache", ACTION_COLOR["cache"]),
           ("jump·small", JUMP_SMALL), ("jump·mid", JUMP_MID), ("jump·large", JUMP_BIG)]
    handles = [plt.Line2D([0], [0], color=c, lw=9) for _, c in leg]
    ax_s.legend(handles, [n for n, _ in leg], ncol=5, fontsize=7.5, facecolor=PANEL,
                edgecolor=LINE, labelcolor=INK, loc="lower center", bbox_to_anchor=(0.5, 1.28))

    # bottom: accumulated score + headroom for both
    for tr, col, lab in [(sea, ACC, f"SeaCache acc relL1 (τ={_tau_of(sea_m):g})"),
                         (hor, ACC2, f"HorizonCache acc relL1 (τ={tau_h:g})")]:
        steps = [t["step_index"] for t in tr["traces"]]
        acc = [t["acc_rel_l1"] for t in tr["traces"]]
        ax_t.plot(steps, acc, "-o", color=col, markersize=3, label=lab)
    if tau_h:
        ax_t.axhline(tau_h, color=WARN, ls="--", lw=1.1, label=f"refresh τ={tau_h:g}")
    # mark HorizonCache jumps
    jsteps = [j["step"] for j in hor.get("jumps", [])]
    for js in jsteps:
        ax_t.axvline(js, color="#d2a8ff", lw=0.8, alpha=0.5)
    ax_t.set_xlabel("step index"); ax_t.set_ylabel("accumulated SeaCache score")
    ax_t.set_xlim(-0.5, max(len(sea["traces"]), len(hor["traces"])) - 0.5)
    ax_t.legend(fontsize=8, facecolor=PANEL, edgecolor=LINE, labelcolor=INK, loc="upper left")
    fig.text(0.5, -0.03,
             "SeaCache buys speed by letting the cache go stale (few fresh, long holds); HorizonCache "
             "refreshes as often but removes whole nodes with small safe jumps (purple).",
             ha="center", va="top", color=MUT, fontsize=8.5)
    return _save(fig, out)


# ------------------------------------------------------- 3.2 sigma-schedule diagram
def sigma_schedule(gen: Path, out: Path) -> Path:
    """Semantics of a cached step vs fixed regrid vs adaptive headroom jump."""
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    _style(ax)
    ax.set_ylim(0.2, 3.8); ax.set_xlim(-0.05, 1.05)
    ax.set_yticks([1, 2, 3]); ax.set_yticklabels(["cached step", "fixed regrid 1.25", "adaptive headroom"])
    ax.invert_xaxis()  # sigma decreases along integration
    ax.set_xlabel("σ (noise level) — integration runs right → left")

    def node(x, y, c=INK, r=95):
        ax.scatter([x], [y], s=r, color=c, zorder=5, edgecolors=BG, linewidths=1.2)

    # row 1: normal cached step σ_i -> σ_{i+1}
    xs = [1.0, 0.86, 0.72, 0.58, 0.44, 0.30, 0.16, 0.02]
    for x in xs:
        node(x, 3, ACC)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(b, 3), xytext=(a, 3), arrowprops=dict(arrowstyle="->", color=MUT, lw=1.3))
    ax.text(0.5, 3.32, "every node executed (cache freezes the block residual, σ-grid unchanged)",
            color=MUT, fontsize=8.5, ha="center")

    # row 2: fixed regrid — jump σ_i -> σ_target (skip a node), respace tail
    node(1.0, 2, ACC)
    node(0.86, 2, "#6b7684")  # skipped
    ax.scatter([0.86], [2], s=180, facecolors="none", edgecolors=BAD, linewidths=1.6, zorder=6)
    node(0.72, 2, ACC2)
    ax.annotate("", xy=(0.72, 2), xytext=(1.0, 2),
                arrowprops=dict(arrowstyle="->", color=ACC2, lw=2.0))
    tail = [0.72, 0.576, 0.432, 0.288, 0.144, 0.0]
    for x in tail:
        node(x, 2, ACC)
    for a, b in zip(tail[:-1], tail[1:]):
        ax.annotate("", xy=(b, 2), xytext=(a, 2), arrowprops=dict(arrowstyle="->", color=MUT, lw=1.1))
    ax.text(0.86, 1.66, "fixed factor: one target, then re-space the whole tail (net −1 node)",
            color=MUT, fontsize=8.5, ha="center")

    # row 3: adaptive headroom — small stride when headroom low, larger when high, capped
    real = None
    for cand in sorted((gen / "traces").glob("*__horizon_adaptive_1.25_t0.4.json")):
        real = json.loads(cand.read_text()); break
    if real is None:
        for cand in sorted((gen / "traces").glob("*__horizon_adaptive_*.json")):
            real = json.loads(cand.read_text()); break
    node(1.0, 1, ACC)
    if real and real.get("jumps"):
        # draw a few real jumps: arrow length ∝ jump factor
        prev = 1.0
        drawn = 0
        for j in real["jumps"]:
            tgt = j["target_sigma"]; jf = j["jump_factor"]
            col = ACC2 if jf < 1.15 else (WARN if jf < 1.4 else BAD)
            ax.annotate("", xy=(tgt, 1), xytext=(prev, 1),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.4 + 1.6 * (jf - 1)))
            node(tgt, 1, ACC)
            prev = tgt; drawn += 1
            if drawn >= 6:
                break
    ax.text(0.5, 0.66, "adaptive: stride = 1+(jf_max−1)·headroom, headroom = 1−acc/τ  "
                       "(green=small safe · amber/red=larger · capped at jf_max)",
            color=MUT, fontsize=8.5, ha="center")
    ax.set_title("σ-schedule semantics: cache keeps the grid, jump removes a node")
    return _save(fig, out)


# ------------------------------------------------------- 3.3 compute accounting
def compute_accounting(gen: Path, out: Path, target_speed: float = 2.5) -> Path:
    df = pd.read_csv(gen / "metrics.csv")
    sp = _mean_speedup(df)
    sea_m = _closest_method(sp, "seacache_t", target_speed)
    hor_m = (_closest_method(sp, "horizon_adaptive_1.25_t", target_speed)
             or _closest_method(sp, "horizon_", target_speed))
    rows = {}
    for lab, m in [("SeaCache", sea_m), ("HorizonCache\nadaptive→1.25", hor_m)]:
        sub = df[df.method == m]
        rows[lab] = dict(
            fresh=sub.num_full_forwards.mean(),
            cached=sub.num_cached_forwards.mean(),
            skipped=sub.num_skipped_nodes.mean(),
            cost=sub.block_stack_equiv_cost.mean(),
            speed=sp[m],
        )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.3), gridspec_kw={"width_ratios": [2, 1]})
    _style(ax1); _style(ax2)
    labels = list(rows)
    x = np.arange(len(labels))
    fresh = [rows[l]["fresh"] for l in labels]
    cached = [rows[l]["cached"] for l in labels]
    skipped = [rows[l]["skipped"] for l in labels]
    ax1.bar(x, fresh, color=BAD, label="fresh (full forward, cost 1)")
    ax1.bar(x, cached, bottom=fresh, color=ACC, label="cached (≈1/L)")
    ax1.bar(x, skipped, bottom=np.array(fresh) + np.array(cached), color="#d2a8ff",
            label="skipped node (jump, whole node removed)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("count (per 28-step run)")
    ax1.set_title(f"Where the steps go (~{target_speed:g}× matched)")
    ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=LINE, labelcolor=INK, loc="upper center")
    for i, l in enumerate(labels):
        ax1.text(i, fresh[i] + cached[i] + skipped[i] + 0.4, f"{rows[l]['speed']:.2f}×",
                 ha="center", color=INK, fontsize=9)
    # effective cost
    cost = [rows[l]["cost"] for l in labels]
    ax2.bar(x, cost, color=[ACC, ACC2])
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("block-stack-equiv cost ↓")
    ax2.set_title("Effective cost")
    for i, c in enumerate(cost):
        ax2.text(i, c + 0.2, f"{c:.1f}", ha="center", color=INK, fontsize=9)
    fig.text(0.5, -0.02, "cache saves the expensive block stack *inside* a node; a jump saves the *whole* node.",
             ha="center", color=MUT, fontsize=8.6)
    fig.tight_layout()
    return _save(fig, out)


# ------------------------------------------------------- 3.4 staleness vs stride
def staleness_stride(gen: Path, out: Path) -> Path:
    """Per-step: headroom, jump factor, cached-residual age — HorizonCache trades old
    residual for a small safe stride extension."""
    real = None; tau = 0.4
    for cand in sorted((gen / "traces").glob("*__horizon_adaptive_1.25_t0.4.json")):
        real = json.loads(cand.read_text()); tau = 0.4; break
    if real is None:
        for cand in sorted((gen / "traces").glob("*__horizon_adaptive_*.json")):
            real = json.loads(cand.read_text()); tau = _tau_of(cand.name.split("__")[1][:-5]) or 0.4; break
    if real is None:
        raise SystemExit("no adaptive trace for staleness_stride")
    tr = real["traces"]
    steps = [t["step_index"] for t in tr]
    acc = np.array([t["acc_rel_l1"] for t in tr])
    headroom = np.clip(1 - acc / tau, 0, 1)
    refresh_dist = [t.get("refresh_distance", 0) for t in tr]
    jf_by_step = {j["step"]: j["jump_factor"] for j in real.get("jumps", [])}
    jf = [jf_by_step.get(s, 1.0) for s in steps]

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    _style(ax)
    ax.plot(steps, headroom, "-o", color=ACC2, markersize=3, label="headroom = 1−acc/τ")
    ax.plot(steps, [d / max(refresh_dist + [1]) for d in refresh_dist], "-", color=MUT, alpha=0.7,
            label="cached-residual age (norm)")
    ax.set_ylim(-0.05, 1.15); ax.set_xlabel("step index"); ax.set_ylabel("headroom / residual age")
    axb = ax.twinx()
    axb.stem(steps, jf, linefmt="#d2a8ff", markerfmt="D", basefmt=" ")
    axb.set_ylabel("jump factor", color="#d2a8ff"); axb.tick_params(colors="#d2a8ff")
    axb.axhline(1.25, color=BAD, ls=":", lw=1.0)
    axb.set_ylim(0.95, 1.6)
    for s in axb.spines.values():
        s.set_color(LINE)
    ax.set_title("Staleness ↔ stride: bigger jumps only when headroom is high (capped at jf_max)")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=LINE, labelcolor=INK, loc="upper right")
    return _save(fig, out)


# ------------------------------------------------------- 3.5 failure visual
def failure_visual(gen: Path, out: Path) -> Path:
    """Why aggressive jumps overshoot: pick the most aggressive available method (adaptive_2.0
    if present, else the highest-τ adaptive) and show action timeline + score blow-past + the
    quality it costs."""
    df = pd.read_csv(gen / "metrics.csv")
    sp = _mean_speedup(df)
    # prefer adaptive_2.0, then highest-speed horizon (overshoot end)
    aggressive = _closest_method(sp, "horizon_adaptive_2.0_t", 3.5)
    if aggressive is None:
        hor = {m: s for m, s in sp.items() if m.startswith("horizon_")}
        aggressive = max(hor, key=hor.get) if hor else None
    if aggressive is None:
        raise SystemExit("no horizon method for failure_visual")
    key = _pick_prompt(gen, aggressive)
    tr = _load_trace(gen, key, aggressive)
    tau = _tau_of(aggressive)
    sub = df[df.method == aggressive]
    psnr = sub.psnr.mean()
    # matched seacache for delta context
    sea_m = _closest_method(sp, "seacache_t", sp[aggressive])
    sea_psnr = df[df.method == sea_m].psnr.mean() if sea_m else float("nan")

    fig = plt.figure(figsize=(9.4, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    _style(ax1); _style(ax2)
    _draw_actions(ax1, tr, len(tr["traces"]) - 1)
    ax1.set_ylabel(aggressive.replace("horizon_", ""), fontsize=9)
    ax1.set_title(f"Aggressive-jump overshoot — {aggressive}  ·  {sp[aggressive]:.2f}×  ·  "
                  f"{psnr:.1f} dB  (SeaCache @matched {sea_psnr:.1f} dB)", fontsize=10.5)
    steps = [t["step_index"] for t in tr["traces"]]
    acc = [t["acc_rel_l1"] for t in tr["traces"]]
    ax2.plot(steps, acc, "-o", color=BAD, markersize=3, label="accumulated relL1")
    if tau:
        ax2.axhline(tau, color=WARN, ls="--", lw=1.1, label=f"refresh τ={tau:g}")
    jf_by_step = {j["step"]: j["jump_factor"] for j in tr.get("jumps", [])}
    big = [(s, jf_by_step[s]) for s in steps if s in jf_by_step and jf_by_step[s] > 1.4]
    for s, f in big:
        ax2.axvline(s, color="#d2a8ff", lw=1.2, alpha=0.7)
        ax2.text(s, max(acc) * 0.9, f"jf={f:.1f}", color="#d2a8ff", fontsize=7, rotation=90, va="top")
    ax2.set_xlabel("step index"); ax2.set_ylabel("SeaCache score")
    ax2.legend(fontsize=8, facecolor=PANEL, edgecolor=LINE, labelcolor=INK, loc="upper left")
    ax2.text(0.99, 0.03, "large strides remove nodes where the field is still curving → integration error\n"
                         "compounds to the end; the deck's DP-surrogate lesson, made visible.",
             transform=ax2.transAxes, ha="right", va="bottom", color=MUT, fontsize=8.3)
    return _save(fig, out)


# ------------------------------------------------------- qualitative grids (§5)
def qualitative_grids(gen: Path, out: Path, n_prompts: int = 3, target_speed: float = 2.5) -> Path | None:
    """Rows = representative prompts; cols = full · SeaCache@matched · adaptive_1.25@matched ·
    aggressive · error-heatmap(adaptive_1.25). Uses whatever sample PNGs exist; captions carry
    PSNR / LPIPS / ΔPSNR / action counts. Returns None if no samples on disk."""
    sdir = gen / "samples"
    if not sdir.exists() or not any(sdir.glob("*.png")):
        return None
    df = pd.read_csv(gen / "metrics.csv")
    sp = _mean_speedup(df)
    sea_m = _closest_method(sp, "seacache_t", target_speed)
    a125 = _closest_method(sp, "horizon_adaptive_1.25_t", target_speed)
    aggr = _closest_method(sp, "horizon_adaptive_2.0_t", 3.4) or _closest_method(sp, "horizon_adaptive_1.5_t", 3.0)
    cols = [("full", "full"), (sea_m, f"SeaCache {sp.get(sea_m,0):.2f}×"),
            (a125, f"adaptive→1.25 {sp.get(a125,0):.2f}×")]
    if aggr:
        cols.append((aggr, f"aggressive {sp.get(aggr,0):.2f}×"))
    cols.append(("__err__", "|adaptive−full|"))

    # representative prompt keys (seed 0) that have full + adaptive images
    keys = []
    for p in sorted(sdir.glob("*__full.png")):
        k = p.name[: -len("__full.png")]
        if a125 and (sdir / f"{k}__{a125}.png").exists():
            keys.append(k)
        if len(keys) >= n_prompts:
            break
    if not keys:
        return None

    import matplotlib.image as mpimg
    fig, axes = plt.subplots(len(keys), len(cols), figsize=(2.5 * len(cols), 2.7 * len(keys)),
                             squeeze=False)
    fig.patch.set_facecolor(BG)

    def _metric(key, method, col):
        r = df[(df.key == key) & (df.method == method)]
        return float(r[col].iloc[0]) if not r.empty and col in r else None

    for i, key in enumerate(keys):
        full_img = mpimg.imread(sdir / f"{key}__full.png") if (sdir / f"{key}__full.png").exists() else None
        a_img = mpimg.imread(sdir / f"{key}__{a125}.png") if a125 and (sdir / f"{key}__{a125}.png").exists() else None
        for j, (method, label) in enumerate(cols):
            ax = axes[i][j]; ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(LINE)
            if method == "__err__":
                if full_img is not None and a_img is not None:
                    err = np.abs(full_img[..., :3].astype(float) - a_img[..., :3].astype(float)).mean(-1)
                    ax.imshow(err, cmap="magma")
                cap = ""
            else:
                ip = sdir / f"{key}__{method}.png"
                if ip.exists():
                    ax.imshow(mpimg.imread(ip))
                ps = _metric(key, method, "psnr"); lp = _metric(key, method, "lpips")
                cap = ""
                if ps is not None and method != "full":
                    dp = None
                    if sea_m:
                        sps = _metric(key, sea_m, "psnr")
                        dp = (ps - sps) if sps is not None else None
                    cap = f"{ps:.1f}dB" + (f" Δ{dp:+.1f}" if dp is not None else "") + (f"\nLPIPS {lp:.3f}" if lp else "")
            if i == 0:
                ax.set_title(label, color=INK, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(key.replace("geneval_", "").replace("_s0", "")[:16], color=MUT, fontsize=7.5)
            if cap:
                ax.text(0.5, -0.02, cap, transform=ax.transAxes, ha="center", va="top",
                        color=MUT, fontsize=7)
    fig.suptitle("Qualitative — full vs SeaCache vs adaptive→1.25 vs aggressive (matched speed) + error",
                 color=INK, fontsize=10, y=1.005)
    fig.tight_layout()
    return _save(fig, out)


# ----------------------------------------------------------------------- driver
FIGS = {
    "mechanism_pair": mechanism_pair,
    "sigma_schedule": sigma_schedule,
    "compute_accounting": compute_accounting,
    "staleness_stride": staleness_stride,
    "failure_visual": failure_visual,
}


def build_all(gen: Path, out: Path, target_speed: float = 2.5) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    for name, fn in FIGS.items():
        try:
            kw = {"target_speed": target_speed} if name in ("mechanism_pair", "compute_accounting") else {}
            p = fn(gen, out / f"mech_{name}.png", **kw)
            made[name] = str(p)
            print(f"[ok] {name} -> {p}")
        except SystemExit as e:
            print(f"[skip] {name}: {e}")
        except Exception as e:
            print(f"[fail] {name}: {type(e).__name__}: {e}")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-speed", type=float, default=2.5)
    args = ap.parse_args()
    build_all(Path(args.gen_dir), Path(args.out), args.target_speed)


if __name__ == "__main__":
    main()
