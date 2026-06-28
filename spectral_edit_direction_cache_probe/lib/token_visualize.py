"""E52 token-autopsy figures (anaconda env, matplotlib + PIL). All saved under
outputs/.../token_autopsy/. Every figure is referenced + explained in the report's
"Text-Token Modulation Autopsy" section. All functions are guarded so a partial run
still renders what data exists.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import config as C

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.titlesize": 10})


def _obs(eid):
    fp = C.DIAG / f"token_obs_{eid}.json"
    return json.loads(fp.read_text()) if fp.exists() else None


def _stack(blocks, stat, steps, n_tok):
    arr = np.full((len(steps), n_tok), np.nan)
    for si, s in enumerate(steps):
        vals = [blocks[b][str(s)][stat] for b in blocks if str(s) in blocks[b]]
        if vals:
            arr[si] = np.mean(np.array(vals, dtype=float), axis=0)
    return arr


def _ids():
    return [fp.stem.replace("token_obs_", "") for fp in sorted(C.DIAG.glob("token_obs_*.json"))]


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return path


# ----------------------------------------------------------- 1+3+4 attention curves/heatmaps
def attention_figs(eid, topk=10):
    d = _obs(eid)
    if d is None:
        return
    blocks = d["obs"]; toks = d["tokens"]; n = len(toks)
    steps = sorted({int(s) for b in blocks.values() for s in b})
    mass = _stack(blocks, "mass", steps, n)
    contrib = _stack(blocks, "contrib", steps, n)
    edit_idx = []
    al = C.DIAG / f"token_align_{eid}.json"
    if al.exists():
        a = json.loads(al.read_text())["align"]; edit_idx = [j for j in a["changed"] + a["inserted"] if j < n]
    mean_mass = np.nanmean(mass, axis=0)
    show = sorted(set(list(np.argsort(mean_mass)[::-1][:topk]) + edit_idx))

    # 1. per-token attention over timesteps
    fig, ax = plt.subplots(figsize=(8, 4))
    for j in show:
        lw = 2.6 if j in edit_idx else 1.2
        ax.plot(steps, mass[:, j], lw=lw, label=f"{toks[j]}{'*' if j in edit_idx else ''}")
    ax.set_xlabel("denoising step"); ax.set_ylabel("image→token attention mass")
    ax.set_title(f"{eid}: per-token attention over time (* = edited token)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    _save(fig, C.TOK_HEAT / f"per_token_attention_{eid}.png")

    # 3. token × timestep heatmap (attention mass)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.28), 4))
    im = ax.imshow(mass.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(n)); ax.set_yticklabels(toks, fontsize=6)
    ax.set_xlabel("denoising step"); ax.set_title(f"{eid}: token × timestep attention mass")
    for j in edit_idx:
        ax.get_yticklabels()[j].set_color("#c0392b"); ax.get_yticklabels()[j].set_weight("bold")
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, C.TOK_HEAT / f"token_timestep_heatmap_{eid}.png")

    # 4. token × layer/block heatmap (contribution)
    bids = sorted(int(b) for b in blocks)
    bc = np.array([np.nanmean(np.array([blocks[str(b)][str(s)]["contrib"]
                  for s in steps if str(s) in blocks[str(b)]], dtype=float), axis=0) for b in bids])
    fig, ax = plt.subplots(figsize=(max(6, n * 0.28), 3.4))
    im = ax.imshow(bc, aspect="auto", cmap="magma")
    ax.set_xticks(range(n)); ax.set_xticklabels(toks, rotation=90, fontsize=6)
    ax.set_yticks(range(len(bids))); ax.set_yticklabels([f"block {b}" for b in bids], fontsize=7)
    ax.set_title(f"{eid}: token × block contribution (where each token enters strongly)")
    for j in edit_idx:
        ax.get_xticklabels()[j].set_color("#c0392b"); ax.get_xticklabels()[j].set_weight("bold")
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, C.TOK_HEAT / f"token_layer_heatmap_{eid}.png")


# ----------------------------------------------------------- 2 edit influence over time
def influence_fig(eid):
    d = _obs(eid)
    if d is None or not d.get("ablation"):
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for r, steps_d in d["ablation"].items():
        if not steps_d:
            continue
        ss = sorted(int(s) for s in steps_d)
        ax.plot(ss, [steps_d[str(s)]["norm"] for s in ss], "-o", label=r, ms=4)
    ax.set_xlabel("denoising step"); ax.set_ylabel("‖Δ_edit contribution‖ (token ablation)")
    ax.set_title(f"{eid}: per-token causal influence on the edit direction over time")
    ax.legend(fontsize=8)
    _save(fig, C.TOK_HEAT / f"edit_influence_{eid}.png")


# ----------------------------------------------------------- 5 spatial token maps + overlay
def spatial_fig(eid):
    npz = C.DIAG / f"token_spatial_{eid}.npz"
    d = _obs(eid)
    if not npz.exists() or d is None:
        return
    arr = np.load(npz)
    roles = d.get("roles", {})
    inp = C.SAMP / eid / "input.png"
    base = Image.open(inp).convert("RGB").resize((192, 192)) if inp.exists() else None
    # pick, per role token, the latest available (block,step) map
    picks = []
    for role, tok in roles.items():
        if not tok:
            continue
        ti = tok["index"]
        keys = [k for k in arr.files if k.endswith(f"_t{ti}")]
        if keys:
            picks.append((role, tok["word"], sorted(keys)[-1]))
    if not picks:
        return
    cols = len(picks)
    fig, ax = plt.subplots(2, cols, figsize=(2.4 * cols, 4.8), squeeze=False)
    for c, (role, word, key) in enumerate(picks):
        m = arr[key]; mn = (m - m.min()) / (m.max() - m.min() + 1e-9)
        ax[0, c].imshow(mn, cmap="inferno"); ax[0, c].set_title(f"{role}\n'{word}'", fontsize=8)
        if base is not None:
            heat = Image.fromarray((plt.cm.inferno(np.asarray(
                Image.fromarray((mn * 255).astype("uint8")).resize((192, 192))) / 255.0)[..., :3] * 255
            ).astype("uint8"))
            over = Image.blend(base, heat, 0.5)
            ax[1, c].imshow(over)
        ax[1, c].set_title("overlay on input", fontsize=8)
        for a in (ax[0, c], ax[1, c]):
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle(f"{eid}: per-token spatial attention maps (object / attribute / style / bg / control)",
                 weight="bold", fontsize=10)
    _save(fig, C.TOK_SPATIAL / f"spatial_{eid}.png")


# ----------------------------------------------------------- 6 frequency/token coupling
def freq_coupling_fig(eid):
    d = _obs(eid)
    if d is None or not d.get("ablation"):
        return
    roles, lows, mids, highs = [], [], [], []
    for r, steps_d in d["ablation"].items():
        if not steps_d:
            continue
        roles.append(r)
        lows.append(np.mean([v["low"] for v in steps_d.values()]))
        mids.append(np.mean([v["mid"] for v in steps_d.values()]))
        highs.append(np.mean([v["high"] for v in steps_d.values()]))
    if not roles:
        return
    x = np.arange(len(roles)); w = 0.26
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - w, lows, w, label="low (layout)", color="#2471a3")
    ax.bar(x, mids, w, label="mid (shape)", color="#b9770e")
    ax.bar(x + w, highs, w, label="high (texture)", color="#922b21")
    ax.set_xticks(x); ax.set_xticklabels(roles, rotation=15, fontsize=8)
    ax.set_ylabel("fractional band energy of token's Δ_edit effect")
    ax.set_title(f"{eid}: frequency/token coupling"); ax.legend(fontsize=8)
    _save(fig, C.TOK_HEAT / f"freq_coupling_{eid}.png")


# ----------------------------------------------------------- 7 weight curves
def weight_curve_figs():
    s = _summary()
    if not s or not s.get("interventions", {}).get("mechanisms"):
        return
    mechs = s["interventions"]["mechanisms"]
    for ycol, ylab, fname, title in [
        ("clipT_gain", "CLIP-T edit gain ↑", "weight_edit_strength.png", "Token weight vs edit strength"),
        ("lpips_to_ref", "LPIPS to unmodified edit ↓", "weight_preservation.png", "Token weight vs preservation (drift)"),
        ("delta_smoothness", "Δ_edit adjacent change", "weight_delta_smoothness.png", "Token weight vs Δ_edit spectral stability")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for mech, m in mechs.items():
            cv = m["curve"]
            xs = [c["weight"] for c in cv]; ys = [c.get(ycol, np.nan) for c in cv]
            ax.plot(xs, ys, "-o", label=mech, ms=4)
        ax.axvline(1.0, ls=":", c="#888", lw=.8)
        ax.set_xlabel("token weight (×)"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.legend(fontsize=8)
        _save(fig, C.TOK_CURVES / fname)


# ----------------------------------------------------------- 8 cache correlation
def cache_corr_figs():
    s = _summary()
    cc = (s or {}).get("cache_connection", {})
    tab = cc.get("table")
    if not tab:
        return
    d = {k: np.array([r.get(k, np.nan) for r in tab], float)
         for k in ("attn_stability_cv", "attn_entropy", "peak_step", "delta_smoothness",
                   "cache_lpips")}
    plots = [("attn_stability_cv", "cache_lpips", "token attention instability (CV)", "delta-cache LPIPS ↓",
              "Token attention stability vs cache quality", "stability_vs_cache.png"),
             ("attn_entropy", "cache_lpips", "edited-token attention entropy", "delta-cache LPIPS ↓",
              "Token entropy vs cache failure", "entropy_vs_cache.png"),
             ("peak_step", "delta_smoothness", "edited-token attention peak step", "Δ_edit adjacent change",
              "Edited-token peak timestep vs Δ_edit smoothness", "peakstep_vs_smoothness.png")]
    for xk, yk, xl, yl, title, fname in plots:
        x, y = d[xk], d[yk]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 4))
        ax.scatter(x[m], y[m], c="#1e8449", s=40)
        if m.sum() >= 3 and np.std(x[m]) > 1e-9:
            a, b = np.polyfit(x[m], y[m], 1)
            xs = np.linspace(x[m].min(), x[m].max(), 20); ax.plot(xs, a * xs + b, "--", c="#888")
            ax.text(0.05, 0.92, f"r={np.corrcoef(x[m], y[m])[0,1]:.2f}", transform=ax.transAxes)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
        _save(fig, C.TOK_CACHE / fname)


# ----------------------------------------------------------- 9 intervention montage
def intervention_montages(per_role=True):
    grids = sorted(C.TOK_GRIDS.glob("*/"))
    sz = 150
    for gdir in grids:
        eid = gdir.name
        imgs = {}
        for fp in gdir.glob("*.png"):
            stem = fp.stem  # role__mech__wX.XX
            parts = stem.split("__")
            if len(parts) != 3:
                continue
            role, mech, wtag = parts
            imgs.setdefault((role, mech), {})[wtag] = fp
        for (role, mech), wmap in imgs.items():
            wtags = sorted(wmap)
            row = Image.new("RGB", (sz * len(wtags), sz + 18), "white")
            from PIL import ImageDraw
            for j, wt in enumerate(wtags):
                im = Image.open(wmap[wt]).convert("RGB").resize((sz, sz))
                row.paste(im, (j * sz, 18))
                ImageDraw.Draw(row).text((j * sz + 4, 3), wt.replace("w", "×"), fill="black")
            (C.TOK_GRIDS / "_montage").mkdir(exist_ok=True)
            row.save(C.TOK_GRIDS / "_montage" / f"{eid}__{role}__{mech}.png")


def _summary():
    fp = C.TOK / "token_summary.json"
    return json.loads(fp.read_text()) if fp.exists() else None


def all_token_figs():
    ids = _ids()
    for eid in ids:
        for fn in (attention_figs, influence_fig, spatial_fig, freq_coupling_fig):
            try:
                fn(eid)
            except Exception as e:
                print(f"[token-viz] {fn.__name__}({eid}) failed: {e}")
    for fn in (weight_curve_figs, cache_corr_figs, intervention_montages):
        try:
            fn()
        except Exception as e:
            print(f"[token-viz] {fn.__name__} failed: {e}")
    print(f"[token-viz] figures for {len(ids)} examples -> {C.TOK}")


if __name__ == "__main__":
    all_token_figs()
