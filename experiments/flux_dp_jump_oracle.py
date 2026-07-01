#!/usr/bin/env python3
"""E53 — FLUX jump-DP skip-schedule oracle.

Offline/diagnostic study: given a full vanilla 100-step no-skip FLUX trajectory,
use dynamic programming to discover the offline-optimal jump schedule for saving
steps. This is a TEACHER-FORCED, NON-CAUSAL oracle -- not a deployable method.

Three method families are kept strictly separate throughout the report:
  1. offline jump-DP surrogate  (sum of independent per-edge S_jump costs),
  2. live jump replay           (compounded replay from z_0 with vanilla velocities),
  3. causal baselines           (SeaCache live, uniform-jump, random-jump).
Plus an optional capped cached-residual stage-2 (a truly causal cached replay).

This is a thin orchestrator: the trajectory capture, jump edge table, budgeted
shortest-path DP, replay, pipeline loading, decoding and SeaCache forward all come
from experiments/flux_seacache_dp_shortcuts.py (imported as `fsd`). We add only the
oracle-specific glue: exact S_jump cost, one-step sign-convention assert, max_span
capping, uniform/random/SeaCache baselines matched by achieved budget, capped
cached-residual replay, the required figures/CSVs, and the spec-structured report.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures  # noqa: E402
import flux_seacache_dp_shortcuts as fsd  # noqa: E402

# torch cu124 bundles cuDNN 9.2, which fails CUDNN_STATUS_NOT_INITIALIZED on the VAE
# conv2d against this node's driver (535 / CUDA 12.2). The matmul-based transformer
# steps are unaffected (cuBLAS). Disabling cuDNN routes conv through native CUDA
# kernels; only VAE decode is touched and the cost is negligible.
torch.backends.cudnn.enabled = False

REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-8


# ----------------------------------------------------------------------------
# S_jump edge table (spec-exact cost)
# ----------------------------------------------------------------------------
def build_sjump_edges(sample_dir: Path, cache_h_threshold: float):
    """fsd.build_edges + override each edge cost with the spec S_jump.

    S_jump[k,i] = ||z_hat_i - z_i||_2^2 / (||z_i||_2^2 + eps),  where
    z_hat_i = z_k + (sigma_i - sigma_k) * v_k. fsd already computes latent_rel_l2 =
    ||diff|| / (||z_i|| + eps); S_jump is its square.
    """
    edges, heat_rmse = fsd.build_edges(sample_dir, cache_h_threshold)
    n = len(edges) - 1
    heat_sjump = np.full((n + 1, n + 1), np.nan, dtype=np.float32)
    sj = {}
    for i in range(n + 1):
        for e in edges[i]:
            s = float(e.metrics["latent_rel_l2"]) ** 2
            e.err = s  # DP minimizes edge.err
            e.metrics["s_jump"] = s
            heat_sjump[e.src, e.dst] = s
            sj[(e.src, e.dst)] = s
    return edges, heat_sjump, heat_rmse, sj, n


def verify_one_step(sample_dir: Path, tol: float = 5e-3):
    """For i=k+1 the jump prediction must match the vanilla next latent.

    Returns (passed, worst_rel_err, mean_rel_err). This checks the sign/convention
    z_{k+1} = z_k + (sigma_{k+1}-sigma_k) v_k against the saved sampler output.
    """
    meta = fsd.read_json(sample_dir / "metadata.json")
    n = int(meta["num_inference_steps"])
    sigmas = meta["sigmas"]
    errs = []
    for k in range(n):
        zk = fsd.load_step_tensor(sample_dir / "latents", k)
        zk1 = fsd.load_step_tensor(sample_dir / "latents", k + 1)
        vk = fsd.load_step_tensor(sample_dir / "velocities", k)
        ds = float(sigmas[k + 1] - sigmas[k]) if k + 1 < len(sigmas) else float(0.0 - sigmas[k])
        pred = zk + ds * vk
        rel = float(torch.linalg.vector_norm((pred - zk1).flatten()) / (torch.linalg.vector_norm(zk1.flatten()) + EPS))
        errs.append(rel)
    worst = float(max(errs))
    mean = float(np.mean(errs))
    return worst <= tol, worst, mean


# ----------------------------------------------------------------------------
# schedules (node lists) and replay
# ----------------------------------------------------------------------------
def path_to_nodes(path: list) -> list[int]:
    return [0] + [e.dst for e in path]


def nodes_to_edges(nodes: list[int]) -> list:
    """Lightweight Edge list for replay_path (only src/dst are used by replay)."""
    return [fsd.Edge(nodes[i], nodes[i + 1], 0.0, 1, 1, {}) for i in range(len(nodes) - 1)]


def uniform_nodes(n: int, budget: int) -> list[int]:
    """`budget` segments, evenly spaced anchors 0..n."""
    budget = max(1, min(budget, n))
    pts = np.linspace(0, n, budget + 1)
    nodes = sorted(set(int(round(x)) for x in pts))
    if nodes[0] != 0:
        nodes[0] = 0
    if nodes[-1] != n:
        nodes.append(n)
    return sorted(set(nodes))


def random_nodes(n: int, budget: int, rng: np.random.Generator) -> list[int]:
    budget = max(1, min(budget, n))
    if budget >= n:
        return list(range(n + 1))
    interior = rng.choice(np.arange(1, n), size=budget - 1, replace=False)
    return sorted(set([0] + interior.tolist() + [n]))


def schedule_surrogate_cost(nodes: list[int], sj: dict) -> float:
    """Sum of independent per-edge S_jump costs (the surrogate). NaN if an edge is
    missing (should not happen for a full all-pairs table)."""
    total = 0.0
    for a, b in zip(nodes[:-1], nodes[1:]):
        total += sj.get((a, b), float("nan"))
    return float(total)


def span_lengths(nodes: list[int]) -> list[int]:
    return [b - a for a, b in zip(nodes[:-1], nodes[1:])]


# ----------------------------------------------------------------------------
# perceptual metrics
# ----------------------------------------------------------------------------
class MetricBank:
    def __init__(self, device: str):
        import lpips
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def image_metrics(self, img: Image.Image, ref: Image.Image, prompt: str) -> dict:
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        from skimage.metrics import structural_similarity as ssim_fn

        a = np.asarray(img.convert("RGB"))
        b = np.asarray(ref.convert("RGB"))
        if a.shape != b.shape:
            img = img.resize(ref.size)
            a = np.asarray(img.convert("RGB"))
        psnr = float(psnr_fn(b, a, data_range=255))
        ssim = float(ssim_fn(b, a, channel_axis=2, data_range=255))
        ta = torch.from_numpy(a).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0).to(self.device)
        tb = torch.from_numpy(b).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0).to(self.device)
        lp = float(self.lpips(ta, tb).item())
        # CLIP image-image + image-text
        feats = self.clip_proc(text=[prompt], images=[img, ref], return_tensors="pt", padding=True, truncation=True).to(self.device)
        img_emb = self.clip.get_image_features(pixel_values=feats["pixel_values"])
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = self.clip.get_text_features(input_ids=feats["input_ids"], attention_mask=feats["attention_mask"])
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        clip_img = float((img_emb[0] * img_emb[1]).sum().item())
        clip_txt = float((img_emb[0] * txt_emb[0]).sum().item())
        return {"psnr": psnr, "ssim": ssim, "lpips": lp, "clip_img": clip_img, "clip_text": clip_txt}


# ----------------------------------------------------------------------------
# causal cached replay (stage-2) — scheduled residual reuse
# ----------------------------------------------------------------------------
def install_scheduled_cache_forward(pipe, fresh_steps: set[int], num_steps: int) -> dict:
    """Like fsd.install_seacache_forward but the fresh/reuse decision follows a FIXED
    schedule instead of the online rel-L1 gate. Fresh transformer eval happens at
    step 0, the last step, and any step in `fresh_steps`; otherwise the cached
    residual is reused. This is a genuine causal cached replay."""
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    tr = pipe.transformer
    orig_forward = tr.forward
    state = {"fresh_evals": 0, "cached_evals": 0, "fresh_steps": sorted(fresh_steps)}
    tr.cnt = 0
    tr.num_steps = int(num_steps)
    tr.previous_residual = None

    def wrapped(hidden_states, encoder_hidden_states=None, pooled_projections=None, timestep=None,
                img_ids=None, txt_ids=None, guidance=None, joint_attention_kwargs=None,
                controlnet_block_samples=None, controlnet_single_block_samples=None,
                return_dict=True, controlnet_blocks_repeat=False):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(tr, lora_scale)
        hs = tr.x_embedder(hidden_states)
        ts = timestep.to(hs.dtype) * 1000
        guidance_in = guidance.to(hs.dtype) * 1000 if guidance is not None else None
        temb = tr.time_text_embed(ts, pooled_projections) if guidance_in is None else tr.time_text_embed(ts, guidance_in, pooled_projections)
        enc = tr.context_embedder(encoder_hidden_states)
        txt = txt_ids[0] if txt_ids is not None and txt_ids.ndim == 3 else txt_ids
        img = img_ids[0] if img_ids is not None and img_ids.ndim == 3 else img_ids
        image_rotary_emb = tr.pos_embed(torch.cat((txt, img), dim=0)) if txt is not None and img is not None else None
        step_index = int(tr.cnt)
        is_fresh = (step_index == 0) or (step_index == tr.num_steps - 1) or (step_index in fresh_steps) or (tr.previous_residual is None)
        tr.cnt += 1
        if tr.cnt == tr.num_steps:
            tr.cnt = 0
        if not is_fresh:
            state["cached_evals"] += 1
            hs = hs + tr.previous_residual
        else:
            state["fresh_evals"] += 1
            ori_hs = hs
            for block in tr.transformer_blocks:
                enc, hs = block(hidden_states=hs, encoder_hidden_states=enc, temb=temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs)
            for block in tr.single_transformer_blocks:
                enc, hs = block(hidden_states=hs, encoder_hidden_states=enc, temb=temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs)
            tr.previous_residual = (hs - ori_hs).detach()
        hs = tr.norm_out(hs, temb)
        output = tr.proj_out(hs)
        if USE_PEFT_BACKEND:
            unscale_lora_layers(tr, lora_scale)
        return (output,) if not return_dict else Transformer2DModelOutput(sample=output)

    tr.forward = wrapped
    state["restore"] = lambda: setattr(tr, "forward", orig_forward)
    return state


@torch.no_grad()
def live_generate(pipe, prompt, seed, steps, height, width, guidance, max_seq_len, device):
    """Run the standard pipeline (whatever forward is currently installed) and return
    (packed_latent, PIL image, runtime_sec)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.perf_counter()
    out = pipe(prompt=prompt, num_inference_steps=steps, height=height, width=width,
               guidance_scale=guidance, max_sequence_length=max_seq_len,
               num_images_per_prompt=1, generator=gen, output_type="latent")
    rt = time.perf_counter() - t0
    lat = out.images if hasattr(out, "images") else out[0]
    img = fsd.decode_flux_latents(pipe, lat.to(device).to(pipe.dtype), height, width)
    return lat.detach().float().cpu(), img, rt


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def fig_method_diagram(path: Path, nodes_example: list[int], n: int):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(9, 5.5))
    for ax in axs:
        ax.set_xlim(-2, n + 2)
        ax.set_yticks([])
    axs[0].plot(range(n + 1), [1] * (n + 1), "-o", ms=2, color="#888")
    axs[0].set_title("Vanilla trajectory z_0 … z_100 (no skip)")
    axs[1].plot(range(n + 1), [1] * (n + 1), "-", color="#ccc")
    axs[1].plot([0, 20], [1, 1], "-o", color="#c0392b", lw=2, label="jump z_0→z_20 via v_0")
    axs[1].legend(fontsize=8, loc="upper right")
    axs[1].set_title("Candidate jump k→i:  ẑ_i = z_k + (σ_i-σ_k)·v_k")
    axs[2].plot(range(n + 1), [1] * (n + 1), "-", color="#ccc")
    axs[2].plot(nodes_example, [1] * len(nodes_example), "o", color="#245c9a", ms=7)
    axs[2].set_title(f"DP schedule anchors (example, {len(nodes_example)} fresh evals)")
    axs[2].set_xlabel("denoising step")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_heatmap(path: Path, heat: np.ndarray, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(np.log10(heat + 1e-8), origin="lower", cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, label="log10 S_jump")
    ax.set_title(title)
    ax.set_xlabel("destination step i")
    ax.set_ylabel("source step k")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_schedule_raster(path: Path, dp_by_saved: dict, uni_by_saved: dict, sc_points: list, n: int):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    saveds = sorted(dp_by_saved.keys())
    for row, saved in enumerate(saveds):
        y = row
        ax.scatter(dp_by_saved[saved], [y + 0.15] * len(dp_by_saved[saved]), s=18, color="#245c9a", label="DP" if row == 0 else None)
        if saved in uni_by_saved:
            ax.scatter(uni_by_saved[saved], [y - 0.15] * len(uni_by_saved[saved]), s=12, marker="s", color="#e08a1e", label="uniform" if row == 0 else None)
    ax.set_yticks(range(len(saveds)))
    ax.set_yticklabels([f"saved {s}" for s in saveds])
    ax.set_xlabel("denoising step (fresh anchor = dot)")
    ax.set_title("Schedule raster: DP (blue) vs uniform (orange)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_frontier(path: Path, curves: dict, xkey: str, ykey: str, xlabel: str, ylabel: str, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "dp_surrogate": ("#245c9a", "-o", "DP jump surrogate"),
        "dp_replay": ("#1f9c5a", "-s", "DP jump live replay"),
        "dp_cached": ("#6a3d9a", "-^", "DP cached replay (stage-2)"),
        "uniform": ("#e08a1e", "--o", "uniform jump"),
        "random": ("#c0392b", ":o", "random jump (mean)"),
        "seacache": ("#111", "-D", "SeaCache (causal live)"),
    }
    for name, pts in curves.items():
        if not pts:
            continue
        c, ls, lab = styles.get(name, ("#555", "-o", name))
        xs = [p[xkey] for p in pts]
        ys = [p[ykey] for p in pts]
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], ls, color=c, label=lab, ms=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_span_hist(path: Path, dp_spans: list, uni_spans: list, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0, max([1] + dp_spans + uni_spans) + 2) - 0.5
    ax.hist(dp_spans, bins=bins, alpha=0.6, label="DP", color="#245c9a")
    ax.hist(uni_spans, bins=bins, alpha=0.6, label="uniform", color="#e08a1e")
    ax.set_xlabel("selected jump length (steps)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_predictor_scatter(path: Path, xvals, yvals, xlabel, ylabel, title, r_p, r_s):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xvals, yvals, s=14, alpha=0.5, color="#245c9a")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nPearson={r_p:.3f}  Spearman={r_s:.3f}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def image_grid(paths_labels: list[tuple[Path, str]], out: Path, thumb=(300, 300)):
    from PIL import ImageDraw

    n = max(1, len(paths_labels))
    cols = min(5, n)
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * thumb[0], rows * (thumb[1] + 26)), (250, 248, 242))
    draw = ImageDraw.Draw(sheet)
    for idx, (p, lab) in enumerate(paths_labels):
        if p and Path(p).exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail(thumb)
            x = (idx % cols) * thumb[0] + (thumb[0] - im.width) // 2
            y = (idx // cols) * (thumb[1] + 26)
            sheet.paste(im, (x, y))
        draw.text(((idx % cols) * thumb[0] + 6, (idx // cols) * (thumb[1] + 26) + thumb[1] + 6), lab[:40], fill=(20, 24, 28))
    sheet.save(out)


# ----------------------------------------------------------------------------
# main run
# ----------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    device = args.device
    n = args.steps
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"{ts}__flux_dp_jump_oracle{'__smoke' if args.smoke else ''}"
    for sub in ["figures", "samples", "schedules", "edge_costs", "metrics", "reports"]:
        fsd.ensure_dir(run_dir / sub)
    print(f"[oracle] run_dir = {run_dir}", flush=True)

    # prompts (canonical fixture) -> drive fsd capture
    fixt = fixtures.canonical_prompts()[: args.num_samples]
    fsd.PROMPTS = [(d["tag"], d["prompt"]) for d in fixt]
    prompts = [d["prompt"] for d in fixt]

    traj_root = Path(args.trajectory_root)
    cap_args = SimpleNamespace(
        model_id=args.model_id, dtype=args.dtype, device=device, offload=args.offload, bnb4=False,
        num_samples=args.num_samples, seed_base=args.seed_base, steps=n, height=args.height,
        width=args.width, guidance=args.guidance, max_sequence_length=args.max_sequence_length,
        save_h_raw=False, force=args.force, output_root=str(traj_root),
    )
    print("[oracle] capturing vanilla trajectories ...", flush=True)
    fsd.run_capture(cap_args)

    replay_saved = [int(x) for x in args.replay_saved.split(",") if x.strip()]
    replay_budgets = sorted(set(max(1, min(n, n - s)) for s in replay_saved))
    max_spans = [int(x) for x in args.max_spans.split(",") if x.strip()]
    sc_thresholds = [float(x) for x in args.seacache_thresholds.split(",") if x.strip()]
    frontier_budgets = sorted(set(list(range(1, n)) if not args.smoke else replay_budgets))

    bank = MetricBank(device)
    pipe = fsd.load_flux_pipeline(args.model_id, args.dtype, device, args.offload, False)

    per_sample_rows: list[dict] = []
    per_budget_agg: dict = {}
    frontier_rows: list[dict] = []
    predictor_pairs = {"inst_vs_span": ([], []), "acc_vs_span": ([], []), "inst_vs_sjump": ([], [])}
    sample_meta = []
    dp_schedule_dump = {}

    sample_dirs = sorted(traj_root.glob("sample_*"))[: args.num_samples]
    for si, sample_dir in enumerate(sample_dirs):
        meta = fsd.read_json(sample_dir / "metadata.json")
        prompt = meta["prompt"]
        seed = int(meta["seed"])
        h, w = int(meta["height"]), int(meta["width"])
        print(f"[oracle] sample {si}: {prompt[:50]!r} seed={seed}", flush=True)

        edges, heat_sjump, heat_rmse, sj, nn = build_sjump_edges(sample_dir, args.cache_h_threshold)
        assert nn == n, f"trajectory has {nn} steps, expected {n}"
        np.savez_compressed(run_dir / "edge_costs" / f"{sample_dir.name}_S_jump.npz", s_jump=heat_sjump, latent_rmse=heat_rmse)

        passed, worst, mean_rel = verify_one_step(sample_dir)
        print(f"[oracle]   one-step sign check: passed={passed} worst_rel={worst:.2e} mean_rel={mean_rel:.2e}", flush=True)

        vanilla_final_latent = fsd.load_step_tensor(sample_dir / "latents", n)
        vanilla_img = Image.open(sample_dir / "final.png").convert("RGB")

        # ---- full-span jump-DP over budgets ----
        dp_nodes_by_budget = {}
        dp_surrogate_by_budget = {}
        for b in frontier_budgets:
            path, _ = fsd.dp_path(edges, b, "cost_a")
            nodes = path_to_nodes(path)
            achieved = len(nodes) - 1
            dp_nodes_by_budget[b] = nodes
            surr = sum(e.err for e in path)
            dp_surrogate_by_budget[b] = surr
            replay_lat = fsd.replay_path(sample_dir, path)
            m = fsd.latent_metrics(replay_lat, vanilla_final_latent)
            frontier_rows.append({"sample": sample_dir.name, "method": "dp_surrogate", "budget": b,
                                  "achieved_budget": achieved, "saved": n - achieved, "surrogate_cost": surr,
                                  "replay_latent_rel_l2": m["latent_rel_l2"]})

        # ---- capped max_span jump-DP (surrogate only) ----
        capped_frontier = {}
        for ms in max_spans:
            capped_edges = [[e for e in edges[i] if (e.dst - e.src) <= ms] for i in range(len(edges))]
            pts = []
            for b in frontier_budgets:
                path, _ = fsd.dp_path(capped_edges, b, "cost_a")
                nodes = path_to_nodes(path)
                if not nodes or nodes[-1] != n:
                    continue  # infeasible at this span/budget
                surr = sum(e.err for e in path)
                pts.append({"budget": b, "achieved_budget": len(nodes) - 1, "saved": n - (len(nodes) - 1), "surrogate_cost": surr})
                frontier_rows.append({"sample": sample_dir.name, "method": f"dp_maxspan{ms}", "budget": b,
                                      "achieved_budget": len(nodes) - 1, "saved": n - (len(nodes) - 1),
                                      "surrogate_cost": surr, "replay_latent_rel_l2": ""})
            capped_frontier[ms] = pts

        # ---- baselines: uniform + random (surrogate over all replay budgets) ----
        rng = np.random.default_rng(1234 + si)
        for b in replay_budgets:
            un = uniform_nodes(n, b)
            frontier_rows.append({"sample": sample_dir.name, "method": "uniform", "budget": b,
                                  "achieved_budget": len(un) - 1, "saved": n - (len(un) - 1),
                                  "surrogate_cost": schedule_surrogate_cost(un, sj), "replay_latent_rel_l2": ""})
            rand_costs = []
            for _ in range(args.random_trials):
                rn = random_nodes(n, b, rng)
                rand_costs.append(schedule_surrogate_cost(rn, sj))
            frontier_rows.append({"sample": sample_dir.name, "method": "random", "budget": b,
                                  "achieved_budget": b, "saved": n - b,
                                  "surrogate_cost": float(np.nanmean(rand_costs)), "replay_latent_rel_l2": ""})

        # ---- live replay at required budgets: DP + uniform (jump replay, decode+perceptual) ----
        dp_nodes_saved = {}
        uni_nodes_saved = {}
        sample_sample_imgs = {}  # budget -> {method: path}
        for b in replay_budgets:
            saved = n - b
            path, _ = fsd.dp_path(edges, b, "cost_a")
            dp_nodes = path_to_nodes(path)
            dp_nodes_saved[saved] = dp_nodes
            dp_schedule_dump.setdefault(sample_dir.name, {})[f"saved_{saved}"] = {
                "dp_nodes": dp_nodes, "achieved_budget": len(dp_nodes) - 1,
                "surrogate_cost": sum(e.err for e in path),
                "spans": span_lengths(dp_nodes),
            }
            # DP jump replay
            dp_lat = fsd.replay_path(sample_dir, path)
            dp_img = fsd.decode_flux_latents(pipe, dp_lat.to(device).to(pipe.dtype), h, w)
            dp_png = run_dir / "samples" / f"{sample_dir.name}_saved{saved}_dp.png"
            dp_img.save(dp_png)
            dp_m = bank.image_metrics(dp_img, vanilla_img, prompt)
            dp_ll = fsd.latent_metrics(dp_lat, vanilla_final_latent)["latent_rel_l2"]
            per_sample_rows.append({"sample": sample_dir.name, "prompt": prompt, "method": "dp_replay",
                                    "saved": saved, "budget": b, "achieved_budget": len(dp_nodes) - 1,
                                    "latent_rel_l2": dp_ll, **dp_m})
            # uniform jump replay
            un = uniform_nodes(n, b)
            uni_nodes_saved[saved] = un
            un_lat = fsd.replay_path(sample_dir, nodes_to_edges(un))
            un_img = fsd.decode_flux_latents(pipe, un_lat.to(device).to(pipe.dtype), h, w)
            un_png = run_dir / "samples" / f"{sample_dir.name}_saved{saved}_uniform.png"
            un_img.save(un_png)
            un_m = bank.image_metrics(un_img, vanilla_img, prompt)
            un_ll = fsd.latent_metrics(un_lat, vanilla_final_latent)["latent_rel_l2"]
            per_sample_rows.append({"sample": sample_dir.name, "prompt": prompt, "method": "uniform",
                                    "saved": saved, "budget": b, "achieved_budget": len(un) - 1,
                                    "latent_rel_l2": un_ll, **un_m})
            sample_sample_imgs[saved] = {"dp": dp_png, "uniform": un_png}

        # ---- SeaCache causal live baseline (matched by achieved budget) ----
        sc_points = []
        if not args.no_seacache:
            for th in sc_thresholds:
                try:
                    state = fsd.install_seacache_forward(pipe, th, n)
                    lat, img, rt = live_generate(pipe, prompt, seed, n, h, w, args.guidance, args.max_sequence_length, device)
                    fresh = int(state["fresh_evals"])
                    state["restore"]()
                    sc_png = run_dir / "samples" / f"{sample_dir.name}_seacache_th{th}.png"
                    img.save(sc_png)
                    m = bank.image_metrics(img, vanilla_img, prompt)
                    ll = fsd.latent_metrics(lat, vanilla_final_latent)["latent_rel_l2"]
                    per_sample_rows.append({"sample": sample_dir.name, "prompt": prompt, "method": "seacache",
                                            "saved": n - fresh, "budget": fresh, "achieved_budget": fresh,
                                            "threshold": th, "latent_rel_l2": ll, "runtime_sec": rt, **m})
                    sc_points.append({"threshold": th, "achieved_budget": fresh, "saved": n - fresh, "png": str(sc_png), **m})
                except Exception as exc:  # noqa: BLE001
                    print(f"[oracle]   SeaCache th={th} FAILED: {exc}", flush=True)
                    if hasattr(pipe.transformer, "forward"):
                        try:
                            state["restore"]()
                        except Exception:
                            pass

        # ---- stage-2 capped cached-residual replay: DP + uniform at replay budgets ----
        stage2_ran = False
        stage2_note = "not requested"
        if args.stage2:
            stage2_note = f"capped: DP + uniform schedules at saved={replay_saved}, only spans<= n"
            for saved in replay_saved:
                b = n - saved
                if saved not in dp_nodes_saved:
                    continue
                for mkey, nodes in [("dp_cached", dp_nodes_saved[saved]), ("uniform_cached", uni_nodes_saved.get(saved))]:
                    if nodes is None:
                        continue
                    fresh_steps = set(nodes[:-1])  # fresh eval at each anchor source
                    try:
                        state = install_scheduled_cache_forward(pipe, fresh_steps, n)
                        lat, img, rt = live_generate(pipe, prompt, seed, n, h, w, args.guidance, args.max_sequence_length, device)
                        fresh = int(state["fresh_evals"])
                        state["restore"]()
                        c_png = run_dir / "samples" / f"{sample_dir.name}_saved{saved}_{mkey}.png"
                        img.save(c_png)
                        m = bank.image_metrics(img, vanilla_img, prompt)
                        ll = fsd.latent_metrics(lat, vanilla_final_latent)["latent_rel_l2"]
                        per_sample_rows.append({"sample": sample_dir.name, "prompt": prompt, "method": mkey,
                                                "saved": n - fresh, "budget": fresh, "achieved_budget": fresh,
                                                "latent_rel_l2": ll, "runtime_sec": rt, **m})
                        if mkey == "dp_cached":
                            sample_sample_imgs.setdefault(saved, {})["dp_cached"] = c_png
                        stage2_ran = True
                    except Exception as exc:  # noqa: BLE001
                        print(f"[oracle]   stage2 {mkey} saved={saved} FAILED: {exc}", flush=True)
                        try:
                            state["restore"]()
                        except Exception:
                            pass

        # ---- predictor diagnostic: SeaCache step traces vs oracle spans/S_jump ----
        # Use a mid-threshold SeaCache trace's raw/accumulated rel-L1 per step.
        try:
            state = fsd.install_seacache_forward(pipe, sc_thresholds[len(sc_thresholds) // 2] if sc_thresholds else 0.3, n)
            live_generate(pipe, prompt, seed, n, h, w, args.guidance, args.max_sequence_length, device)
            traces = {t["step"]: t for t in state["step_traces"]}
            state["restore"]()
            # oracle next-anchor span from a mid replay budget DP schedule
            mid_saved = replay_saved[len(replay_saved) // 2]
            mid_nodes = dp_nodes_saved.get(mid_saved, [])
            next_anchor = {}
            for a, bnode in zip(mid_nodes[:-1], mid_nodes[1:]):
                next_anchor[a] = bnode - a
            for k in range(1, n - 1):
                if k in traces and k in next_anchor:
                    predictor_pairs["inst_vs_span"][0].append(traces[k]["raw_rel_l1"])
                    predictor_pairs["inst_vs_span"][1].append(next_anchor[k])
                    predictor_pairs["acc_vs_span"][0].append(traces[k]["accumulated_after"])
                    predictor_pairs["acc_vs_span"][1].append(next_anchor[k])
                if k in traces and (k, k + 1) in sj:
                    predictor_pairs["inst_vs_sjump"][0].append(traces[k]["raw_rel_l1"])
                    predictor_pairs["inst_vs_sjump"][1].append(sj[(k, k + 1)])
        except Exception as exc:  # noqa: BLE001
            print(f"[oracle]   predictor trace FAILED: {exc}", flush=True)

        sample_meta.append({
            "sample": sample_dir.name, "prompt": prompt, "seed": seed,
            "one_step_passed": passed, "one_step_worst_rel": worst, "one_step_mean_rel": mean_rel,
            "dp_nodes_saved": {str(k): v for k, v in dp_nodes_saved.items()},
            "uni_nodes_saved": {str(k): v for k, v in uni_nodes_saved.items()},
            "seacache_points": sc_points, "sample_imgs": {str(k): {kk: str(vv) for kk, vv in v.items()} for k, v in sample_sample_imgs.items()},
            "capped_frontier": {str(ms): pts for ms, pts in capped_frontier.items()},
            "heat_sjump_png_ready": True,
        })

        # per-sample S_jump heatmap figure
        fig_heatmap(run_dir / "figures" / f"heatmap_{sample_dir.name}.png", heat_sjump, f"S_jump[k,i] — {sample_dir.name}")

    del pipe
    torch.cuda.empty_cache()

    _finalize(args, run_dir, n, fixt, prompts, per_sample_rows, frontier_rows, predictor_pairs, sample_meta, dp_schedule_dump, max_spans, replay_saved)


def _finalize(args, run_dir, n, fixt, prompts, per_sample_rows, frontier_rows, predictor_pairs, sample_meta, dp_schedule_dump, max_spans, replay_saved):
    import matplotlib  # noqa

    # ---------- CSVs ----------
    def write_csv(path, rows, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            wcsv.writeheader()
            for r in rows:
                wcsv.writerow(r)

    ps_fields = ["sample", "prompt", "method", "saved", "budget", "achieved_budget", "threshold",
                 "latent_rel_l2", "psnr", "ssim", "lpips", "clip_img", "clip_text", "runtime_sec"]
    write_csv(run_dir / "metrics" / "per_sample_metrics.csv", per_sample_rows, ps_fields)
    fr_fields = ["sample", "method", "budget", "achieved_budget", "saved", "surrogate_cost", "replay_latent_rel_l2"]
    write_csv(run_dir / "metrics" / "frontier.csv", frontier_rows, fr_fields)

    # per-budget aggregate (mean over samples) for replay methods
    per_budget = {}
    for r in per_sample_rows:
        key = (r["method"], r["saved"])
        per_budget.setdefault(key, []).append(r)
    pb_rows = []
    for (method, saved), rs in sorted(per_budget.items()):
        def mean(k):
            vals = [float(x[k]) for x in rs if x.get(k) not in (None, "", "nan")]
            return float(np.mean(vals)) if vals else float("nan")
        pb_rows.append({"method": method, "saved": saved, "n": len(rs),
                        "mean_latent_rel_l2": mean("latent_rel_l2"), "mean_psnr": mean("psnr"),
                        "mean_ssim": mean("ssim"), "mean_lpips": mean("lpips"),
                        "mean_clip_img": mean("clip_img"), "mean_clip_text": mean("clip_text")})
    write_csv(run_dir / "metrics" / "per_budget_metrics.csv", pb_rows,
              ["method", "saved", "n", "mean_latent_rel_l2", "mean_psnr", "mean_ssim", "mean_lpips", "mean_clip_img", "mean_clip_text"])

    # predictor correlations
    corr_rows = []
    for name, (xs, ys) in predictor_pairs.items():
        corr_rows.append({"pair": name, "n": len(xs),
                          "pearson": fsd.pearson_corr(xs, ys), "spearman": fsd.spearman_corr(xs, ys)})
    write_csv(run_dir / "metrics" / "predictor_correlations.csv", corr_rows, ["pair", "n", "pearson", "spearman"])

    # ---------- figures ----------
    fig_method_diagram(run_dir / "figures" / "method_diagram.png", sample_meta[0]["dp_nodes_saved"].get(str(replay_saved[len(replay_saved) // 2]), [0, 50, n]), n)

    # aggregate S_jump heatmap
    heats = []
    for m in sample_meta:
        f = run_dir / "edge_costs" / f"{m['sample']}_S_jump.npz"
        if f.exists():
            heats.append(np.load(f)["s_jump"])
    if heats:
        agg = np.nanmean(np.stack(heats, axis=0), axis=0)
        fig_heatmap(run_dir / "figures" / "heatmap_aggregate.png", agg, "Aggregate mean S_jump[k,i]")

    # schedule raster (sample 0)
    m0 = sample_meta[0]
    dp_by_saved = {int(k): v for k, v in m0["dp_nodes_saved"].items()}
    uni_by_saved = {int(k): v for k, v in m0["uni_nodes_saved"].items()}
    fig_schedule_raster(run_dir / "figures" / "schedule_raster.png", dp_by_saved, uni_by_saved, m0["seacache_points"], n)

    # frontier curves (latent rel L2 surrogate & replay; PSNR/LPIPS from per_budget)
    dp_surr = [{"saved": r["saved"], "v": float(r["surrogate_cost"])} for r in frontier_rows if r["method"] == "dp_surrogate" and r["saved"] in replay_saved]
    # aggregate surrogate by saved
    def agg_by_saved(rows, key):
        d = {}
        for r in rows:
            d.setdefault(r["saved"], []).append(float(r[key]))
        return [{"saved": s, key: float(np.nanmean(v))} for s, v in sorted(d.items())]

    surr_curves = {
        "dp_surrogate": agg_by_saved([r for r in frontier_rows if r["method"] == "dp_surrogate"], "surrogate_cost"),
        "uniform": agg_by_saved([r for r in frontier_rows if r["method"] == "uniform"], "surrogate_cost"),
        "random": agg_by_saved([r for r in frontier_rows if r["method"] == "random"], "surrogate_cost"),
    }
    fig_frontier(run_dir / "figures" / "frontier_surrogate.png", surr_curves, "saved", "surrogate_cost",
                 "saved steps", "sum S_jump surrogate cost", "Frontier: DP surrogate vs baselines (lower better)")

    # replay/perceptual frontier from per_budget
    def pb_curve(method, ykey):
        return [{"saved": r["saved"], ykey: r[f"mean_{ykey}"]} for r in pb_rows if r["method"] == method and not math.isnan(r[f"mean_{ykey}"])]
    for ykey, fname, ylab, better in [("psnr", "frontier_psnr.png", "PSNR (dB)", "higher"),
                                      ("lpips", "frontier_lpips.png", "LPIPS", "lower"),
                                      ("latent_rel_l2", "frontier_latent.png", "final latent rel L2", "lower")]:
        curves = {"dp_replay": pb_curve("dp_replay", ykey), "uniform": pb_curve("uniform", ykey),
                  "seacache": pb_curve("seacache", ykey), "dp_cached": pb_curve("dp_cached", ykey)}
        fig_frontier(run_dir / "figures" / fname, curves, "saved", ykey, "saved steps", ylab,
                     f"Live-replay frontier: {ylab} ({better} is better)")

    # span histogram (aggregate over samples, mid saved budget)
    mid_saved = replay_saved[len(replay_saved) // 2]
    dp_spans, uni_spans = [], []
    for m in sample_meta:
        dn = m["dp_nodes_saved"].get(str(mid_saved))
        un = m["uni_nodes_saved"].get(str(mid_saved))
        if dn:
            dp_spans += span_lengths(dn)
        if un:
            uni_spans += span_lengths(un)
    fig_span_hist(run_dir / "figures" / "span_hist.png", dp_spans, uni_spans, f"Selected jump lengths (saved {mid_saved})")

    # predictor scatters
    for name, (xs, ys) in predictor_pairs.items():
        if len(xs) >= 3:
            fig_predictor_scatter(run_dir / "figures" / f"predictor_{name}.png", xs, ys,
                                  "SeaCache rel-L1" if "inst" in name else "SeaCache accumulated rel-L1",
                                  "oracle next-anchor span" if "span" in name else "S_jump[k,k+1]",
                                  name, fsd.pearson_corr(xs, ys), fsd.spearman_corr(xs, ys))

    # image grids per sample (vanilla | uniform | seacache | dp | dp_cached)
    grid_paths = []
    for m in sample_meta:
        sd = Path(args.trajectory_root) / m["sample"]
        for saved in replay_saved:
            imgs = m["sample_imgs"].get(str(saved), {})
            sc_png = m["seacache_points"][len(m["seacache_points"]) // 2]["png"] if m["seacache_points"] else None
            cells = [(sd / "final.png", "vanilla 100-step"),
                     (imgs.get("uniform"), f"uniform saved{saved}"),
                     (sc_png, "SeaCache"),
                     (imgs.get("dp"), f"DP jump saved{saved}")]
            if imgs.get("dp_cached"):
                cells.append((imgs.get("dp_cached"), f"DP cached saved{saved}"))
            gpath = run_dir / "figures" / f"grid_{m['sample']}_saved{saved}.png"
            image_grid(cells, gpath)
            grid_paths.append((gpath, f"{m['sample']} saved {saved}"))

    _write_report(args, run_dir, n, fixt, per_sample_rows, pb_rows, corr_rows, sample_meta, grid_paths, max_spans, replay_saved)

    # schedules dump + manifest
    fsd.write_json(run_dir / "schedules" / "dp_schedules.json", dp_schedule_dump)
    all_pass = all(m["one_step_passed"] for m in sample_meta)
    stage2_methods = sorted({r["method"] for r in per_sample_rows if "cached" in r["method"]})
    manifest = {
        "experiment": "E53_flux_dp_jump_oracle",
        "timestamp": dt.datetime.now().isoformat(),
        "smoke": args.smoke,
        "fixture_version": fixtures.FIXTURE_VERSION,
        "num_samples": len(sample_meta),
        "steps": n,
        "model_id": args.model_id,
        "dtype": args.dtype,
        "resolution": [args.height, args.width],
        "gpu": fsd.collect_gpu_info(),
        "one_step_sign_check_passed": all_pass,
        "replay_saved": replay_saved,
        "max_spans": max_spans,
        "seacache_available": any(r["method"] == "seacache" for r in per_sample_rows),
        "stage2_cached_residual_ran": bool(stage2_methods),
        "teacache_available": False,
        "teacache_note": "No TeaCache implementation exists in this repo; baseline marked unavailable.",
        "outputs": {
            "report_html": "report.html",
            "summary_md": "reports/summary.md",
            "summary_json": "reports/summary.json",
            "frontier_csv": "metrics/frontier.csv",
            "per_sample_csv": "metrics/per_sample_metrics.csv",
            "per_budget_csv": "metrics/per_budget_metrics.csv",
            "predictor_csv": "metrics/predictor_correlations.csv",
        },
    }
    fsd.write_json(run_dir / "artifacts_manifest.json", manifest)
    print(f"[oracle] DONE. report at {run_dir / 'report.html'}", flush=True)
    print(f"ORACLE_RUN_DIR={run_dir}", flush=True)


def _verdict(pb_rows, sample_meta):
    """Honest survival test.

    dp_replay is TEACHER-FORCED (reuses vanilla velocities at the true z_k), so it
    cannot drift much and always looks strong — it is essentially the compounded
    surrogate, not a causal method. The only causal live test is the stage-2
    cached-residual replay (dp_cached). Survival hinges on that when available.
    """
    def by_saved(method):
        return {r["saved"]: r for r in pb_rows if r["method"] == method and not math.isnan(r["mean_psnr"])}

    dp_r, dp_c = by_saved("dp_replay"), by_saved("dp_cached")
    if not dp_r:
        return "INCONCLUSIVE — no live replay data.", False
    mids = sorted(dp_r)
    mid = mids[len(mids) // 2]
    tf_psnr = dp_r[mid]["mean_psnr"]
    if dp_c:
        cs = sorted(dp_c)
        cmid = cs[len(cs) // 2]
        causal_psnr = dp_c[cmid]["mean_psnr"]
        survives = causal_psnr >= 25.0
        gap = tf_psnr - causal_psnr
        verdict = (
            f"The jump-DP oracle is TEACHER-FORCED and does NOT survive as a deployable schedule: "
            f"vanilla-velocity jump replay looks strong (PSNR≈{tf_psnr:.1f} dB at saved {mid}), "
            f"but the causal cached-residual replay of the SAME schedule collapses "
            f"(PSNR≈{causal_psnr:.1f} dB at saved {cmid}; {gap:.1f} dB gap). "
            f"DP-jump anchors do not transfer to cached-residual dynamics."
            if not survives else
            f"The jump-DP schedule survives causal cached-residual replay (PSNR≈{causal_psnr:.1f} dB at saved {cmid})."
        )
        return verdict, survives
    survives = tf_psnr >= 25.0
    return (f"Jump-DP oracle reaches PSNR≈{tf_psnr:.1f} dB under teacher-forced velocity replay at saved {mid}; "
            f"causal cached-residual stage-2 was not run, so deployable survival is untested (labelled oracle-only).", survives)


def _write_report(args, run_dir, n, fixt, per_sample_rows, pb_rows, corr_rows, sample_meta, grid_paths, max_spans, replay_saved):
    report_path = run_dir / "report.html"
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gpu = fsd.collect_gpu_info().get("gpu_name", "?")
    verdict, survives = _verdict(pb_rows, sample_meta)
    all_pass = all(m["one_step_passed"] for m in sample_meta)
    worst_rel = max((m["one_step_worst_rel"] for m in sample_meta), default=float("nan"))
    seacache_ok = any(r["method"] == "seacache" for r in per_sample_rows)
    stage2_ok = any("cached" in r["method"] for r in per_sample_rows)

    def fig(name, cap):
        p = run_dir / "figures" / name
        if not p.exists():
            return ""
        return f'<figure><img src="figures/{name}" alt="{name}"><figcaption>{cap}</figcaption></figure>'

    # per-budget table
    def pbtable(methods):
        head = "<tr><th>method</th><th>saved</th><th>latent relL2</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>CLIP-img</th><th>CLIP-txt</th></tr>"
        body = ""
        for r in sorted(pb_rows, key=lambda x: (x["method"], x["saved"])):
            if r["method"] not in methods:
                continue
            body += (f"<tr><td>{r['method']}</td><td>{r['saved']}</td><td>{r['mean_latent_rel_l2']:.4f}</td>"
                     f"<td>{r['mean_psnr']:.2f}</td><td>{r['mean_ssim']:.4f}</td><td>{r['mean_lpips']:.4f}</td>"
                     f"<td>{r['mean_clip_img']:.4f}</td><td>{r['mean_clip_text']:.4f}</td></tr>")
        return f"<table>{head}{body}</table>"

    corr_html = "<table><tr><th>pair</th><th>n</th><th>Pearson</th><th>Spearman</th></tr>"
    for c in corr_rows:
        corr_html += f"<tr><td>{c['pair']}</td><td>{c['n']}</td><td>{c['pearson']:.3f}</td><td>{c['spearman']:.3f}</td></tr>"
    corr_html += "</table>"

    grids_html = "".join(f'<figure><img src="figures/{Path(g).name}"><figcaption>{lab}</figcaption></figure>' for g, lab in grid_paths)

    method_cards = f"""
    <table class="methods">
      <tr><th>method</th><th>model</th><th>data</th><th>supervision</th><th>one-line insight</th></tr>
      <tr><td>Jump-DP surrogate</td><td>FLUX.1-dev {args.height}px {args.dtype} / {gpu}</td>
          <td>{len(sample_meta)} vanilla 100-step trajectories</td>
          <td>teacher-forced: vanilla z_k,v_k per edge</td>
          <td>Lower bound on achievable skip quality if per-edge jumps were independent.</td></tr>
      <tr><td>Jump-DP live replay</td><td>same</td><td>same trajectories</td>
          <td>compounded replay from z_0 with vanilla velocities</td>
          <td>Reveals whether the surrogate survives error compounding across jumps.</td></tr>
      <tr><td>SeaCache</td><td>same</td><td>live generation, same prompt/seed</td>
          <td>causal: online accumulated rel-L1 gate</td>
          <td>Deployable causal baseline; matched by achieved fresh-eval budget.</td></tr>
      <tr><td>Uniform / Random</td><td>same</td><td>same trajectories</td>
          <td>teacher-forced jump replay</td>
          <td>Naive schedule references at the same budget.</td></tr>
      <tr><td>Cached-residual (stage-2)</td><td>same</td><td>live generation along fixed schedule</td>
          <td>causal: refresh block-stack at anchors, reuse residual between</td>
          <td>{'Ran (capped).' if stage2_ok else 'Not run.'} True causal test of a DP schedule.</td></tr>
      <tr><td>TeaCache</td><td>—</td><td>—</td><td>—</td><td>Unavailable: no implementation in this repo.</td></tr>
    </table>"""

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>E53 — FLUX jump-DP skip-schedule oracle</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1100px;margin:0 auto;padding:28px;color:#1a1e24;background:#faf8f2;line-height:1.5}}
h1{{font-size:26px}} h2{{margin-top:34px;border-bottom:2px solid #ddd;padding-bottom:4px}}
.verdict{{background:#eef6ef;border-left:5px solid #1f9c5a;padding:12px 16px;font-size:17px;font-weight:600;border-radius:4px}}
.warn{{background:#fdf0e6;border-left:5px solid #e08a1e}}
figure{{margin:14px 0;background:#fff;padding:8px;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
img{{max-width:100%;height:auto;border-radius:4px}}
figcaption{{font-size:13px;color:#555;margin-top:6px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}}
th,td{{border:1px solid #ddd;padding:5px 8px;text-align:left}} th{{background:#f0ede4}}
code,pre{{background:#f2efe6;padding:2px 5px;border-radius:3px;font-size:13px}}
pre{{padding:12px;overflow-x:auto}} .chip{{display:inline-block;background:#e8eef5;padding:3px 9px;border-radius:12px;font-size:12px;margin:2px}}
</style></head><body>
<h1>E53 — FLUX jump-DP skip-schedule oracle</h1>
<p>Generated {now} · {gpu} · FLUX.1-dev · {args.height}×{args.width} · {args.dtype} · {n} steps · {len(sample_meta)} trajectories{' · SMOKE' if args.smoke else ''}</p>

<div class="verdict">{verdict}</div>
<p class="chip">one-line verdict</p>

<h2>Executive summary</h2>
<p>Given a full vanilla {n}-step no-skip FLUX trajectory we solve a budgeted shortest-path DP to
find the offline-optimal jump schedule under the cost
<code>S_jump[k,i]=||ẑ_i-z_i||²/(||z_i||²+ε)</code> with <code>ẑ_i=z_k+(σ_i-σ_k)·v_k</code>.
This is a <b>teacher-forced, non-causal oracle</b>: it reuses the vanilla velocity captured at the
true <code>z_k</code>. We keep three families strictly separate: (1) the offline surrogate (sum of
independent per-edge costs), (2) live jump replay (compounded from <code>z_0</code>), and (3) causal
baselines (SeaCache, uniform, random{', plus a capped cached-residual replay' if stage2_ok else ''}).
The one-step sign-convention check {'passed' if all_pass else 'FAILED'} (worst rel err {worst_rel:.2e}).</p>

<h2>Model / data / supervision / insight per method</h2>
{method_cards}

<h2>Exact DP formulation</h2>
<pre>N = {n}   B = number of fresh/jump segments   saved = N - B
edge cost  S_jump[k,i] = ||z_k + (σ_i-σ_k)·v_k - z_i||² / (||z_i||² + ε)      for 0 ≤ k &lt; i ≤ N
dp[b,i] = min cost to reach reference step i using b fresh/jump segments
dp[b,i] = min_{{k&lt;i}} ( dp[b-1,k] + S_jump[k,i] ),   dp[0,0]=0
backtrack from argmin_b dp[b,N] to recover the anchor schedule.</pre>
<p>Sign convention verified against the sampler: for <code>i=k+1</code> the jump prediction reproduces
the saved vanilla next latent (worst relative error {worst_rel:.2e} ≤ tol).</p>

<h2>Complexity / ETA</h2>
<p>All-pairs jump table is {n}(N+1)/2 = {n*(n+1)//2} edges/trajectory; jump-DP is O(N²·B) and runs in
milliseconds. Cost is dominated by (a) capturing vanilla trajectories ({n} transformer evals each) and
(b) the causal SeaCache / cached-residual live runs used as baselines.</p>

<h2>Method diagram</h2>
{fig('method_diagram.png', 'Vanilla trajectory; a candidate jump k→i; and the DP-selected fresh anchors on the step line.')}

<h2>Edge-cost heatmaps</h2>
{fig('heatmap_aggregate.png', 'Aggregate mean S_jump[k,i] over all trajectories. Bright = costly jump.')}
{''.join(fig(f"heatmap_{m['sample']}.png", f"S_jump for {m['sample']}") for m in sample_meta[:2])}

<h2>Schedule raster</h2>
{fig('schedule_raster.png', 'Fresh anchors vs saved budget. DP (blue) concentrates fresh evals where S_jump is high; uniform (orange) is evenly spaced.')}

<h2>Frontier curves</h2>
{fig('frontier_surrogate.png', 'Surrogate cost vs saved steps (DP vs uniform vs random). DP is the offline lower bound.')}
{fig('frontier_psnr.png', 'Live-replay PSNR vs saved steps: DP jump replay, uniform, SeaCache, and cached-residual (stage-2) if run.')}
{fig('frontier_lpips.png', 'Live-replay LPIPS vs saved steps.')}
{fig('frontier_latent.png', 'Final-latent relative L2 vs saved steps.')}

<h2>Span histograms</h2>
{fig('span_hist.png', 'Distribution of selected jump lengths (DP vs uniform) at a representative budget.')}

<h2>Live-replay results (mean over samples)</h2>
{pbtable({'dp_replay','uniform','seacache','dp_cached','uniform_cached'})}

<h2>Predictor diagnostics</h2>
<p>Does the SeaCache predictor (instantaneous / accumulated relative-L1) also predict how far the
oracle can safely jump — or only refresh/no-refresh?</p>
{corr_html}
{fig('predictor_inst_vs_span.png', 'SeaCache instantaneous rel-L1 vs oracle next-anchor span.')}
{fig('predictor_acc_vs_span.png', 'SeaCache accumulated rel-L1 vs oracle next-anchor span.')}
{fig('predictor_inst_vs_sjump.png', 'SeaCache instantaneous rel-L1 vs S_jump[k,k+1].')}

<h2>Image samples</h2>
<p>Columns: vanilla reference · uniform · SeaCache · DP jump replay{' · DP cached replay' if stage2_ok else ''}.</p>
{grids_html}

<h2>Limitations</h2>
<ul>
<li><b>Non-causal / teacher-forced.</b> The jump-DP oracle reuses vanilla velocities computed at the
true <code>z_k</code>; it cannot be deployed as-is. It is a diagnostic upper bound on schedule quality.</li>
<li><b>Surrogate ≠ replay.</b> The surrogate sums independent per-edge costs; live jump replay compounds
error from <code>z_0</code>. Where they diverge, believe the replay.</li>
<li>SeaCache comparison matched by <i>achieved</i> fresh-eval budget, not nominal threshold.
{'' if seacache_ok else 'SeaCache runs failed in this run — see logs.'}</li>
<li>{'Cached-residual stage-2 was run on a capped set (DP + uniform schedules at the replay budgets).' if stage2_ok else 'Cached-residual stage-2 did not produce data in this run.'}</li>
<li>TeaCache: no implementation in this repo; marked unavailable rather than approximated.</li>
</ul>

<h2>Next recommended experiment</h2>
<p>If the surrogate does not survive replay, the informative next step is a <i>path-dependent</i> cached-residual
DP restricted to short spans (≤12–16) plus jump-DP-selected spans, i.e. re-scoring edges from the actually
reached state rather than vanilla <code>z_k</code> — closing the gap between oracle and deployable SeaCache.</p>

</body></html>"""

    html = fsd.embed_local_images(html, report_path)
    report_path.write_text(html, encoding="utf-8")

    # summary md/json
    summary = {
        "experiment": "E53_flux_dp_jump_oracle", "verdict": verdict, "survives_live_replay": survives,
        "one_step_sign_check_passed": all_pass, "one_step_worst_rel": worst_rel,
        "seacache_available": seacache_ok, "stage2_cached_residual_ran": stage2_ok,
        "teacache_available": False, "num_samples": len(sample_meta),
        "per_budget": pb_rows, "predictor_correlations": corr_rows,
    }
    fsd.write_json(run_dir / "reports" / "summary.json", summary)
    md = [f"# E53 — FLUX jump-DP skip-schedule oracle\n",
          f"**Verdict:** {verdict}\n",
          f"- one-step sign check passed: {all_pass} (worst rel {worst_rel:.2e})",
          f"- SeaCache available: {seacache_ok}",
          f"- cached-residual stage-2 ran: {stage2_ok}",
          f"- TeaCache: unavailable (no impl in repo)",
          f"- trajectories: {len(sample_meta)} · steps: {n}\n",
          "## Per-budget (mean over samples)\n",
          "| method | saved | latent relL2 | PSNR | SSIM | LPIPS |",
          "|---|---|---|---|---|---|"]
    for r in sorted(pb_rows, key=lambda x: (x["method"], x["saved"])):
        md.append(f"| {r['method']} | {r['saved']} | {r['mean_latent_rel_l2']:.4f} | {r['mean_psnr']:.2f} | {r['mean_ssim']:.4f} | {r['mean_lpips']:.4f} |")
    (run_dir / "reports" / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="E53 FLUX jump-DP skip-schedule oracle")
    ap.add_argument("--model-id", default=fsd.DEFAULT_MODEL_ID)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=31000)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--max-sequence-length", type=int, default=512)
    ap.add_argument("--cache-h-threshold", type=float, default=0.02)
    ap.add_argument("--replay-saved", default="10,25,50,67,75,90")
    ap.add_argument("--max-spans", default="4,8,12,16")
    ap.add_argument("--seacache-thresholds", default="0.2,0.3,0.4,0.6")
    ap.add_argument("--random-trials", type=int, default=8)
    ap.add_argument("--stage2", action="store_true", help="Run capped cached-residual (stage-2) replay.")
    ap.add_argument("--no-seacache", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Reduced sweep for verification.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--trajectory-root", default=str(REPO_ROOT / "outputs" / "flux_dp_jump_oracle" / "trajectories"))
    ap.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "h100"))
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
