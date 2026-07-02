#!/usr/bin/env python3
"""E54 — causal runtime jump-size hints for FLUX.

This experiment asks whether cheap, causal local-error hints can decide when a
FLUX sampler may safely take a larger sigma step. It intentionally treats the
E53 jump-DP result as a negative/offline baseline: saved vanilla latents and
velocities are used only for references and post-hoc error diagnostics, never
for runtime controller decisions or updates.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import inspect
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
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures  # noqa: E402
import flux_dp_jump_oracle as e53  # noqa: E402
import flux_seacache_dp_shortcuts as fsd  # noqa: E402

N = 100
EPS = 1e-8
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ControllerSpec:
    method: str
    threshold: float
    update: str = "euler"
    max_jump: int = 4
    weights: tuple[float, float, float] = (1.0, 0.0, 0.0)

    @property
    def key(self) -> str:
        if self.method == "uniform":
            return f"uniform_ret{int(self.threshold)}"
        bits = [self.method, f"th{self.threshold:g}"]
        if self.update != "euler":
            bits.append(self.update)
        if self.method == "hybrid":
            bits.append("w" + "-".join(f"{w:g}" for w in self.weights))
        if self.max_jump != 4:
            bits.append(f"m{self.max_jump}")
        return "_".join(bits).replace(".", "p")


class CallCounter:
    def __init__(self, pipe: Any):
        self.pipe = pipe
        self.n = 0
        self._orig = pipe.transformer.forward

        def wrapped(*args, **kwargs):
            self.n += 1
            return self._orig(*args, **kwargs)

        pipe.transformer.forward = wrapped

    def reset(self) -> None:
        self.n = 0

    def restore(self) -> None:
        self.pipe.transformer.forward = self._orig


def ensure_tree(run_dir: Path) -> None:
    for sub in [
        "figures",
        "samples",
        "metrics/runtime_traces",
        "reports",
        "trajectories",
    ]:
        fsd.ensure_dir(run_dir / sub)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    fsd.ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm((a.float() - b.float()).flatten()) / (torch.linalg.vector_norm(b.float().flatten()) + EPS))


def norm_l2(a: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a.float().flatten()))


def get_default_steps(pipe: Any) -> int:
    sig = inspect.signature(pipe.__call__)
    return int(sig.parameters["num_inference_steps"].default)


def prep_inputs(pipe: Any, prompt: str, seed: int, steps: int, height: int, width: int, guidance: float, max_seq_len: int, device: str) -> dict[str, Any]:
    pe, ppe, text_ids, latents, image_ids, timesteps, guidance_tensor = fsd.prepare_flux_inputs(
        pipe, prompt, seed, steps, height, width, guidance, device, max_seq_len
    )
    return {
        "pe": pe,
        "ppe": ppe,
        "text_ids": text_ids,
        "z0": latents,
        "image_ids": image_ids,
        "timesteps": timesteps,
        "guidance": guidance_tensor,
    }


@torch.no_grad()
def velocity_call(pipe: Any, prep: dict[str, Any], z: torch.Tensor, k: int) -> torch.Tensor:
    ts = prep["timesteps"][k].expand(z.shape[0]).to(z.dtype)
    return pipe.transformer(
        hidden_states=z,
        timestep=ts / 1000,
        guidance=prep["guidance"],
        pooled_projections=prep["ppe"],
        encoder_hidden_states=prep["pe"],
        txt_ids=prep["text_ids"],
        img_ids=prep["image_ids"],
        return_dict=False,
    )[0]


@torch.no_grad()
def sea_feature(pipe: Any, prep: dict[str, Any], z: torch.Tensor, k: int, sea_filter: bool = True) -> torch.Tensor:
    ts = prep["timesteps"][k].expand(z.shape[0]).to(z.dtype)
    _, h = fsd.flux_h_decision_tensor(
        pipe.transformer,
        z,
        ts / 1000,
        prep["pe"],
        prep["ppe"],
        prep["image_ids"],
        prep["text_ids"],
        prep["guidance"],
        pipe.scheduler,
        k,
        sea_filter=sea_filter and k not in (0, len(prep["timesteps"]) - 1),
    )
    return h.float()


def uniform_nodes(n: int, retained: int) -> list[int]:
    retained = max(1, min(n, int(retained)))
    pts = np.linspace(0, n, retained + 1)
    nodes = sorted(set(int(round(x)) for x in pts))
    if nodes[0] != 0:
        nodes.insert(0, 0)
    if nodes[-1] != n:
        nodes.append(n)
    return sorted(set(nodes))


def choose_uniform_target(k: int, nodes: list[int]) -> int:
    for node in nodes[1:]:
        if node > k:
            return node
    return N


def candidate_ms(k: int, max_jump: int) -> list[int]:
    out = [m for m in (1, 2, 3, 4, 6, 8) if m <= max_jump and k + m <= N]
    if not out:
        return [N - k]
    return out


def controller_score(
    spec: ControllerSpec,
    m: int,
    k: int,
    z: torch.Tensor,
    v: torch.Tensor,
    v_prev: torch.Tensor | None,
    prev_k: int | None,
    sigmas: list[float],
    h_k: torch.Tensor | None,
    h_i: torch.Tensor | None,
) -> tuple[float, dict[str, float]]:
    i = min(N, k + m)
    h = float(sigmas[i] - sigmas[k])
    abs_h = abs(h)
    v_norm = norm_l2(v) + EPS
    z_norm = norm_l2(z) + EPS
    curv = math.inf
    curv_rel = math.inf
    lte = math.inf
    ab2 = math.inf
    sea = math.inf
    hcos = math.inf
    if v_prev is not None and prev_k is not None:
        h_prev = float(sigmas[k] - sigmas[prev_k])
        denom = abs(h_prev) + EPS
        curv = norm_l2(v - v_prev) / denom
        curv_rel = norm_l2(v - v_prev) / v_norm
        lte = 0.5 * (abs_h**2) * curv / v_norm
        r = h / (h_prev + math.copysign(EPS, h_prev if h_prev != 0 else -1.0))
        z_euler = z.float() + h * v.float()
        z_ab2 = z.float() + h * ((1.0 + r / 2.0) * v.float() - (r / 2.0) * v_prev.float())
        ab2 = norm_l2(z_ab2 - z_euler) / (1e-8 + 1e-3 * max(z_norm, norm_l2(z_ab2)))
    if h_k is not None and h_i is not None:
        sea = fsd.rel_l1(h_i, h_k)
        hk = h_k.flatten()
        hi = h_i.flatten()
        hcos = float(1.0 - torch.nn.functional.cosine_similarity(hi[None], hk[None]).item())
    if spec.method == "curvature":
        score = lte
    elif spec.method == "ab2":
        score = ab2
    elif spec.method == "sea_defect":
        score = sea
    elif spec.method == "hybrid":
        wc, wa, ws = spec.weights
        c = 0.0 if not math.isfinite(lte) else lte
        a = 0.0 if not math.isfinite(ab2) else ab2
        s = 0.0 if not math.isfinite(sea) else sea
        score = wc * c + wa * a + ws * s
    else:
        score = 0.0
    return float(score), {
        "curv_abs": float(curv),
        "curv_rel": float(curv_rel),
        "lte_norm": float(lte),
        "ab2_disagreement": float(ab2),
        "sea_rel_l1": float(sea),
        "sea_cos_defect": float(hcos),
    }


@torch.no_grad()
def adaptive_replay(
    pipe: Any,
    prep: dict[str, Any],
    spec: ControllerSpec,
    sample_dir: Path,
    sigmas: list[float],
    height: int,
    width: int,
    device: str,
) -> tuple[torch.Tensor, Image.Image, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    z = prep["z0"].clone()
    k = 0
    calls = 0
    prefix_calls = 0
    v_prev: torch.Tensor | None = None
    prev_k: int | None = None
    h_prev_anchor: torch.Tensor | None = None
    trace: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    nodes = [0]
    t0 = time.perf_counter()
    uniform = uniform_nodes(N, int(spec.threshold)) if spec.method == "uniform" else None

    while k < N:
        v = velocity_call(pipe, prep, z, k)
        calls += 1
        h_k = None
        if spec.method in ("sea_defect", "hybrid"):
            h_k = sea_feature(pipe, prep, z, k)
            prefix_calls += 1
        if spec.method == "uniform":
            chosen_i = choose_uniform_target(k, uniform or [0, N])
            chosen_m = chosen_i - k
            chosen_score = 0.0
            chosen_stats: dict[str, float] = {}
            chosen_use_ab2 = False
        else:
            chosen_m = 1
            chosen_i = min(N, k + 1)
            chosen_score = math.inf
            chosen_stats = {}
            chosen_use_ab2 = spec.update == "ab2" and v_prev is not None
            for m in candidate_ms(k, spec.max_jump):
                i = min(N, k + m)
                h = float(sigmas[i] - sigmas[k])
                h_i = None
                if spec.method in ("sea_defect", "hybrid"):
                    z_prop = z.float() + h * v.float()
                    h_i = sea_feature(pipe, prep, z_prop.to(z.dtype), i if i < N else N - 1)
                    prefix_calls += 1
                score, stats = controller_score(spec, m, k, z, v, v_prev, prev_k, sigmas, h_k, h_i)
                z_prop = z.float() + h * v.float()
                z_ref = fsd.load_step_tensor(sample_dir / "latents", i)
                err_local = rel_l2(z_prop.detach().cpu(), z_ref) ** 2
                diagnostics.append({
                    "method": spec.key,
                    "base_method": spec.method,
                    "threshold": spec.threshold,
                    "k": k,
                    "i": i,
                    "m": m,
                    "accepted": False,
                    "score": score,
                    "err_local": err_local,
                    "sigma_k": float(sigmas[k]),
                    "sigma_i": float(sigmas[i]),
                    "sigma_region": sigma_region(k),
                    **stats,
                })
                if v_prev is None and m > 1:
                    continue
                if score < spec.threshold and m >= chosen_m:
                    chosen_m = m
                    chosen_i = i
                    chosen_score = score
                    chosen_stats = stats
                    chosen_use_ab2 = spec.update == "ab2" and v_prev is not None
            if not math.isfinite(chosen_score):
                score, stats = controller_score(spec, 1, k, z, v, v_prev, prev_k, sigmas, h_k, None)
                chosen_score = score
                chosen_stats = stats
        h = float(sigmas[chosen_i] - sigmas[k])
        if chosen_use_ab2 and v_prev is not None and prev_k is not None:
            h_prev = float(sigmas[k] - sigmas[prev_k])
            r = h / (h_prev + math.copysign(EPS, h_prev if h_prev != 0 else -1.0))
            z_next = z.float() + h * ((1.0 + r / 2.0) * v.float() - (r / 2.0) * v_prev.float())
        else:
            z_next = z.float() + h * v.float()
        z_ref_i = fsd.load_step_tensor(sample_dir / "latents", chosen_i)
        reached = rel_l2(z_next.detach().cpu(), z_ref_i)
        for row in diagnostics:
            if row["k"] == k and row["i"] == chosen_i and row["method"] == spec.key:
                row["accepted"] = True
        trace.append({
            "method": spec.key,
            "base_method": spec.method,
            "threshold": spec.threshold,
            "seg": len(trace),
            "k": k,
            "i": chosen_i,
            "jump": chosen_m,
            "sigma_k": float(sigmas[k]),
            "sigma_i": float(sigmas[chosen_i]),
            "delta_sigma": h,
            "score": chosen_score,
            "full_call": True,
            "saved_velocity_used": False,
            "prefix_calls_so_far": prefix_calls,
            "latent_rel_l2_to_ref_at_i": reached,
            **chosen_stats,
        })
        v_prev = v.detach()
        prev_k = k
        h_prev_anchor = h_k if h_k is not None else h_prev_anchor
        z = z_next.to(prep["z0"].dtype)
        k = chosen_i
        nodes.append(k)
    wall = time.perf_counter() - t0
    img = fsd.decode_flux_latents(pipe, z.to(device).to(pipe.dtype), height, width)
    audit = {
        "actual_full_calls": calls,
        "cheap_prefix_calls": prefix_calls,
        "actual_segments": len(nodes) - 1,
        "nodes": nodes,
        "wall_sec": wall,
    }
    return z.detach().float().cpu(), img, audit, trace, diagnostics


def sigma_region(k: int) -> str:
    if k < N / 3:
        return "early_high_noise"
    if k < 2 * N / 3:
        return "mid"
    return "late_low_noise"


def offline_saved_velocity_replay(sample_dir: Path, sigmas: list[float], nodes: list[int]) -> torch.Tensor:
    z = fsd.load_step_tensor(sample_dir / "latents", 0)
    for k, i in zip(nodes[:-1], nodes[1:]):
        v = fsd.load_step_tensor(sample_dir / "velocities", k)
        z = z + float(sigmas[i] - sigmas[k]) * v
    return z


def method_specs(args: argparse.Namespace) -> list[ControllerSpec]:
    specs: list[ControllerSpec] = []
    for retained in args.uniform_retained:
        specs.append(ControllerSpec("uniform", float(retained), max_jump=N))
    for tau in args.curvature_thresholds:
        specs.append(ControllerSpec("curvature", tau, max_jump=args.max_jump))
    for tau in args.ab2_thresholds:
        specs.append(ControllerSpec("ab2", tau, "euler", args.max_jump))
        specs.append(ControllerSpec("ab2", tau, "ab2", args.max_jump))
    for tau in args.sea_thresholds:
        specs.append(ControllerSpec("sea_defect", tau, max_jump=args.max_jump))
    for tau in args.hybrid_thresholds:
        specs.append(ControllerSpec("hybrid", tau, max_jump=args.max_jump, weights=(1.0, 1.0, 0.0)))
        if args.sea_thresholds:
            specs.append(ControllerSpec("hybrid", tau, max_jump=args.max_jump, weights=(1.0, 1.0, 1.0)))
    return specs


def run(args: argparse.Namespace) -> None:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"{ts}__flux_runtime_jump_hints{'__smoke' if args.smoke else ''}"
    ensure_tree(run_dir)
    print(f"[e54] run_dir={run_dir}", flush=True)

    fixture = fixtures.canonical_prompts()[: args.num_samples]
    fsd.PROMPTS = [(x["tag"], x["prompt"]) for x in fixture]
    traj_root = Path(args.trajectory_root)
    cap_args = SimpleNamespace(
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        offload=args.offload,
        bnb4=False,
        num_samples=args.num_samples,
        seed_base=args.seed_base,
        steps=N,
        height=args.height,
        width=args.width,
        guidance=args.guidance,
        max_sequence_length=args.max_sequence_length,
        save_h_raw=False,
        force=args.force_capture,
        output_root=str(traj_root),
    )
    print("[e54] capturing/reusing 100-step vanilla trajectories", flush=True)
    fsd.run_capture(cap_args)

    pipe = fsd.load_flux_pipeline(args.model_id, args.dtype, args.device, args.offload, False)
    bank = e53.MetricBank(args.device)
    default_steps = get_default_steps(pipe)
    specs = method_specs(args)
    if args.smoke:
        specs = specs[: min(len(specs), args.smoke_methods)]

    all_metric_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    sample_grid: dict[tuple[str, str], Path] = {}
    sea_trace_for_plot: list[dict[str, Any]] = []
    sample_dirs = sorted(traj_root.glob("sample_*"))[: args.num_samples]

    counter = CallCounter(pipe)
    try:
        for si, sd in enumerate(sample_dirs):
            meta = fsd.read_json(sd / "metadata.json")
            prompt = meta["prompt"]
            seed = int(meta["seed"])
            height = int(meta["height"])
            width = int(meta["width"])
            sigmas = [float(x) for x in meta["sigmas"]]
            reference_wall_sec = float(meta.get("runtime_sec", 0.0) or 0.0)
            vanilla_img = Image.open(sd / "final.png").convert("RGB")
            vanilla_lat = fsd.load_step_tensor(sd / "latents", N)
            prep = prep_inputs(pipe, prompt, seed, N, height, width, args.guidance, args.max_sequence_length, args.device)
            print(f"[e54] sample={sd.name} seed={seed} prompt={prompt[:60]!r}", flush=True)

            # 100-step reference audit row.
            call_rows.append({
                "sample": sd.name,
                "method": "vanilla_100step",
                "intended_calls": N,
                "actual_full_calls": N,
                "cheap_prefix_calls": 0,
                "valid": True,
                "note": "captured reference trajectory",
            })
            sample_grid[(sd.name, "vanilla_100step")] = sd / "final.png"

            # Programmatic default FLUX.
            counter.reset()
            d_lat, d_img, d_wall = e53.live_generate(
                pipe, prompt, seed, default_steps, height, width, args.guidance, args.max_sequence_length, args.device
            )
            d_calls = counter.n
            d_png = run_dir / "samples" / f"{sd.name}_default{default_steps}.png"
            d_img.save(d_png)
            dm = bank.image_metrics(d_img, vanilla_img, prompt)
            dll = fsd.latent_metrics(d_lat, vanilla_lat)["latent_rel_l2"]
            row = metric_row(sd.name, prompt, "default_flux", "default_flux", "", True, default_steps, d_calls, 0, d_wall, dll, dm, d_png, reference_wall_sec=reference_wall_sec)
            per_sample_rows.append(row)
            call_rows.append(call_audit(sd.name, "default_flux", default_steps, d_calls, 0, True, "pipeline default num_inference_steps"))
            leakage_rows.append(leakage("default_flux", False, False, "standard pipeline"))
            sample_grid[(sd.name, "default_flux")] = d_png

            # SeaCache threshold sweep.
            for th in args.seacache_thresholds:
                state = fsd.install_seacache_forward(pipe, th, N)
                counter.reset()
                lat, img, wall = e53.live_generate(
                    pipe, prompt, seed, N, height, width, args.guidance, args.max_sequence_length, args.device
                )
                counted = counter.n
                traces = list(state["step_traces"])
                fresh = int(state["fresh_evals"])
                cached = int(state["cached_evals"])
                state["restore"]()
                png = run_dir / "samples" / f"{sd.name}_seacache_th{th:g}.png"
                img.save(png)
                metrics = bank.image_metrics(img, vanilla_img, prompt)
                ll = fsd.latent_metrics(lat, vanilla_lat)["latent_rel_l2"]
                per_sample_rows.append(metric_row(sd.name, prompt, "seacache", "seacache", th, True, N, fresh, 0, wall, ll, metrics, png, reference_wall_sec=reference_wall_sec, cached_calls=cached, counted_transformer_calls=counted))
                call_rows.append(call_audit(sd.name, f"seacache_th{th:g}", N, fresh, 0, counted == N, f"fresh={fresh}; cached={cached}; wrapper calls include cache reuses"))
                leakage_rows.append(leakage("seacache", False, False, "official accumulated SEA rel-L1 gate"))
                if si == 0 and (not sea_trace_for_plot or abs(th - args.seacache_thresholds[len(args.seacache_thresholds) // 2]) < 1e-12):
                    sea_trace_for_plot = traces
                sample_grid.setdefault((sd.name, "seacache"), png)

            # Runtime hint controllers.
            for spec in specs:
                counter.reset()
                lat, img, audit, tr, diags = adaptive_replay(pipe, prep, spec, sd, sigmas, height, width, args.device)
                counted = counter.n
                if counted != audit["actual_full_calls"]:
                    raise RuntimeError(f"{spec.key} counted {counted} != audit {audit['actual_full_calls']}")
                png = run_dir / "samples" / f"{sd.name}_{spec.key}.png"
                img.save(png)
                metrics = bank.image_metrics(img, vanilla_img, prompt)
                ll = fsd.latent_metrics(lat, vanilla_lat)["latent_rel_l2"]
                row = metric_row(
                    sd.name,
                    prompt,
                    spec.key,
                    spec.method,
                    spec.threshold,
                    True,
                    audit["actual_segments"],
                    audit["actual_full_calls"],
                    audit["cheap_prefix_calls"],
                    audit["wall_sec"],
                    ll,
                    metrics,
                    png,
                    reference_wall_sec=reference_wall_sec,
                    update=spec.update,
                    nodes=json.dumps(audit["nodes"]),
                )
                per_sample_rows.append(row)
                trace_rows.extend({"sample": sd.name, **x} for x in tr)
                diag_rows.extend({"sample": sd.name, **x} for x in diags)
                call_rows.append(call_audit(sd.name, spec.key, audit["actual_segments"], audit["actual_full_calls"], audit["cheap_prefix_calls"], True, "causal adaptive replay"))
                leakage_rows.append(leakage(spec.key, False, False, "uses live current/previous velocity only"))
                if spec.method in ("uniform", "curvature", "ab2", "hybrid") and (sd.name, spec.method) not in sample_grid:
                    sample_grid[(sd.name, spec.method)] = png

            # Optional non-causal diagnostic, clearly separated.
            for retained in args.oracle_retained:
                nodes = uniform_nodes(N, retained)
                lat = offline_saved_velocity_replay(sd, sigmas, nodes)
                img = fsd.decode_flux_latents(pipe, lat.to(args.device).to(pipe.dtype), height, width)
                png = run_dir / "samples" / f"{sd.name}_offline_saved_velocity_ret{retained}.png"
                img.save(png)
                metrics = bank.image_metrics(img, vanilla_img, prompt)
                ll = fsd.latent_metrics(lat, vanilla_lat)["latent_rel_l2"]
                per_sample_rows.append(metric_row(sd.name, prompt, f"offline_saved_velocity_ret{retained}", "offline_saved_velocity_oracle", retained, False, retained, 0, 0, 0.0, ll, metrics, png, reference_wall_sec=reference_wall_sec))
                call_rows.append(call_audit(sd.name, f"offline_saved_velocity_ret{retained}", 0, 0, 0, True, "non-causal oracle; saved vanilla velocity replay"))
                leakage_rows.append(leakage("offline_saved_velocity_oracle", True, True, "diagnostic only; invalid as deployable sampler"))

    finally:
        counter.restore()
        del pipe
        torch.cuda.empty_cache()

    all_metric_rows = aggregate_budget_rows(per_sample_rows)
    corr_rows = local_error_correlations(diag_rows)
    write_outputs(run_dir, per_sample_rows, all_metric_rows, trace_rows, diag_rows, corr_rows, call_rows, leakage_rows)
    make_figures(run_dir, all_metric_rows, per_sample_rows, trace_rows, diag_rows, corr_rows, sea_trace_for_plot, sample_grid, sample_dirs)
    write_report(run_dir, args, default_steps, per_sample_rows, all_metric_rows, corr_rows, call_rows, leakage_rows)
    write_manifest(run_dir, args)
    print(f"[e54] DONE report={run_dir / 'report.html'}", flush=True)


def metric_row(sample: str, prompt: str, method: str, base_method: str, threshold: Any, causal: bool, intended_calls: int, actual_calls: int, prefix_calls: int, wall_sec: float, latent_rel_l2: float, metrics: dict[str, float], png: Path, **extra: Any) -> dict[str, Any]:
    reference_wall_sec = extra.pop("reference_wall_sec", "")
    try:
        wall_speedup = float(reference_wall_sec) / float(wall_sec) if float(reference_wall_sec) > 0 and float(wall_sec) > 0 else ""
    except Exception:
        wall_speedup = ""
    out = {
        "sample": sample,
        "prompt": prompt,
        "method": method,
        "base_method": base_method,
        "threshold": threshold,
        "causal": causal,
        "intended_calls": intended_calls,
        "actual_full_calls": actual_calls,
        "cheap_prefix_calls": prefix_calls,
        "cached_calls": extra.pop("cached_calls", ""),
        "counted_transformer_calls": extra.pop("counted_transformer_calls", actual_calls),
        "wall_sec": wall_sec,
        "speedup_vs_100": N / max(1, actual_calls) if actual_calls > 0 else "",
        "reference_wall_sec": reference_wall_sec,
        "wall_speedup_vs_100": wall_speedup,
        "latent_rel_l2": latent_rel_l2,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "lpips": metrics["lpips"],
        "clip_img": metrics["clip_img"],
        "clip_text": metrics["clip_text"],
        "image_path": str(png),
        "image_sha256": sha256_file(png),
    }
    out.update(extra)
    return out


def call_audit(sample: str, method: str, intended: int, actual: int, prefix: int, valid: bool, note: str) -> dict[str, Any]:
    return {
        "sample": sample,
        "method": method,
        "intended_calls": intended,
        "actual_full_calls": actual,
        "cheap_prefix_calls": prefix,
        "valid": valid,
        "note": note,
    }


def leakage(method: str, saved_vel: bool, saved_update: bool, note: str) -> dict[str, Any]:
    return {
        "method": method,
        "saved_vanilla_velocities_used_for_decision": saved_vel,
        "saved_vanilla_velocities_used_for_update": saved_update,
        "valid_causal": not (saved_vel or saved_update),
        "note": note,
    }


def aggregate_budget_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        if r["actual_full_calls"] == "":
            continue
        groups.setdefault((r["method"], int(r["actual_full_calls"])), []).append(r)
    out = []
    for (method, calls), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        def mean(key: str) -> float:
            vals = [float(r[key]) for r in rs if r.get(key) not in ("", None)]
            return float(np.mean(vals)) if vals else float("nan")
        out.append({
            "method": method,
            "base_method": rs[0]["base_method"],
            "threshold": rs[0]["threshold"],
            "causal": rs[0]["causal"],
            "actual_full_calls": calls,
            "cheap_prefix_calls_mean": mean("cheap_prefix_calls"),
            "wall_sec_mean": mean("wall_sec"),
            "speedup_vs_100": N / max(1, calls),
            "reference_wall_sec_mean": mean("reference_wall_sec"),
            "wall_speedup_vs_100_mean": mean("wall_speedup_vs_100"),
            "latent_rel_l2_mean": mean("latent_rel_l2"),
            "psnr_mean": mean("psnr"),
            "ssim_mean": mean("ssim"),
            "lpips_mean": mean("lpips"),
            "clip_text_mean": mean("clip_text"),
            "n": len(rs),
        })
    return out


def local_error_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if r["score"] in ("", None) or not math.isfinite(float(r["score"])):
            continue
        groups.setdefault((r["base_method"], "all"), []).append(r)
        groups.setdefault((r["base_method"], r["sigma_region"]), []).append(r)
    for (method, region), rs in sorted(groups.items()):
        xs = [float(r["score"]) for r in rs]
        ys = [float(r["err_local"]) for r in rs]
        out.append({
            "hint": method,
            "sigma_region": region,
            "n": len(rs),
            "pearson": fsd.pearson_corr(xs, ys),
            "spearman": fsd.spearman_corr(xs, ys),
        })
    return out


def write_outputs(run_dir: Path, per_sample: list[dict[str, Any]], per_budget: list[dict[str, Any]], traces: list[dict[str, Any]], diags: list[dict[str, Any]], corrs: list[dict[str, Any]], calls: list[dict[str, Any]], leaks: list[dict[str, Any]]) -> None:
    metric_fields = [
        "sample",
        "prompt",
        "method",
        "base_method",
        "threshold",
        "causal",
        "intended_calls",
        "actual_full_calls",
        "cheap_prefix_calls",
        "cached_calls",
        "counted_transformer_calls",
        "wall_sec",
        "speedup_vs_100",
        "reference_wall_sec",
        "wall_speedup_vs_100",
        "latent_rel_l2",
        "psnr",
        "ssim",
        "lpips",
        "clip_img",
        "clip_text",
        "image_path",
        "image_sha256",
        "update",
        "nodes",
    ]
    write_csv(run_dir / "metrics" / "per_sample_metrics.csv", per_sample, metric_fields)
    write_csv(run_dir / "metrics" / "per_method_budget_metrics.csv", per_budget, [
        "method",
        "base_method",
        "threshold",
        "causal",
        "actual_full_calls",
        "cheap_prefix_calls_mean",
        "wall_sec_mean",
        "speedup_vs_100",
        "reference_wall_sec_mean",
        "wall_speedup_vs_100_mean",
        "latent_rel_l2_mean",
        "psnr_mean",
        "ssim_mean",
        "lpips_mean",
        "clip_text_mean",
        "n",
    ])
    trace_fields = [
        "sample",
        "method",
        "base_method",
        "threshold",
        "seg",
        "k",
        "i",
        "jump",
        "sigma_k",
        "sigma_i",
        "delta_sigma",
        "score",
        "full_call",
        "saved_velocity_used",
        "prefix_calls_so_far",
        "latent_rel_l2_to_ref_at_i",
        "curv_abs",
        "curv_rel",
        "lte_norm",
        "ab2_disagreement",
        "sea_rel_l1",
        "sea_cos_defect",
    ]
    write_csv(run_dir / "metrics" / "runtime_traces" / "all_runtime_traces.csv", traces, trace_fields)
    for method in sorted({r["method"] for r in traces}):
        write_csv(run_dir / "metrics" / "runtime_traces" / f"{method}.csv", [r for r in traces if r["method"] == method], trace_fields)
    diag_fields = [
        "sample",
        "method",
        "base_method",
        "threshold",
        "k",
        "i",
        "m",
        "accepted",
        "score",
        "err_local",
        "sigma_k",
        "sigma_i",
        "sigma_region",
        "curv_abs",
        "curv_rel",
        "lte_norm",
        "ab2_disagreement",
        "sea_rel_l1",
        "sea_cos_defect",
    ]
    write_csv(run_dir / "metrics" / "local_error_diagnostics.csv", diags, diag_fields)
    write_csv(run_dir / "metrics" / "local_error_correlations.csv", corrs, ["hint", "sigma_region", "n", "pearson", "spearman"])
    write_csv(run_dir / "metrics" / "call_counter_audit.csv", calls, ["sample", "method", "intended_calls", "actual_full_calls", "cheap_prefix_calls", "valid", "note"])
    # De-duplicate leakage rows by method.
    seen = {}
    for r in leaks:
        seen.setdefault(r["method"], r)
    write_csv(run_dir / "metrics" / "leakage_audit.csv", list(seen.values()), ["method", "saved_vanilla_velocities_used_for_decision", "saved_vanilla_velocities_used_for_update", "valid_causal", "note"])


def make_figures(run_dir: Path, budget_rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]], traces: list[dict[str, Any]], diags: list[dict[str, Any]], corrs: list[dict[str, Any]], sea_trace: list[dict[str, Any]], sample_grid: dict[tuple[str, str], Path], sample_dirs: list[Path]) -> None:
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "figures"

    # Quality frontier: the required figure includes both PSNR and LPIPS.
    fig, (ax_psnr, ax_lpips) = plt.subplots(1, 2, figsize=(13.5, 5.4), sharex=True)
    style = {
        "uniform": ("#d98b2b", "o"),
        "seacache": ("#111111", "D"),
        "curvature": ("#2878b5", "o"),
        "ab2": ("#2f9d63", "s"),
        "sea_defect": ("#8c5fbf", "^"),
        "hybrid": ("#c43b3b", "P"),
        "default_flux": ("#777777", "*"),
        "offline_saved_velocity_oracle": ("#aaaaaa", "x"),
    }
    for base in sorted({r["base_method"] for r in budget_rows}):
        rs = [r for r in budget_rows if r["base_method"] == base]
        if not rs:
            continue
        rs = sorted(rs, key=lambda r: float(r["actual_full_calls"]))
        c, m = style.get(base, ("#555", "o"))
        xs = [r["actual_full_calls"] for r in rs]
        ax_psnr.plot(xs, [r["psnr_mean"] for r in rs], marker=m, color=c, lw=1.8, label=base)
        ax_lpips.plot(xs, [r["lpips_mean"] for r in rs], marker=m, color=c, lw=1.8, label=base)
    ax_psnr.set_xlabel("actual full transformer / velocity calls")
    ax_psnr.set_ylabel("PSNR to 100-step vanilla (dB)")
    ax_psnr.set_title("PSNR frontier")
    ax_psnr.grid(alpha=0.25)
    ax_lpips.set_xlabel("actual full transformer / velocity calls")
    ax_lpips.set_ylabel("LPIPS to 100-step vanilla (lower better)")
    ax_lpips.set_title("LPIPS frontier")
    ax_lpips.grid(alpha=0.25)
    ax_lpips.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "frontier_quality_vs_calls.png", dpi=160)
    plt.close(fig)

    # LPIPS companion retained as a direct single-metric view.
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for base in sorted({r["base_method"] for r in budget_rows}):
        rs = sorted([r for r in budget_rows if r["base_method"] == base], key=lambda r: float(r["actual_full_calls"]))
        c, m = style.get(base, ("#555", "o"))
        ax.plot([r["actual_full_calls"] for r in rs], [r["lpips_mean"] for r in rs], marker=m, color=c, lw=1.8, label=base)
    ax.set_xlabel("actual full transformer / velocity calls")
    ax.set_ylabel("LPIPS to 100-step vanilla (lower better)")
    ax.set_title("E54 frontier: LPIPS vs actual calls")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "frontier_lpips_vs_calls.png", dpi=160)
    plt.close(fig)

    # Wall-clock vs quality.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for base in sorted({r["base_method"] for r in budget_rows}):
        rs = [r for r in budget_rows if r["base_method"] == base and r["causal"]]
        if not rs:
            continue
        c, m = style.get(base, ("#555", "o"))
        points = [
            (safe_float(r.get("wall_speedup_vs_100_mean", "")), r["psnr_mean"])
            for r in rs
            if safe_float(r.get("wall_speedup_vs_100_mean", "")) > 0
        ]
        if points:
            ax.scatter([x for x, _ in points], [y for _, y in points], label=base, c=c, marker=m, s=48)
    ax.set_xlabel("measured wall-clock speedup vs captured 100-step vanilla")
    ax.set_ylabel("PSNR to 100-step vanilla (dB)")
    ax.set_title("Measured wall-clock frontier: speedup vs quality")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "wallclock_vs_quality.png", dpi=160)
    plt.close(fig)

    # Runtime schedule raster.
    raster = traces[:]
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.22 * len({r["method"] for r in raster}) + 1)))
    methods = sorted({r["method"] for r in raster})
    ymap = {m: i for i, m in enumerate(methods)}
    for r in raster:
        ax.scatter(float(r["k"]), ymap[r["method"]], s=8 + 8 * float(r["jump"]), color=style.get(r["base_method"], ("#555", "o"))[0], alpha=0.75)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_xlabel("accepted anchor step")
    ax.set_title("Runtime jump schedule raster")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "runtime_jump_schedule_raster.png", dpi=170)
    plt.close(fig)

    # Jump histograms.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for base in sorted({r["base_method"] for r in traces if r["base_method"] not in ("seacache",)}):
        vals = [int(r["jump"]) for r in traces if r["base_method"] == base]
        if vals:
            ax.hist(vals, bins=np.arange(1, max(vals + [4]) + 2) - 0.5, alpha=0.45, label=base)
    ax.set_xlabel("accepted jump size (100-step index span)")
    ax.set_ylabel("count")
    ax.set_title("Accepted jump-size distributions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "jump_size_histograms.png", dpi=160)
    plt.close(fig)

    # Local error scatter/correlation.
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    for base in sorted({r["base_method"] for r in diags}):
        rs = [r for r in diags if r["base_method"] == base and math.isfinite(float(r["score"]))]
        if not rs:
            continue
        c, _ = style.get(base, ("#555", "o"))
        ax.scatter([float(r["score"]) for r in rs], [float(r["err_local"]) for r in rs], s=12, alpha=0.35, label=base, c=c)
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_yscale("log")
    ax.set_xlabel("runtime hint score")
    ax.set_ylabel("post-hoc local jump error")
    ax.set_title("Runtime hint score vs actual local error")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "local_error_correlation.png", dpi=160)
    plt.close(fig)

    # SeaCache vs runtime hint trace.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if sea_trace:
        ax.plot([t["step"] for t in sea_trace], [t["accumulated_after"] for t in sea_trace], color="#111", label="SeaCache accumulated rel-L1")
    for method in methods[:5]:
        rs = [r for r in traces if r["method"] == method]
        if rs:
            ax.plot([r["k"] for r in rs], [safe_float(r["score"]) for r in rs], lw=1, alpha=0.75, label=method)
    ax.set_xlabel("step")
    ax.set_ylabel("score")
    ax.set_title("SeaCache accumulated score and runtime hint traces")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "sea_vs_runtime_hint_trace.png", dpi=160)
    plt.close(fig)

    # Sample grid.
    grid_cols = ["vanilla_100step", "default_flux", "uniform", "seacache", "curvature", "ab2", "hybrid"]
    labels = ["100-step vanilla", "default FLUX", "uniform", "SeaCache", "best curvature", "best AB2", "best hybrid"]
    make_image_grid(sample_grid, sample_dirs, grid_cols, labels, fig_dir / "sample_grid_runtime_hints.png")

    # Failure cases: worst causal rows by PSNR.
    worst = sorted([r for r in sample_rows if r["causal"] and r["base_method"] not in ("default_flux",)], key=lambda r: float(r["psnr"]))[: min(8, len(sample_rows))]
    failure_grid = {(r["sample"], f"{r['method']}"): Path(r["image_path"]) for r in worst}
    make_failure_grid(worst, fig_dir / "failure_cases.png")


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


def make_image_grid(sample_grid: dict[tuple[str, str], Path], sample_dirs: list[Path], cols: list[str], labels: list[str], out: Path) -> None:
    thumb = (210, 210)
    row_h = thumb[1] + 28
    left = 130
    W = left + len(cols) * thumb[0]
    H = 30 + min(4, len(sample_dirs)) * row_h
    sheet = Image.new("RGB", (W, H), (248, 246, 239))
    draw = ImageDraw.Draw(sheet)
    for ci, lab in enumerate(labels):
        draw.text((left + ci * thumb[0] + 4, 8), lab[:24], fill=(20, 24, 28))
    for ri, sd in enumerate(sample_dirs[:4]):
        y0 = 30 + ri * row_h
        draw.text((6, y0 + 92), sd.name, fill=(20, 24, 28))
        for ci, key in enumerate(cols):
            p = sample_grid.get((sd.name, key))
            if p and Path(p).exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail(thumb)
                x = left + ci * thumb[0] + (thumb[0] - im.width) // 2
                y = y0 + (thumb[1] - im.height) // 2
                sheet.paste(im, (x, y))
    sheet.save(out)


def make_failure_grid(rows: list[dict[str, Any]], out: Path) -> None:
    if not rows:
        Image.new("RGB", (800, 200), (248, 246, 239)).save(out)
        return
    thumb = (220, 220)
    cols = min(4, len(rows))
    row_h = thumb[1] + 36
    sheet = Image.new("RGB", (cols * thumb[0], math.ceil(len(rows) / cols) * row_h), (248, 246, 239))
    draw = ImageDraw.Draw(sheet)
    for idx, r in enumerate(rows):
        p = Path(r["image_path"])
        if p.exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail(thumb)
            x = (idx % cols) * thumb[0] + (thumb[0] - im.width) // 2
            y = (idx // cols) * row_h
            sheet.paste(im, (x, y))
        lab = f"{r['sample']} {r['method']} PSNR {float(r['psnr']):.1f}"
        draw.text(((idx % cols) * thumb[0] + 4, (idx // cols) * row_h + thumb[1] + 6), lab[:38], fill=(20, 24, 28))
    sheet.save(out)


def best_by_base(rows: list[dict[str, Any]], base: str, target: int | None = None) -> dict[str, Any] | None:
    rs = [
        r for r in rows
        if r["base_method"] == base
        and r["causal"]
        and is_finite_number(r.get("psnr_mean"))
    ]
    if not rs:
        return None
    if target is None:
        return max(rs, key=lambda r: float(r["psnr_mean"]))
    return min(
        rs,
        key=lambda r: (
            abs(float(r["actual_full_calls"]) - target),
            -float(r["psnr_mean"]),
            float(r["actual_full_calls"]),
        ),
    )


def write_report(run_dir: Path, args: argparse.Namespace, default_steps: int, sample_rows: list[dict[str, Any]], budget_rows: list[dict[str, Any]], corr_rows: list[dict[str, Any]], call_rows: list[dict[str, Any]], leakage_rows: list[dict[str, Any]]) -> None:
    best_uniform = best_by_base(budget_rows, "uniform", 50)
    best_sea = best_by_base(budget_rows, "seacache", 50)
    best_curv = best_by_base(budget_rows, "curvature", 50)
    best_ab2 = best_by_base(budget_rows, "ab2", 50)
    best_sea_defect = best_by_base(budget_rows, "sea_defect", 50)
    best_hybrid = best_by_base(budget_rows, "hybrid", 50)
    best_hint = max(
        [x for x in [best_curv, best_ab2, best_sea_defect, best_hybrid] if x],
        key=lambda r: float(r["psnr_mean"]),
        default=None,
    )
    beat_uniform = bool(best_hint and best_uniform and float(best_hint["psnr_mean"]) > float(best_uniform["psnr_mean"]))
    beat_sea = bool(best_hint and best_sea and float(best_hint["psnr_mean"]) > float(best_sea["psnr_mean"]))
    best_corr = max(corr_rows, key=lambda r: abs(float(r["spearman"])) if str(r["spearman"]) != "nan" else -1, default=None)

    def fmt(row: dict[str, Any] | None, key: str = "psnr_mean") -> str:
        return "n/a" if not row else f"{float(row[key]):.2f}"

    def point_label(row: dict[str, Any] | None) -> str:
        if not row:
            return "n/a"
        return f"{float(row['psnr_mean']):.2f} dB @ {int(row['actual_full_calls'])} calls"

    method_summary = table(
        ["method", "causal?", "extra full calls?", "prefix calls", "calls", "speedup", "PSNR", "LPIPS"],
        [
            [
                r["method"],
                str(r["causal"]),
                "no",
                f"{float(r['cheap_prefix_calls_mean']):.1f}",
                r["actual_full_calls"],
                f"{float(r.get('wall_speedup_vs_100_mean', float('nan'))):.2f}x wall / {float(r['speedup_vs_100']):.2f}x calls",
                f"{float(r['psnr_mean']):.2f}",
                f"{float(r['lpips_mean']):.4f}",
            ]
            for r in sorted(budget_rows, key=lambda x: (x["base_method"], float(x["actual_full_calls"])))[:80]
        ],
    )
    targets = [80, 60, 50, 40, 33, 28, 25, 20, 14]
    matched_rows = []
    for target in targets:
        for base in ["uniform", "seacache", "curvature", "ab2", "sea_defect", "hybrid"]:
            b = best_by_base(budget_rows, base, target)
            if b:
                matched_rows.append([target, base, b["method"], b["actual_full_calls"], f"{float(b['psnr_mean']):.2f}", f"{float(b['lpips_mean']):.4f}"])
    matched_table = table(["target calls", "base", "best point", "actual calls", "PSNR", "LPIPS"], matched_rows)
    corr_table = table(["hint", "region", "n", "Pearson", "Spearman"], [[r["hint"], r["sigma_region"], r["n"], f"{float(r['pearson']):.3f}", f"{float(r['spearman']):.3f}"] for r in corr_rows])
    call_table = table(["sample", "method", "intended", "actual full", "prefix", "valid", "note"], [[r["sample"], r["method"], r["intended_calls"], r["actual_full_calls"], r["cheap_prefix_calls"], r["valid"], r["note"]] for r in call_rows[:120]])
    leak_seen = {}
    for r in leakage_rows:
        leak_seen.setdefault(r["method"], r)
    leak_table = table(["method", "decision leakage?", "update leakage?", "causal valid?", "note"], [[r["method"], r["saved_vanilla_velocities_used_for_decision"], r["saved_vanilla_velocities_used_for_update"], r["valid_causal"], r["note"]] for r in leak_seen.values()])
    default_rs = [r for r in sample_rows if r["base_method"] == "default_flux"]
    default_psnr = float(np.mean([float(r["psnr"]) for r in default_rs])) if default_rs else float("nan")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>E54 — Causal runtime jump-size hints for FLUX</title>
<style>
body{{margin:0;background:#f7f1e7;color:#18201f;font:15px/1.55 Georgia,serif}}
main{{max-width:1180px;margin:auto;padding:34px 28px 64px}}
h1{{font:700 38px/1.05 ui-serif,Georgia,serif;margin:0 0 8px}}
h2{{font:700 24px/1.15 ui-serif,Georgia,serif;margin:32px 0 10px;border-top:1px solid #d8cbb8;padding-top:18px}}
h3{{font:700 18px/1.2 ui-serif,Georgia,serif;margin:22px 0 8px}}
.lede{{font-size:18px;max-width:920px}}
.verdict{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}}
.card{{background:#fffaf1;border:1px solid #dfd2bd;border-radius:12px;padding:14px}}
.card b{{display:block;font-size:22px}}
figure{{margin:18px 0;background:#fffaf1;border:1px solid #dfd2bd;border-radius:12px;padding:10px}}
figure img{{max-width:100%;display:block;margin:auto}}
figcaption{{font-size:13px;color:#51483b;margin-top:6px}}
table{{border-collapse:collapse;width:100%;font-size:12px;background:#fffaf1;margin:10px 0 22px}}
th,td{{border:1px solid #d8cbb8;padding:5px 7px;text-align:left;vertical-align:top}}
th{{background:#efe2ce}}
code{{background:#efe2ce;padding:1px 4px;border-radius:4px}}
.warn{{border-left:5px solid #b43b2f;background:#fff6e8;padding:12px 16px;border-radius:8px}}
</style></head><body><main>
<h1>E54 — Causal runtime jump-size hints for FLUX</h1>
<p class="lede">Runtime adaptive jump controllers for FLUX.1-dev at 1024×1024 bf16, compared by <b>actual full transformer calls</b> against uniform jumps, default FLUX, 100-step vanilla, and the official SeaCache accumulated SEA rel-L1 gate.</p>
<div class="verdict">
<div class="card"><span>Best runtime hint beats uniform?</span><b>{beat_uniform}</b><small>best hint {point_label(best_hint)} vs uniform {point_label(best_uniform)}</small></div>
<div class="card"><span>Best runtime hint beats SeaCache?</span><b>{beat_sea}</b><small>best hint {point_label(best_hint)} vs SeaCache {point_label(best_sea)}</small></div>
<div class="card"><span>Best local-error correlation</span><b>{best_corr['hint'] if best_corr else 'n/a'}</b><small>Spearman {float(best_corr['spearman']):.3f} in {best_corr['sigma_region'] if best_corr else 'n/a'}</small></div>
</div>
<div class="warn"><b>Do not overclaim:</b> all runtime controllers are causal; the offline saved-velocity rows, if present, are labelled non-causal diagnostics. A controller is only useful if it improves quality at matched actual calls and produces real call-count reduction.</div>

<h2>Executive Summary</h2>
<p>Default FLUX uses <b>{default_steps}</b> inference steps programmatically; mean default-vs-100-step PSNR is <b>{default_psnr:.2f} dB</b>. At the inspected ~50-call range, the best causal runtime hint is <b>{point_label(best_hint)}</b>. It {'does' if beat_uniform else 'does not'} beat uniform ({point_label(best_uniform)}) and {'does' if beat_sea else 'does not'} beat SeaCache ({point_label(best_sea)}). The dominant option remains SeaCache unless the frontier above shows a hint strictly above it at the same actual full-call count.</p>

<h2>Prior Result Recap</h2>
<p>E53 ran FLUX.1-dev, 1024×1024, bf16 on H100 NVL with a 100-step reference and 4 trajectories. The jump-DP oracle was teacher-forced: offline saved-velocity replay was strong, but causal DP-schedule replay only improved over causal uniform by +2.81 dB and trailed SeaCache by −6.54 dB at matched calls. E54 therefore tests causal runtime error hints rather than another offline DP schedule sweep.</p>

<h2>Numerical Framing</h2>
<p>FLUX flow sampling is an ODE integration problem over sigma. A jump from anchor <code>k</code> to <code>i</code> uses an Euler-like update <code>z_i = z_k + (sigma_i - sigma_k) v_k</code>. Local truncation error scales like field curvature, approximately <code>0.5 h² ||dv/dsigma||</code>. The curvature controller estimates this from consecutive live velocities. The AB2-vs-Euler controller uses a zero-extra-full-call embedded disagreement. The SEA-defect controller uses prefix/modulation features only; prefix calls are counted separately and are not counted as full transformer calls.</p>

<h2>Figures</h2>
<figure><img src="figures/frontier_quality_vs_calls.png"><figcaption>Primary comparison: PSNR and LPIPS vs actual full transformer calls.</figcaption></figure>
<figure><img src="figures/frontier_lpips_vs_calls.png"><figcaption>LPIPS vs actual full transformer calls.</figcaption></figure>
<figure><img src="figures/wallclock_vs_quality.png"><figcaption>Measured wall-clock speedup vs quality; call-count speedup is reported separately in metrics tables.</figcaption></figure>
<figure><img src="figures/runtime_jump_schedule_raster.png"><figcaption>Accepted runtime anchors by method and threshold.</figcaption></figure>
<figure><img src="figures/jump_size_histograms.png"><figcaption>Accepted jump-size distributions.</figcaption></figure>
<figure><img src="figures/local_error_correlation.png"><figcaption>Hint score vs post-hoc local jump error.</figcaption></figure>
<figure><img src="figures/sea_vs_runtime_hint_trace.png"><figcaption>SeaCache accumulated score and runtime hint traces over sigma.</figcaption></figure>
<figure><img src="figures/sample_grid_runtime_hints.png"><figcaption>Columns: 100-step vanilla, default FLUX, uniform, SeaCache, best curvature, best AB2, best hybrid.</figcaption></figure>
<figure><img src="figures/failure_cases.png"><figcaption>Worst drifting causal runtime-jump outputs.</figcaption></figure>

<h2>Methods</h2>
<p><b>Uniform jump:</b> causal live velocities, fixed retained-call target. <b>SeaCache:</b> official accumulated SEA rel-L1 gate; fresh and cached evaluations logged. <b>Velocity-curvature:</b> largest jump with normalized LTE below threshold. <b>Euler-vs-AB2:</b> largest jump with embedded disagreement below threshold; Euler and AB2 update variants are separated. <b>SEA-defect:</b> prefix-only h/SEA drift if available, counted as cheap prefix calls. <b>Hybrid:</b> hand-set sums of curvature, AB2 and optionally SEA defect. <b>Offline saved-velocity replay:</b> non-causal oracle diagnostic only.</p>

<h2>Tables</h2>
<h3>Method Summary</h3>{method_summary}
<h3>Best Point Per Matched Call Range</h3>{matched_table}
<h3>Local-Error Correlations</h3>{corr_table}
<h3>Call-Counter Audit</h3>{call_table}
<h3>Leakage Audit</h3>{leak_table}

<h2>Outputs</h2>
<p>Metrics and traces are saved under <code>metrics/</code>; report summaries under <code>reports/</code>; samples under <code>samples/</code>; figures under <code>figures/</code>.</p>
</main></body></html>"""
    report_path = run_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    report_path.write_text(fsd.embed_local_images(html, report_path), encoding="utf-8")

    summary = {
        "title": "E54 — Causal runtime jump-size hints for FLUX",
        "default_steps": default_steps,
        "best_hint": best_hint,
        "beat_uniform": beat_uniform,
        "beat_seacache": beat_sea,
        "best_correlation": best_corr,
        "run_dir": str(run_dir),
    }
    fsd.write_json(run_dir / "reports" / "summary.json", summary)
    (run_dir / "reports" / "summary.md").write_text(
        f"# E54 — Causal runtime jump-size hints for FLUX\n\n"
        f"- Best hint beats uniform: **{beat_uniform}**\n"
        f"- Best hint beats SeaCache: **{beat_sea}**\n"
        f"- Best local-error correlation: **{best_corr['hint'] if best_corr else 'n/a'}**\n"
        f"- Default FLUX steps: **{default_steps}**\n"
        f"- Report: `{run_dir / 'report.html'}`\n",
        encoding="utf-8",
    )


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{head}{body}</table>"


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def write_manifest(run_dir: Path, args: argparse.Namespace) -> None:
    files = [str(p.relative_to(run_dir)) for p in sorted(run_dir.rglob("*")) if p.is_file()]
    fsd.write_json(run_dir / "artifacts_manifest.json", {
        "experiment": "E54_flux_runtime_jump_hints",
        "created": dt.datetime.now().isoformat(),
        "args": vars(args),
        "files": files,
    })


def parse_csv_list(raw: str, typ: Any) -> list[Any]:
    return [typ(x) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E54 causal runtime jump-size hints for FLUX")
    ap.add_argument("--model-id", default=fsd.DEFAULT_MODEL_ID)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "h100"))
    ap.add_argument("--trajectory-root", default=str(REPO_ROOT / "outputs" / "flux_runtime_jump_hints" / "trajectories"))
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--max-sequence-length", type=int, default=512)
    ap.add_argument("--max-jump", type=int, default=4)
    ap.add_argument("--uniform-retained", type=lambda s: parse_csv_list(s, int), default=[80, 50, 33, 28, 25, 20, 14, 10])
    ap.add_argument("--curvature-thresholds", type=lambda s: parse_csv_list(s, float), default=[0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002])
    ap.add_argument("--ab2-thresholds", type=lambda s: parse_csv_list(s, float), default=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ap.add_argument("--sea-thresholds", type=lambda s: parse_csv_list(s, float), default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--hybrid-thresholds", type=lambda s: parse_csv_list(s, float), default=[0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--seacache-thresholds", type=lambda s: parse_csv_list(s, float), default=[0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    ap.add_argument("--oracle-retained", type=lambda s: parse_csv_list(s, int), default=[50, 20])
    ap.add_argument("--force-capture", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-methods", type=int, default=8)
    args = ap.parse_args()
    if args.smoke:
        args.num_samples = min(args.num_samples, 2)
        args.seacache_thresholds = args.seacache_thresholds[:3]
        args.uniform_retained = [50, 20]
        args.curvature_thresholds = args.curvature_thresholds[:2]
        args.ab2_thresholds = args.ab2_thresholds[:2]
        args.sea_thresholds = args.sea_thresholds[:1]
        args.hybrid_thresholds = args.hybrid_thresholds[:1]
        args.oracle_retained = [50]
    return args


if __name__ == "__main__":
    run(parse_args())
