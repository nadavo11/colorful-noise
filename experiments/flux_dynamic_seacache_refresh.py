#!/usr/bin/env python3
"""E55 distilled SEA-defect for dynamic SeaCache refresh.

This run treats the expensive SEA-defect signal as an offline teacher only.
The online methods remain prefix-free and causal: they can inspect the current
SeaCache cheap feature path, but they may not run extra full transformer calls,
use future vanilla features, or use saved teacher labels online.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
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

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SLUG = "flux_seadefect_distilled_seacache"
N = 100
EPS = 1e-8
TEACHER_HORIZONS = (1, 2, 3, 4, 6, 8)
FEATURE_NAMES = [
    "raw_rel_l1",
    "prev_raw_rel_l1",
    "accumulated_before",
    "accumulated_after",
    "normalized_accumulated_before",
    "normalized_accumulated_after",
    "age_since_refresh",
    "last_refresh_step",
    "step_frac",
    "sigma",
    "sigma_sq",
    "raw_x_step",
    "delta_raw",
    "raw_h_norm_mean",
    "raw_h_norm_std",
    "sea_h_norm_mean",
    "sea_h_norm_std",
]


@dataclass(frozen=True)
class PolicySpec:
    method: str
    threshold: float
    sea_threshold: float | None = None
    probe_band: float = 0.2
    probe_teacher_threshold: float | None = None

    @property
    def key(self) -> str:
        bits = [self.method, f"th{self.threshold:g}"]
        if self.sea_threshold is not None:
            bits.append(f"sea{self.sea_threshold:g}")
        if self.method == "rare_probe":
            bits.append(f"band{self.probe_band:g}")
            bits.append(f"teacher{self.probe_teacher_threshold:g}")
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
        "metrics",
        "metrics/runtime_traces",
        "reports",
        "samples",
        "schedules",
        "trajectories",
    ]:
        fsd.ensure_dir(run_dir / sub)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    fsd.ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    write_csv(path, rows, list(rows[0].keys()))


def write_parquet_or_csv(path_base: Path, rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        write_csv(path_base.with_suffix(".csv"), rows, fields)
        return str(path_base.with_suffix(".csv"))
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist([{k: row.get(k) for k in fields} for row in rows])
        out = path_base.with_suffix(".parquet")
        pq.write_table(table, out)
        return str(out)
    except Exception:
        out = path_base.with_suffix(".csv")
        write_csv(out, rows, fields)
        return str(out)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_default_steps(pipe: Any) -> int:
    import inspect

    sig = inspect.signature(pipe.__call__)
    return int(sig.parameters["num_inference_steps"].default)


def prep_inputs(
    pipe: Any,
    prompt: str,
    seed: int,
    steps: int,
    height: int,
    width: int,
    guidance: float,
    max_seq_len: int,
    device: str,
) -> dict[str, Any]:
    pe, ppe, text_ids, latents, image_ids, timesteps, guidance_tensor = fsd.prepare_flux_inputs(
        pipe,
        prompt,
        seed,
        steps,
        height,
        width,
        guidance,
        device,
        max_seq_len,
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


def rel_l2_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    return float(torch.linalg.vector_norm(af - bf) / (torch.linalg.vector_norm(bf) + EPS))


def logistic_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def rank_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(np.int64)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_ranks = ranks[y == 1].sum()
    return float((pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg))


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(y_true) == 0 or int(np.sum(y_true)) == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = y_true[order].astype(np.int64)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int(y.sum()))
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def calibration_bins(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for bi in range(bins):
        lo = edges[bi]
        hi = edges[bi + 1]
        mask = (scores >= lo) & (scores <= hi if bi == bins - 1 else scores < hi)
        if not mask.any():
            continue
        rows.append({
            "bin_index": bi,
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "count": int(mask.sum()),
            "score_mean": float(scores[mask].mean()),
            "empirical_positive_rate": float(y_true[mask].mean()),
        })
    return rows


def sigma_region(step_index: int) -> str:
    if step_index < N / 3:
        return "early_high_noise"
    if step_index < 2 * N / 3:
        return "mid"
    return "late_low_noise"


def build_transformer_context(
    tr: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None,
    joint_attention_kwargs: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    hs = tr.x_embedder(hidden_states)
    ts = timestep.to(hs.dtype) * 1000
    guidance_in = guidance.to(hs.dtype) * 1000 if guidance is not None else None
    temb = tr.time_text_embed(ts, pooled_projections) if guidance_in is None else tr.time_text_embed(ts, guidance_in, pooled_projections)
    enc = tr.context_embedder(encoder_hidden_states)
    txt = txt_ids[0] if txt_ids is not None and txt_ids.ndim == 3 else txt_ids
    img = img_ids[0] if img_ids is not None and img_ids.ndim == 3 else img_ids
    image_rotary_emb = tr.pos_embed(torch.cat((txt, img), dim=0)) if txt is not None and img is not None else None
    return hs, enc, temb, image_rotary_emb


def run_transformer_stack(
    tr: Any,
    hs: torch.Tensor,
    enc: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: torch.Tensor | None,
    joint_attention_kwargs: dict[str, Any] | None,
    controlnet_block_samples: Any = None,
    controlnet_single_block_samples: Any = None,
    controlnet_blocks_repeat: bool = False,
) -> torch.Tensor:
    for index_block, block in enumerate(tr.transformer_blocks):
        enc, hs = block(
            hidden_states=hs,
            encoder_hidden_states=enc,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        if controlnet_block_samples is not None:
            interval_control = int(np.ceil(len(tr.transformer_blocks) / len(controlnet_block_samples)))
            hs = hs + (
                controlnet_block_samples[index_block % len(controlnet_block_samples)]
                if controlnet_blocks_repeat
                else controlnet_block_samples[index_block // interval_control]
            )
    for index_block, block in enumerate(tr.single_transformer_blocks):
        enc, hs = block(
            hidden_states=hs,
            encoder_hidden_states=enc,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        if controlnet_single_block_samples is not None:
            interval_control = int(np.ceil(len(tr.single_transformer_blocks) / len(controlnet_single_block_samples)))
            hs = hs + controlnet_single_block_samples[index_block // interval_control]
    return hs


def decision_feature_stats(tensor: torch.Tensor) -> tuple[float, float]:
    norms = tensor.float().norm(dim=-1)
    return float(norms.mean().item()), float(norms.std(unbiased=False).item())


def current_step_decision_feature(
    tr: Any,
    hs: torch.Tensor,
    temb: torch.Tensor,
    img_ids: torch.Tensor,
    step_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    modulated, *_ = tr.transformer_blocks[0].norm1(hs, emb=temb)
    decision_feature = modulated
    if step_index not in (0, N - 1):
        img = img_ids[0] if img_ids is not None and img_ids.ndim == 3 else img_ids
        grid = modulated.reshape(modulated.shape[0], int(img[:, 1].max().item() + 1), int(img[:, 2].max().item() + 1), modulated.shape[-1])
        a, b = fsd.ab_from_scheduler(tr.scheduler, step_index)
        decision_feature = fsd.apply_sea_from_ab(grid, a, b, dims=(-2, -3), norm_mode="mean").reshape(modulated.shape[0], -1, modulated.shape[-1])
    return modulated, decision_feature


def sea_feature_at_step(
    tr: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None,
    scheduler: Any,
    step_index: int,
) -> torch.Tensor:
    _, h = fsd.flux_h_decision_tensor(
        tr,
        hidden_states,
        timestep,
        encoder_hidden_states,
        pooled_projections,
        img_ids,
        txt_ids,
        guidance,
        scheduler,
        step_index,
        sea_filter=step_index not in (0, N - 1),
    )
    return h.float()


def build_feature_values(
    raw_rel_l1: float,
    prev_raw_rel_l1: float,
    accumulated_before: float,
    accumulated_after: float,
    age_since_refresh: int,
    last_refresh_step: int,
    step_index: int,
    sigma_value: float,
    fixed_threshold: float,
    raw_h_norm_mean: float,
    raw_h_norm_std: float,
    sea_h_norm_mean: float,
    sea_h_norm_std: float,
) -> dict[str, float]:
    step_frac = float(step_index) / float(max(1, N - 1))
    return {
        "feature_raw_rel_l1": float(raw_rel_l1),
        "feature_prev_raw_rel_l1": float(prev_raw_rel_l1),
        "feature_accumulated_before": float(accumulated_before),
        "feature_accumulated_after": float(accumulated_after),
        "feature_normalized_accumulated_before": float(accumulated_before / max(EPS, fixed_threshold)),
        "feature_normalized_accumulated_after": float(accumulated_after / max(EPS, fixed_threshold)),
        "feature_age_since_refresh": float(age_since_refresh),
        "feature_last_refresh_step": float(last_refresh_step),
        "feature_step_frac": float(step_frac),
        "feature_sigma": float(sigma_value),
        "feature_sigma_sq": float(sigma_value * sigma_value),
        "feature_raw_x_step": float(raw_rel_l1 * step_frac),
        "feature_delta_raw": float(raw_rel_l1 - prev_raw_rel_l1),
        "feature_raw_h_norm_mean": float(raw_h_norm_mean),
        "feature_raw_h_norm_std": float(raw_h_norm_std),
        "feature_sea_h_norm_mean": float(sea_h_norm_mean),
        "feature_sea_h_norm_std": float(sea_h_norm_std),
    }


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row[f"feature_{name}"]) for name in FEATURE_NAMES] for row in rows], dtype=np.float64)


def split_protocol(total: int) -> tuple[list[str], str]:
    if total >= 8:
        n_train = max(1, int(round(total * 0.6)))
        n_val = max(1, int(round(total * 0.2)))
        if n_train + n_val >= total:
            n_val = max(1, total - n_train - 1)
        n_eval = total - n_train - n_val
        if n_eval <= 0:
            n_eval = 1
            n_train = max(1, n_train - 1)
        labels = ["train"] * n_train + ["val"] * n_val + ["eval"] * n_eval
        return labels[:total], "trajectory_split_60_20_20"
    if total == 1:
        return ["train"], "single_trajectory_no_holdout"
    if total == 2:
        return ["train", "eval"], "leave_one_trajectory_out_fallback"
    if total == 3:
        return ["train", "val", "eval"], "minimal_train_val_eval_fallback"
    return ["train", "train", "val", "eval"][:total], "minimal_train_val_eval_fallback"


def install_teacher_capture_forward(
    pipe: Any,
    sample_name: str,
    split: str,
    prompt: str,
    seed: int,
    sea_threshold: float,
    num_steps: int,
) -> dict[str, Any]:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    tr = pipe.transformer
    orig_forward = tr.forward
    state: dict[str, Any] = {
        "rows": [],
        "prev_raw_rel_l1": 0.0,
        "sim_acc": 0.0,
        "age_since_refresh": 0,
        "last_refresh_step": 0,
        "previous_modulated_input": None,
        "previous_residual": None,
        "fresh_evals": 0,
        "hook_forward_calls": 0,
    }
    tr.cnt = 0
    tr.num_steps = int(num_steps)
    tr.scheduler = pipe.scheduler

    def wrapped_forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        state["hook_forward_calls"] += 1
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(tr, lora_scale)

        hs, enc, temb, image_rotary_emb = build_transformer_context(
            tr,
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance,
            joint_attention_kwargs,
        )
        step_index = int(tr.cnt)
        sigma_value = float(tr.scheduler.sigmas[step_index]) if hasattr(tr.scheduler, "sigmas") else float("nan")
        timestep_value = float(timestep.flatten()[0].detach().cpu())
        raw_feature, decision_feature = current_step_decision_feature(tr, hs, temb, img_ids, step_index)
        raw_h_norm_mean, raw_h_norm_std = decision_feature_stats(raw_feature)
        sea_h_norm_mean, sea_h_norm_std = decision_feature_stats(decision_feature)

        raw_rel_l1 = 0.0
        if step_index not in (0, num_steps - 1) and state["previous_modulated_input"] is not None:
            raw_rel_l1 = fsd.rel_l1(decision_feature, state["previous_modulated_input"])
        acc_before = float(state["sim_acc"])
        acc_after = float(acc_before + raw_rel_l1)
        age_before = int(state["age_since_refresh"])
        last_refresh_step = int(state["last_refresh_step"])
        fixed_refresh = (
            step_index in (0, num_steps - 1)
            or state["previous_residual"] is None
            or not (acc_after < sea_threshold)
        )
        feature_values = build_feature_values(
            raw_rel_l1,
            float(state["prev_raw_rel_l1"]),
            acc_before,
            acc_after,
            age_before,
            last_refresh_step,
            step_index,
            sigma_value,
            sea_threshold,
            raw_h_norm_mean,
            raw_h_norm_std,
            sea_h_norm_mean,
            sea_h_norm_std,
        )

        ori_hs = hs
        hs_fresh = run_transformer_stack(
            tr,
            hs,
            enc,
            temb,
            image_rotary_emb,
            joint_attention_kwargs,
            controlnet_block_samples,
            controlnet_single_block_samples,
            controlnet_blocks_repeat,
        )
        fresh_residual = (hs_fresh - ori_hs).detach()
        fresh_norm = tr.norm_out(hs_fresh, temb)
        fresh_output = tr.proj_out(fresh_norm)

        cache_error = float("nan")
        teacher_defect = float("nan")
        if state["previous_residual"] is not None and step_index < num_steps - 1:
            hs_cached = ori_hs + state["previous_residual"]
            cached_norm = tr.norm_out(hs_cached, temb)
            cached_output = tr.proj_out(cached_norm)
            ds = float(tr.scheduler.sigmas[step_index + 1] - tr.scheduler.sigmas[step_index])
            z_next_fresh = hidden_states.float() + ds * fresh_output.float()
            z_next_cached = hidden_states.float() + ds * cached_output.float()
            cache_error = rel_l2_torch(z_next_cached, z_next_fresh)
            next_timestep = tr.scheduler.timesteps[step_index + 1].expand(hidden_states.shape[0]).to(hidden_states.dtype) / 1000
            h_next_fresh = sea_feature_at_step(
                tr,
                z_next_fresh.to(hidden_states.dtype),
                encoder_hidden_states,
                pooled_projections,
                next_timestep,
                img_ids,
                txt_ids,
                guidance,
                tr.scheduler,
                step_index + 1,
            )
            h_next_cached = sea_feature_at_step(
                tr,
                z_next_cached.to(hidden_states.dtype),
                encoder_hidden_states,
                pooled_projections,
                next_timestep,
                img_ids,
                txt_ids,
                guidance,
                tr.scheduler,
                step_index + 1,
            )
            teacher_defect = fsd.rel_l1(h_next_cached, h_next_fresh)

        row = {
            "sample": sample_name,
            "prompt": prompt,
            "seed": seed,
            "split": split,
            "step": step_index,
            "sigma": sigma_value,
            "scheduler_timestep": timestep_value,
            "sigma_region": sigma_region(step_index),
            "raw_rel_l1": float(raw_rel_l1),
            "prev_raw_rel_l1": float(state["prev_raw_rel_l1"]),
            "accumulated_before": acc_before,
            "accumulated_after": acc_after,
            "normalized_accumulated_before": acc_before / max(EPS, sea_threshold),
            "normalized_accumulated_after": acc_after / max(EPS, sea_threshold),
            "age_since_refresh": age_before,
            "last_refresh_step": last_refresh_step,
            "steps_until_end": int(num_steps - 1 - step_index),
            "fixed_threshold": float(sea_threshold),
            "fixed_refresh": bool(fixed_refresh),
            "teacher_defect": float(teacher_defect),
            "cache_error": float(cache_error),
            "raw_h_norm_mean": raw_h_norm_mean,
            "raw_h_norm_std": raw_h_norm_std,
            "sea_h_norm_mean": sea_h_norm_mean,
            "sea_h_norm_std": sea_h_norm_std,
        }
        row.update(feature_values)
        state["rows"].append(row)

        state["previous_modulated_input"] = decision_feature.detach()
        state["prev_raw_rel_l1"] = float(raw_rel_l1)
        state["sim_acc"] = 0.0 if fixed_refresh else acc_after
        state["age_since_refresh"] = 0 if fixed_refresh else age_before + 1
        state["last_refresh_step"] = step_index if fixed_refresh else last_refresh_step
        state["previous_residual"] = fresh_residual
        state["fresh_evals"] += 1
        tr.cnt += 1
        if tr.cnt == tr.num_steps:
            tr.cnt = 0

        if USE_PEFT_BACKEND:
            unscale_lora_layers(tr, lora_scale)
        return (fresh_output,) if not return_dict else Transformer2DModelOutput(sample=fresh_output)

    tr.forward = wrapped_forward
    state["restore"] = lambda: setattr(tr, "forward", orig_forward)
    return state


def enrich_teacher_rows(
    teacher_rows: list[dict[str, Any]],
    sample_dirs: list[Path],
    teacher_safe_threshold: float,
) -> None:
    sample_to_dir = {sd.name: sd for sd in sample_dirs}
    rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in teacher_rows:
        rows_by_sample.setdefault(str(row["sample"]), []).append(row)
    for sample_name, rows in rows_by_sample.items():
        rows.sort(key=lambda r: int(r["step"]))
        sd = sample_to_dir[sample_name]
        meta = fsd.read_json(sd / "metadata.json")
        sigmas = [float(x) for x in meta["sigmas"]]
        latents = [fsd.load_step_tensor(sd / "latents", i) for i in range(N + 1)]
        velocities = [fsd.load_step_tensor(sd / "velocities", i) for i in range(N)]
        h_values = [fsd.load_step_tensor(sd / "h_values", i) for i in range(N)]
        for row in rows:
            step = int(row["step"])
            max_safe = 0
            for m in TEACHER_HORIZONS:
                sea_key = f"teacher_sea_defect_m{m}"
                end_key = f"teacher_endpoint_latent_error_m{m}"
                safe_key = f"teacher_safe_cache_m{m}"
                if step >= N - 1 or step + m > N:
                    row[sea_key] = float("nan")
                    row[end_key] = float("nan")
                    row[safe_key] = ""
                    continue
                future_idx = min(N - 1, step + m - 1)
                row[sea_key] = float(fsd.rel_l1(h_values[step], h_values[future_idx]))
                ds = float(sigmas[step + m] - sigmas[step]) if step + m < len(sigmas) else float(0.0 - sigmas[step])
                pred = latents[step] + ds * velocities[step]
                row[end_key] = float(fsd.latent_metrics(pred, latents[step + m])["latent_rel_l2"])
                is_safe = int(float(row[sea_key]) < teacher_safe_threshold)
                row[safe_key] = is_safe
                if is_safe:
                    max_safe = m
            row["teacher_max_safe_m"] = max_safe


def choose_label_threshold(train_rows: list[dict[str, Any]], label_quantile: float) -> float:
    values = [float(r["cache_error"]) for r in train_rows if math.isfinite(float(r["cache_error"]))]
    threshold = float(np.quantile(values, label_quantile)) if values else 1e-6
    if threshold <= 0:
        threshold = max(1e-6, float(np.mean(values))) if values else 1e-6
    return threshold


def build_labels(rows: list[dict[str, Any]], threshold: float) -> None:
    for row in rows:
        cache_error = float(row["cache_error"]) if math.isfinite(float(row["cache_error"])) else float("nan")
        row["refresh_needed"] = int(math.isfinite(cache_error) and cache_error >= threshold)


def fit_predictor_suite(
    rows: list[dict[str, Any]],
    split_strategy: str,
    label_quantile: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], float]:
    train_rows = [r for r in rows if r["split"] == "train" and math.isfinite(float(r["cache_error"])) and int(r["step"]) not in (0, N - 1)]
    val_rows = [r for r in rows if r["split"] == "val" and math.isfinite(float(r["cache_error"])) and int(r["step"]) not in (0, N - 1)]
    eval_rows = [r for r in rows if r["split"] == "eval" and math.isfinite(float(r["cache_error"])) and int(r["step"]) not in (0, N - 1)]
    if not train_rows:
        raise RuntimeError("need at least one train trajectory to fit predictors")
    label_threshold = choose_label_threshold(train_rows, label_quantile)
    build_labels(rows, label_threshold)
    train_rows = [r for r in rows if r["split"] == "train" and math.isfinite(float(r["cache_error"])) and int(r["step"]) not in (0, N - 1)]
    val_rows = [r for r in rows if r["split"] == "val" and math.isfinite(float(r["cache_error"])) and int(r["step"]) not in (0, N - 1)]
    eval_rows = [r for r in rows if r["split"] == "eval" and math.isfinite(float(r["cache_error"])) and int(r["step"]) not in (0, N - 1)]
    if not eval_rows:
        eval_rows = train_rows

    x_train = feature_matrix(train_rows)
    y_train = np.asarray([int(r["refresh_needed"]) for r in train_rows], dtype=np.int64)

    bundles: dict[str, dict[str, Any]] = {}

    feature_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    raw_idx = feature_idx["raw_rel_l1"]
    acc_idx = feature_idx["normalized_accumulated_after"]
    sigma_idx = feature_idx["sigma"]

    raw_train = x_train[:, raw_idx]
    acc_train = x_train[:, acc_idx]
    sigma_train = x_train[:, sigma_idx]
    heuristic = 0.55 * raw_train + 0.35 * acc_train + 0.10 * (1.0 - np.clip(sigma_train, 0.0, 1.0))
    h_lo = float(np.min(heuristic))
    h_hi = float(np.max(heuristic))

    def heuristic_score(x: np.ndarray) -> np.ndarray:
        score = 0.55 * x[:, raw_idx] + 0.35 * x[:, acc_idx] + 0.10 * (1.0 - np.clip(x[:, sigma_idx], 0.0, 1.0))
        if h_hi <= h_lo + EPS:
            return np.full(len(x), 0.5, dtype=np.float64)
        return np.clip((score - h_lo) / (h_hi - h_lo), 0.0, 1.0)

    bundles["dynamic_threshold_heuristic"] = {
        "model_name": "dynamic_threshold_heuristic",
        "score_fn": heuristic_score,
        "fit_note": "closed-form risk heuristic using raw_rel_l1, normalized accumulator, sigma",
        "runtime_family": "heuristic",
    }

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("E55 predictor suite requires scikit-learn at runtime") from exc

    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    logistic.fit(x_train, y_train)
    bundles["logistic_regression"] = {
        "model_name": "logistic_regression",
        "predictor": logistic,
        "score_fn": lambda x, model=logistic: model.predict_proba(x)[:, 1],
        "runtime_family": "risk_model",
    }

    ridge = make_pipeline(StandardScaler(), RidgeClassifier(class_weight="balanced"))
    ridge.fit(x_train, y_train)
    bundles["ridge_classifier"] = {
        "model_name": "ridge_classifier",
        "predictor": ridge,
        "score_fn": lambda x, model=ridge: logistic_sigmoid(np.asarray(model.decision_function(x), dtype=np.float64)),
        "runtime_family": "risk_model",
    }

    gbt = GradientBoostingClassifier(random_state=0)
    gbt.fit(x_train, y_train)
    bundles["gradient_boosted_tree"] = {
        "model_name": "gradient_boosted_tree",
        "predictor": gbt,
        "score_fn": lambda x, model=gbt: model.predict_proba(x)[:, 1],
        "runtime_family": "risk_model",
    }

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_train, y_train)
    bundles["isotonic_calibration"] = {
        "model_name": "isotonic_calibration",
        "predictor": iso,
        "score_fn": lambda x, model=iso: np.clip(model.predict(x[:, raw_idx]), 0.0, 1.0),
        "runtime_family": "risk_model",
    }

    selection_rows = val_rows if val_rows else eval_rows
    should_try_mlp = True
    simple_auc = []
    for name in ["logistic_regression", "ridge_classifier", "gradient_boosted_tree", "isotonic_calibration"]:
        scores = bundles[name]["score_fn"](feature_matrix(selection_rows))
        labels = np.asarray([int(r["refresh_needed"]) for r in selection_rows], dtype=np.int64)
        auc = rank_auc(labels, scores)
        if math.isfinite(auc):
            simple_auc.append(auc)
    if simple_auc and max(simple_auc) >= 0.72:
        should_try_mlp = False
    if should_try_mlp and len(train_rows) >= 64:
        mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), max_iter=800, random_state=0))
        mlp.fit(x_train, y_train)
        bundles["tiny_mlp_optional"] = {
            "model_name": "tiny_mlp_optional",
            "predictor": mlp,
            "score_fn": lambda x, model=mlp: model.predict_proba(x)[:, 1],
            "runtime_family": "risk_model",
        }

    training_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    best_name = "logistic_regression"
    best_auc = -float("inf")
    selection_split = "val" if val_rows else "eval"
    selected_summary: dict[str, Any] | None = None

    for model_name, bundle in bundles.items():
        for split_name, split_rows in [("train", train_rows), ("val", val_rows), ("eval", eval_rows)]:
            if not split_rows:
                continue
            x = feature_matrix(split_rows)
            y = np.asarray([int(r["refresh_needed"]) for r in split_rows], dtype=np.int64)
            scores = np.asarray(bundle["score_fn"](x), dtype=np.float64)
            auc = rank_auc(y, scores)
            pr = pr_auc(y, scores)
            brier = float(np.mean((scores - y) ** 2))
            bins = calibration_bins(y, scores, bins=10)
            ece = 0.0
            for b in bins:
                ece += abs(float(b["score_mean"]) - float(b["empirical_positive_rate"])) * (float(b["count"]) / max(1, len(y)))
            training_rows.append({
                "model": model_name,
                "split": split_name,
                "split_strategy": split_strategy,
                "count": len(split_rows),
                "positives": int(y.sum()),
                "roc_auc": auc,
                "pr_auc": pr,
                "brier": brier,
                "ece": ece,
                "label_threshold": label_threshold,
                "selected_for_runtime": False,
            })
            if model_name == "logistic_regression":
                for row, score in zip(split_rows, scores):
                    row["logistic_score"] = float(score)
            if split_name == selection_split and bundle["runtime_family"] == "risk_model" and math.isfinite(auc):
                if auc > best_auc:
                    best_auc = auc
                    best_name = model_name
                    selected_summary = training_rows[-1]

    best_bundle = bundles[best_name]
    for row in training_rows:
        if row["model"] == best_name:
            row["selected_for_runtime"] = True

    for split_name, split_rows in [("train", train_rows), ("val", val_rows), ("eval", eval_rows)]:
        if not split_rows:
            continue
        x = feature_matrix(split_rows)
        y = np.asarray([int(r["refresh_needed"]) for r in split_rows], dtype=np.int64)
        scores = np.asarray(best_bundle["score_fn"](x), dtype=np.float64)
        for row, score in zip(split_rows, scores):
            row["predictor_score"] = float(score)
        bins = calibration_bins(y, scores, bins=10)
        ece = 0.0
        for b in bins:
            ece += abs(float(b["score_mean"]) - float(b["empirical_positive_rate"])) * (float(b["count"]) / max(1, len(y)))
            calibration_rows.append({
                "model": best_name,
                "split": split_name,
                "metric": "calibration_bin",
                **b,
                "roc_auc": rank_auc(y, scores),
                "pr_auc": pr_auc(y, scores),
                "brier": float(np.mean((scores - y) ** 2)),
                "ece": ece,
                "label_threshold": label_threshold,
            })
        calibration_rows.append({
            "model": best_name,
            "split": split_name,
            "metric": "summary",
            "bin_index": "",
            "bin_lo": "",
            "bin_hi": "",
            "count": len(split_rows),
            "score_mean": float(scores.mean()) if len(scores) else float("nan"),
            "empirical_positive_rate": float(y.mean()) if len(y) else float("nan"),
            "roc_auc": rank_auc(y, scores),
            "pr_auc": pr_auc(y, scores),
            "brier": float(np.mean((scores - y) ** 2)),
            "ece": ece,
            "label_threshold": label_threshold,
        })

    best_bundle = dict(best_bundle)
    best_bundle["selected_model_name"] = best_name
    best_bundle["label_threshold"] = label_threshold
    best_bundle["split_strategy"] = split_strategy
    best_bundle["selection_split"] = selection_split
    best_bundle["selected_summary"] = selected_summary
    best_bundle["feature_names"] = FEATURE_NAMES
    return best_bundle, training_rows, calibration_rows, label_threshold


def score_row(bundle: dict[str, Any], row: dict[str, Any]) -> float:
    x = np.asarray([[float(row[f"feature_{name}"]) for name in FEATURE_NAMES]], dtype=np.float64)
    return float(np.asarray(bundle["score_fn"](x), dtype=np.float64)[0])


def predictor_feature_importance(bundle: dict[str, Any]) -> list[tuple[str, float]]:
    predictor = bundle.get("predictor")
    if predictor is None:
        return [("raw_rel_l1", 1.0), ("normalized_accumulated_after", 0.7), ("sigma", 0.2)]
    if hasattr(predictor, "named_steps"):
        clf = list(predictor.named_steps.values())[-1]
        if hasattr(clf, "coef_"):
            coef = np.abs(np.asarray(clf.coef_)).reshape(-1)
            return sorted(zip(FEATURE_NAMES, coef.tolist()), key=lambda x: x[1], reverse=True)[:10]
    if hasattr(predictor, "feature_importances_"):
        imp = np.asarray(predictor.feature_importances_).reshape(-1)
        return sorted(zip(FEATURE_NAMES, imp.tolist()), key=lambda x: x[1], reverse=True)[:10]
    return [("raw_rel_l1", 1.0), ("normalized_accumulated_after", 0.7), ("sigma", 0.2)]


def policy_specs(args: argparse.Namespace) -> list[PolicySpec]:
    specs: list[PolicySpec] = []
    for th in args.dynamic_risk_thresholds:
        specs.append(PolicySpec("dynamic_risk_gate", th))
    for sea_th in args.seacache_thresholds[: min(3, len(args.seacache_thresholds))]:
        for th in args.dynamic_threshold_thresholds:
            specs.append(PolicySpec("dynamic_threshold_gate", th, sea_threshold=sea_th))
        for th in args.override_thresholds:
            specs.append(PolicySpec("conservative_override", th, sea_threshold=sea_th))
    if args.run_rare_probe:
        for sea_th in args.seacache_thresholds[:2]:
            specs.append(PolicySpec("rare_probe", args.rare_probe_dynamic_threshold, sea_threshold=sea_th, probe_band=args.rare_probe_band, probe_teacher_threshold=args.rare_probe_teacher_threshold))
    return specs


def install_policy_forward(
    pipe: Any,
    bundle: dict[str, Any],
    spec: PolicySpec,
    num_steps: int,
) -> dict[str, Any]:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    tr = pipe.transformer
    orig_forward = tr.forward
    state: dict[str, Any] = {
        "fresh_evals": 0,
        "cached_evals": 0,
        "cheap_prefix_calls": 0,
        "expensive_probe_calls": 0,
        "hook_forward_calls": 0,
        "step_traces": [],
        "previous_modulated_input": None,
        "previous_residual": None,
        "prev_raw_rel_l1": 0.0,
        "accumulated_since_refresh": 0.0,
        "age_since_refresh": 0,
        "last_refresh_step": 0,
    }
    tr.cnt = 0
    tr.num_steps = int(num_steps)
    tr.scheduler = pipe.scheduler

    def wrapped_forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        state["hook_forward_calls"] += 1
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(tr, lora_scale)

        hs, enc, temb, image_rotary_emb = build_transformer_context(
            tr,
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance,
            joint_attention_kwargs,
        )
        step_index = int(tr.cnt)
        sigma_value = float(tr.scheduler.sigmas[step_index]) if hasattr(tr.scheduler, "sigmas") else float("nan")
        timestep_value = float(timestep.flatten()[0].detach().cpu())
        raw_feature, decision_feature = current_step_decision_feature(tr, hs, temb, img_ids, step_index)
        raw_h_norm_mean, raw_h_norm_std = decision_feature_stats(raw_feature)
        sea_h_norm_mean, sea_h_norm_std = decision_feature_stats(decision_feature)

        raw_rel_l1 = 0.0
        if step_index not in (0, num_steps - 1) and state["previous_modulated_input"] is not None:
            raw_rel_l1 = fsd.rel_l1(decision_feature, state["previous_modulated_input"])
        acc_before = float(state["accumulated_since_refresh"])
        acc_after = float(acc_before + raw_rel_l1)
        age_before = int(state["age_since_refresh"])
        last_refresh_step = int(state["last_refresh_step"])
        sea_threshold = float(spec.sea_threshold if spec.sea_threshold is not None else 0.15)
        feature_row = build_feature_values(
            raw_rel_l1,
            float(state["prev_raw_rel_l1"]),
            acc_before,
            acc_after,
            age_before,
            last_refresh_step,
            step_index,
            sigma_value,
            sea_threshold,
            raw_h_norm_mean,
            raw_h_norm_std,
            sea_h_norm_mean,
            sea_h_norm_std,
        )
        state["cheap_prefix_calls"] += 1
        risk = score_row(bundle, feature_row)

        refresh = step_index in (0, num_steps - 1) or state["previous_residual"] is None
        dynamic_threshold = float("nan")
        fixed_would_refresh = step_index in (0, num_steps - 1) or state["previous_residual"] is None or not (acc_after < sea_threshold)

        if not refresh:
            if spec.method == "dynamic_risk_gate":
                refresh = risk >= spec.threshold
            elif spec.method == "dynamic_threshold_gate":
                dynamic_threshold = sea_threshold * np.clip(1.6 - spec.threshold * risk, 0.35, 1.9)
                refresh = not (acc_after < dynamic_threshold)
            elif spec.method == "conservative_override":
                refresh = fixed_would_refresh or (risk >= spec.threshold)
            elif spec.method == "rare_probe":
                refresh = fixed_would_refresh or (risk >= spec.threshold)
            else:
                raise ValueError(f"unknown policy {spec.method}")

        probe_teacher_defect = float("nan")
        if spec.method == "rare_probe" and not refresh and step_index < num_steps - 1 and spec.sea_threshold is not None:
            band = float(spec.probe_band)
            lower = (1.0 - band) * spec.sea_threshold
            upper = (1.0 + band) * spec.sea_threshold
            if lower <= acc_after <= upper:
                state["expensive_probe_calls"] += 2
                ds = float(tr.scheduler.sigmas[step_index + 1] - tr.scheduler.sigmas[step_index])
                hs_cached = hs + state["previous_residual"]
                cached_norm = tr.norm_out(hs_cached, temb)
                cached_output = tr.proj_out(cached_norm)
                ori_hs = hs
                hs_fresh = run_transformer_stack(
                    tr,
                    hs,
                    enc,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                    controlnet_block_samples,
                    controlnet_single_block_samples,
                    controlnet_blocks_repeat,
                )
                fresh_norm = tr.norm_out(hs_fresh, temb)
                fresh_output = tr.proj_out(fresh_norm)
                next_timestep = tr.scheduler.timesteps[step_index + 1].expand(hidden_states.shape[0]).to(hidden_states.dtype) / 1000
                z_next_fresh = hidden_states.float() + ds * fresh_output.float()
                z_next_cached = hidden_states.float() + ds * cached_output.float()
                h_next_fresh = sea_feature_at_step(
                    tr,
                    z_next_fresh.to(hidden_states.dtype),
                    encoder_hidden_states,
                    pooled_projections,
                    next_timestep,
                    img_ids,
                    txt_ids,
                    guidance,
                    tr.scheduler,
                    step_index + 1,
                )
                h_next_cached = sea_feature_at_step(
                    tr,
                    z_next_cached.to(hidden_states.dtype),
                    encoder_hidden_states,
                    pooled_projections,
                    next_timestep,
                    img_ids,
                    txt_ids,
                    guidance,
                    tr.scheduler,
                    step_index + 1,
                )
                probe_teacher_defect = fsd.rel_l1(h_next_cached, h_next_fresh)
                refresh = bool(probe_teacher_defect >= float(spec.probe_teacher_threshold))

        if refresh:
            ori_hs = hs
            hs = run_transformer_stack(
                tr,
                hs,
                enc,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
                controlnet_block_samples,
                controlnet_single_block_samples,
                controlnet_blocks_repeat,
            )
            state["previous_residual"] = (hs - ori_hs).detach()
            state["fresh_evals"] += 1
            decision = "fresh_eval"
        else:
            hs = hs + state["previous_residual"]
            state["cached_evals"] += 1
            decision = "cache_reuse"

        hs = tr.norm_out(hs, temb)
        output = tr.proj_out(hs)
        state["step_traces"].append({
            "method": spec.key,
            "base_method": spec.method,
            "step": step_index,
            "sigma": sigma_value,
            "scheduler_timestep": timestep_value,
            "raw_rel_l1": float(raw_rel_l1),
            "prev_raw_rel_l1": float(state["prev_raw_rel_l1"]),
            "accumulated_before": acc_before,
            "accumulated_after": acc_after,
            "normalized_accumulated_after": acc_after / max(EPS, sea_threshold),
            "age_since_refresh": age_before,
            "last_refresh_step": last_refresh_step,
            "predictor_score": float(risk),
            "decision": decision,
            "dynamic_threshold": float(dynamic_threshold),
            "fixed_threshold": float(sea_threshold),
            "fixed_would_refresh": bool(fixed_would_refresh),
            "cheap_prefix_calls_so_far": int(state["cheap_prefix_calls"]),
            "expensive_probe_calls_so_far": int(state["expensive_probe_calls"]),
            "teacher_probe_defect": float(probe_teacher_defect),
        })
        state["previous_modulated_input"] = decision_feature.detach()
        state["prev_raw_rel_l1"] = float(raw_rel_l1)
        state["accumulated_since_refresh"] = 0.0 if refresh else acc_after
        state["age_since_refresh"] = 0 if refresh else age_before + 1
        state["last_refresh_step"] = step_index if refresh else last_refresh_step
        tr.cnt += 1
        if tr.cnt == tr.num_steps:
            tr.cnt = 0
        if USE_PEFT_BACKEND:
            unscale_lora_layers(tr, lora_scale)
        return (output,) if not return_dict else Transformer2DModelOutput(sample=output)

    tr.forward = wrapped_forward
    state["restore"] = lambda: setattr(tr, "forward", orig_forward)
    return state


def metric_row(
    sample: str,
    prompt: str,
    split: str,
    method: str,
    base_method: str,
    threshold: Any,
    sea_threshold: Any,
    actual_calls: int,
    cheap_prefix_calls: int,
    expensive_probe_calls: int,
    cached_calls: int,
    hook_forward_calls: int,
    wall_sec: float,
    latent_rel_l2: float,
    metrics: dict[str, float],
    png: Path,
    reference_wall_sec: float,
) -> dict[str, Any]:
    wall_speedup = reference_wall_sec / wall_sec if reference_wall_sec > 0 and wall_sec > 0 else float("nan")
    return {
        "sample": sample,
        "prompt": prompt,
        "split": split,
        "method": method,
        "base_method": base_method,
        "threshold": threshold,
        "sea_threshold": sea_threshold,
        "actual_full_calls": actual_calls,
        "cheap_prefix_calls": cheap_prefix_calls,
        "expensive_probe_calls": expensive_probe_calls,
        "cached_calls": cached_calls,
        "hook_forward_calls": hook_forward_calls,
        "wall_sec": wall_sec,
        "speedup_vs_100": N / max(1, actual_calls),
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


def aggregate_budget_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["threshold"], row["sea_threshold"]), []).append(row)
    out = []
    for (method, threshold, sea_threshold), rs in sorted(groups.items(), key=lambda x: (x[0][0], str(x[0][1]), str(x[0][2]))):
        def mean(key: str) -> float:
            vals = [float(r[key]) for r in rs if r.get(key) not in ("", None)]
            return float(np.mean(vals)) if vals else float("nan")
        out.append({
            "method": method,
            "base_method": rs[0]["base_method"],
            "threshold": threshold,
            "sea_threshold": sea_threshold,
            "actual_full_calls": mean("actual_full_calls"),
            "cheap_prefix_calls_mean": mean("cheap_prefix_calls"),
            "expensive_probe_calls_mean": mean("expensive_probe_calls"),
            "cached_calls_mean": mean("cached_calls"),
            "hook_forward_calls_mean": mean("hook_forward_calls"),
            "wall_sec_mean": mean("wall_sec"),
            "speedup_vs_100": N / max(1.0, mean("actual_full_calls")),
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


def nearest_by_calls(rows: list[dict[str, Any]], base_method: str, target: float) -> dict[str, Any] | None:
    candidates = [r for r in rows if r["base_method"] == base_method and math.isfinite(float(r["psnr_mean"]))]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (abs(float(r["actual_full_calls"]) - target), -float(r["psnr_mean"])))


def nearest_by_wall(rows: list[dict[str, Any]], base_method: str, target: float) -> dict[str, Any] | None:
    candidates = [r for r in rows if r["base_method"] == base_method and math.isfinite(float(r["psnr_mean"])) and math.isfinite(float(r["wall_speedup_vs_100_mean"]))]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (abs(float(r["wall_speedup_vs_100_mean"]) - target), -float(r["psnr_mean"])))


def best_dynamic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["base_method"] in ("dynamic_risk_gate", "dynamic_threshold_gate", "conservative_override", "rare_probe")]


def make_image_grid(sample_grid: dict[tuple[str, str], Path], sample_dirs: list[Path], cols: list[str], labels: list[str], out: Path) -> None:
    thumb = (230, 230)
    row_h = thumb[1] + 28
    left = 150
    sheet = Image.new("RGB", (left + len(cols) * thumb[0], 30 + min(4, len(sample_dirs)) * row_h), (248, 246, 239))
    draw = ImageDraw.Draw(sheet)
    for ci, lab in enumerate(labels):
        draw.text((left + ci * thumb[0] + 6, 8), lab[:28], fill=(20, 24, 28))
    for ri, sd in enumerate(sample_dirs[:4]):
        y0 = 30 + ri * row_h
        draw.text((6, y0 + 96), sd.name, fill=(20, 24, 28))
        for ci, key in enumerate(cols):
            p = sample_grid.get((sd.name, key))
            if p and Path(p).exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail(thumb)
                x = left + ci * thumb[0] + (thumb[0] - im.width) // 2
                y = y0 + (thumb[1] - im.height) // 2
                sheet.paste(im, (x, y))
    sheet.save(out)


def make_failure_case_grid(rows: list[dict[str, Any]], sample_grid: dict[tuple[str, str], Path], out: Path) -> None:
    dynamic_rows = [r for r in rows if r["base_method"] == "dynamic_risk_gate" and math.isfinite(float(r["lpips"]))]
    if not dynamic_rows:
        return
    worst = sorted(dynamic_rows, key=lambda r: (float(r["psnr"]), -float(r["lpips"])))[:2]
    thumb = (260, 260)
    cols = ["vanilla_100step", "seacache_fixed", "dynamic_refresh"]
    labels = ["100-step vanilla", "fixed SeaCache", "dynamic risk gate"]
    sheet = Image.new("RGB", (140 + len(cols) * thumb[0], 30 + len(worst) * (thumb[1] + 28)), (248, 246, 239))
    draw = ImageDraw.Draw(sheet)
    for ci, label in enumerate(labels):
        draw.text((140 + ci * thumb[0] + 6, 8), label, fill=(20, 24, 28))
    for ri, row in enumerate(worst):
        y0 = 30 + ri * (thumb[1] + 28)
        draw.text((6, y0 + 100), f"{row['sample']}\nPSNR {float(row['psnr']):.2f}", fill=(20, 24, 28))
        for ci, key in enumerate(cols):
            p = sample_grid.get((str(row["sample"]), key))
            if p and Path(p).exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail(thumb)
                x = 140 + ci * thumb[0] + (thumb[0] - im.width) // 2
                y = y0 + (thumb[1] - im.height) // 2
                sheet.paste(im, (x, y))
    sheet.save(out)


def make_figures(
    run_dir: Path,
    teacher_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    sample_grid: dict[tuple[str, str], Path],
    sample_dirs: list[Path],
    feature_importance: list[tuple[str, float]],
    per_sample_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "figures"
    colors = {
        "seacache_fixed": "#111111",
        "dynamic_risk_gate": "#2878b5",
        "dynamic_threshold_gate": "#2f9e44",
        "conservative_override": "#b85c38",
        "rare_probe": "#8c5fbf",
        "default_flux": "#9a6700",
    }

    eval_teacher = [r for r in teacher_rows if r["split"] == "eval" and math.isfinite(float(r["teacher_defect"])) and math.isfinite(float(r["cache_error"]))]
    fig, ax = plt.subplots(figsize=(8.3, 5.4))
    ax.scatter([float(r["teacher_defect"]) for r in eval_teacher], [float(r["cache_error"]) for r in eval_teacher], s=16, alpha=0.35, color=colors["dynamic_risk_gate"])
    ax.set_xlabel("SEA-defect teacher")
    ax.set_ylabel("actual cache error")
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_yscale("log")
    ax.set_title("SEA-defect teacher vs cache error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "teacher_signal_vs_cache_error.png", dpi=170)
    plt.close(fig)

    eval_cal = [r for r in calibration_rows if r["split"] == "eval" and r["metric"] == "summary"]
    eval_bins = [r for r in calibration_rows if r["split"] == "eval" and r["metric"] == "calibration_bin"]
    eval_rows = [r for r in teacher_rows if r["split"] == "eval" and "predictor_score" in r and math.isfinite(float(r["cache_error"])) and int(r["step"]) not in (0, N - 1)]
    scores = np.asarray([float(r["predictor_score"]) for r in eval_rows], dtype=np.float64)
    labels = np.asarray([int(r["refresh_needed"]) for r in eval_rows], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    ax.plot([0, 1], [0, 1], "--", color="#999999")
    if eval_bins:
        ax.plot([float(r["score_mean"]) for r in eval_bins], [float(r["empirical_positive_rate"]) for r in eval_bins], "-o", color=colors["dynamic_risk_gate"])
    ax.set_xlabel("predicted refresh probability")
    ax.set_ylabel("empirical refresh-needed rate")
    ax.set_title("Predictor calibration")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "predictor_calibration_curve.png", dpi=170)
    plt.close(fig)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.8, 5.0))
    if len(scores):
        thresholds = np.unique(scores)
        tprs, fprs, precs, recalls = [], [], [], []
        pos = max(1, int(labels.sum()))
        neg = max(1, int(len(labels) - labels.sum()))
        for th in thresholds:
            pred = scores >= th
            tp = int(np.logical_and(pred, labels == 1).sum())
            fp = int(np.logical_and(pred, labels == 0).sum())
            fn = int(np.logical_and(~pred, labels == 1).sum())
            tprs.append(tp / pos)
            fprs.append(fp / neg)
            precs.append(tp / max(1, tp + fp))
            recalls.append(tp / max(1, tp + fn))
        ax0.plot(fprs, tprs, color=colors["dynamic_risk_gate"])
        ax0.plot([0, 1], [0, 1], "--", color="#999999")
        ax0.set_xlabel("false positive rate")
        ax0.set_ylabel("true positive rate")
        ax0.set_title(f"ROC (AUC={rank_auc(labels, scores):.3f})")
        ax1.plot(recalls, precs, color=colors["dynamic_risk_gate"])
        ax1.set_xlabel("recall")
        ax1.set_ylabel("precision")
        ax1.set_title(f"PR (AUC={pr_auc(labels, scores):.3f})")
    ax0.grid(alpha=0.25)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "predictor_roc_pr.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    xs = [name for name, _ in feature_importance][:10]
    ys = [val for _, val in feature_importance][:10]
    ax.barh(range(len(xs)), ys[::-1], color=colors["dynamic_risk_gate"])
    ax.set_yticks(range(len(xs)))
    ax.set_yticklabels(xs[::-1], fontsize=9)
    ax.set_title("Selected predictor feature importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for base_method in sorted({r["base_method"] for r in budget_rows}):
        rs = [r for r in budget_rows if r["base_method"] == base_method and base_method != "default_flux"]
        if not rs:
            continue
        rs = sorted(rs, key=lambda r: float(r["actual_full_calls"]))
        ax.plot([float(r["actual_full_calls"]) for r in rs], [float(r["psnr_mean"]) for r in rs], marker="o", lw=1.7, color=colors.get(base_method, "#666666"), label=base_method)
    ax.set_xlabel("actual fresh full transformer calls")
    ax.set_ylabel("PSNR to 100-step vanilla (dB)")
    ax.set_title("Frontier by actual calls")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "dynamic_vs_fixed_seacache_frontier_calls.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for base_method in sorted({r["base_method"] for r in budget_rows}):
        rs = [r for r in budget_rows if r["base_method"] == base_method and base_method != "default_flux" and math.isfinite(float(r["wall_speedup_vs_100_mean"]))]
        if not rs:
            continue
        rs = sorted(rs, key=lambda r: float(r["wall_speedup_vs_100_mean"]))
        ax.plot([float(r["wall_speedup_vs_100_mean"]) for r in rs], [float(r["psnr_mean"]) for r in rs], marker="o", lw=1.7, color=colors.get(base_method, "#666666"), label=base_method)
    ax.set_xlabel("measured wall-clock speedup vs 100-step vanilla")
    ax.set_ylabel("PSNR to 100-step vanilla (dB)")
    ax.set_title("Frontier by wall-clock")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "dynamic_vs_fixed_seacache_frontier_wallclock.png", dpi=170)
    plt.close(fig)

    methods = sorted({r["method"] for r in trace_rows})
    ymap = {m: i for i, m in enumerate(methods)}
    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.28 * len(methods) + 1)))
    for row in trace_rows:
        color = colors.get(str(row["base_method"]), "#666666")
        ax.scatter(int(row["step"]), ymap[str(row["method"])], s=20, color=color if row["decision"] == "fresh_eval" else "#c8c8c8", alpha=0.85)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_xlabel("step")
    ax.set_title("Refresh schedule raster")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "refresh_schedule_raster.png", dpi=170)
    plt.close(fig)

    dynamic_trace = next((r["method"] for r in trace_rows if r["base_method"] == "dynamic_risk_gate"), None)
    if dynamic_trace is not None:
        rows = [r for r in trace_rows if r["method"] == dynamic_trace]
        fig, ax0 = plt.subplots(figsize=(9.6, 5.2))
        ax0.plot([int(r["step"]) for r in rows], [float(r["accumulated_after"]) for r in rows], color=colors["dynamic_risk_gate"], label="accumulator")
        ax0.plot([int(r["step"]) for r in rows], [float(r["predictor_score"]) for r in rows], color=colors["conservative_override"], label="risk score")
        ax0.set_xlabel("step")
        ax0.set_title(f"Accumulator and risk trace: {dynamic_trace}")
        ax0.grid(alpha=0.25)
        ax0.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "accumulator_and_risk_trace.png", dpi=170)
        plt.close(fig)

    make_image_grid(
        sample_grid,
        sample_dirs,
        ["vanilla_100step", "default_flux", "seacache_fixed", "dynamic_refresh"],
        ["100-step vanilla", "default FLUX", "fixed SeaCache", "best dynamic"],
        fig_dir / "sample_grid_dynamic_seacache.png",
    )
    make_failure_case_grid(per_sample_rows, sample_grid, fig_dir / "failure_cases_dynamic_gate.png")


def write_outputs(
    run_dir: Path,
    teacher_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    per_sample_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    call_rows: list[dict[str, Any]],
    wallclock_rows: list[dict[str, Any]],
) -> str:
    teacher_fields = [
        "sample",
        "prompt",
        "seed",
        "split",
        "step",
        "sigma",
        "scheduler_timestep",
        "sigma_region",
        "raw_rel_l1",
        "prev_raw_rel_l1",
        "accumulated_before",
        "accumulated_after",
        "normalized_accumulated_before",
        "normalized_accumulated_after",
        "age_since_refresh",
        "last_refresh_step",
        "steps_until_end",
        "raw_h_norm_mean",
        "raw_h_norm_std",
        "sea_h_norm_mean",
        "sea_h_norm_std",
        "fixed_threshold",
        "fixed_refresh",
        "teacher_defect",
        "cache_error",
        "refresh_needed",
        "predictor_score",
    ] + [f"feature_{name}" for name in FEATURE_NAMES] + [f"teacher_sea_defect_m{m}" for m in TEACHER_HORIZONS] + [f"teacher_endpoint_latent_error_m{m}" for m in TEACHER_HORIZONS] + [f"teacher_safe_cache_m{m}" for m in TEACHER_HORIZONS] + ["teacher_max_safe_m"]
    teacher_dataset_path = write_parquet_or_csv(run_dir / "metrics" / "teacher_label_dataset", teacher_rows, teacher_fields)

    write_csv(
        run_dir / "metrics" / "predictor_training_metrics.csv",
        training_rows,
        ["model", "split", "split_strategy", "count", "positives", "roc_auc", "pr_auc", "brier", "ece", "label_threshold", "selected_for_runtime"],
    )
    write_csv(
        run_dir / "metrics" / "predictor_calibration.csv",
        calibration_rows,
        ["model", "split", "metric", "bin_index", "bin_lo", "bin_hi", "count", "score_mean", "empirical_positive_rate", "roc_auc", "pr_auc", "brier", "ece", "label_threshold"],
    )
    write_csv(
        run_dir / "metrics" / "per_sample_metrics.csv",
        per_sample_rows,
        [
            "sample",
            "prompt",
            "split",
            "method",
            "base_method",
            "threshold",
            "sea_threshold",
            "actual_full_calls",
            "cheap_prefix_calls",
            "expensive_probe_calls",
            "cached_calls",
            "hook_forward_calls",
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
        ],
    )
    write_csv(
        run_dir / "metrics" / "per_method_budget_metrics.csv",
        budget_rows,
        [
            "method",
            "base_method",
            "threshold",
            "sea_threshold",
            "actual_full_calls",
            "cheap_prefix_calls_mean",
            "expensive_probe_calls_mean",
            "cached_calls_mean",
            "hook_forward_calls_mean",
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
        ],
    )
    write_csv(
        run_dir / "metrics" / "fixed_seacache_frontier.csv",
        [r for r in budget_rows if r["base_method"] == "seacache_fixed"],
        [
            "method",
            "base_method",
            "threshold",
            "sea_threshold",
            "actual_full_calls",
            "wall_speedup_vs_100_mean",
            "psnr_mean",
            "lpips_mean",
            "n",
        ],
    )
    write_csv(
        run_dir / "metrics" / "dynamic_seacache_frontier.csv",
        [r for r in budget_rows if r["base_method"] in ("dynamic_risk_gate", "dynamic_threshold_gate", "conservative_override", "rare_probe")],
        [
            "method",
            "base_method",
            "threshold",
            "sea_threshold",
            "actual_full_calls",
            "cheap_prefix_calls_mean",
            "expensive_probe_calls_mean",
            "wall_speedup_vs_100_mean",
            "psnr_mean",
            "lpips_mean",
            "n",
        ],
    )
    write_csv(
        run_dir / "metrics" / "call_counter_audit.csv",
        call_rows,
        ["sample", "method", "actual_full_calls", "cheap_prefix_calls", "expensive_probe_calls", "cached_calls", "hook_forward_calls", "wrapper_calls", "valid", "note"],
    )
    write_csv(
        run_dir / "metrics" / "wallclock_audit.csv",
        wallclock_rows,
        ["sample", "method", "reference_wall_sec", "method_wall_sec", "wall_speedup_vs_100", "actual_full_calls", "cheap_prefix_calls", "expensive_probe_calls", "valid", "note"],
    )
    leak_seen: dict[str, dict[str, Any]] = {}
    for row in leakage_rows:
        leak_seen.setdefault(str(row["method"]), row)
    write_csv(
        run_dir / "metrics" / "leakage_audit.csv",
        list(leak_seen.values()),
        ["method", "saved_vanilla_velocities_used_for_decision", "saved_vanilla_velocities_used_for_update", "valid_causal", "note"],
    )

    trace_fields = [
        "sample",
        "method",
        "base_method",
        "step",
        "sigma",
        "scheduler_timestep",
        "raw_rel_l1",
        "prev_raw_rel_l1",
        "accumulated_before",
        "accumulated_after",
        "normalized_accumulated_after",
        "age_since_refresh",
        "last_refresh_step",
        "predictor_score",
        "decision",
        "dynamic_threshold",
        "fixed_threshold",
        "fixed_would_refresh",
        "cheap_prefix_calls_so_far",
        "expensive_probe_calls_so_far",
        "teacher_probe_defect",
    ]
    write_csv(run_dir / "metrics" / "runtime_traces" / "all_runtime_traces.csv", trace_rows, trace_fields)
    for method in sorted({str(r["method"]) for r in trace_rows}):
        method_rows = [r for r in trace_rows if str(r["method"]) == method]
        write_csv(run_dir / "metrics" / "runtime_traces" / f"{method}.csv", method_rows, trace_fields)
        fsd.write_json(run_dir / "schedules" / f"{method}.json", {"method": method, "rows": method_rows})
    return teacher_dataset_path


def report_verdict(budget_rows: list[dict[str, Any]], teacher_corr: float) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
    fixed_rows = [r for r in budget_rows if r["base_method"] == "seacache_fixed"]
    dynamic_rows = best_dynamic(budget_rows)
    if not fixed_rows or not dynamic_rows:
        return "insufficient frontier data", "negative", None, None
    target_calls = float(np.median([float(r["actual_full_calls"]) for r in fixed_rows]))
    fixed_call = nearest_by_calls(fixed_rows, "seacache_fixed", target_calls)
    dynamic_call = min(dynamic_rows, key=lambda r: (abs(float(r["actual_full_calls"]) - float(fixed_call["actual_full_calls"])), -float(r["psnr_mean"]))) if fixed_call else None
    target_wall = float(np.median([float(r["wall_speedup_vs_100_mean"]) for r in fixed_rows if math.isfinite(float(r["wall_speedup_vs_100_mean"]))]))
    fixed_wall = nearest_by_wall(fixed_rows, "seacache_fixed", target_wall)
    dynamic_wall = min(dynamic_rows, key=lambda r: (abs(float(r["wall_speedup_vs_100_mean"]) - float(fixed_wall["wall_speedup_vs_100_mean"])), -float(r["psnr_mean"]))) if fixed_wall else None

    call_win = bool(dynamic_call and fixed_call and float(dynamic_call["psnr_mean"]) > float(fixed_call["psnr_mean"]))
    wall_win = bool(dynamic_wall and fixed_wall and float(dynamic_wall["psnr_mean"]) > float(fixed_wall["psnr_mean"]))
    if call_win and wall_win:
        return "dynamic SeaCache wins on matched calls and matched wall-clock", "positive", fixed_call, dynamic_call
    if call_win and not wall_win:
        return "call-count win but wall-clock loss", "negative", fixed_call, dynamic_call
    if teacher_corr > 0 and not call_win and not wall_win:
        return "teacher correlation exists but no method improvement", "negative", fixed_call, dynamic_call
    return "dynamic SeaCache does not beat fixed SeaCache on the required frontiers", "negative", fixed_call, dynamic_call


def write_report(
    run_dir: Path,
    default_steps: int,
    split_strategy: str,
    teacher_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    call_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    label_threshold: float,
    teacher_dataset_path: str,
    selected_model: dict[str, Any],
) -> None:
    eval_teacher = [r for r in teacher_rows if r["split"] == "eval" and math.isfinite(float(r["teacher_defect"])) and math.isfinite(float(r["cache_error"]))]
    teacher_corr = fsd.spearman_corr([float(r["teacher_defect"]) for r in eval_teacher], [float(r["cache_error"]) for r in eval_teacher]) if eval_teacher else float("nan")
    verdict, verdict_class, fixed_point, dynamic_point = report_verdict(budget_rows, teacher_corr)
    summary_row = next((r for r in calibration_rows if r["split"] == "eval" and r["metric"] == "summary"), None)

    def table(headers: list[str], rows: list[list[Any]]) -> str:
        head = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        return f"<table>{head}{body}</table>"

    frontier_rows = table(
        ["method", "calls", "wall", "PSNR", "LPIPS"],
        [
            [
                r["method"],
                f"{float(r['actual_full_calls']):.1f}",
                f"{float(r['wall_speedup_vs_100_mean']):.2f}x",
                f"{float(r['psnr_mean']):.2f}",
                f"{float(r['lpips_mean']):.4f}",
            ]
            for r in budget_rows
            if r["base_method"] != "default_flux"
        ],
    )
    predictor_table = table(
        ["model", "split", "count", "ROC AUC", "PR AUC", "Brier", "ECE", "selected"],
        [
            [
                r["model"],
                r["split"],
                r["count"],
                f"{float(r['roc_auc']):.3f}" if math.isfinite(float(r["roc_auc"])) else "nan",
                f"{float(r['pr_auc']):.3f}" if math.isfinite(float(r["pr_auc"])) else "nan",
                f"{float(r['brier']):.4f}",
                f"{float(r['ece']):.4f}",
                r["selected_for_runtime"],
            ]
            for r in training_rows
        ],
    )
    call_table = table(
        ["sample", "method", "full", "prefix", "probes", "cached", "hook", "wrapper", "valid", "note"],
        [[r["sample"], r["method"], r["actual_full_calls"], r["cheap_prefix_calls"], r["expensive_probe_calls"], r["cached_calls"], r["hook_forward_calls"], r["wrapper_calls"], r["valid"], r["note"]] for r in call_rows[:160]],
    )
    leak_seen: dict[str, dict[str, Any]] = {}
    for row in leakage_rows:
        leak_seen.setdefault(str(row["method"]), row)
    leak_table = table(
        ["method", "decision leakage?", "update leakage?", "causal valid?", "note"],
        [[r["method"], r["saved_vanilla_velocities_used_for_decision"], r["saved_vanilla_velocities_used_for_update"], r["valid_causal"], r["note"]] for r in leak_seen.values()],
    )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>E55 distilled SEA-defect for dynamic SeaCache</title>
<style>
body{{margin:0;background:#f7f1e7;color:#18201f;font:15px/1.55 Georgia,serif}}
main{{max-width:1200px;margin:auto;padding:34px 28px 64px}}
h1{{font:700 38px/1.05 ui-serif,Georgia,serif;margin:0 0 8px}}
h2{{font:700 24px/1.15 ui-serif,Georgia,serif;margin:32px 0 10px;border-top:1px solid #d8cbb8;padding-top:18px}}
h3{{font:700 18px/1.2 ui-serif,Georgia,serif;margin:22px 0 8px}}
.lede{{font-size:18px;max-width:960px}}
.verdict{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}}
.card{{background:#fffaf1;border:1px solid #dfd2bd;border-radius:12px;padding:14px}}
.card b{{display:block;font-size:22px}}
figure{{margin:18px 0;background:#fffaf1;border:1px solid #dfd2bd;border-radius:12px;padding:10px}}
figure img{{max-width:100%;display:block;margin:auto}}
figcaption{{font-size:13px;color:#51483b;margin-top:6px}}
table{{border-collapse:collapse;width:100%;font-size:12px;background:#fffaf1;margin:10px 0 22px}}
th,td{{border:1px solid #d8cbb8;padding:5px 7px;text-align:left;vertical-align:top}}
th{{background:#efe2ce}}
.warn{{border-left:5px solid #b43b2f;background:#fff6e8;padding:12px 16px;border-radius:8px}}
</style></head><body><main>
<h1>E55 distilled SEA-defect for dynamic SeaCache refresh</h1>
<p class="lede">Offline teacher: SEA-defect. Online methods: fixed SeaCache, dynamic risk gate, dynamic threshold gate, conservative override, and optional rare-probe ablation. Runtime methods remain causal and prefix-free.</p>
<div class="verdict">
<div class="card"><span>Verdict</span><b>{verdict}</b><small>{verdict_class}</small></div>
<div class="card"><span>Teacher Spearman</span><b>{teacher_corr:.3f}</b><small>eval split</small></div>
<div class="card"><span>Predictor AUC</span><b>{float(summary_row['roc_auc']) if summary_row else float('nan'):.3f}</b><small>{selected_model['selected_model_name']} on eval</small></div>
<div class="card"><span>Split Protocol</span><b>{split_strategy}</b><small>label threshold {label_threshold:.4g}</small></div>
</div>

<h2>Contract</h2>
<p>Run slug: <code>{EXPERIMENT_SLUG}</code>. Teacher dataset: <code>{teacher_dataset_path}</code>. Default FLUX steps detected programmatically: <b>{default_steps}</b>. Selected runtime risk model: <b>{selected_model['selected_model_name']}</b>.</p>

<h2>Outcome</h2>
<p>{verdict}. Fixed matched-call point: <b>{fixed_point['method'] if fixed_point else 'n/a'}</b>. Dynamic matched-call point: <b>{dynamic_point['method'] if dynamic_point else 'n/a'}</b>. The report only declares a method win when a dynamic method beats fixed SeaCache on the matched frontiers instead of merely showing teacher correlation.</p>

<h2>Figures</h2>
<figure><img src="figures/teacher_signal_vs_cache_error.png"><figcaption>SEA-defect teacher vs actual cache error.</figcaption></figure>
<figure><img src="figures/predictor_roc_pr.png"><figcaption>Selected predictor ROC and PR curves.</figcaption></figure>
<figure><img src="figures/predictor_calibration_curve.png"><figcaption>Selected predictor calibration curve.</figcaption></figure>
<figure><img src="figures/feature_importance.png"><figcaption>Selected predictor feature importance.</figcaption></figure>
<figure><img src="figures/dynamic_vs_fixed_seacache_frontier_calls.png"><figcaption>Matched-call frontier.</figcaption></figure>
<figure><img src="figures/dynamic_vs_fixed_seacache_frontier_wallclock.png"><figcaption>Matched-wall-clock frontier.</figcaption></figure>
<figure><img src="figures/refresh_schedule_raster.png"><figcaption>Refresh schedule raster across fixed and dynamic policies.</figcaption></figure>
<figure><img src="figures/accumulator_and_risk_trace.png"><figcaption>Accumulator and risk trace for a dynamic risk gate run.</figcaption></figure>
<figure><img src="figures/sample_grid_dynamic_seacache.png"><figcaption>Vanilla, default FLUX, fixed SeaCache, best dynamic policy.</figcaption></figure>
<figure><img src="figures/failure_cases_dynamic_gate.png"><figcaption>Failure cases for the dynamic risk gate.</figcaption></figure>

<h2>Predictor Comparison</h2>
{predictor_table}

<h2>Frontiers</h2>
{frontier_rows}

<h2>Audits</h2>
<div class="warn"><b>Call audit note:</b> the old invalid teacher assertion on the outer transformer wrapper count was removed. This run audits both the wrapper count and the hook's own state instead.</div>
<h3>Call Counter Audit</h3>{call_table}
<h3>Leakage Audit</h3>{leak_table}
</main></body></html>"""
    report_path = run_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    report_path.write_text(fsd.embed_local_images(html, report_path), encoding="utf-8")

    summary = {
        "title": "E55 distilled SEA-defect for dynamic SeaCache refresh",
        "slug": EXPERIMENT_SLUG,
        "split_strategy": split_strategy,
        "selected_model_name": selected_model["selected_model_name"],
        "predictor_eval_auc": float(summary_row["roc_auc"]) if summary_row else float("nan"),
        "teacher_spearman": teacher_corr,
        "label_threshold": label_threshold,
        "teacher_dataset_path": teacher_dataset_path,
        "verdict": verdict,
        "run_dir": str(run_dir),
    }
    fsd.write_json(run_dir / "reports" / "summary.json", summary)
    (run_dir / "reports" / "summary.md").write_text(
        f"# E55 distilled SEA-defect for dynamic SeaCache refresh\n\n"
        f"- Slug: `{EXPERIMENT_SLUG}`\n"
        f"- Split protocol: **{split_strategy}**\n"
        f"- Selected model: **{selected_model['selected_model_name']}**\n"
        f"- Teacher/cache-error Spearman: **{teacher_corr:.3f}**\n"
        f"- Predictor eval AUC: **{float(summary_row['roc_auc']) if summary_row else float('nan'):.3f}**\n"
        f"- Verdict: **{verdict}**\n",
        encoding="utf-8",
    )


def write_manifest(run_dir: Path, args: argparse.Namespace) -> None:
    files = [str(p.relative_to(run_dir)) for p in sorted(run_dir.rglob("*")) if p.is_file()]
    fsd.write_json(run_dir / "artifacts_manifest.json", {
        "experiment": "E55_flux_seadefect_distilled_seacache",
        "slug": EXPERIMENT_SLUG,
        "created": dt.datetime.now().isoformat(),
        "args": vars(args),
        "files": files,
    })


def run(args: argparse.Namespace) -> None:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "__smoke" if args.smoke else ""
    run_dir = Path(args.run_root) / f"{ts}__{EXPERIMENT_SLUG}{suffix}"
    ensure_tree(run_dir)
    print(f"[e55] run_dir={run_dir}", flush=True)

    fixture = fixtures.canonical_prompts()[: args.num_samples]
    split_labels, split_strategy = split_protocol(len(fixture))
    fsd.PROMPTS = [(x["id"], x["prompt"]) for x in fixture]
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
    print("[e55] capturing/reusing vanilla 100-step trajectories", flush=True)
    fsd.run_capture(cap_args)

    pipe = fsd.load_flux_pipeline(args.model_id, args.dtype, args.device, args.offload, False)
    bank = e53.MetricBank(args.device)
    default_steps = get_default_steps(pipe)
    sample_dirs = sorted(traj_root.glob("sample_*"))[: args.num_samples]
    teacher_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    wallclock_rows: list[dict[str, Any]] = []
    sample_grid: dict[tuple[str, str], Path] = {}

    counter = CallCounter(pipe)
    try:
        for si, sd in enumerate(sample_dirs):
            meta = fsd.read_json(sd / "metadata.json")
            prompt = meta["prompt"]
            seed = int(meta["seed"])
            split = split_labels[si]
            prep_inputs(pipe, prompt, seed, N, args.height, args.width, args.guidance, args.max_sequence_length, args.device)
            print(f"[e55] teacher capture sample={sd.name} split={split}", flush=True)
            teacher_state = install_teacher_capture_forward(pipe, sd.name, split, prompt, seed, args.teacher_seacache_threshold, N)
            counter.reset()
            _, _, _ = e53.live_generate(pipe, prompt, seed, N, args.height, args.width, args.guidance, args.max_sequence_length, args.device)
            teacher_state["restore"]()
            teacher_rows.extend(teacher_state["rows"])
            call_rows.append({
                "sample": sd.name,
                "method": "teacher_capture",
                "actual_full_calls": int(teacher_state["fresh_evals"]),
                "cheap_prefix_calls": 0,
                "expensive_probe_calls": 0,
                "cached_calls": 0,
                "hook_forward_calls": int(teacher_state["hook_forward_calls"]),
                "wrapper_calls": int(counter.n),
                "valid": int(teacher_state["hook_forward_calls"]) == N and int(counter.n) == N,
                "note": "teacher audit uses hook state plus outer wrapper count",
            })
            sample_grid[(sd.name, "vanilla_100step")] = sd / "final.png"

        enrich_teacher_rows(teacher_rows, sample_dirs, args.teacher_safe_threshold)
        selected_model, training_rows, calibration_rows, label_threshold = fit_predictor_suite(teacher_rows, split_strategy, args.label_quantile)
        predictor_summary = {
            "slug": EXPERIMENT_SLUG,
            "selected_model_name": selected_model["selected_model_name"],
            "split_strategy": split_strategy,
            "selection_split": selected_model["selection_split"],
            "feature_names": selected_model["feature_names"],
            "label_threshold": label_threshold,
            "selected_summary": selected_model["selected_summary"],
        }
        fsd.write_json(run_dir / "reports" / "predictor_model_summary.json", predictor_summary)

        specs = policy_specs(args)
        feature_importance = predictor_feature_importance(selected_model)
        for si, sd in enumerate(sample_dirs):
            meta = fsd.read_json(sd / "metadata.json")
            prompt = meta["prompt"]
            seed = int(meta["seed"])
            split = split_labels[si]
            if split not in ("eval", "val"):
                continue
            reference_wall_sec = float(meta.get("runtime_sec", 0.0) or 0.0)
            vanilla_img = Image.open(sd / "final.png").convert("RGB")
            vanilla_lat = fsd.load_step_tensor(sd / "latents", N)
            print(f"[e55] eval sample={sd.name} split={split}", flush=True)

            counter.reset()
            d_lat, d_img, d_wall = e53.live_generate(pipe, prompt, seed, default_steps, args.height, args.width, args.guidance, args.max_sequence_length, args.device)
            d_png = run_dir / "samples" / f"{sd.name}_default_flux.png"
            d_img.save(d_png)
            dm = bank.image_metrics(d_img, vanilla_img, prompt)
            dll = fsd.latent_metrics(d_lat, vanilla_lat)["latent_rel_l2"]
            per_sample_rows.append(metric_row(sd.name, prompt, split, "default_flux", "default_flux", "", "", counter.n, 0, 0, 0, counter.n, d_wall, dll, dm, d_png, reference_wall_sec))
            sample_grid[(sd.name, "default_flux")] = d_png
            leakage_rows.append({
                "method": "default_flux",
                "saved_vanilla_velocities_used_for_decision": False,
                "saved_vanilla_velocities_used_for_update": False,
                "valid_causal": True,
                "note": "standard pipeline",
            })
            call_rows.append({
                "sample": sd.name,
                "method": "default_flux",
                "actual_full_calls": int(counter.n),
                "cheap_prefix_calls": 0,
                "expensive_probe_calls": 0,
                "cached_calls": 0,
                "hook_forward_calls": int(counter.n),
                "wrapper_calls": int(counter.n),
                "valid": True,
                "note": "pipeline default num_inference_steps",
            })
            wallclock_rows.append({
                "sample": sd.name,
                "method": "default_flux",
                "reference_wall_sec": reference_wall_sec,
                "method_wall_sec": d_wall,
                "wall_speedup_vs_100": reference_wall_sec / d_wall if reference_wall_sec > 0 and d_wall > 0 else float("nan"),
                "actual_full_calls": int(counter.n),
                "cheap_prefix_calls": 0,
                "expensive_probe_calls": 0,
                "valid": True,
                "note": "default pipeline baseline",
            })

            for th in args.seacache_thresholds:
                state = fsd.install_seacache_forward(pipe, th, N)
                counter.reset()
                lat, img, wall = e53.live_generate(pipe, prompt, seed, N, args.height, args.width, args.guidance, args.max_sequence_length, args.device)
                wrapper_calls = counter.n
                state["restore"]()
                fresh = int(state["fresh_evals"])
                cached = int(state["cached_evals"])
                png = run_dir / "samples" / f"{sd.name}_seacache_fixed_th{th:g}.png"
                img.save(png)
                metrics = bank.image_metrics(img, vanilla_img, prompt)
                ll = fsd.latent_metrics(lat, vanilla_lat)["latent_rel_l2"]
                per_sample_rows.append(metric_row(sd.name, prompt, split, f"seacache_fixed_th{th:g}", "seacache_fixed", th, th, fresh, 0, 0, cached, wrapper_calls, wall, ll, metrics, png, reference_wall_sec))
                leakage_rows.append({
                    "method": "seacache_fixed",
                    "saved_vanilla_velocities_used_for_decision": False,
                    "saved_vanilla_velocities_used_for_update": False,
                    "valid_causal": True,
                    "note": "official accumulated SEA rel-L1 gate",
                })
                call_rows.append({
                    "sample": sd.name,
                    "method": f"seacache_fixed_th{th:g}",
                    "actual_full_calls": fresh,
                    "cheap_prefix_calls": 0,
                    "expensive_probe_calls": 0,
                    "cached_calls": cached,
                    "hook_forward_calls": wrapper_calls,
                    "wrapper_calls": wrapper_calls,
                    "valid": wrapper_calls == N,
                    "note": f"fresh={fresh}; cached={cached}",
                })
                wallclock_rows.append({
                    "sample": sd.name,
                    "method": f"seacache_fixed_th{th:g}",
                    "reference_wall_sec": reference_wall_sec,
                    "method_wall_sec": wall,
                    "wall_speedup_vs_100": reference_wall_sec / wall if reference_wall_sec > 0 and wall > 0 else float("nan"),
                    "actual_full_calls": fresh,
                    "cheap_prefix_calls": 0,
                    "expensive_probe_calls": 0,
                    "valid": wrapper_calls == N,
                    "note": "fixed SeaCache sweep",
                })
                sample_grid.setdefault((sd.name, "seacache_fixed"), png)

            best_dynamic_for_grid: tuple[str, float] | None = None
            best_dynamic_png: Path | None = None
            for spec in specs:
                state = install_policy_forward(pipe, selected_model, spec, N)
                counter.reset()
                lat, img, wall = e53.live_generate(pipe, prompt, seed, N, args.height, args.width, args.guidance, args.max_sequence_length, args.device)
                wrapper_calls = counter.n
                state["restore"]()
                fresh = int(state["fresh_evals"])
                cached = int(state["cached_evals"])
                prefix = int(state["cheap_prefix_calls"])
                probes = int(state["expensive_probe_calls"])
                png = run_dir / "samples" / f"{sd.name}_{spec.key}.png"
                img.save(png)
                metrics = bank.image_metrics(img, vanilla_img, prompt)
                ll = fsd.latent_metrics(lat, vanilla_lat)["latent_rel_l2"]
                per_sample_rows.append(metric_row(sd.name, prompt, split, spec.key, spec.method, spec.threshold, spec.sea_threshold, fresh, prefix, probes, cached, int(state["hook_forward_calls"]), wall, ll, metrics, png, reference_wall_sec))
                trace_rows.extend({"sample": sd.name, **row} for row in state["step_traces"])
                leakage_rows.append({
                    "method": spec.method,
                    "saved_vanilla_velocities_used_for_decision": False,
                    "saved_vanilla_velocities_used_for_update": False,
                    "valid_causal": True,
                    "note": "online gate uses only cheap current-step features; no future vanilla states used online",
                })
                call_rows.append({
                    "sample": sd.name,
                    "method": spec.key,
                    "actual_full_calls": fresh,
                    "cheap_prefix_calls": prefix,
                    "expensive_probe_calls": probes,
                    "cached_calls": cached,
                    "hook_forward_calls": int(state["hook_forward_calls"]),
                    "wrapper_calls": wrapper_calls,
                    "valid": int(state["hook_forward_calls"]) == N and wrapper_calls == N,
                    "note": "policy hook audit",
                })
                wallclock_rows.append({
                    "sample": sd.name,
                    "method": spec.key,
                    "reference_wall_sec": reference_wall_sec,
                    "method_wall_sec": wall,
                    "wall_speedup_vs_100": reference_wall_sec / wall if reference_wall_sec > 0 and wall > 0 else float("nan"),
                    "actual_full_calls": fresh,
                    "cheap_prefix_calls": prefix,
                    "expensive_probe_calls": probes,
                    "valid": int(state["hook_forward_calls"]) == N and wrapper_calls == N,
                    "note": "dynamic policy wallclock audit",
                })
                if spec.method in ("dynamic_risk_gate", "dynamic_threshold_gate", "conservative_override", "rare_probe"):
                    if best_dynamic_for_grid is None or float(metrics["psnr"]) > best_dynamic_for_grid[1]:
                        best_dynamic_for_grid = (spec.key, float(metrics["psnr"]))
                        best_dynamic_png = png
            if best_dynamic_png is not None:
                sample_grid[(sd.name, "dynamic_refresh")] = best_dynamic_png

        budget_rows = aggregate_budget_rows(per_sample_rows)
        teacher_dataset_path = write_outputs(run_dir, teacher_rows, training_rows, calibration_rows, per_sample_rows, budget_rows, trace_rows, leakage_rows, call_rows, wallclock_rows)
        make_figures(run_dir, teacher_rows, training_rows, calibration_rows, budget_rows, trace_rows, sample_grid, sample_dirs, feature_importance, per_sample_rows)
        write_report(run_dir, default_steps, split_strategy, teacher_rows, training_rows, calibration_rows, budget_rows, call_rows, leakage_rows, label_threshold, teacher_dataset_path, selected_model)
        write_manifest(run_dir, args)
    finally:
        counter.restore()
        del pipe
        torch.cuda.empty_cache()

    print(f"[e55] DONE report={run_dir / 'report.html'}", flush=True)


def parse_csv_list(raw: str, typ: Any) -> list[Any]:
    return [typ(x) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E55 distilled SEA-defect for dynamic SeaCache refresh")
    ap.add_argument("--model-id", default=fsd.DEFAULT_MODEL_ID)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "h100"))
    ap.add_argument("--trajectory-root", default=str(REPO_ROOT / "outputs" / EXPERIMENT_SLUG / "trajectories"))
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--max-sequence-length", type=int, default=512)
    ap.add_argument("--teacher-seacache-threshold", type=float, default=0.15)
    ap.add_argument("--teacher-safe-threshold", type=float, default=0.15)
    ap.add_argument("--label-quantile", type=float, default=0.75)
    ap.add_argument("--seacache-thresholds", type=lambda s: parse_csv_list(s, float), default=[0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    ap.add_argument("--dynamic-risk-thresholds", type=lambda s: parse_csv_list(s, float), default=[0.35, 0.45, 0.55, 0.65, 0.75])
    ap.add_argument("--dynamic-threshold-thresholds", type=lambda s: parse_csv_list(s, float), default=[0.6, 0.8, 1.0])
    ap.add_argument("--override-thresholds", type=lambda s: parse_csv_list(s, float), default=[0.35, 0.5, 0.65])
    ap.add_argument("--run-rare-probe", action="store_true")
    ap.add_argument("--rare-probe-dynamic-threshold", type=float, default=0.55)
    ap.add_argument("--rare-probe-teacher-threshold", type=float, default=0.02)
    ap.add_argument("--rare-probe-band", type=float, default=0.2)
    ap.add_argument("--force-capture", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.num_samples = min(args.num_samples, 2)
        args.seacache_thresholds = args.seacache_thresholds[:3]
        args.dynamic_risk_thresholds = args.dynamic_risk_thresholds[:2]
        args.dynamic_threshold_thresholds = args.dynamic_threshold_thresholds[:2]
        args.override_thresholds = args.override_thresholds[:2]
    return args


if __name__ == "__main__":
    run(parse_args())
