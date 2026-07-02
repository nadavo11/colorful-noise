#!/usr/bin/env python3
"""E55 — distilled SEA-defect for dynamic SeaCache refresh.

This experiment treats the expensive SEA-defect signal as an offline teacher and
fits a cheap, prefix-free, causal refresh gate for SeaCache. The online policy
uses only the current raw SeaCache relative-L1 signal and simple causal step
context; expensive SEA-defect stays offline-only except for an optional rare-
probe ablation.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
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
class DynamicPolicySpec:
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
        "trajectories",
    ]:
        fsd.ensure_dir(run_dir / sub)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    fsd.ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


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


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    steps: int = 2500,
    l2: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    if y.ndim != 1:
        raise ValueError("y must be rank-1")
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    xs = (x - mu) / sigma
    w = np.zeros(xs.shape[1], dtype=np.float64)
    b = 0.0
    n = float(max(1, xs.shape[0]))
    for _ in range(steps):
        logits = xs @ w + b
        p = logistic_sigmoid(logits)
        grad_w = (xs.T @ (p - y)) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return np.concatenate([[b], w]), np.concatenate([[0.0], mu, sigma])


def predict_logistic(params: np.ndarray, stats: np.ndarray, x: np.ndarray) -> np.ndarray:
    b = float(params[0])
    w = params[1:]
    d = x.shape[1]
    mu = stats[1 : 1 + d]
    sigma = stats[1 + d : 1 + 2 * d]
    xs = (x - mu) / sigma
    return logistic_sigmoid(xs @ w + b)


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


def make_feature_vector(
    raw_rel_l1: float,
    prev_raw_rel_l1: float,
    step_index: int,
    sigma_value: float,
) -> list[float]:
    step_frac = float(step_index) / float(max(1, N - 1))
    return [
        1.0,
        raw_rel_l1,
        prev_raw_rel_l1,
        step_frac,
        sigma_value,
        raw_rel_l1 * step_frac,
        raw_rel_l1 - prev_raw_rel_l1,
    ]


def is_eval_sample(sample_idx: int, total: int) -> bool:
    split = max(1, total // 2)
    return sample_idx >= split


def sigma_region(step_index: int) -> str:
    if step_index < N / 3:
        return "early_high_noise"
    if step_index < 2 * N / 3:
        return "mid"
    return "late_low_noise"


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
        "previous_modulated_input": None,
        "previous_residual": None,
        "fresh_evals": 0,
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
        modulated, *_ = tr.transformer_blocks[0].norm1(hs, emb=temb)
        decision_feature = modulated
        raw_rel_l1 = 0.0
        if step_index not in (0, num_steps - 1) and state["previous_modulated_input"] is not None:
            img = img_ids[0] if img_ids is not None and img_ids.ndim == 3 else img_ids
            grid = modulated.reshape(modulated.shape[0], int(img[:, 1].max().item() + 1), int(img[:, 2].max().item() + 1), modulated.shape[-1])
            a, b = fsd.ab_from_scheduler(tr.scheduler, step_index)
            decision_feature = fsd.apply_sea_from_ab(grid, a, b, dims=(-2, -3), norm_mode="mean").reshape(modulated.shape[0], -1, modulated.shape[-1])
            raw_rel_l1 = fsd.rel_l1(decision_feature, state["previous_modulated_input"])

        acc_before = float(state["sim_acc"])
        acc_after = float(acc_before + raw_rel_l1)
        fixed_refresh = (
            step_index in (0, num_steps - 1)
            or state["previous_residual"] is None
            or not (acc_after < sea_threshold)
        )
        feature_vec = make_feature_vector(raw_rel_l1, float(state["prev_raw_rel_l1"]), step_index, sigma_value)

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

        state["rows"].append({
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
            "feature_bias": float(feature_vec[0]),
            "feature_raw_rel_l1": float(feature_vec[1]),
            "feature_prev_raw_rel_l1": float(feature_vec[2]),
            "feature_step_frac": float(feature_vec[3]),
            "feature_sigma": float(feature_vec[4]),
            "feature_raw_x_step": float(feature_vec[5]),
            "feature_delta_raw": float(feature_vec[6]),
            "fixed_threshold": float(sea_threshold),
            "fixed_refresh": bool(fixed_refresh),
            "teacher_defect": float(teacher_defect),
            "cache_error": float(cache_error),
        })

        state["previous_modulated_input"] = decision_feature.detach()
        state["prev_raw_rel_l1"] = float(raw_rel_l1)
        state["sim_acc"] = 0.0 if fixed_refresh else acc_after
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


def fit_predictor_from_rows(rows: list[dict[str, Any]], label_quantile: float) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    train_rows = [r for r in rows if r["split"] == "train" and math.isfinite(float(r["cache_error"])) and int(r["step"]) not in (0, N - 1)]
    eval_rows = [r for r in rows if r["split"] == "eval" and math.isfinite(float(r["cache_error"])) and int(r["step"]) not in (0, N - 1)]
    if not train_rows or not eval_rows:
        raise RuntimeError("need both train and eval teacher rows to fit predictor")
    threshold = float(np.quantile([float(r["cache_error"]) for r in train_rows], label_quantile))
    if threshold <= 0:
        threshold = max(1e-6, float(np.mean([float(r["cache_error"]) for r in train_rows])))
    for r in rows:
        cache_error = float(r["cache_error"]) if math.isfinite(float(r["cache_error"])) else float("nan")
        r["refresh_needed"] = int(math.isfinite(cache_error) and cache_error >= threshold)
    feats = np.asarray([
        [
            float(r["feature_raw_rel_l1"]),
            float(r["feature_prev_raw_rel_l1"]),
            float(r["feature_step_frac"]),
            float(r["feature_sigma"]),
            float(r["feature_raw_x_step"]),
            float(r["feature_delta_raw"]),
        ]
        for r in train_rows
    ], dtype=np.float64)
    labels = np.asarray([int(r["refresh_needed"]) for r in train_rows], dtype=np.float64)
    params, stats = fit_logistic_regression(feats, labels)
    for r in rows:
        x = np.asarray([[
            float(r["feature_raw_rel_l1"]),
            float(r["feature_prev_raw_rel_l1"]),
            float(r["feature_step_frac"]),
            float(r["feature_sigma"]),
            float(r["feature_raw_x_step"]),
            float(r["feature_delta_raw"]),
        ]], dtype=np.float64)
        r["predictor_score"] = float(predict_logistic(params, stats, x)[0])
    eval_y = np.asarray([int(r["refresh_needed"]) for r in eval_rows], dtype=np.int64)
    eval_scores = np.asarray([float(r["predictor_score"]) for r in eval_rows], dtype=np.float64)
    train_y = np.asarray([int(r["refresh_needed"]) for r in train_rows], dtype=np.int64)
    train_scores = np.asarray([float(r["predictor_score"]) for r in train_rows], dtype=np.float64)
    calib_rows = []
    for split_name, yy, ss in [("train", train_y, train_scores), ("eval", eval_y, eval_scores)]:
        auc = rank_auc(yy, ss)
        brier = float(np.mean((ss - yy) ** 2))
        bins = calibration_bins(yy, ss, bins=10)
        ece = 0.0
        for b in bins:
            ece += abs(float(b["score_mean"]) - float(b["empirical_positive_rate"])) * (float(b["count"]) / max(1, len(yy)))
            calib_rows.append({
                "split": split_name,
                "metric": "calibration_bin",
                **b,
                "roc_auc": auc,
                "brier": brier,
                "ece": ece,
                "label_threshold": threshold,
            })
        calib_rows.append({
            "split": split_name,
            "metric": "summary",
            "bin_index": "",
            "bin_lo": "",
            "bin_hi": "",
            "count": len(yy),
            "score_mean": float(ss.mean()) if len(ss) else float("nan"),
            "empirical_positive_rate": float(yy.mean()) if len(yy) else float("nan"),
            "roc_auc": auc,
            "brier": brier,
            "ece": ece,
            "label_threshold": threshold,
        })
    model = {
        "params": params.tolist(),
        "stats": stats.tolist(),
        "feature_names": [
            "raw_rel_l1",
            "prev_raw_rel_l1",
            "step_frac",
            "sigma",
            "raw_x_step",
            "delta_raw",
        ],
    }
    return model, calib_rows, threshold


def install_dynamic_refresh_forward(
    pipe: Any,
    model: dict[str, Any],
    spec: DynamicPolicySpec,
    num_steps: int,
) -> dict[str, Any]:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    params = np.asarray(model["params"], dtype=np.float64)
    stats = np.asarray(model["stats"], dtype=np.float64)
    tr = pipe.transformer
    orig_forward = tr.forward
    state: dict[str, Any] = {
        "fresh_evals": 0,
        "cached_evals": 0,
        "cheap_prefix_calls": 0,
        "expensive_probe_calls": 0,
        "step_traces": [],
        "previous_modulated_input": None,
        "previous_residual": None,
        "prev_raw_rel_l1": 0.0,
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
        modulated, *_ = tr.transformer_blocks[0].norm1(hs, emb=temb)
        decision_feature = modulated
        raw_rel_l1 = 0.0
        if step_index not in (0, num_steps - 1) and state["previous_modulated_input"] is not None:
            img = img_ids[0] if img_ids is not None and img_ids.ndim == 3 else img_ids
            grid = modulated.reshape(modulated.shape[0], int(img[:, 1].max().item() + 1), int(img[:, 2].max().item() + 1), modulated.shape[-1])
            a, b = fsd.ab_from_scheduler(tr.scheduler, step_index)
            decision_feature = fsd.apply_sea_from_ab(grid, a, b, dims=(-2, -3), norm_mode="mean").reshape(modulated.shape[0], -1, modulated.shape[-1])
            raw_rel_l1 = fsd.rel_l1(decision_feature, state["previous_modulated_input"])
        state["cheap_prefix_calls"] += 1

        x = np.asarray([[
            float(raw_rel_l1),
            float(state["prev_raw_rel_l1"]),
            float(step_index) / float(max(1, N - 1)),
            float(sigma_value),
            float(raw_rel_l1 * (float(step_index) / float(max(1, N - 1)))),
            float(raw_rel_l1 - state["prev_raw_rel_l1"]),
        ]], dtype=np.float64)
        score = float(predict_logistic(params, stats, x)[0])
        refresh = (
            step_index in (0, num_steps - 1)
            or state["previous_residual"] is None
            or score >= spec.threshold
        )
        probe_teacher_defect = float("nan")
        if spec.method == "rare_probe" and not refresh and spec.sea_threshold is not None:
            lower = 0.8 * spec.sea_threshold
            upper = 1.2 * spec.sea_threshold
            if lower <= raw_rel_l1 <= upper and step_index < num_steps - 1:
                state["expensive_probe_calls"] += 2
                ds = float(tr.scheduler.sigmas[step_index + 1] - tr.scheduler.sigmas[step_index])
                if state["previous_residual"] is not None:
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
                    h_next_fresh = sea_feature_at_step(tr, z_next_fresh.to(hidden_states.dtype), encoder_hidden_states, pooled_projections, next_timestep, img_ids, txt_ids, guidance, tr.scheduler, step_index + 1)
                    h_next_cached = sea_feature_at_step(tr, z_next_cached.to(hidden_states.dtype), encoder_hidden_states, pooled_projections, next_timestep, img_ids, txt_ids, guidance, tr.scheduler, step_index + 1)
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
            "predictor_score": float(score),
            "decision": decision,
            "expensive_probe_calls_so_far": int(state["expensive_probe_calls"]),
            "cheap_prefix_calls_so_far": int(state["cheap_prefix_calls"]),
            "teacher_probe_defect": float(probe_teacher_defect),
        })
        state["previous_modulated_input"] = decision_feature.detach()
        state["prev_raw_rel_l1"] = float(raw_rel_l1)
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
    method: str,
    base_method: str,
    threshold: Any,
    causal: bool,
    actual_calls: int,
    cheap_prefix_calls: int,
    expensive_probe_calls: int,
    cached_calls: int,
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
        "method": method,
        "base_method": base_method,
        "threshold": threshold,
        "causal": causal,
        "actual_full_calls": actual_calls,
        "cheap_prefix_calls": cheap_prefix_calls,
        "expensive_probe_calls": expensive_probe_calls,
        "cached_calls": cached_calls,
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
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
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
            "expensive_probe_calls_mean": mean("expensive_probe_calls"),
            "cached_calls_mean": mean("cached_calls"),
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


def best_by_base(rows: list[dict[str, Any]], base: str, target: int | None = None) -> dict[str, Any] | None:
    rs = [r for r in rows if r["base_method"] == base and math.isfinite(float(r["psnr_mean"]))]
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


def make_image_grid(sample_grid: dict[tuple[str, str], Path], sample_dirs: list[Path], cols: list[str], labels: list[str], out: Path) -> None:
    thumb = (230, 230)
    row_h = thumb[1] + 28
    left = 130
    sheet = Image.new("RGB", (left + len(cols) * thumb[0], 30 + min(4, len(sample_dirs)) * row_h), (248, 246, 239))
    draw = ImageDraw.Draw(sheet)
    for ci, lab in enumerate(labels):
        draw.text((left + ci * thumb[0] + 6, 8), lab[:24], fill=(20, 24, 28))
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


def make_figures(
    run_dir: Path,
    teacher_rows: list[dict[str, Any]],
    calib_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    sample_grid: dict[tuple[str, str], Path],
    sample_dirs: list[Path],
) -> None:
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "figures"
    style = {
        "seacache_fixed": ("#111111", "D"),
        "dynamic_refresh": ("#2878b5", "o"),
        "rare_probe": ("#8c5fbf", "^"),
    }

    eval_teacher = [r for r in teacher_rows if r["split"] == "eval" and math.isfinite(float(r["teacher_defect"])) and math.isfinite(float(r["cache_error"]))]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.scatter([float(r["teacher_defect"]) for r in eval_teacher], [float(r["cache_error"]) for r in eval_teacher], s=16, alpha=0.35, color="#2878b5")
    ax.set_xlabel("SEA-defect teacher")
    ax.set_ylabel("actual cache error")
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_yscale("log")
    ax.set_title("SEA-defect teacher vs actual cache error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "teacher_signal_vs_cache_error.png", dpi=160)
    plt.close(fig)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    bins = [r for r in calib_rows if r["split"] == "eval" and r["metric"] == "calibration_bin"]
    if bins:
        ax0.plot([0, 1], [0, 1], "--", color="#999")
        ax0.plot([float(r["score_mean"]) for r in bins], [float(r["empirical_positive_rate"]) for r in bins], "-o", color="#2878b5")
        ax0.set_xlabel("predicted refresh probability")
        ax0.set_ylabel("empirical refresh-needed rate")
        ax0.set_title("Calibration curve")
        ax0.grid(alpha=0.25)
    eval_rows = [r for r in teacher_rows if r["split"] == "eval" and "predictor_score" in r and math.isfinite(float(r["cache_error"]))]
    scores = np.asarray([float(r["predictor_score"]) for r in eval_rows], dtype=np.float64)
    labels = np.asarray([int(r["refresh_needed"]) for r in eval_rows], dtype=np.int64)
    order = np.argsort(scores)
    if len(scores):
        thresholds = np.unique(scores[order])
        tprs, fprs = [], []
        pos = max(1, int(labels.sum()))
        neg = max(1, int(len(labels) - labels.sum()))
        for th in thresholds:
            pred = scores >= th
            tp = int(np.logical_and(pred, labels == 1).sum())
            fp = int(np.logical_and(pred, labels == 0).sum())
            tprs.append(tp / pos)
            fprs.append(fp / neg)
        ax1.plot(fprs, tprs, color="#2878b5")
        ax1.plot([0, 1], [0, 1], "--", color="#999")
        auc = rank_auc(labels, scores)
        ax1.set_title(f"ROC curve (AUC={auc:.3f})")
        ax1.set_xlabel("false positive rate")
        ax1.set_ylabel("true positive rate")
        ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "predictor_calibration_curve.png", dpi=160)
    plt.close(fig)

    fig, (ax_calls, ax_wall) = plt.subplots(1, 2, figsize=(13.2, 5.2))
    for base in sorted({r["base_method"] for r in budget_rows}):
        rs = sorted([r for r in budget_rows if r["base_method"] == base], key=lambda r: float(r["actual_full_calls"]))
        c, m = style.get(base, ("#555555", "o"))
        ax_calls.plot([float(r["actual_full_calls"]) for r in rs], [float(r["psnr_mean"]) for r in rs], marker=m, color=c, lw=1.8, label=base)
        wall_points = [(float(r["wall_speedup_vs_100_mean"]), float(r["psnr_mean"])) for r in rs if math.isfinite(float(r["wall_speedup_vs_100_mean"]))]
        if wall_points:
            ax_wall.scatter([x for x, _ in wall_points], [y for _, y in wall_points], color=c, marker=m, s=44, label=base)
    ax_calls.set_xlabel("actual fresh full transformer calls")
    ax_calls.set_ylabel("PSNR to 100-step vanilla (dB)")
    ax_calls.set_title("Quality vs actual fresh calls")
    ax_calls.grid(alpha=0.25)
    ax_wall.set_xlabel("measured wall-clock speedup vs 100-step vanilla")
    ax_wall.set_ylabel("PSNR to 100-step vanilla (dB)")
    ax_wall.set_title("Wall-clock vs quality")
    ax_wall.grid(alpha=0.25)
    ax_wall.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "dynamic_vs_fixed_seacache_frontier.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.28 * len({r['method'] for r in trace_rows}) + 1)))
    methods = sorted({r["method"] for r in trace_rows})
    ymap = {m: i for i, m in enumerate(methods)}
    for row in trace_rows:
        color = style.get(row["base_method"], ("#555", "o"))[0]
        ax.scatter(int(row["step"]), ymap[row["method"]], s=20, color=color if row["decision"] == "fresh_eval" else "#c8c8c8", alpha=0.85)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_xlabel("step")
    ax.set_title("Refresh schedule raster")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "refresh_schedule_raster.png", dpi=170)
    plt.close(fig)

    make_image_grid(
        sample_grid,
        sample_dirs,
        ["vanilla_100step", "default_flux", "seacache_fixed", "dynamic_refresh"],
        ["100-step vanilla", "default FLUX", "fixed SeaCache", "best dynamic"],
        fig_dir / "sample_grid_dynamic_seacache.png",
    )


def write_outputs(
    run_dir: Path,
    teacher_rows: list[dict[str, Any]],
    calib_rows: list[dict[str, Any]],
    per_sample_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    call_rows: list[dict[str, Any]],
) -> None:
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
        "feature_bias",
        "feature_raw_rel_l1",
        "feature_prev_raw_rel_l1",
        "feature_step_frac",
        "feature_sigma",
        "feature_raw_x_step",
        "feature_delta_raw",
        "fixed_threshold",
        "fixed_refresh",
        "teacher_defect",
        "cache_error",
        "refresh_needed",
        "predictor_score",
    ]
    write_csv(run_dir / "metrics" / "teacher_label_dataset.csv", teacher_rows, teacher_fields)
    write_csv(
        run_dir / "metrics" / "predictor_auc_calibration.csv",
        calib_rows,
        ["split", "metric", "bin_index", "bin_lo", "bin_hi", "count", "score_mean", "empirical_positive_rate", "roc_auc", "brier", "ece", "label_threshold"],
    )
    frontier_rows = [r for r in budget_rows if r["base_method"] in ("seacache_fixed", "dynamic_refresh", "rare_probe")]
    write_csv(
        run_dir / "metrics" / "seacache_dynamic_frontier.csv",
        frontier_rows,
        [
            "method",
            "base_method",
            "threshold",
            "actual_full_calls",
            "cheap_prefix_calls_mean",
            "expensive_probe_calls_mean",
            "cached_calls_mean",
            "wall_sec_mean",
            "wall_speedup_vs_100_mean",
            "psnr_mean",
            "lpips_mean",
            "latent_rel_l2_mean",
            "n",
        ],
    )
    write_csv(
        run_dir / "metrics" / "per_sample_metrics.csv",
        per_sample_rows,
        [
            "sample",
            "prompt",
            "method",
            "base_method",
            "threshold",
            "causal",
            "actual_full_calls",
            "cheap_prefix_calls",
            "expensive_probe_calls",
            "cached_calls",
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
            "causal",
            "actual_full_calls",
            "cheap_prefix_calls_mean",
            "expensive_probe_calls_mean",
            "cached_calls_mean",
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
    trace_fields = [
        "sample",
        "method",
        "base_method",
        "step",
        "sigma",
        "scheduler_timestep",
        "raw_rel_l1",
        "prev_raw_rel_l1",
        "predictor_score",
        "decision",
        "cheap_prefix_calls_so_far",
        "expensive_probe_calls_so_far",
        "teacher_probe_defect",
    ]
    write_csv(run_dir / "metrics" / "runtime_traces" / "all_runtime_traces.csv", trace_rows, trace_fields)
    for method in sorted({r["method"] for r in trace_rows}):
        write_csv(run_dir / "metrics" / "runtime_traces" / f"{method}.csv", [r for r in trace_rows if r["method"] == method], trace_fields)
    write_csv(run_dir / "metrics" / "call_counter_audit.csv", call_rows, ["sample", "method", "actual_full_calls", "cheap_prefix_calls", "expensive_probe_calls", "cached_calls", "valid", "note"])
    leak_seen = {}
    for row in leakage_rows:
        leak_seen.setdefault(row["method"], row)
    write_csv(run_dir / "metrics" / "leakage_audit.csv", list(leak_seen.values()), ["method", "saved_vanilla_velocities_used_for_decision", "saved_vanilla_velocities_used_for_update", "valid_causal", "note"])


def write_report(
    run_dir: Path,
    default_steps: int,
    teacher_rows: list[dict[str, Any]],
    calib_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    call_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    label_threshold: float,
    rare_probe_run: bool,
) -> None:
    eval_teacher = [r for r in teacher_rows if r["split"] == "eval" and math.isfinite(float(r["teacher_defect"])) and math.isfinite(float(r["cache_error"]))]
    teacher_corr = fsd.spearman_corr([float(r["teacher_defect"]) for r in eval_teacher], [float(r["cache_error"]) for r in eval_teacher]) if eval_teacher else float("nan")
    best_fixed = best_by_base(budget_rows, "seacache_fixed", 50)
    best_dynamic = best_by_base(budget_rows, "dynamic_refresh", 50)
    best_rare = best_by_base(budget_rows, "rare_probe", 50) if rare_probe_run else None
    beat_fixed = bool(best_dynamic and best_fixed and float(best_dynamic["psnr_mean"]) > float(best_fixed["psnr_mean"]) and float(best_dynamic["wall_speedup_vs_100_mean"]) >= float(best_fixed["wall_speedup_vs_100_mean"]))
    summary_row = next((r for r in calib_rows if r["split"] == "eval" and r["metric"] == "summary"), None)

    def table(headers: list[str], rows: list[list[Any]]) -> str:
        head = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        return f"<table>{head}{body}</table>"

    def point(row: dict[str, Any] | None) -> str:
        if not row:
            return "n/a"
        return f"{float(row['psnr_mean']):.2f} dB @ {int(row['actual_full_calls'])} calls / {float(row['wall_speedup_vs_100_mean']):.2f}x wall"

    method_rows = table(
        ["method", "calls", "prefix", "probes", "speedup", "PSNR", "LPIPS"],
        [
            [
                r["method"],
                r["actual_full_calls"],
                f"{float(r['cheap_prefix_calls_mean']):.1f}",
                f"{float(r['expensive_probe_calls_mean']):.1f}",
                f"{float(r['wall_speedup_vs_100_mean']):.2f}x wall / {float(r['speedup_vs_100']):.2f}x calls",
                f"{float(r['psnr_mean']):.2f}",
                f"{float(r['lpips_mean']):.4f}",
            ]
            for r in budget_rows
            if r["base_method"] in ("seacache_fixed", "dynamic_refresh", "rare_probe")
        ],
    )
    matched_targets = [80, 60, 50, 40, 33, 28, 25, 20]
    matched = []
    for target in matched_targets:
        for base in ["seacache_fixed", "dynamic_refresh"] + (["rare_probe"] if rare_probe_run else []):
            row = best_by_base(budget_rows, base, target)
            if row:
                matched.append([target, base, row["method"], row["actual_full_calls"], f"{float(row['psnr_mean']):.2f}", f"{float(row['lpips_mean']):.4f}", f"{float(row['wall_speedup_vs_100_mean']):.2f}x"])
    matched_rows = table(["target calls", "base", "best point", "actual calls", "PSNR", "LPIPS", "wall"], matched)
    call_table = table(
        ["sample", "method", "actual full", "prefix", "probes", "cached", "valid", "note"],
        [[r["sample"], r["method"], r["actual_full_calls"], r["cheap_prefix_calls"], r["expensive_probe_calls"], r["cached_calls"], r["valid"], r["note"]] for r in call_rows[:120]],
    )
    leak_seen = {}
    for row in leakage_rows:
        leak_seen.setdefault(row["method"], row)
    leak_table = table(
        ["method", "decision leakage?", "update leakage?", "causal valid?", "note"],
        [[r["method"], r["saved_vanilla_velocities_used_for_decision"], r["saved_vanilla_velocities_used_for_update"], r["valid_causal"], r["note"]] for r in leak_seen.values()],
    )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>E55 — Distilled SEA-defect for dynamic SeaCache refresh</title>
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
.warn{{border-left:5px solid #b43b2f;background:#fff6e8;padding:12px 16px;border-radius:8px}}
</style></head><body><main>
<h1>E55 — Distilled SEA-defect for dynamic SeaCache refresh</h1>
<p class="lede">Use expensive SEA-defect as an offline teacher to fit a cheap, prefix-free, causal dynamic SeaCache refresh gate. Primary comparison is fixed SeaCache vs dynamic SeaCache by actual fresh full transformer calls and measured wall-clock.</p>
<div class="verdict">
<div class="card"><span>Dynamic beats fixed SeaCache?</span><b>{beat_fixed}</b><small>dynamic {point(best_dynamic)} vs fixed {point(best_fixed)}</small></div>
<div class="card"><span>Teacher correlates with cache error?</span><b>{teacher_corr:.3f}</b><small>Spearman on eval split</small></div>
<div class="card"><span>Predictor AUC</span><b>{float(summary_row['roc_auc']) if summary_row else float('nan'):.3f}</b><small>eval split, label threshold {label_threshold:.4g}</small></div>
</div>
<div class="warn"><b>Corrected E54 context:</b> live SEA-defect jump was quality-promising but wall-clock negative. E55 therefore distills SEA-defect into a cheap refresh controller instead of using it live as the method.</div>

<h2>Executive Summary</h2>
<p>Default FLUX uses <b>{default_steps}</b> steps programmatically. The question here is not whether SEA-defect is useful as an expensive online controller; E54 already showed that live prefix probes kill wall-clock. The E55 question is whether that expensive signal can supervise a cheap refresh gate. At the inspected ~50-call range, dynamic SeaCache is <b>{point(best_dynamic)}</b> versus fixed SeaCache <b>{point(best_fixed)}</b>. Rare-probe was {'run as a small ablation' if rare_probe_run else 'not run in this pass'}.</p>

<h2>E54 Recap</h2>
<p>E54 corrected result: SeaCache still beats jump controllers at matched calls. The only promising branch was SEA-defect, but its live prefix computation was wall-clock negative. That makes dynamic refresh the plausible follow-up rather than dynamic jumping.</p>

<h2>Method</h2>
<p>Offline teacher rows are captured on the vanilla 100-step path. At each step the experiment logs the cheap SeaCache raw relative-L1 feature, step context, the expensive SEA-defect teacher, and actual post-hoc cache error from cached-vs-fresh next-latent predictions. A simple logistic gate is fit on train prompts and evaluated on held-out prompts. The online gate uses only cheap current-step features and never calls the expensive teacher.</p>

<h2>Figures</h2>
<figure><img src="figures/teacher_signal_vs_cache_error.png"><figcaption>SEA-defect teacher vs actual cache error.</figcaption></figure>
<figure><img src="figures/predictor_calibration_curve.png"><figcaption>Predictor calibration curve and ROC on the eval split.</figcaption></figure>
<figure><img src="figures/dynamic_vs_fixed_seacache_frontier.png"><figcaption>Fixed SeaCache vs dynamic SeaCache frontier: calls and wall-clock.</figcaption></figure>
<figure><img src="figures/refresh_schedule_raster.png"><figcaption>Refresh schedule raster for fixed and dynamic policies.</figcaption></figure>
<figure><img src="figures/sample_grid_dynamic_seacache.png"><figcaption>100-step vanilla, default FLUX, fixed SeaCache, best dynamic SeaCache.</figcaption></figure>

<h2>Tables</h2>
<h3>Method Summary</h3>{method_rows}
<h3>Best Point Per Matched Call Range</h3>{matched_rows}
<h3>Call-Counter Audit</h3>{call_table}
<h3>Leakage Audit</h3>{leak_table}
</main></body></html>"""
    report_path = run_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    report_path.write_text(fsd.embed_local_images(html, report_path), encoding="utf-8")

    summary = {
        "title": "E55 — Distilled SEA-defect for dynamic SeaCache refresh",
        "default_steps": default_steps,
        "best_fixed": best_fixed,
        "best_dynamic": best_dynamic,
        "best_rare_probe": best_rare,
        "beat_fixed_seacache": beat_fixed,
        "teacher_spearman": teacher_corr,
        "predictor_eval_auc": float(summary_row["roc_auc"]) if summary_row else float("nan"),
        "label_threshold": label_threshold,
        "rare_probe_run": rare_probe_run,
        "run_dir": str(run_dir),
    }
    fsd.write_json(run_dir / "reports" / "summary.json", summary)
    (run_dir / "reports" / "summary.md").write_text(
        f"# E55 — Distilled SEA-defect for dynamic SeaCache refresh\n\n"
        f"- Dynamic beats fixed SeaCache: **{beat_fixed}**\n"
        f"- Teacher/cache-error Spearman: **{teacher_corr:.3f}**\n"
        f"- Predictor eval AUC: **{float(summary_row['roc_auc']) if summary_row else float('nan'):.3f}**\n"
        f"- Rare-probe run: **{rare_probe_run}**\n"
        f"- Report: `{run_dir / 'report.html'}`\n",
        encoding="utf-8",
    )


def write_manifest(run_dir: Path, args: argparse.Namespace) -> None:
    files = [str(p.relative_to(run_dir)) for p in sorted(run_dir.rglob("*")) if p.is_file()]
    fsd.write_json(run_dir / "artifacts_manifest.json", {
        "experiment": "E55_flux_dynamic_seacache_refresh",
        "created": dt.datetime.now().isoformat(),
        "args": vars(args),
        "files": files,
    })


def dynamic_specs(args: argparse.Namespace) -> list[DynamicPolicySpec]:
    specs = [DynamicPolicySpec("dynamic_refresh", th) for th in args.dynamic_thresholds]
    if args.run_rare_probe:
        for sea_th in args.seacache_thresholds[:2]:
            specs.append(DynamicPolicySpec("rare_probe", args.rare_probe_dynamic_threshold, sea_threshold=sea_th, probe_band=0.2, probe_teacher_threshold=args.rare_probe_teacher_threshold))
    return specs


def run(args: argparse.Namespace) -> None:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"{ts}__flux_dynamic_seacache_refresh{'__smoke' if args.smoke else ''}"
    ensure_tree(run_dir)
    print(f"[e55] run_dir={run_dir}", flush=True)

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
    sample_grid: dict[tuple[str, str], Path] = {}

    counter = CallCounter(pipe)
    try:
        for si, sd in enumerate(sample_dirs):
            meta = fsd.read_json(sd / "metadata.json")
            prompt = meta["prompt"]
            seed = int(meta["seed"])
            split = "eval" if is_eval_sample(si, len(sample_dirs)) else "train"
            reference_wall_sec = float(meta.get("runtime_sec", 0.0) or 0.0)
            prep = prep_inputs(pipe, prompt, seed, N, args.height, args.width, args.guidance, args.max_sequence_length, args.device)
            print(f"[e55] teacher capture sample={sd.name} split={split}", flush=True)
            teacher_state = install_teacher_capture_forward(pipe, sd.name, split, prompt, seed, args.teacher_seacache_threshold, N)
            counter.reset()
            lat, img, _ = e53.live_generate(pipe, prompt, seed, N, args.height, args.width, args.guidance, args.max_sequence_length, args.device)
            teacher_state["restore"]()
            if counter.n != N:
                raise RuntimeError(f"teacher capture expected {N} wrapper calls, got {counter.n}")
            teacher_rows.extend(teacher_state["rows"])
            sample_grid[(sd.name, "vanilla_100step")] = sd / "final.png"
            del lat, img

        model, calib_rows, label_threshold = fit_predictor_from_rows(teacher_rows, args.label_quantile)
        fsd.write_json(run_dir / "reports" / "predictor_model.json", model)

        specs = dynamic_specs(args)
        for si, sd in enumerate(sample_dirs):
            meta = fsd.read_json(sd / "metadata.json")
            prompt = meta["prompt"]
            seed = int(meta["seed"])
            split = "eval" if is_eval_sample(si, len(sample_dirs)) else "train"
            if split != "eval":
                continue
            reference_wall_sec = float(meta.get("runtime_sec", 0.0) or 0.0)
            vanilla_img = Image.open(sd / "final.png").convert("RGB")
            vanilla_lat = fsd.load_step_tensor(sd / "latents", N)
            print(f"[e55] eval sample={sd.name}", flush=True)

            # default FLUX baseline on eval split.
            counter.reset()
            d_lat, d_img, d_wall = e53.live_generate(pipe, prompt, seed, default_steps, args.height, args.width, args.guidance, args.max_sequence_length, args.device)
            d_png = run_dir / "samples" / f"{sd.name}_default_flux.png"
            d_img.save(d_png)
            dm = bank.image_metrics(d_img, vanilla_img, prompt)
            dll = fsd.latent_metrics(d_lat, vanilla_lat)["latent_rel_l2"]
            per_sample_rows.append(metric_row(sd.name, prompt, "default_flux", "default_flux", "", True, counter.n, 0, 0, 0, d_wall, dll, dm, d_png, reference_wall_sec))
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
                "valid": True,
                "note": "pipeline default num_inference_steps",
            })

            # fixed SeaCache sweep.
            for th in args.seacache_thresholds:
                state = fsd.install_seacache_forward(pipe, th, N)
                counter.reset()
                lat, img, wall = e53.live_generate(pipe, prompt, seed, N, args.height, args.width, args.guidance, args.max_sequence_length, args.device)
                counted = counter.n
                state["restore"]()
                fresh = int(state["fresh_evals"])
                cached = int(state["cached_evals"])
                png = run_dir / "samples" / f"{sd.name}_seacache_fixed_th{th:g}.png"
                img.save(png)
                metrics = bank.image_metrics(img, vanilla_img, prompt)
                ll = fsd.latent_metrics(lat, vanilla_lat)["latent_rel_l2"]
                per_sample_rows.append(metric_row(sd.name, prompt, f"seacache_fixed_th{th:g}", "seacache_fixed", th, True, fresh, 0, 0, cached, wall, ll, metrics, png, reference_wall_sec))
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
                    "valid": counted == N,
                    "note": f"wrapper calls={counted}; fresh={fresh}; cached={cached}",
                })
                sample_grid.setdefault((sd.name, "seacache_fixed"), png)

            # dynamic policies.
            for spec in specs:
                state = install_dynamic_refresh_forward(pipe, model, spec, N)
                counter.reset()
                lat, img, wall = e53.live_generate(pipe, prompt, seed, N, args.height, args.width, args.guidance, args.max_sequence_length, args.device)
                counted = counter.n
                state["restore"]()
                fresh = int(state["fresh_evals"])
                cached = int(state["cached_evals"])
                prefix = int(state["cheap_prefix_calls"])
                probes = int(state["expensive_probe_calls"])
                png = run_dir / "samples" / f"{sd.name}_{spec.key}.png"
                img.save(png)
                metrics = bank.image_metrics(img, vanilla_img, prompt)
                ll = fsd.latent_metrics(lat, vanilla_lat)["latent_rel_l2"]
                per_sample_rows.append(metric_row(sd.name, prompt, spec.key, spec.method, spec.threshold, True, fresh, prefix, probes, cached, wall, ll, metrics, png, reference_wall_sec))
                trace_rows.extend({"sample": sd.name, **row} for row in state["step_traces"])
                leakage_rows.append({
                    "method": spec.method,
                    "saved_vanilla_velocities_used_for_decision": False,
                    "saved_vanilla_velocities_used_for_update": False,
                    "valid_causal": True,
                    "note": "online gate uses only current cheap SeaCache features",
                })
                call_rows.append({
                    "sample": sd.name,
                    "method": spec.key,
                    "actual_full_calls": fresh,
                    "cheap_prefix_calls": prefix,
                    "expensive_probe_calls": probes,
                    "cached_calls": cached,
                    "valid": counted == N,
                    "note": f"wrapper calls={counted}; dynamic refresh policy",
                })
                if spec.method == "dynamic_refresh":
                    sample_grid.setdefault((sd.name, "dynamic_refresh"), png)

    finally:
        counter.restore()
        del pipe
        torch.cuda.empty_cache()

    budget_rows = aggregate_budget_rows(per_sample_rows)
    write_outputs(run_dir, teacher_rows, calib_rows, per_sample_rows, budget_rows, trace_rows, leakage_rows, call_rows)
    make_figures(run_dir, teacher_rows, calib_rows, budget_rows, trace_rows, sample_grid, sample_dirs)
    write_report(run_dir, default_steps, teacher_rows, calib_rows, budget_rows, call_rows, leakage_rows, label_threshold, args.run_rare_probe)
    write_manifest(run_dir, args)
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
    ap.add_argument("--trajectory-root", default=str(REPO_ROOT / "outputs" / "flux_dynamic_seacache_refresh" / "trajectories"))
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--max-sequence-length", type=int, default=512)
    ap.add_argument("--teacher-seacache-threshold", type=float, default=0.15)
    ap.add_argument("--label-quantile", type=float, default=0.75)
    ap.add_argument("--seacache-thresholds", type=lambda s: parse_csv_list(s, float), default=[0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    ap.add_argument("--dynamic-thresholds", type=lambda s: parse_csv_list(s, float), default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--run-rare-probe", action="store_true")
    ap.add_argument("--rare-probe-dynamic-threshold", type=float, default=0.5)
    ap.add_argument("--rare-probe-teacher-threshold", type=float, default=0.02)
    ap.add_argument("--force-capture", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.num_samples = min(args.num_samples, 2)
        args.seacache_thresholds = args.seacache_thresholds[:3]
        args.dynamic_thresholds = args.dynamic_thresholds[:3]
    return args


if __name__ == "__main__":
    run(parse_args())
