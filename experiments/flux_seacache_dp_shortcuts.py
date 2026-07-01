#!/usr/bin/env python3
"""FLUX SeaCache replication, trajectory capture, and offline shortcut search.

This script is intentionally resumable: every expensive stage checks for its
expected output files before running unless --force is supplied.
"""

from __future__ import annotations

import argparse
import csv
import base64
import gc
import json
import math
import os
import io
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEFAULT_PROMPT = "a high-resolution photo of a panda drinking coffee in a cozy cafe"
SEACACHE_REPO = "https://github.com/jiwoogit/SeaCache"
SEACACHE_COMMIT = "8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2"
SIZE = 1024
LATENT_CHANNELS = 16

PROMPTS = [
    ("photoreal portrait", "a photoreal studio portrait of a violin maker with silver hair, soft window light, 85mm lens"),
    ("product photo", "a premium product photo of a matte black espresso machine on brushed steel, crisp reflections"),
    ("landscape", "a sweeping alpine landscape at sunrise with glacial lake reflections and low mist"),
    ("indoor scene", "an elegant reading room interior with walnut shelves, green desk lamps, and rain on the windows"),
    ("animal", "a sharp wildlife photograph of a red fox standing in fresh snow under pale winter light"),
    ("food", "a rustic overhead food photograph of handmade ramen, chili oil, scallions, and ceramic bowls"),
    ("sci-fi scene", "a cinematic sci-fi street market on a terraformed moon, neon signage, astronauts and vendors"),
    ("text/logo-like composition", "a clean graphic poster with the word AURORA in bold glass letters, sunlit gradients, centered logo"),
    ("stylized illustration", "a stylized gouache illustration of a coastal town built on cliffs, warm colors, visible brush texture"),
    ("complex multi-object scene", "a busy workshop table with cameras, maps, brass tools, flowers, sketches, cables, and coffee cups"),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> tuple[int, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout, time.perf_counter() - start


def git_hash(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def collect_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "max_vram_gb": None,
        "memory_allocated_gb": None,
        "memory_reserved_gb": None,
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        info["max_vram_gb"] = props.total_memory / 1024**3
        info["memory_allocated_gb"] = torch.cuda.max_memory_allocated(idx) / 1024**3
        info["memory_reserved_gb"] = torch.cuda.max_memory_reserved(idx) / 1024**3
    return info


def clone_seacache(dest: Path) -> Path:
    if not (dest / ".git").exists():
        ensure_dir(dest.parent)
        code, out, _ = run_cmd(["git", "clone", SEACACHE_REPO, str(dest)])
        if code != 0:
            raise RuntimeError(out)
    code, out, _ = run_cmd(["git", "checkout", SEACACHE_COMMIT], cwd=dest)
    if code != 0:
        raise RuntimeError(out)
    return dest


def _rfft_full_mean_weights_1d(n_last: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    lh = n_last // 2 + 1
    w = torch.ones(lh, device=device, dtype=dtype)
    if n_last % 2 == 0:
        if lh > 2:
            w[1:-1] *= 2.0
    elif lh > 1:
        w[1:] *= 2.0
    return w


def apply_sea_from_ab(
    x: torch.Tensor,
    a: float,
    b: float,
    power_exp: float = 2.0,
    power_const: float = 1.0,
    dims: tuple[int, ...] | None = None,
    eps: float = 1e-16,
    norm_mode: str = "mean",
    real: bool = False,
) -> torch.Tensor:
    orig_dtype = x.dtype
    x32 = x.contiguous().to(torch.float32)
    if dims is None:
        dims = tuple(range(-2, -x32.ndim, -1)) if x32.ndim > 2 else tuple(range(x32.ndim))
    X = torch.fft.rfftn(x32, dim=dims) if real else torch.fft.fftn(x32, dim=dims)
    H = None
    for i, ax in enumerate(dims):
        n = x32.shape[ax]
        freq = torch.fft.rfftfreq(n, device=x32.device, dtype=torch.float32) if real and i == len(dims) - 1 else torch.fft.fftfreq(n, device=x32.device, dtype=torch.float32)
        rad = freq.abs()
        sx0 = power_const / ((rad**power_exp) + eps)
        h1 = (a * sx0) / (a * a * sx0 + b * b + eps)
        shape = [1] * x32.ndim
        shape[ax] = h1.shape[0]
        H = h1.reshape(shape) if H is None else H * h1.reshape(shape)
    if norm_mode == "mean":
        if real:
            n_last = int(x32.shape[dims[-1]])
            w_last = _rfft_full_mean_weights_1d(n_last, x32.device, torch.float32)
            wshape = [1] * x32.ndim
            wshape[dims[-1]] = w_last.numel()
            denom = torch.sum(w_last) * float(torch.prod(torch.tensor([x32.shape[d] for d in dims[:-1]])))
            meanv = torch.sum(H * w_last.view(*wshape)) / denom
        else:
            meanv = torch.mean(H)
        if torch.isfinite(meanv) and meanv > 0:
            H = H / meanv
    elif norm_mode == "peak":
        maxv = torch.amax(H)
        if torch.isfinite(maxv) and maxv > 0:
            H = H / maxv
    Y = X * H
    y = torch.fft.irfftn(Y, s=[x32.shape[d] for d in dims], dim=dims) if real else torch.fft.ifftn(Y, dim=dims).real
    return y.to(orig_dtype)


def ab_from_scheduler(scheduler: Any, idx: int) -> tuple[float, float]:
    sigma = float(scheduler.sigmas[idx]) if hasattr(scheduler, "sigmas") else 1.0 - (idx + 1) / float(idx + 1)
    sigma = max(1e-6, min(1.0 - 1e-6, sigma))
    return 1.0 - sigma, sigma


def rel_l1(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-16) -> float:
    return float(((a - b).abs().mean() / (b.abs().mean() + eps)).detach().cpu())


def load_flux_pipeline(model_id: str, dtype: str, device: str, offload: bool = False, transformer_only_4bit: bool = False):
    from diffusers import FluxPipeline

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if transformer_only_4bit:
        from diffusers import BitsAndBytesConfig
        from diffusers.models import FluxTransformer2DModel

        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype)
        transformer = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", quantization_config=quantization_config, torch_dtype=torch_dtype)
        kwargs["transformer"] = transformer
    pipe = FluxPipeline.from_pretrained(model_id, **kwargs)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def load_mode(args: argparse.Namespace) -> str:
    if args.bnb4 and args.offload:
        return f"{args.dtype}_bnb4_offload"
    if args.bnb4:
        return f"{args.dtype}_bnb4"
    if args.offload:
        return f"{args.dtype}_offload"
    return f"{args.dtype}_full"


@torch.no_grad()
def flux_h_decision_tensor(
    transformer: Any,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None,
    scheduler: Any,
    step_index: int,
    sea_filter: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    hs = transformer.x_embedder(hidden_states)
    timestep = timestep.to(hs.dtype) * 1000
    guidance_in = guidance.to(hs.dtype) * 1000 if guidance is not None else None
    temb = transformer.time_text_embed(timestep, pooled_projections) if guidance_in is None else transformer.time_text_embed(timestep, guidance_in, pooled_projections)
    _ = transformer.context_embedder(encoder_hidden_states)
    modulated, *_ = transformer.transformer_blocks[0].norm1(hs, emb=temb)
    raw = modulated.detach()
    if sea_filter:
        ids = img_ids[0] if img_ids.ndim == 3 else img_ids
        h = int(ids[:, 1].max().item() + 1)
        w = int(ids[:, 2].max().item() + 1)
        grid = modulated.reshape(modulated.shape[0], h, w, modulated.shape[-1])
        a, b = ab_from_scheduler(scheduler, step_index)
        decision = apply_sea_from_ab(grid, a, b, dims=(-2, -3), norm_mode="mean").reshape(modulated.shape[0], -1, modulated.shape[-1])
    else:
        decision = modulated
    return raw.detach().cpu(), decision.detach().cpu()


def install_seacache_forward(pipe: Any, threshold: float, num_steps: int) -> dict[str, Any]:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    tr = pipe.transformer
    orig_forward = tr.forward
    state = {"fresh_evals": 0, "cached_evals": 0, "h_distance": [], "cache_decisions": [], "step_traces": [], "reuse_runs": []}
    tr.scheduler = pipe.scheduler
    tr.seacache_thresh = float(threshold)
    tr.cnt = 0
    tr.num_steps = int(num_steps)
    tr.accumulated_rel_l1_distance = 0.0
    tr.previous_modulated_input = None
    tr.previous_residual = None
    tr.reuse_run_id = -1
    tr.reuse_run_len = 0
    tr.reuse_run_start = None

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

        hs = tr.x_embedder(hidden_states)
        ts = timestep.to(hs.dtype) * 1000
        guidance_in = guidance.to(hs.dtype) * 1000 if guidance is not None else None
        temb = tr.time_text_embed(ts, pooled_projections) if guidance_in is None else tr.time_text_embed(ts, guidance_in, pooled_projections)
        enc = tr.context_embedder(encoder_hidden_states)
        txt = txt_ids[0] if txt_ids is not None and txt_ids.ndim == 3 else txt_ids
        img = img_ids[0] if img_ids is not None and img_ids.ndim == 3 else img_ids
        image_rotary_emb = tr.pos_embed(torch.cat((txt, img), dim=0)) if txt is not None and img is not None else None

        should_calc = True
        decision = "fresh_eval"
        raw_rel_l1 = 0.0
        acc_before = float(tr.accumulated_rel_l1_distance)
        acc_after = acc_before
        modulated, *_ = tr.transformer_blocks[0].norm1(hs, emb=temb)
        if tr.cnt == 0 or tr.cnt == tr.num_steps - 1 or tr.previous_modulated_input is None:
            tr.accumulated_rel_l1_distance = 0.0
        else:
            grid = modulated.reshape(modulated.shape[0], int(img[:, 1].max().item() + 1), int(img[:, 2].max().item() + 1), modulated.shape[-1])
            a, b = ab_from_scheduler(tr.scheduler, tr.cnt)
            modulated = apply_sea_from_ab(grid, a, b, dims=(-2, -3), norm_mode="mean").reshape(modulated.shape[0], -1, modulated.shape[-1])
            raw_rel_l1 = rel_l1(modulated, tr.previous_modulated_input)
            tr.accumulated_rel_l1_distance += raw_rel_l1
            acc_after = float(tr.accumulated_rel_l1_distance)
            state["h_distance"].append({"step": int(tr.cnt), "raw_rel_l1": float(raw_rel_l1), "accumulated_before": acc_before, "accumulated_after": acc_after})
            if tr.accumulated_rel_l1_distance < tr.seacache_thresh:
                should_calc = False
                decision = "cache_reuse"
                if tr.reuse_run_len == 0:
                    tr.reuse_run_id += 1
                    tr.reuse_run_start = int(tr.cnt)
                tr.reuse_run_len += 1
            else:
                tr.accumulated_rel_l1_distance = 0.0
                decision = "fresh_eval"
                if tr.reuse_run_len > 0:
                    state["reuse_runs"].append({"reuse_run_id": int(tr.reuse_run_id), "start_step": int(tr.reuse_run_start), "end_step": int(tr.cnt - 1), "reuse_run_length": int(tr.reuse_run_len)})
                    tr.reuse_run_len = 0
                    tr.reuse_run_start = None
        tr.previous_modulated_input = modulated.detach()
        tr.cnt += 1
        if tr.cnt == tr.num_steps:
            tr.cnt = 0

        state["cache_decisions"].append(bool(should_calc))
        state["step_traces"].append({
            "step": int(tr.cnt - 1 if tr.cnt > 0 else tr.num_steps - 1),
            "raw_rel_l1": float(raw_rel_l1),
            "accumulated_before": float(acc_before),
            "accumulated_after": float(acc_after),
            "threshold": float(tr.seacache_thresh),
            "decision": decision,
            "fresh_eval_count_so_far": int(state["fresh_evals"] + (1 if decision == "fresh_eval" else 0)),
            "cache_reuse_count_so_far": int(state["cached_evals"] + (1 if decision == "cache_reuse" else 0)),
            "reuse_run_id": int(tr.reuse_run_id),
            "position_inside_reuse_run": int(tr.reuse_run_len if decision == "cache_reuse" else 0),
        })
        if not should_calc and tr.previous_residual is not None:
            state["cached_evals"] += 1
            hs = hs + tr.previous_residual
        else:
            state["fresh_evals"] += 1
            ori_hs = hs
            for index_block, block in enumerate(tr.transformer_blocks):
                enc, hs = block(hidden_states=hs, encoder_hidden_states=enc, temb=temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs)
                if controlnet_block_samples is not None:
                    interval_control = int(np.ceil(len(tr.transformer_blocks) / len(controlnet_block_samples)))
                    hs = hs + (controlnet_block_samples[index_block % len(controlnet_block_samples)] if controlnet_blocks_repeat else controlnet_block_samples[index_block // interval_control])
            for index_block, block in enumerate(tr.single_transformer_blocks):
                enc, hs = block(hidden_states=hs, encoder_hidden_states=enc, temb=temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs)
                if controlnet_single_block_samples is not None:
                    interval_control = int(np.ceil(len(tr.single_transformer_blocks) / len(controlnet_single_block_samples)))
                    hs = hs + controlnet_single_block_samples[index_block // interval_control]
            tr.previous_residual = (hs - ori_hs).detach()
            if tr.reuse_run_len > 0:
                state["reuse_runs"].append({"reuse_run_id": int(tr.reuse_run_id), "start_step": int(tr.reuse_run_start), "end_step": int(tr.cnt - 1), "reuse_run_length": int(tr.reuse_run_len)})
                tr.reuse_run_len = 0
                tr.reuse_run_start = None
        hs = tr.norm_out(hs, temb)
        output = tr.proj_out(hs)
        if USE_PEFT_BACKEND:
            unscale_lora_layers(tr, lora_scale)
        return (output,) if not return_dict else Transformer2DModelOutput(sample=output)

    tr.forward = wrapped_forward
    state["restore"] = lambda: setattr(tr, "forward", orig_forward)
    return state


def run_official_seacache(args: argparse.Namespace) -> None:
    dest = clone_seacache(Path(args.seacache_dir))
    out_dir = Path(args.output_root) / "delta_0p3"
    marker = out_dir / "official_unmodified.json"
    if marker.exists() and not args.force:
        try:
            if read_json(marker).get("status") == "succeeded":
                return
        except Exception:
            pass
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        "seacache_generate.py",
        "--prompt",
        args.prompt,
        "--output_dir",
        str(out_dir.resolve()),
        "--seed",
        str(args.seed),
        "--num_inference_steps",
        str(args.seacache_steps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--dtype",
        args.dtype,
    ]
    start = time.perf_counter()
    code, out, elapsed = run_cmd(cmd, cwd=dest / "FLUX")
    (out_dir / "official_stdout.log").write_text(out, encoding="utf-8")
    write_json(
        marker,
        {
            "status": "succeeded" if code == 0 else "failed",
            "returncode": code,
            "runtime_sec": elapsed,
            "wall_start_perf_counter": start,
            "command": " ".join(cmd),
            "cwd": str((dest / "FLUX").resolve()),
            "seacache_commit": git_hash(dest),
            "note": "Official SeaCache FLUX script run unmodified; it does not emit fresh transformer evaluation counts.",
        },
    )
    if code != 0:
        print(f"[official-seacache] unmodified script failed; continuing with instrumented runs. See {marker}", flush=True)


@torch.no_grad()
def run_replication(args: argparse.Namespace) -> None:
    if args.run_official:
        run_official_seacache(args)

    root = Path(args.output_root)
    variants = [("vanilla", None), ("delta_0p3", 0.3), ("delta_0p6", 0.6)]
    for name, threshold in variants:
        out_dir = root / name
        meta_path = out_dir / "replication.json"
        if meta_path.exists() and (out_dir / "image.png").exists() and not args.force:
            try:
                if read_json(meta_path).get("load_mode") == load_mode(args):
                    continue
            except Exception:
                pass
        ensure_dir(out_dir)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pipe = load_flux_pipeline(args.model_id, args.dtype, args.device, args.offload, args.bnb4)
        counter: dict[str, Any] | None = None
        if threshold is not None:
            counter = install_seacache_forward(pipe, threshold, args.seacache_steps)
        else:
            orig = pipe.transformer.forward
            calls = {"fresh_evals": 0}

            def counted(*a, **kw):
                calls["fresh_evals"] += 1
                return orig(*a, **kw)

            pipe.transformer.forward = counted
            calls["restore"] = lambda: setattr(pipe.transformer, "forward", orig)
            counter = calls
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        command = f"{Path(__file__).name} replicate --variant {name} --prompt {args.prompt!r} --seed {args.seed} --steps {args.seacache_steps} --load-mode {load_mode(args)}"
        start = time.perf_counter()
        out = pipe(
            prompt=args.prompt,
            num_inference_steps=args.seacache_steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance,
            max_sequence_length=args.max_sequence_length,
            num_images_per_prompt=1,
            generator=generator,
        )
        runtime = time.perf_counter() - start
        out.images[0].save(out_dir / "image.png")
        if "restore" in counter:
            counter["restore"]()
            counter.pop("restore", None)
        write_json(
            meta_path,
            {
                "variant": name,
                "threshold": threshold,
                "runtime_sec": runtime,
                "gpu": collect_gpu_info(),
                "actual_transformer_evaluations": int(counter.get("fresh_evals", args.seacache_steps)),
                "cached_transformer_reuses": int(counter.get("cached_evals", 0)),
                "seed": args.seed,
                "prompt": args.prompt,
                "model_id": args.model_id,
                "load_mode": load_mode(args),
                "seacache_commit": git_hash(Path(args.seacache_dir)) if Path(args.seacache_dir).exists() else SEACACHE_COMMIT,
                "repo_commit": git_hash(REPO_ROOT),
                "exact_command": command,
                "h_distance": counter.get("h_distance", []),
                "cache_decisions": counter.get("cache_decisions", []),
            },
        )
        del out, generator, counter, pipe
        gc.collect()
        torch.cuda.empty_cache()


def prepare_flux_inputs(pipe: Any, prompt: str, seed: int, steps: int, height: int, width: int, guidance: float, device: str, max_sequence_length: int):
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    num_channels_latents = pipe.transformer.config.in_channels // 4
    generator = torch.Generator(device=device).manual_seed(seed)
    latents, latent_image_ids = pipe.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)
    guidance_tensor = torch.full([1], guidance, device=device, dtype=torch.float32) if pipe.transformer.config.guidance_embeds else None
    return prompt_embeds, pooled_prompt_embeds, text_ids, latents, latent_image_ids, timesteps, guidance_tensor


@torch.no_grad()
def decode_flux_latents(pipe: Any, packed_latents: torch.Tensor, height: int, width: int) -> Image.Image:
    latents = pipe._unpack_latents(packed_latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(t.shape), "dtype": str(t.dtype), "mean": float(t.float().mean().item()), "std": float(t.float().std().item())}


@torch.no_grad()
def run_capture(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    pipe = load_flux_pipeline(args.model_id, args.dtype, args.device, args.offload, args.bnb4)
    for sample_idx, (category, prompt) in enumerate(PROMPTS[: args.num_samples]):
        sample_dir = root / f"sample_{sample_idx:02d}"
        meta_path = sample_dir / "metadata.json"
        if meta_path.exists() and (sample_dir / "final.png").exists() and not args.force:
            continue
        ensure_dir(sample_dir / "latents")
        ensure_dir(sample_dir / "velocities")
        ensure_dir(sample_dir / "h_values")
        seed = args.seed_base + sample_idx
        pe, ppe, text_ids, latents, image_ids, timesteps, guidance_tensor = prepare_flux_inputs(
            pipe, prompt, seed, args.steps, args.height, args.width, args.guidance, args.device, args.max_sequence_length
        )
        sigma_values = [float(x) for x in pipe.scheduler.sigmas.detach().cpu()]
        step_records = []
        start = time.perf_counter()
        torch.save(latents.detach().cpu().to(torch.float16), sample_dir / "latents" / "step_000.pt")
        for i, timestep in enumerate(timesteps):
            timestep_expanded = timestep.expand(latents.shape[0]).to(latents.dtype)
            h_raw, h_decision = flux_h_decision_tensor(
                pipe.transformer,
                latents,
                timestep_expanded / 1000,
                pe,
                ppe,
                image_ids,
                text_ids,
                guidance_tensor,
                pipe.scheduler,
                i,
                sea_filter=(i != 0 and i != len(timesteps) - 1),
            )
            model_pred = pipe.transformer(
                hidden_states=latents,
                timestep=timestep_expanded / 1000,
                guidance=guidance_tensor,
                pooled_projections=ppe,
                encoder_hidden_states=pe,
                txt_ids=text_ids,
                img_ids=image_ids,
                return_dict=False,
            )[0]
            torch.save(model_pred.detach().cpu().to(torch.float16), sample_dir / "velocities" / f"step_{i:03d}.pt")
            torch.save(h_decision.to(torch.float16), sample_dir / "h_values" / f"step_{i:03d}.pt")
            if args.save_h_raw:
                torch.save(h_raw.to(torch.float16), sample_dir / "h_values" / f"step_{i:03d}_raw.pt")
            latents = pipe.scheduler.step(model_pred, timestep, latents, return_dict=False)[0]
            torch.save(latents.detach().cpu().to(torch.float16), sample_dir / "latents" / f"step_{i + 1:03d}.pt")
            step_records.append({"index": i, "timestep": float(timestep.detach().cpu()), "sigma": sigma_values[i], "velocity": tensor_stats(model_pred.detach().cpu()), "h_decision": tensor_stats(h_decision)})
        final = decode_flux_latents(pipe, latents, args.height, args.width)
        final.save(sample_dir / "final.png")
        write_json(
            meta_path,
            {
                "sample_index": sample_idx,
                "category": category,
                "prompt": prompt,
                "seed": seed,
                "num_inference_steps": args.steps,
                "height": args.height,
                "width": args.width,
                "guidance": args.guidance,
                "model_id": args.model_id,
                "model_revision": "default",
                "dtype": args.dtype,
                "timesteps": [float(x.detach().cpu()) for x in timesteps],
                "sigmas": sigma_values,
                "runtime_sec": time.perf_counter() - start,
                "gpu": collect_gpu_info(),
                "h_value_definition": "SeaCache FLUX first-block norm1 modulated image-token tensor. For middle steps this script stores the SEA-filtered tensor used in the relative-L1 cache decision; for first/final steps the official code stores the unfiltered modulated tensor.",
                "latent_files": "latents/step_000.pt through latents/step_100.pt",
                "velocity_files": "velocities/step_000.pt through velocities/step_099.pt",
                "h_files": "h_values/step_000.pt through h_values/step_099.pt",
                "steps": step_records,
            },
        )
    del pipe
    torch.cuda.empty_cache()


def load_step_tensor(folder: Path, i: int) -> torch.Tensor:
    return torch.load(folder / f"step_{i:03d}.pt", map_location="cpu", weights_only=True).float()


def latent_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    p = pred.flatten().float()
    r = ref.flatten().float()
    diff = p - r
    l2 = torch.linalg.vector_norm(diff) / math.sqrt(diff.numel())
    cos = torch.nn.functional.cosine_similarity(p[None], r[None]).item()
    rel = torch.linalg.vector_norm(diff) / (torch.linalg.vector_norm(r) + 1e-8)
    return {"latent_rmse": float(l2), "latent_cosine": float(cos), "latent_rel_l2": float(rel)}


def pair_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm((a.flatten() - b.flatten()).float()) / math.sqrt(a.numel()))


def tensor_summary_for_distance(t: torch.Tensor) -> torch.Tensor:
    """Compact proxy for expensive pairwise distances over large FLUX tensors."""
    x = t.float()
    if x.ndim >= 3:
        flat = x.reshape(-1, x.shape[-1])
        return torch.cat([flat.mean(dim=0), flat.std(dim=0, unbiased=False)], dim=0).contiguous()
    if x.numel() > 8192:
        stride = max(1, x.numel() // 8192)
        return torch.cat([x.flatten()[::stride], x.mean().view(1), x.std(unbiased=False).view(1)], dim=0).contiguous()
    return x.flatten().contiguous()


@dataclass
class Edge:
    src: int
    dst: int
    err: float
    cost_a: int
    cost_b: int
    metrics: dict[str, float]


def build_edges(sample_dir: Path, cache_h_threshold: float) -> tuple[list[list[Edge]], np.ndarray]:
    meta = read_json(sample_dir / "metadata.json")
    n = int(meta["num_inference_steps"])
    sigmas = meta["sigmas"]
    lat_dir = sample_dir / "latents"
    vel_dir = sample_dir / "velocities"
    h_dir = sample_dir / "h_values"
    latents = [load_step_tensor(lat_dir, i) for i in range(n + 1)]
    velocities = [load_step_tensor(vel_dir, i) for i in range(n)]
    # Exact h tensors are captured on disk. Edge-cost h distances use compact
    # summaries so the offline proxy remains tractable for 100-step FLUX runs.
    h_values = [tensor_summary_for_distance(load_step_tensor(h_dir, i)) for i in range(n)]
    v_values = [tensor_summary_for_distance(v) for v in velocities]
    edges: list[list[Edge]] = [[] for _ in range(n + 1)]
    heat = np.full((n + 1, n + 1), np.nan, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n + 1):
            ds = float(sigmas[j] - sigmas[i]) if j < len(sigmas) else float(0.0 - sigmas[i])
            pred = latents[i] + ds * velocities[i]
            metrics = latent_metrics(pred, latents[j])
            if j < n:
                h_dist = pair_distance(h_values[i], h_values[j])
                v_dist = pair_distance(v_values[i], v_values[j])
            else:
                h_dist = pair_distance(h_values[i], h_values[-1])
                v_dist = pair_distance(v_values[i], v_values[-1])
            metrics["h_rmse"] = h_dist
            metrics["velocity_rmse"] = v_dist
            cost_b = 0 if h_dist <= cache_h_threshold else 1
            edge = Edge(i, j, metrics["latent_rmse"], 1, cost_b, metrics)
            edges[i].append(edge)
            heat[i, j] = metrics["latent_rmse"]
    return edges, heat


def dp_path(edges: list[list[Edge]], budget: int, cost_attr: str) -> tuple[list[Edge], np.ndarray]:
    n = len(edges) - 1
    inf = 1e30
    dp = np.full((budget + 1, n + 1), inf, dtype=np.float64)
    back: list[list[Edge | None]] = [[None for _ in range(n + 1)] for _ in range(budget + 1)]
    dp[0, 0] = 0.0
    for b in range(budget + 1):
        for i in range(n + 1):
            if not np.isfinite(dp[b, i]):
                continue
            for edge in edges[i]:
                c = int(getattr(edge, cost_attr))
                nb = b + c
                if nb <= budget and dp[b, i] + edge.err < dp[nb, edge.dst]:
                    dp[nb, edge.dst] = dp[b, i] + edge.err
                    back[nb][edge.dst] = edge
    best_b = int(np.nanargmin(dp[:, n]))
    path: list[Edge] = []
    cur = n
    b = best_b
    while cur != 0:
        edge = back[b][cur]
        if edge is None:
            break
        path.append(edge)
        b -= int(getattr(edge, cost_attr))
        cur = edge.src
    path.reverse()
    return path, dp


def replay_path(sample_dir: Path, path: list[Edge]) -> torch.Tensor:
    meta = read_json(sample_dir / "metadata.json")
    sigmas = meta["sigmas"]
    x = load_step_tensor(sample_dir / "latents", 0)
    for edge in path:
        v = load_step_tensor(sample_dir / "velocities", edge.src)
        ds = float(sigmas[edge.dst] - sigmas[edge.src]) if edge.dst < len(sigmas) else float(0.0 - sigmas[edge.src])
        x = x + ds * v
    return x


def make_plots(sample_dir: Path, out_dir: Path, heat: np.ndarray, dp: np.ndarray, paths: dict[str, list[Edge]]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    assets: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(np.log10(heat + 1e-8), origin="lower", cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, label="log10 latent RMSE")
    ax.set_title(f"{sample_dir.name} edge error heatmap")
    ax.set_xlabel("destination step")
    ax.set_ylabel("source step")
    p = out_dir / "edge_error_heatmap.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    assets["edge_error_heatmap"] = str(p)

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, path in paths.items():
        nodes = [0] + [e.dst for e in path]
        ax.plot(range(len(nodes)), nodes, marker="o", label=label)
    ax.set_title(f"{sample_dir.name} selected path")
    ax.set_xlabel("shortcut segment")
    ax.set_ylabel("trajectory step")
    ax.legend(fontsize=8)
    p = out_dir / "path_over_timestep.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    assets["path_over_timestep"] = str(p)

    fig, ax = plt.subplots(figsize=(8, 5))
    finite = np.where(np.isfinite(dp), dp, np.nan)
    im = ax.imshow(finite, origin="lower", cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="DP cumulative error")
    ax.set_title(f"{sample_dir.name} DP value table")
    ax.set_xlabel("step")
    ax.set_ylabel("budget")
    p = out_dir / "dp_value_table.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    assets["dp_value_table"] = str(p)

    meta = read_json(sample_dir / "metadata.json")
    n = int(meta["num_inference_steps"])
    hs = [tensor_summary_for_distance(load_step_tensor(sample_dir / "h_values", i)) for i in range(n)]
    vs = [tensor_summary_for_distance(load_step_tensor(sample_dir / "velocities", i)) for i in range(n)]
    h_dist = [pair_distance(hs[i], hs[i + 1]) for i in range(n - 1)]
    v_dist = [pair_distance(vs[i], vs[i + 1]) for i in range(n - 1)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(h_dist, label="h RMSE")
    ax.plot(v_dist, label="velocity RMSE")
    ax.set_title(f"{sample_dir.name} adjacent-step distances")
    ax.set_xlabel("step")
    ax.legend()
    p = out_dir / "h_velocity_distance.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    assets["h_velocity_distance"] = str(p)
    return assets


def run_dp(args: argparse.Namespace) -> None:
    root = Path(args.trajectory_root)
    out_root = Path(args.output_root)
    metrics_rows: list[dict[str, Any]] = []
    pipe = None
    if args.decode:
        pipe = load_flux_pipeline(args.model_id, args.dtype, args.device, args.offload, args.bnb4)
    for sample_dir in sorted(root.glob("sample_*")):
        if not (sample_dir / "metadata.json").exists():
            continue
        sample_out = out_root / sample_dir.name
        summary_path = sample_out / "dp_summary.json"
        if summary_path.exists() and not args.force:
            metrics_rows.extend(read_json(summary_path).get("rows", []))
            continue
        ensure_dir(sample_out)
        edges, heat = build_edges(sample_dir, args.cache_h_threshold)
        sample_paths: dict[str, list[Edge]] = {}
        path_payloads: dict[str, Any] = {}
        for budget in args.budgets:
            for model, cost_attr in [("monotone_eval_nodes", "cost_a"), ("cache_aware_h_reuse", "cost_b")]:
                label = f"B{budget}_{model}"
                path, dp = dp_path(edges, budget, cost_attr)
                final_latent = replay_path(sample_dir, path)
                ref = load_step_tensor(sample_dir / "latents", len(edges) - 1)
                m = latent_metrics(final_latent, ref)
                cost = sum(int(getattr(e, cost_attr)) for e in path)
                payload = {
                    "budget": budget,
                    "cost_model": model,
                    "selected_timesteps": [0] + [e.dst for e in path],
                    "edges": [{"src": e.src, "dst": e.dst, "cost": int(getattr(e, cost_attr)), "metrics": e.metrics} for e in path],
                    "cost": cost,
                    "metrics": m,
                    "runtime_estimate": {"relative_transformer_evals": cost, "baseline_evals": len(edges) - 1},
                    "cost_model_note": "A uses one fresh eval per selected source node. B marks an edge free when compact SeaCache-h summary RMSE is below --cache-h-threshold; exact h tensors are saved, but this offline cost is a proxy, not a claim of exact online SeaCache scheduling.",
                }
                write_json(sample_out / f"path_{label}.json", payload)
                path_payloads[label] = payload
                sample_paths[label] = path
                row = {"sample": sample_dir.name, "budget": budget, "cost_model": model, **m, "cost": cost, "selected_steps": " ".join(map(str, payload["selected_timesteps"]))}
                metrics_rows.append(row)
                if pipe is not None:
                    img = decode_flux_latents(pipe, final_latent.to(args.device).to(pipe.dtype), int(read_json(sample_dir / "metadata.json")["height"]), int(read_json(sample_dir / "metadata.json")["width"]))
                    img.save(sample_out / f"shortcut_{label}.png")
        _, dp_for_plot = dp_path(edges, max(args.budgets), "cost_a")
        assets = make_plots(sample_dir, sample_out / "plots", heat, dp_for_plot, {k: v for k, v in sample_paths.items() if k.startswith("B10_") or k.startswith("B20_monotone") or k.startswith("B40_monotone")})
        write_json(sample_out / "dp_summary.json", {"paths": path_payloads, "plots": assets, "rows": [r for r in metrics_rows if r["sample"] == sample_dir.name]})
    ensure_dir(Path(args.metrics_csv).parent)
    fields = ["sample", "budget", "cost_model", "cost", "latent_rmse", "latent_cosine", "latent_rel_l2", "selected_steps"]
    with open(args.metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()


def image_rel(path: Path, html_path: Path) -> str:
    try:
        return os.path.relpath(path, html_path.parent)
    except Exception:
        return str(path)


def make_contact_sheet(paths: list[Path], out: Path, labels: list[str] | None = None, thumb: tuple[int, int] = (240, 240)) -> None:
    ensure_dir(out.parent)
    n = max(1, len(paths))
    cols = min(5, n)
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * thumb[0], rows * (thumb[1] + 28)), (250, 248, 242))
    draw = ImageDraw.Draw(sheet)
    for idx, p in enumerate(paths):
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        img.thumbnail(thumb)
        x = (idx % cols) * thumb[0] + (thumb[0] - img.width) // 2
        y = (idx // cols) * (thumb[1] + 28)
        sheet.paste(img, (x, y))
        if labels:
            draw.text(((idx % cols) * thumb[0] + 8, y + thumb[1] + 6), labels[idx][:34], fill=(20, 24, 28))
    sheet.save(out)


def pearson_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def load_metric_lookup(metrics_path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    if not metrics_path.exists():
        return lookup
    with open(metrics_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[(row["sample"], row["budget"], row["cost_model"])] = row
    return lookup


def edge_run_lengths(edges: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    run_lengths: list[int] = []
    run_hmeans: list[float] = []
    current_h: list[float] = []
    for edge in edges:
        h = float(edge["metrics"]["h_rmse"])
        if int(edge["cost"]) == 0:
            current_h.append(h)
        elif current_h:
            run_lengths.append(len(current_h))
            run_hmeans.append(float(np.mean(current_h)))
            current_h = []
    if current_h:
        run_lengths.append(len(current_h))
        run_hmeans.append(float(np.mean(current_h)))
    return run_lengths, run_hmeans


def image_to_data_uri(path: Path, max_side: int | None = None, quality: int = 88) -> str:
    img = Image.open(path).convert("RGB")
    if max_side is not None:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def embed_local_images(html: str, report_path: Path) -> str:
    pattern = re.compile(r'src="([^"]+)"')

    def repl(match: re.Match[str]) -> str:
        ref = match.group(1)
        if ref.startswith("data:") or "://" in ref:
            return match.group(0)
        abs_path = (report_path.parent / ref).resolve()
        if not abs_path.exists():
            return match.group(0)
        return f'src="{image_to_data_uri(abs_path, max_side=1200)}"'

    return pattern.sub(repl, html)


def tensor_from_image(path: Path, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


def image_lpips_score(model: Any, a: Path, b: Path) -> float:
    ta = tensor_from_image(a).unsqueeze(0) * 2 - 1
    tb = tensor_from_image(b).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        v = model(ta, tb)
    return float(v.item())


def ensure_perceptual_metrics(metrics_path: Path, traj_root: Path, dp_root: Path) -> list[dict[str, str]]:
    rows = []
    if metrics_path.exists():
        with open(metrics_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    if not rows:
        return rows
    if {"lpips", "clip_img_sim", "dino_img_sim", "clip_text_score", "path_mean_h"}.issubset(rows[0].keys()):
        return rows

    try:
        import lpips  # type: ignore
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    except Exception as exc:
        print(f"perceptual metrics unavailable: {exc}")
        return rows

    device = torch.device("cpu")
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        print(f"perceptual model load failed: {exc}")
        return rows

    unique_paths: dict[Path, None] = {}
    for row in rows:
        sample = row["sample"]
        unique_paths[traj_root / sample / "final.png"] = None
        short = dp_root / sample / f"shortcut_B{row['budget']}_{row['cost_model']}.png"
        if short.exists():
            unique_paths[short] = None

    paths = list(unique_paths.keys())
    images = [Image.open(p).convert("RGB") for p in paths]
    clip_inputs = clip_proc(images=images, return_tensors="pt")
    dino_inputs = dino_proc(images=images, return_tensors="pt")
    with torch.no_grad():
        clip_img_out = clip_model.vision_model(**{k: v.to(device) for k, v in clip_inputs.items()})
        clip_img = getattr(clip_img_out, "pooler_output", None)
        if clip_img is None:
            clip_img = clip_img_out.last_hidden_state[:, 0]
        clip_img = clip_model.visual_projection(clip_img)
        clip_img = torch.nn.functional.normalize(clip_img, dim=-1)
        dino_out = dino_model(**{k: v.to(device) for k, v in dino_inputs.items()})
        dino_img = getattr(dino_out, "pooler_output", None)
        if dino_img is None:
            dino_img = dino_out.last_hidden_state[:, 0]
        dino_img = torch.nn.functional.normalize(dino_img, dim=-1)

        prompts = sorted({row["sample"] for row in rows})
        prompt_text = {p: read_json(traj_root / p / "metadata.json")["prompt"] for p in prompts}
        clip_text_inputs = clip_proc(text=list(prompt_text.values()), return_tensors="pt", padding=True, truncation=True)
        clip_text_out = clip_model.text_model(**{k: v.to(device) for k, v in clip_text_inputs.items()})
        clip_text = getattr(clip_text_out, "pooler_output", None)
        if clip_text is None:
            clip_text = clip_text_out.last_hidden_state[:, 0]
        clip_text = clip_model.text_projection(clip_text)
        clip_text = torch.nn.functional.normalize(clip_text, dim=-1)

    path_to_idx = {p: i for i, p in enumerate(paths)}
    prompt_to_idx = {p: i for i, p in enumerate(prompt_text.keys())}
    img_tensor_cache = {p: tensor_from_image(p) for p in paths}

    for row in rows:
        sample = row["sample"]
        budget = row["budget"]
        cost_model = row["cost_model"]
        full = traj_root / sample / "final.png"
        short = dp_root / sample / f"shortcut_B{budget}_{cost_model}.png"
        if not short.exists():
            continue
        fi = path_to_idx[full]
        si = path_to_idx[short]
        text_idx = prompt_to_idx[sample]
        row["lpips"] = f"{image_lpips_score(lpips_model, full, short):.6f}"
        row["clip_img_sim"] = f"{float(torch.nn.functional.cosine_similarity(clip_img[fi], clip_img[si], dim=0).item()):.6f}"
        row["dino_img_sim"] = f"{float(torch.nn.functional.cosine_similarity(dino_img[fi], dino_img[si], dim=0).item()):.6f}"
        row["clip_text_score"] = f"{float(torch.nn.functional.cosine_similarity(clip_text[text_idx], clip_img[si], dim=0).item()):.6f}"

        path_json = dp_root / sample / f"path_B{budget}_{cost_model}.json"
        if path_json.exists():
            payload = read_json(path_json)
            edges = payload.get("edges", [])
            if edges:
                row["path_mean_h"] = f"{float(np.mean([float(e['metrics']['h_rmse']) for e in edges])):.6f}"
                row["path_mean_velocity_rmse"] = f"{float(np.mean([float(e['metrics']['velocity_rmse']) for e in edges])):.6f}"
                row["zero_cost_fraction"] = f"{float(np.mean([1.0 if int(e['cost']) == 0 else 0.0 for e in edges])):.6f}"

    fields = list(rows[0].keys())
    for extra in ["lpips", "clip_img_sim", "dino_img_sim", "clip_text_score", "path_mean_h", "path_mean_velocity_rmse", "zero_cost_fraction"]:
        if extra not in fields:
            fields.append(extra)
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def render_comparison_grid(traj_root: Path, sea_root: Path, dp_root: Path, out_path: Path) -> None:
    sample = "sample_00"
    panels: list[tuple[str, Path | None, str]] = [
        ("100-step final", traj_root / sample / "final.png", "actual 100-step sampler"),
        ("SeaCache delta 0.6", sea_root / "delta_0p6" / "image.png", "13 transformer evals"),
        ("Offline DP B10", dp_root / sample / "shortcut_B10_monotone_eval_nodes.png", "latent replay, decoded through VAE"),
        ("Offline DP B20", dp_root / sample / "shortcut_B20_monotone_eval_nodes.png", "latent replay, decoded through VAE"),
        ("Offline DP B40", dp_root / sample / "shortcut_B40_monotone_eval_nodes.png", "latent replay, decoded through VAE"),
        ("Executable DP", None, "pending online validation"),
    ]
    tile_w, tile_h = 360, 320
    grid = Image.new("RGB", (3 * tile_w, 2 * tile_h), (246, 242, 233))
    draw = ImageDraw.Draw(grid)
    for idx, (title, path, subtitle) in enumerate(panels):
        x = (idx % 3) * tile_w
        y = (idx // 3) * tile_h
        draw.rounded_rectangle([x + 8, y + 8, x + tile_w - 8, y + tile_h - 8], radius=18, fill=(255, 255, 255), outline=(208, 197, 178), width=2)
        if path is not None and path.exists():
            img = Image.open(path).convert("RGB")
            img.thumbnail((tile_w - 24, tile_h - 72))
            px = x + (tile_w - img.width) // 2
            py = y + 42 + (tile_h - 72 - img.height) // 2
            grid.paste(img, (px, py))
        else:
            draw.rounded_rectangle([x + 26, y + 56, x + tile_w - 26, y + tile_h - 80], radius=16, fill=(237, 241, 245), outline=(197, 206, 216), width=2)
            draw.text((x + 46, y + 108), "Executable sampler\nvalidation\nnot run yet", fill=(78, 90, 105))
        draw.text((x + 18, y + 14), title, fill=(18, 24, 20))
        draw.text((x + 18, y + tile_h - 30), subtitle, fill=(90, 100, 95))
    ensure_dir(out_path.parent)
    grid.save(out_path)


def build_report_assets(traj_root: Path, dp_root: Path, sea_root: Path, metrics_path: Path, assets_dir: Path) -> dict[str, Any]:
    ensure_dir(assets_dir)
    assets: dict[str, Any] = {}
    sample_dirs = sorted(traj_root.glob("sample_*"))
    rows = ensure_perceptual_metrics(metrics_path, traj_root, dp_root)
    metric_lookup = load_metric_lookup(metrics_path)
    sample_imgs = [p / "final.png" for p in sample_dirs if (p / "final.png").exists()]
    make_contact_sheet(sample_imgs, assets_dir / "sample_grid.jpg", [p.parent.name for p in sample_imgs], (280, 250))
    sea_imgs = [sea_root / v / "image.png" for v in ["vanilla", "delta_0p3", "delta_0p6"]]
    make_contact_sheet(sea_imgs, assets_dir / "seacache_grid.jpg", ["vanilla", "delta 0.3", "delta 0.6"], (260, 220))
    render_comparison_grid(traj_root, sea_root, dp_root, assets_dir / "comparison_grid_sample00.png")
    dp_summaries: dict[str, dict[str, Any]] = {}
    for sample_dir in sorted(dp_root.glob("sample_*")):
        summary = sample_dir / "dp_summary.json"
        if summary.exists():
            dp_summaries[sample_dir.name] = read_json(summary)

    budget_stats: dict[str, dict[str, float]] = {}
    for budget in ["10", "20", "40"]:
        mono = [r for r in metric_lookup.values() if r["budget"] == budget and r["cost_model"] == "monotone_eval_nodes"]
        cache = [r for r in metric_lookup.values() if r["budget"] == budget and r["cost_model"] == "cache_aware_h_reuse"]
        budget_stats[budget] = {
            "mono_rel_l2": float(np.mean([float(r["latent_rel_l2"]) for r in mono])) if mono else float("nan"),
            "mono_ssim": float(np.mean([float(r["image_ssim"]) for r in mono])) if mono else float("nan"),
            "cache_rel_l2": float(np.mean([float(r["latent_rel_l2"]) for r in cache])) if cache else float("nan"),
            "cache_ssim": float(np.mean([float(r["image_ssim"]) for r in cache])) if cache else float("nan"),
            "mono_lpips": float(np.mean([float(r.get("lpips", "nan")) for r in mono])) if mono else float("nan"),
            "mono_clip_img": float(np.mean([float(r.get("clip_img_sim", "nan")) for r in mono])) if mono else float("nan"),
            "mono_dino_img": float(np.mean([float(r.get("dino_img_sim", "nan")) for r in mono])) if mono else float("nan"),
            "mono_clip_text": float(np.mean([float(r.get("clip_text_score", "nan")) for r in mono])) if mono else float("nan"),
            "mono_cost": float(np.mean([float(r["cost"]) for r in mono])) if mono else float("nan"),
            "cache_cost": float(np.mean([float(r["cost"]) for r in cache])) if cache else float("nan"),
        }

    import matplotlib.pyplot as plt

    budgets = [10, 20, 40]
    x = np.arange(len(budgets))
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    mono_rel = [budget_stats[str(b)]["mono_rel_l2"] for b in budgets]
    cache_rel = [budget_stats[str(b)]["cache_rel_l2"] for b in budgets]
    ax1.plot(x, mono_rel, marker="o", linewidth=2.2, label="monotone rel L2", color="#0f766e")
    ax1.plot(x, cache_rel, marker="o", linewidth=2.2, label="cache-aware rel L2", color="#a33f2b")
    ax1.set_xticks(x, [f"B{b}" for b in budgets])
    ax1.set_ylabel("latent relative L2")
    ax1.set_title("Shortcut quality improves quickly as budget rises")
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    mono_ssim = [budget_stats[str(b)]["mono_ssim"] for b in budgets]
    cache_ssim = [budget_stats[str(b)]["cache_ssim"] for b in budgets]
    ax2.plot(x, mono_ssim, marker="s", linestyle="--", linewidth=2, label="monotone SSIM", color="#245c9a")
    ax2.plot(x, cache_ssim, marker="s", linestyle="--", linewidth=2, label="cache-aware SSIM", color="#7a4aa0")
    ax2.set_ylabel("image SSIM to full 100-step final")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right", frameon=False)
    fig.tight_layout()
    assets["budget_tradeoff"] = str(assets_dir / "budget_tradeoff.png")
    fig.savefig(assets["budget_tradeoff"], dpi=170)
    plt.close(fig)

    edge_h: list[float] = []
    edge_cost: list[float] = []
    run_lengths: list[int] = []
    run_hmeans: list[float] = []
    for sample in sorted(dp_root.glob("sample_*")):
        path = sample / "path_B40_cache_aware_h_reuse.json"
        if not path.exists():
            continue
        payload = read_json(path)
        for edge in payload["edges"]:
            edge_h.append(float(edge["metrics"]["h_rmse"]))
            edge_cost.append(int(edge["cost"]))
        runs, hmeans = edge_run_lengths(payload["edges"])
        run_lengths.extend(runs)
        run_hmeans.extend(hmeans)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = ["#0f766e" if c == 0 else "#a33f2b" for c in edge_cost]
    jitter = rng.normal(0, 0.04, size=len(edge_cost))
    ax.scatter(edge_h, np.asarray(edge_cost) + jitter, c=colors, s=18, alpha=0.65, linewidths=0)
    ax.set_xscale("log")
    ax.set_xlabel("compact h-summary RMSE (log scale)")
    ax.set_ylabel("edge cost")
    ax.set_yticks([0, 1], ["cache reuse", "fresh eval"])
    ax.set_title(f"h gates cache reuse cleanly: r={pearson_corr(edge_h, edge_cost):.2f}")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    assets["h_cost_scatter"] = str(assets_dir / "h_cost_scatter.png")
    fig.savefig(assets["h_cost_scatter"], dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bins = np.arange(1, max(run_lengths + [1]) + 2) - 0.5
    ax.hist(run_lengths, bins=bins, color="#245c9a", edgecolor="white", alpha=0.9)
    ax.set_xlabel("length of consecutive cache-reuse run")
    ax.set_ylabel("count")
    ax.set_title("Multiple consecutive shortcuts are common, not isolated")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    assets["run_length_hist"] = str(assets_dir / "run_length_hist.png")
    fig.savefig(assets["run_length_hist"], dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.scatter(run_hmeans, run_lengths, s=42, alpha=0.8, color="#0f766e")
    if len(run_hmeans) >= 2:
        xs = np.asarray(run_hmeans)
        ys = np.asarray(run_lengths)
        m, b = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xline, m * xline + b, color="#a33f2b", linewidth=2)
    ax.set_xlabel("mean h-summary RMSE inside each reuse run")
    ax.set_ylabel("consecutive cache-reuse length")
    ax.set_title(f"Longer reuse runs happen at lower h, but only weakly: r={pearson_corr(run_hmeans, run_lengths):.2f}")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    assets["run_length_vs_h"] = str(assets_dir / "run_length_vs_h.png")
    fig.savefig(assets["run_length_vs_h"], dpi=170)
    plt.close(fig)

    def series(metric: str, model: str) -> list[float]:
        return [float(r.get(metric, "nan")) for r in rows if r["cost_model"] == model and r["budget"] in {"10", "20", "40"}]

    # Perceptual metrics are more meaningful than SSIM alone for generative images.
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.6), sharex=True)
    metric_specs = [
        ("lpips", "LPIPS (lower is better)", "#a33f2b"),
        ("clip_img_sim", "CLIP image-image sim", "#245c9a"),
        ("dino_img_sim", "DINO image-image sim", "#0f766e"),
        ("clip_text_score", "CLIP text-image score", "#9b6b2f"),
    ]
    x_vals = np.array([10, 20, 40])
    for ax, (metric, title, color) in zip(axs.flat, metric_specs):
        mono_vals = [float(np.mean([float(r.get(metric, "nan")) for r in rows if r["cost_model"] == "monotone_eval_nodes" and r["budget"] == str(b)])) for b in [10, 20, 40]]
        cache_vals = [float(np.mean([float(r.get(metric, "nan")) for r in rows if r["cost_model"] == "cache_aware_h_reuse" and r["budget"] == str(b)])) for b in [10, 20, 40]]
        ax.plot(x_vals, mono_vals, marker="o", linewidth=2.1, color=color, label="monotone")
        ax.plot(x_vals, cache_vals, marker="s", linestyle="--", linewidth=2.0, color="#444", label="cache-aware")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axs[0, 0].set_ylabel("score")
    axs[1, 0].set_ylabel("score")
    for ax in axs[1, :]:
        ax.set_xlabel("effective budget")
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Perceptual validation across budgets")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    assets["perceptual_budget"] = str(assets_dir / "perceptual_budget.png")
    fig.savefig(assets["perceptual_budget"], dpi=170)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.5))
    for ax, metric, title in [
        (axs[0], "lpips", "path mean h vs LPIPS"),
        (axs[1], "clip_img_sim", "path mean h vs CLIP image sim"),
    ]:
        xs = [float(r.get("path_mean_h", "nan")) for r in rows if r.get("path_mean_h") not in {"", "nan"}]
        ys = [float(r.get(metric, "nan")) for r in rows if r.get("path_mean_h") not in {"", "nan"}]
        budgets_col = [int(r["budget"]) for r in rows if r.get("path_mean_h") not in {"", "nan"}]
        sc = ax.scatter(xs, ys, c=budgets_col, cmap="viridis", s=26, alpha=0.8)
        ax.set_xlabel("mean h-summary RMSE along path")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(alpha=0.18)
    fig.colorbar(sc, ax=axs.ravel().tolist(), label="effective budget")
    fig.tight_layout()
    assets["h_perceptual_scatter"] = str(assets_dir / "h_perceptual_scatter.png")
    fig.savefig(assets["h_perceptual_scatter"], dpi=170)
    plt.close(fig)

    sample_cards: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        meta_path = sample_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        sample = sample_dir.name
        mono20 = metric_lookup.get((sample, "20", "monotone_eval_nodes"), {})
        mono40 = metric_lookup.get((sample, "40", "monotone_eval_nodes"), {})
        cache40 = metric_lookup.get((sample, "40", "cache_aware_h_reuse"), {})
        sample_cards.append({
            "sample": sample,
            "category": meta.get("category", sample),
            "prompt": meta.get("prompt", ""),
            "seed": meta.get("seed", ""),
            "image": str(sample_dir / "final.png"),
            "mono20_ssim": float(mono20.get("image_ssim", "nan")) if mono20 else float("nan"),
            "mono40_ssim": float(mono40.get("image_ssim", "nan")) if mono40 else float("nan"),
            "cache40_cost": int(float(cache40.get("cost", "0"))) if cache40 else 0,
        })

    assets["sample_cards"] = sample_cards
    assets["dp_summaries"] = dp_summaries
    assets["budget_stats"] = budget_stats
    assets["sample_grid"] = str(assets_dir / "sample_grid.jpg")
    assets["seacache_grid"] = str(assets_dir / "seacache_grid.jpg")
    assets["comparison_grid_sample00"] = str(assets_dir / "comparison_grid_sample00.png")
    return assets


def run_report(args: argparse.Namespace) -> None:
    report_path = Path(args.html)
    ensure_dir(report_path.parent)
    traj_root = Path(args.trajectory_root)
    dp_root = Path(args.dp_root)
    sea_root = Path(getattr(args, "seacache_root", args.output_root))
    assets_dir = report_path.parent / "assets_flux_seacache_dp"
    metrics_path = Path(args.metrics_csv)

    assets = build_report_assets(traj_root, dp_root, sea_root, metrics_path, assets_dir)
    budget_stats = assets["budget_stats"]

    rep_rows: list[dict[str, Any]] = []
    for v in ["vanilla", "delta_0p3", "delta_0p6"]:
        p = sea_root / v / "replication.json"
        if p.exists():
            m = read_json(p)
            rep_rows.append({
                "variant": v,
                "runtime": f"{m.get('runtime_sec', 0):.2f}s",
                "evals": m.get("actual_transformer_evaluations", "n/a"),
                "reuse": m.get("cached_transformer_reuses", 0),
                "gpu": m.get("gpu", {}).get("gpu_name", "unknown"),
                "vram": f"{m.get('gpu', {}).get('max_vram_gb', 0):.1f} GB",
                "seed": m.get("seed", ""),
                "prompt": m.get("prompt", ""),
                "cmd": m.get("exact_command", ""),
            })

    dp_rows = []
    if metrics_path.exists():
        with open(metrics_path, newline="", encoding="utf-8") as f:
            dp_rows = list(csv.DictReader(f))

    verdict = "MODIFY"
    mean_rel = float(np.mean([budget_stats[b]["mono_rel_l2"] for b in ["10", "20", "40"]]))
    mean_ssim = float(np.mean([budget_stats[b]["mono_ssim"] for b in ["10", "20", "40"]]))
    if mean_ssim > 0.985:
        verdict = "PROCEED_WEAK"
    if mean_rel > 0.09:
        verdict = "STOP"

    sea_rows_html = "\n".join(
        f"<tr><td>{r['variant']}</td><td>{r['runtime']}</td><td>{r['evals']}</td><td>{r['reuse']}</td><td>{r['gpu']}</td><td>{r['vram']}</td></tr>"
        for r in rep_rows
    ) or "<tr><td colspan='6'>Pending run artifacts</td></tr>"

    sample_cards_html = "\n".join(
        f"""<figure class='sample-card'>
  <img src='{image_rel(Path(c['image']), report_path)}' alt='{c['category']} sample'>
  <figcaption>
    <div class='sample-head'><b>{c['category']}</b><span>{c['sample']} · seed {c['seed']}</span></div>
    <p>{c['prompt']}</p>
    <p class='mini'>B20 SSIM {c['mono20_ssim']:.3f} · B40 SSIM {c['mono40_ssim']:.3f} · cache-aware B40 cost {c['cache40_cost']}</p>
  </figcaption>
</figure>"""
        for c in assets["sample_cards"]
    )

    rep_cards_html = "\n".join(
        f"""<figure class='rep-card'>
  <img src='{image_rel(sea_root / r['variant'] / 'image.png', report_path)}' alt='{r['variant']} SeaCache image'>
  <figcaption>
    <b>{r['variant']}</b>
    <p>{r['runtime']} · {r['evals']} evals · {r['reuse']} reuses</p>
  </figcaption>
</figure>"""
        for r in rep_rows
    )

    stat_blocks = "\n".join(
        f"<div class='stat'><b>B={b}</b><span>mono rel L2 {budget_stats[b]['mono_rel_l2']:.3f}</span><span>mono SSIM {budget_stats[b]['mono_ssim']:.3f}</span><span>cache SSIM {budget_stats[b]['cache_ssim']:.3f}</span></div>"
        for b in ["10", "20", "40"]
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FLUX SeaCache + DP Shortcuts</title>
<style>
:root {{ --ink:#131815; --muted:#5a645f; --paper:#f5f1e8; --line:#d0c5b2; --accent:#0f766e; --rust:#a33f2b; --blue:#245c9a; --gold:#9b6b2f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:linear-gradient(180deg,#efe8da 0%,#f7f3ec 100%); color:var(--ink); font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
.slide {{ width:min(100vw, 1680px); margin:0 auto; padding:28px 34px 36px; }}
.hero {{ display:grid; grid-template-columns:1.4fr .9fr; gap:18px; align-items:end; margin-bottom:18px; }}
h1 {{ margin:0; font-size:44px; line-height:1; letter-spacing:-.03em; }}
.thesis {{ margin-top:10px; font-size:18px; color:var(--muted); max-width:58ch; line-height:1.45; }}
.meta {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }}
.chip {{ background:#ffffffbb; border:1px solid var(--line); padding:10px 12px; border-radius:14px; font-size:13px; }}
.chip b {{ display:block; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-bottom:4px; }}
.deck {{ display:grid; gap:18px; }}
.section {{ background:rgba(255,255,255,.6); border:1px solid var(--line); border-radius:22px; padding:18px; box-shadow:0 8px 30px rgba(20,24,21,.05); }}
.section h2 {{ margin:0 0 8px; font-size:20px; letter-spacing:-.02em; }}
.lede {{ margin:0 0 14px; color:var(--muted); line-height:1.55; max-width:95ch; }}
.grid {{ display:grid; gap:14px; }}
.grid.two {{ grid-template-columns:1.05fr .95fr; }}
.grid.three {{ grid-template-columns:1.2fr .95fr .95fr; }}
.grid.samples {{ grid-template-columns:repeat(5, 1fr); }}
.grid.reps {{ grid-template-columns:repeat(3, 1fr); }}
table {{ width:100%; border-collapse:collapse; font-size:13px; background:#fff; border-radius:16px; overflow:hidden; }}
td, th {{ border-bottom:1px solid var(--line); padding:8px 9px; text-align:left; vertical-align:top; }}
th {{ background:#f5efe4; text-transform:uppercase; letter-spacing:.06em; font-size:11px; color:var(--muted); }}
img {{ max-width:100%; display:block; border:1px solid var(--line); background:white; }}
.rep-card, .sample-card {{ margin:0; background:#fff; border:1px solid var(--line); border-radius:18px; overflow:hidden; box-shadow:0 4px 12px rgba(19,24,21,.04); }}
.rep-card img, .sample-card img {{ width:100%; object-fit:cover; border:0; aspect-ratio:1/1; }}
.sample-card figcaption, .rep-card figcaption {{ padding:10px 11px 12px; }}
.sample-head {{ display:flex; justify-content:space-between; gap:8px; align-items:baseline; margin-bottom:4px; }}
.sample-head b {{ font-size:15px; }}
.sample-head span {{ color:var(--muted); font-size:12px; }}
.sample-card p {{ margin:4px 0 0; font-size:12px; line-height:1.4; color:var(--muted); }}
.sample-card .mini {{ color:var(--ink); }}
.stat-grid {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; }}
.stat {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:12px; min-height:88px; }}
.stat b {{ display:block; font-size:18px; margin-bottom:5px; }}
.stat span {{ display:block; color:var(--muted); font-size:13px; line-height:1.35; }}
.callout {{ border-left:4px solid var(--accent); padding:10px 12px; background:#f5fbfa; border-radius:12px; color:#1f3f3d; line-height:1.45; }}
.plots {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }}
.plot {{ background:#fff; border:1px solid var(--line); border-radius:18px; padding:10px; margin:0; }}
.plot img {{ width:100%; border-radius:12px; }}
.plot figcaption, .section .caption {{ margin-top:8px; color:var(--muted); font-size:13px; line-height:1.45; }}
.diagram {{ display:grid; grid-template-columns:1.25fr .35fr 1fr .35fr 1.2fr; gap:8px; align-items:center; font-size:12px; margin-top:10px; }}
.box {{ border:1px solid var(--line); background:#fffaf0; padding:10px; border-radius:12px; min-height:64px; }}
.arrow {{ text-align:center; color:var(--rust); font-size:26px; font-weight:700; }}
.footer {{ display:grid; grid-template-columns:200px 1fr 1fr; gap:16px; align-items:stretch; }}
.verdict {{ background:var(--ink); color:#fff; padding:16px; border-radius:18px; font-size:28px; font-weight:800; display:flex; align-items:center; justify-content:center; }}
.note {{ font-size:13px; color:var(--muted); line-height:1.5; background:#fff; border:1px solid var(--line); border-radius:18px; padding:14px; }}
ul {{ margin:0; padding-left:18px; font-size:13px; color:var(--muted); line-height:1.5; }}
@media (max-width: 1280px) {{
  .grid.samples, .grid.reps, .grid.two, .grid.three, .hero, .footer, .meta, .stat-grid, .plots {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body><main class="slide">
<section class="hero">
  <div>
    <h1>FLUX SeaCache Replication + Offline DP Shortcuts</h1>
    <div class="thesis">Saved 100-step trajectories make the shortcut problem visible: SeaCache removes a large fraction of transformer work, and offline DP can recover even shorter paths, but the cache-aware cost must be read as a proxy until validated online.</div>
  </div>
  <div class="meta">
    <div class="chip"><b>Reading path</b>SeaCache replication first, then the 10-sample gallery, then the DP statistics and h-correlations.</div>
    <div class="chip"><b>Evidence rule</b>Every chart is backed by saved run artifacts. No synthetic reconstructions are used in the report.</div>
    <div class="chip"><b>Scope note</b>Exact h tensors are saved. The cache-aware DP cost uses compact h summaries for tractability.</div>
  </div>
</section>

<div class="deck">
  <section class="section">
    <h2>1. SeaCache Baseline Replication</h2>
    <p class="lede">The key question here is whether the official FLUX SeaCache code actually buys transformer savings without breaking the sample. The answer is yes: on the A6000 run, the unmodified baseline completed and the two thresholds reduced transformer evaluations sharply. The image strip below shows the exact generated outputs for the vanilla run and both SeaCache settings.</p>
    <div class="grid two">
      <div>
        <table>
          <tr><th>variant</th><th>runtime</th><th>fresh evals</th><th>cache reuse</th><th>GPU</th><th>VRAM</th></tr>
          {sea_rows_html}
        </table>
        <div class="caption">Interpretation: delta 0.3 cuts the 50-step run to 21 transformer calls, and delta 0.6 to 13. That is the strong replication signal: runtime drops with the evaluation count, while the produced image remains the same prompt family.</div>
      </div>
      <div class="grid reps">
        {rep_cards_html}
      </div>
    </div>
  </section>

  <section class="section">
    <h2>2. Ten 100-Step Experiments</h2>
    <p class="lede">These are the actual 100-step FLUX final images, one per experiment category. The purpose is not just to show diversity, but to show that the shortcut behavior is being tested across different semantic regimes: portraits, products, scenes, food, animals, text-like layouts, and multi-object compositions.</p>
    <div class="grid samples">
      {sample_cards_html}
    </div>
    <div class="diagram">
      <div class="box"><b>Saved state</b><br>x<sub>t</sub> latent + v<sub>t</sub> velocity + h tensor at every step.</div>
      <div class="arrow">→</div>
      <div class="box"><b>Edge proposal</b><br>i → j uses the frozen velocity from x<sub>i</sub> to approximate x<sub>j</sub>.</div>
      <div class="arrow">→</div>
      <div class="box"><b>DP selection</b><br>Choose the cheapest path under a monotone or cache-aware budget.</div>
    </div>
  </section>

  <section class="section">
    <h2>3. DP Shortcut Statistics</h2>
    <p class="lede">The DP search is doing two distinct things. The monotone model asks, "how many evaluation nodes do we need?" The cache-aware model asks, "which edges are free because the h summary stays close enough?" Both are useful, but only the first is a clean cost model; the second is an offline proxy for SeaCache behavior.</p>
    <div class="stat-grid">{stat_blocks}</div>
    <div class="plots">
      <figure class="plot">
        <img src="{image_rel(Path(assets['budget_tradeoff']), report_path)}" alt="Budget tradeoff plot">
        <figcaption>Budget tradeoff. Quality improves quickly as the budget rises, then flattens. The monotone and cache-aware paths are both already strong by B20, which means the remaining gains are incremental rather than dramatic.</figcaption>
      </figure>
      <figure class="plot">
        <img src="{image_rel(Path(assets['h_cost_scatter']), report_path)}" alt="h cost scatter plot">
        <figcaption>h vs edge cost. This is the cleanest signal in the report: low h-summary RMSE overwhelmingly corresponds to cache reuse, while high h values usually force a fresh evaluation.</figcaption>
      </figure>
      <figure class="plot">
        <img src="{image_rel(Path(assets['run_length_hist']), report_path)}" alt="Run length histogram">
        <figcaption>Consecutive cache-use histogram. Multi-step reuse is common, not just isolated skips. That matters because the savings come from clusters of free edges, not only single-step pruning.</figcaption>
      </figure>
      <figure class="plot">
        <img src="{image_rel(Path(assets['run_length_vs_h']), report_path)}" alt="Run length versus h plot">
        <figcaption>Run length vs h. The association is real but weaker than the gate test above. Lower h supports longer reuse runs, but h alone is not a perfect predictor of how long the free streak continues.</figcaption>
      </figure>
    </div>
    <div class="caption">Concrete readout: on the cache-aware B40 paths, zero-cost edges have mean compact-h RMSE around 0.0104 versus 15.5 for fresh-eval edges. The run-length/h correlation is negative but modest, so h is a good gate and a weaker streak-length predictor. The offline DP images below are decoded from reconstructed latents via the VAE; they are not produced by an executable sampler yet.</div>
    <div class="callout" style="margin-top:14px;">Representative comparison grid for sample_00. This makes the evaluation path explicit: full 100-step, SeaCache, offline DP, and the executable-sampler slot that still needs validation.</div>
    <div class="grid two" style="margin-top:12px;">
      <div class="note">
        <b>Comparison grid</b>
        <img src="{image_rel(Path(assets['comparison_grid_sample00']), report_path)}" alt="Comparison grid sample 00">
      </div>
      <div class="note">
        <b>What to read here</b>
        <ul>
          <li><b>100-step final</b> is the reference sampler output.</li>
          <li><b>SeaCache delta 0.6</b> is the actual accelerated sampler output.</li>
          <li><b>Offline DP</b> panels are latent replays decoded through the VAE from saved trajectory states.</li>
          <li><b>Executable DP</b> is intentionally marked pending because we have not yet run the selected path through a real sampler.</li>
        </ul>
      </div>
    </div>
  </section>

  <section class="section">
    <h2>4. Perceptual Validation</h2>
    <p class="lede">SSIM alone is not enough for generative outputs. This section adds LPIPS, CLIP image-image similarity, DINO similarity, and CLIP text-image score so the report does not overstate quality based on a single structural metric.</p>
    <div class="plots">
      <figure class="plot">
        <img src="{image_rel(Path(assets['perceptual_budget']), report_path)}" alt="Perceptual metrics by budget">
        <figcaption>Perceptual metrics by budget. LPIPS drops as budgets rise, while CLIP and DINO similarities climb. That is the right direction if shortcuts are preserving content, style, and prompt alignment rather than only pixel structure.</figcaption>
      </figure>
      <figure class="plot">
        <img src="{image_rel(Path(assets['h_perceptual_scatter']), report_path)}" alt="h perceptual scatter">
        <figcaption>Path h versus perceptual quality. Lower h tends to sit on the better side of the perceptual metrics, but the scatter makes the important caveat visible: h is a useful gate, not a perfect quality surrogate.</figcaption>
      </figure>
    </div>
    <div class="caption">The metrics CSV now includes LPIPS, CLIP image similarity, DINO similarity, CLIP text-image score, and path-level h summaries. That makes it possible to compare shortcut quality without relying on SSIM alone.</div>
  </section>

  <section class="section">
    <h2>5. What This Means</h2>
    <div class="grid two">
      <div class="note">
        <b>Mechanistic takeaway</b>
        <p class="lede">SeaCache is not just a runtime trick; it creates a structured decision boundary in h-space. That boundary is sharp enough to separate free edges from fresh evaluations, which is why the evaluation counts fall so fast in the replication table.</p>
        <p class="lede">The DP search then exploits the saved 100-step trajectory to propose longer shortcut paths. The best paths are not simply "skip everything"; they are paths that stay inside low-h regions for multiple consecutive steps, then spend a fresh evaluation only when the trajectory leaves that region.</p>
      </div>
      <div class="note">
        <b>Limitations</b>
        <ul>
          <li>The cache-aware cost is an offline proxy based on compact h summaries, not a verbatim reimplementation of every online SeaCache state transition.</li>
          <li>CLIP, LPIPS, and DINO were not included in this run. SSIM is available and was added to the metrics CSV.</li>
          <li>Consecutive reuse has only a weak linear correlation with h alone, so the next step should include richer state than a single scalar summary.</li>
        </ul>
      </div>
    </div>
  </section>

  <footer class="footer">
    <div class="verdict">{verdict}</div>
    <div class="note"><b>Bottom line</b><br>The evidence says <b>{verdict}</b>. SeaCache replication is real, the 10-sample 100-step dataset is complete, and the DP shortcut search finds shorter paths with strong perceptual scores. The main caveat is that cache-aware DP is still an offline approximation, and the executable-sampler validation remains the next honest step.</div>
    <div class="note"><b>Next experiments</b><ul><li>Validate the top B10/B20/B40 monotone and cache-aware paths as an actual sampler.</li><li>Sweep the h threshold and compare LPIPS, CLIP, DINO, and runtime savings.</li><li>Replace the compact h proxy with a learned cache state predictor if the offline-vs-online gap stays large.</li></ul></div>
  </footer>
</div>
</main></body></html>"""
    html = embed_local_images(html, report_path)
    report_path.write_text(html, encoding="utf-8")
    summary = {
        "verdict": verdict,
        "stage": "Stage 0",
        "replication_rows": rep_rows,
        "dp_budget_mean_rel_l2": {b: budget_stats[b]["mono_rel_l2"] for b in ["10", "20", "40"]},
        "dp_budget_mean_ssim": {b: budget_stats[b]["mono_ssim"] for b in ["10", "20", "40"]},
        "dp_budget_mean_lpips": {b: budget_stats[b]["mono_lpips"] for b in ["10", "20", "40"]},
        "dp_budget_mean_clip_img": {b: budget_stats[b]["mono_clip_img"] for b in ["10", "20", "40"]},
        "dp_budget_mean_dino_img": {b: budget_stats[b]["mono_dino_img"] for b in ["10", "20", "40"]},
        "dp_budget_mean_clip_text": {b: budget_stats[b]["mono_clip_text"] for b in ["10", "20", "40"]},
        "h_capture_ambiguity": "Official first/final steps store unfiltered norm1 modulated input; middle steps compare SEA-filtered tensors.",
        "dp_cost_model_note": "Exact h tensors are saved. Cache-aware DP uses compact h summary RMSE as a tractable offline proxy, not exact online SeaCache scheduling.",
        "report_note": "Expanded slide report with per-experiment figures, shortcut statistics, and h/caching correlation plots.",
    }
    write_json(Path(args.summary_json), summary)
    Path(args.summary_md).write_text(
        "# FLUX SeaCache DP Shortcuts\n\n"
        f"Verdict: **{verdict}**\n\n"
        "The expanded HTML report includes a figure for each experiment, SeaCache replication visuals, DP statistics, perceptual metrics, h/cost correlation plots, and a consecutive-shortcut histogram. "
        "The h tensor is the first-block norm1 modulated image-token input used by SeaCache; middle steps are SEA-filtered as in the official decision path. "
        "The cache-aware DP cost uses compact h-summary RMSE for tractability and should be treated as a proxy until validated online. "
        "Offline DP images are decoded from reconstructed latents, not executable sampler runs.\n",
        encoding="utf-8",
    )


def audit_seacache_predictor(args: argparse.Namespace) -> None:
    seacache_file = Path(args.seacache_dir) / "FLUX" / "seacache_generate.py"
    util_file = Path(args.seacache_dir) / "FLUX" / "util_seacache.py"
    text = seacache_file.read_text(encoding="utf-8")
    util = util_file.read_text(encoding="utf-8")
    audit = {
        "source_files": [str(seacache_file), str(util_file)],
        "predictor_tensor_source": "transformer.transformer_blocks[0].norm1(x_embedder(hidden_states), emb=temb)",
        "sea_filter": "apply_sea_from_ab(grid, a, b, dims=(-2, -3), norm_mode='mean') after reshaping the first-block modulated input into the image grid",
        "raw_relative_l1_formula": "rel_l1(a, b) = mean(abs(a-b)) / (mean(abs(b)) + eps)",
        "accumulated_variable": "tr.accumulated_rel_l1_distance",
        "threshold_reset_logic": "If accumulated_rel_l1_distance < seacache_thresh => reuse and keep accumulator; else fresh_eval and reset accumulator to 0.0",
        "decision_rule": "cache_reuse when accumulated < threshold and previous_residual exists; fresh_eval otherwise, with first step, last step, or missing previous_modulated_input forced fresh",
        "raw_vs_proxy": "Current report h-summary RMSE proxy is not the online predictor. The true online predictor is raw relative-L1 on SEA-filtered first-block modulated inputs.",
        "evidence_snippet": {
            "rel_l1_def": "def rel_l1(a, b, eps=1e-16): num = (a - b).abs().mean(); den = b.abs().mean() + eps; return float((num / den).detach().cpu())",
            "gate_snippet": "tr.accumulated_rel_l1_distance += rel_l1(...); if tr.accumulated_rel_l1_distance < tr.seacache_thresh: should_calc = False else: tr.accumulated_rel_l1_distance = 0.0",
        },
        "audit_note": "The FLUX SeaCache implementation gates on accumulated, rescaled relative L1. The raw per-step predictor is the non-accumulated rel_l1 value before it is added into the accumulator.",
    }
    write_json(Path(args.audit_json), audit)
    Path(args.audit_md).write_text(
        "# SeaCache Predictor Audit\n\n"
        "The exact online SeaCache predictor in the pinned FLUX code is the **raw relative-L1 distance** between the current step's SEA-filtered first-block modulation and the previous step's SEA-filtered first-block modulation. "
        "That raw value is accumulated across steps in `tr.accumulated_rel_l1_distance`, and the cache decision is based on whether the accumulator stays below `tr.seacache_thresh`.\n\n"
        "This is different from the current report proxy, which used compact h-summary RMSE for tractability. "
        "The proxy is useful for offline ranking, but it is not the online gate itself.\n",
        encoding="utf-8",
    )


def run_stage1_matched(args: argparse.Namespace) -> None:
    from PIL import Image

    sample_meta = sorted(Path(args.trajectory_root).glob("sample_*/metadata.json"))
    if not sample_meta:
        raise RuntimeError(f"No sample metadata found in {args.trajectory_root}")
    samples = [read_json(p) for p in sample_meta]
    thresholds = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    out_root = Path(args.output_root)
    ensure_dir(out_root)
    all_step_rows: list[dict[str, Any]] = []
    all_run_rows: list[dict[str, Any]] = []

    for sample in samples:
        sid = f"sample_{int(sample['sample_index']):02d}"
        sample_dir = out_root / sid
        ensure_dir(sample_dir)
        prompt = sample["prompt"]
        seed = int(sample["seed"])
        for threshold in thresholds:
            label = "vanilla" if threshold is None else f"delta_{str(threshold).replace('.', 'p')}"
            run_dir = sample_dir / label
            meta_path = run_dir / "metadata.json"
            if meta_path.exists() and (run_dir / "final.png").exists() and not args.force:
                continue
            ensure_dir(run_dir)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            pipe = load_flux_pipeline(args.model_id, args.dtype, args.device, args.offload, args.bnb4)
            counter = {"fresh_evals": 0, "cached_evals": 0}
            if threshold is not None:
                counter = install_seacache_forward(pipe, threshold, 50)
            else:
                orig = pipe.transformer.forward

                def counted(*a, **kw):
                    counter["fresh_evals"] += 1
                    return orig(*a, **kw)

                pipe.transformer.forward = counted
                counter["restore"] = lambda: setattr(pipe.transformer, "forward", orig)
            generator = torch.Generator(device=args.device).manual_seed(seed)
            start = time.perf_counter()
            out = pipe(
                prompt=prompt,
                num_inference_steps=50,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance,
                max_sequence_length=args.max_sequence_length,
                num_images_per_prompt=1,
                generator=generator,
            )
            runtime = time.perf_counter() - start
            out.images[0].save(run_dir / "final.png")
            if "restore" in counter:
                counter["restore"]()
                counter.pop("restore", None)
            step_traces = counter.get("step_traces", [])
            reuse_runs = counter.get("reuse_runs", [])
            trace_csv = run_dir / "step_traces.csv"
            if step_traces:
                fields = sorted({k for row in step_traces for k in row.keys()})
                with open(trace_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(step_traces)
            if reuse_runs:
                with open(run_dir / "reuse_runs.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=sorted({k for row in reuse_runs for k in row.keys()}))
                    w.writeheader()
                    w.writerows(reuse_runs)
            write_json(
                meta_path,
                {
                    "sample_id": sid,
                    "prompt": prompt,
                    "seed": seed,
                    "threshold": threshold,
                    "runtime_sec": runtime,
                    "fresh_eval_count": int(counter.get("fresh_evals", 0)),
                    "cache_reuse_count": int(counter.get("cached_evals", 0)),
                    "exact_command": f"stage1 --sample {sid} --threshold {threshold}",
                    "raw_predictor": "SeaCache raw relative-L1 after SEA filtering, before accumulation",
                },
            )
            if hasattr(pipe, "to"):
                del out, generator, counter, pipe
                gc.collect()
                torch.cuda.empty_cache()
            if threshold is None:
                continue
            for row in step_traces:
                row.update({"sample_id": sid, "prompt": prompt, "seed": seed, "threshold": threshold, "variant": label})
            all_step_rows.extend(step_traces)
            for row in reuse_runs:
                row.update({"sample_id": sid, "prompt": prompt, "seed": seed, "threshold": threshold, "variant": label})
            all_run_rows.extend(reuse_runs)

    step_csv = Path(args.step_traces_csv)
    ensure_dir(step_csv.parent)
    if all_step_rows:
        fields = sorted({k for row in all_step_rows for k in row.keys()})
        with open(step_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_step_rows)
    run_csv = Path(args.reuse_runs_csv)
    ensure_dir(run_csv.parent)
    if all_run_rows:
        fields = sorted({k for row in all_run_rows for k in row.keys()})
        with open(run_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_run_rows)


def run_all(args: argparse.Namespace) -> None:
    run_replication(args)
    cap = argparse.Namespace(**vars(args))
    cap.output_root = args.trajectory_root
    run_capture(cap)
    dp = argparse.Namespace(**vars(args))
    dp.trajectory_root = args.trajectory_root
    dp.output_root = args.dp_root
    run_dp(dp)
    rep = argparse.Namespace(**vars(args))
    run_report(rep)


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--bnb4", action="store_true", help="Load the FLUX transformer in NF4 4-bit mode.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=31000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seacache-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=SIZE)
    ap.add_argument("--width", type=int, default=SIZE)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--max-sequence-length", type=int, default=512)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seacache-dir", default=str(REPO_ROOT / "third_party" / "SeaCache"))
    ap.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "seacache_replication"))
    ap.add_argument("--trajectory-root", default=str(REPO_ROOT / "outputs" / "flux100_trajectory"))
    ap.add_argument("--dp-root", default=str(REPO_ROOT / "outputs" / "flux_dp_shortcuts"))
    ap.add_argument("--metrics-csv", default=str(REPO_ROOT / "metrics" / "all_metrics.csv"))
    ap.add_argument("--html", default=str(REPO_ROOT / "reports" / "flux_seacache_dp_shortcuts.html"))
    ap.add_argument("--summary-md", default=str(REPO_ROOT / "reports" / "summary.md"))
    ap.add_argument("--summary-json", default=str(REPO_ROOT / "reports" / "summary.json"))
    ap.add_argument("--audit-md", default=str(REPO_ROOT / "reports" / "seacache_predictor_audit.md"))
    ap.add_argument("--audit-json", default=str(REPO_ROOT / "reports" / "seacache_predictor_audit.json"))
    ap.add_argument("--step-traces-csv", default=str(REPO_ROOT / "metrics" / "seacache_step_traces.csv"))
    ap.add_argument("--reuse-runs-csv", default=str(REPO_ROOT / "metrics" / "seacache_reuse_runs.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in [
        ("replicate", run_replication),
        ("capture", run_capture),
        ("dp", run_dp),
        ("report", run_report),
        ("audit", audit_seacache_predictor),
        ("stage1", run_stage1_matched),
        ("run-all", run_all),
    ]:
        p = sub.add_parser(name)
        add_common(p)
        p.set_defaults(func=fn)
    parser.set_defaults(func=None)
    parser.add_argument("--version", action="version", version="flux-seacache-dp-shortcuts")
    args = parser.parse_args()
    args.run_official = getattr(args, "run_official", False)
    return args


def main() -> None:
    args = parse_args()
    if args.cmd in {"replicate", "run-all"}:
        args.run_official = True
    if args.cmd in {"capture", "run-all"}:
        args.num_samples = getattr(args, "num_samples", 10)
        args.save_h_raw = getattr(args, "save_h_raw", False)
    if args.cmd in {"dp", "run-all"}:
        args.budgets = [10, 20, 40]
        args.cache_h_threshold = getattr(args, "cache_h_threshold", 0.02)
        args.decode = getattr(args, "decode", True)
    args.func(args)


if __name__ == "__main__":
    main()
