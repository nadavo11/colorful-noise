#!/usr/bin/env python3
"""FLUX SeaCache replication, trajectory capture, and offline shortcut search.

This script is intentionally resumable: every expensive stage checks for its
expected output files before running unless --force is supplied.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
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
    state = {"fresh_evals": 0, "cached_evals": 0, "h_distance": [], "cache_decisions": []}
    tr.scheduler = pipe.scheduler
    tr.seacache_thresh = float(threshold)
    tr.cnt = 0
    tr.num_steps = int(num_steps)
    tr.accumulated_rel_l1_distance = 0.0
    tr.previous_modulated_input = None
    tr.previous_residual = None

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
        modulated, *_ = tr.transformer_blocks[0].norm1(hs, emb=temb)
        if tr.cnt == 0 or tr.cnt == tr.num_steps - 1 or tr.previous_modulated_input is None:
            tr.accumulated_rel_l1_distance = 0.0
        else:
            grid = modulated.reshape(modulated.shape[0], int(img[:, 1].max().item() + 1), int(img[:, 2].max().item() + 1), modulated.shape[-1])
            a, b = ab_from_scheduler(tr.scheduler, tr.cnt)
            modulated = apply_sea_from_ab(grid, a, b, dims=(-2, -3), norm_mode="mean").reshape(modulated.shape[0], -1, modulated.shape[-1])
            dist = rel_l1(modulated, tr.previous_modulated_input)
            tr.accumulated_rel_l1_distance += dist
            state["h_distance"].append({"step": int(tr.cnt), "rel_l1": dist, "accumulated": float(tr.accumulated_rel_l1_distance)})
            if tr.accumulated_rel_l1_distance < tr.seacache_thresh:
                should_calc = False
            else:
                tr.accumulated_rel_l1_distance = 0.0
        tr.previous_modulated_input = modulated.detach()
        tr.cnt += 1
        if tr.cnt == tr.num_steps:
            tr.cnt = 0

        state["cache_decisions"].append(bool(should_calc))
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


def run_report(args: argparse.Namespace) -> None:
    report_path = Path(args.html)
    ensure_dir(report_path.parent)
    traj_root = Path(args.trajectory_root)
    dp_root = Path(args.dp_root)
    sea_root = Path(getattr(args, "seacache_root", args.output_root))
    assets_dir = report_path.parent / "assets_flux_seacache_dp"
    ensure_dir(assets_dir)

    sample_imgs = sorted(traj_root.glob("sample_*/final.png"))
    make_contact_sheet(sample_imgs, assets_dir / "sample_grid.jpg", [p.parent.name for p in sample_imgs])
    sea_imgs = [sea_root / v / "image.png" for v in ["vanilla", "delta_0p3", "delta_0p6"]]
    make_contact_sheet(sea_imgs, assets_dir / "seacache_grid.jpg", ["vanilla", "delta 0.3", "delta 0.6"], (260, 220))

    rep_rows = []
    for v in ["vanilla", "delta_0p3", "delta_0p6"]:
        p = sea_root / v / "replication.json"
        if p.exists():
            m = read_json(p)
            rep_rows.append((v, f"{m.get('runtime_sec', 0):.2f}", m.get("actual_transformer_evaluations", "n/a"), m.get("cached_transformer_reuses", 0), m.get("gpu", {}).get("gpu_name", "unknown")))

    dp_rows = []
    metrics_path = Path(args.metrics_csv)
    if metrics_path.exists():
        with open(metrics_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dp_rows.append(row)
    by_budget: dict[str, list[float]] = {}
    for row in dp_rows:
        if row.get("cost_model") == "monotone_eval_nodes":
            by_budget.setdefault(row["budget"], []).append(float(row["latent_rel_l2"]))
    verdict = "MODIFY"
    if by_budget.get("10") and float(np.mean(by_budget["10"])) < 0.15:
        verdict = "PROCEED"
    elif by_budget.get("40") and float(np.mean(by_budget["40"])) > 0.35:
        verdict = "STOP"

    first_plot = next(iter(sorted(dp_root.glob("sample_*/plots/edge_error_heatmap.png"))), None)
    first_curve = next(iter(sorted(dp_root.glob("sample_*/plots/h_velocity_distance.png"))), None)
    first_path = next(iter(sorted(dp_root.glob("sample_*/plots/path_over_timestep.png"))), None)
    rows_html = "\n".join(f"<tr><td>{a}</td><td>{b}s</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>" for a, b, c, d, e in rep_rows) or "<tr><td colspan='5'>Pending run artifacts</td></tr>"
    budget_html = "\n".join(
        f"<div class='stat'><b>B={b}</b><span>mean rel L2 {np.mean(vals):.3f}</span></div>" for b, vals in sorted(by_budget.items(), key=lambda x: int(x[0]))
    ) or "<div class='stat'><b>DP</b><span>Pending metrics</span></div>"
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FLUX SeaCache + DP Shortcuts</title>
<style>
:root {{ --ink:#151817; --muted:#5e6764; --paper:#f7f4ec; --line:#c9c0ae; --accent:#0f766e; --rust:#a33f2b; --blue:#245c9a; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#e9e1d2; color:var(--ink); font-family: Avenir Next, Segoe UI, sans-serif; }}
.slide {{ width:min(100vw, 1600px); aspect-ratio:16/9; margin:auto; background:linear-gradient(135deg,#fbfaf6 0%,#efe7d7 62%,#d5e1da 100%); padding:34px 42px; display:grid; grid-template-rows:auto 1fr auto; gap:18px; }}
h1 {{ margin:0; font-size:42px; letter-spacing:0; }}
.thesis {{ color:var(--muted); font-size:18px; margin-top:4px; }}
.grid {{ display:grid; grid-template-columns:1.1fr .95fr .95fr; gap:18px; min-height:0; }}
.panel {{ border-top:3px solid var(--ink); padding-top:10px; min-width:0; }}
h2 {{ font-size:18px; margin:0 0 10px; text-transform:uppercase; letter-spacing:.08em; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; background:rgba(255,255,255,.36); }}
td, th {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:left; }}
img {{ max-width:100%; display:block; border:1px solid var(--line); background:white; }}
.sample {{ height:245px; object-fit:cover; width:100%; }}
.plots {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.plots img {{ height:150px; width:100%; object-fit:contain; }}
.stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:8px 0 12px; }}
.stat {{ background:#ffffff80; border-left:5px solid var(--accent); padding:8px; min-height:54px; }}
.stat b {{ display:block; font-size:18px; }}
.stat span {{ color:var(--muted); font-size:13px; }}
.diagram {{ display:grid; grid-template-columns:repeat(5,1fr); gap:6px; align-items:center; font-size:12px; margin-top:8px; }}
.box {{ border:1px solid var(--line); background:#fffaf0; padding:9px; min-height:58px; }}
.arrow {{ text-align:center; color:var(--rust); font-size:24px; }}
.footer {{ display:grid; grid-template-columns:180px 1fr 1fr; gap:18px; align-items:stretch; }}
.verdict {{ background:var(--ink); color:#fff; padding:14px; font-size:26px; font-weight:800; }}
.note {{ font-size:13px; color:var(--muted); line-height:1.35; }}
ul {{ margin:0; padding-left:18px; font-size:13px; }}
</style>
</head>
<body><main class="slide">
<header><h1>FLUX SeaCache Replication + Offline DP Shortcuts</h1><div class="thesis">Thesis: saved 100-step flow trajectories expose shortcut paths, but h-based cache cost remains an offline proxy until validated online.</div></header>
<section class="grid">
<div class="panel"><h2>SeaCache Replication</h2><table><tr><th>variant</th><th>runtime</th><th>fresh evals</th><th>cache reuse</th><th>GPU</th></tr>{rows_html}</table><img src="{image_rel(assets_dir / 'seacache_grid.jpg', report_path)}" style="margin-top:10px;height:220px;width:100%;object-fit:contain"></div>
<div class="panel"><h2>10-Sample 100-Step Dataset</h2><img class="sample" src="{image_rel(assets_dir / 'sample_grid.jpg', report_path)}"><div class="diagram"><div class="box">x_i latent</div><div class="arrow">></div><div class="box">v_i frozen-flow edge</div><div class="arrow">></div><div class="box">compare x_j, h_j, v_j</div></div></div>
<div class="panel"><h2>DP Shortcut Summary</h2><div class="stats">{budget_html}</div><div class="plots"><img src="{image_rel(first_plot, report_path) if first_plot else ''}"><img src="{image_rel(first_curve, report_path) if first_curve else ''}"><img src="{image_rel(first_path, report_path) if first_path else ''}"></div></div>
</section>
<footer class="footer"><div class="verdict">{verdict}</div><div class="note">Cost model A counts selected monotone eval nodes. Cost model B treats low compact-h-summary distance edges as cache-reusable; this is intentionally labeled ambiguous because exact online SeaCache residual reuse state is not fully determined by a static edge alone.</div><div><ul><li>Validate top DP paths online with real transformer calls.</li><li>Sweep h threshold against image-space metrics.</li><li>Replace frozen velocity with two-point or residual-aware edge updates.</li></ul></div></footer>
</main></body></html>"""
    report_path.write_text(html, encoding="utf-8")
    summary = {
        "verdict": verdict,
        "replication_rows": rep_rows,
        "dp_budget_mean_rel_l2": {k: float(np.mean(v)) for k, v in by_budget.items()},
        "h_capture_ambiguity": "Official first/final steps store unfiltered norm1 modulated input; middle steps compare SEA-filtered tensors.",
        "dp_cost_model_note": "Exact h tensors are saved. Cache-aware DP uses compact h summary RMSE as a tractable offline proxy, not exact online SeaCache scheduling.",
    }
    write_json(Path(args.summary_json), summary)
    Path(args.summary_md).write_text(
        "# FLUX SeaCache DP Shortcuts\n\n"
        f"Verdict: **{verdict}**\n\n"
        "The run captures SeaCache replication metadata, 100-step FLUX trajectories, and offline DP shortcut paths. "
        "The h tensor is the first-block norm1 modulated image-token input used by SeaCache; middle steps are SEA-filtered as in the official decision path. "
        "The cache-aware DP cost uses compact h-summary RMSE for tractability and should be treated as a proxy until validated online.\n",
        encoding="utf-8",
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in [
        ("replicate", run_replication),
        ("capture", run_capture),
        ("dp", run_dp),
        ("report", run_report),
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
