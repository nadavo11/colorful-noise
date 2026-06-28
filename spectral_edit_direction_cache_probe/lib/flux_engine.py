"""E51 FLUX engine (uv env, diffusers 0.38). A faithful re-implementation of the
FluxImg2ImgPipeline denoising loop that exposes the per-step transformer prediction so we
can evaluate it under BOTH the source and target prompt at the same x_t.

    v_src(t)   = T(x_t, source_prompt, t)
    v_edit(t)  = T(x_t, target_prompt, t)
    delta(t)   = v_edit(t) - v_src(t)

Closed-loop caching replays a denoise where a precomputed skip-schedule decides, per step,
whether to recompute the cached signal or reuse its stale value.
"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image

import config as C

_D = "cuda"


def load_pipe():
    from diffusers import FluxImg2ImgPipeline, FluxTransformer2DModel, BitsAndBytesConfig
    repo = "black-forest-labs/FLUX.1-dev"
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16)
    tr = FluxTransformer2DModel.from_pretrained(
        repo, subfolder="transformer", quantization_config=qc, torch_dtype=torch.bfloat16)
    pipe = FluxImg2ImgPipeline.from_pretrained(repo, transformer=tr, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to(_D); pipe.text_encoder_2.to(_D); pipe.vae.to(_D)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _img(path, size=C.SIZE):
    im = path if isinstance(path, Image.Image) else Image.open(path)
    return im.convert("RGB").resize((size, size), Image.LANCZOS)


@torch.no_grad()
def encode(pipe, prompt):
    pe, ppe, tid = pipe.encode_prompt(prompt=prompt, prompt_2=prompt, device=_D,
                                      num_images_per_prompt=1, max_sequence_length=512)
    return (pe, ppe, tid)


@torch.no_grad()
def prepare(pipe, image, size=C.SIZE, steps=C.STEPS, strength=C.STRENGTH, seed=C.SEED):
    """Replicate FluxImg2ImgPipeline steps 4-5. Re-runs retrieve_timesteps so the shared
    scheduler is reset; identical seed => identical initial noised latent across variants."""
    from diffusers.pipelines.flux.pipeline_flux_img2img import calculate_shift, retrieve_timesteps
    init = pipe.image_processor.preprocess(_img(image, size), height=size, width=size).to(dtype=torch.float32)
    sigmas = np.linspace(1.0, 1 / steps, steps)
    seq = (size // pipe.vae_scale_factor // 2) * (size // pipe.vae_scale_factor // 2)
    mu = calculate_shift(seq, pipe.scheduler.config.get("base_image_seq_len", 256),
                         pipe.scheduler.config.get("max_image_seq_len", 4096),
                         pipe.scheduler.config.get("base_shift", 0.5),
                         pipe.scheduler.config.get("max_shift", 1.15))
    retrieve_timesteps(pipe.scheduler, steps, _D, sigmas=sigmas, mu=mu)
    timesteps, nsteps = pipe.get_timesteps(steps, strength, _D)
    latent_t = timesteps[:1].repeat(1)
    nch = pipe.transformer.config.in_channels // 4
    gen = torch.Generator(_D).manual_seed(seed)
    latents, ids = pipe.prepare_latents(init, latent_t, 1, nch, size, size,
                                        torch.bfloat16, _D, gen, None)
    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], C.GUIDANCE, device=_D, dtype=torch.float32).expand(latents.shape[0])
    return dict(latents=latents, ids=ids, timesteps=timesteps, guidance=guidance, size=size)


@torch.no_grad()
def _fwd(pipe, latents, t, st, emb):
    pe, ppe, tid = emb
    ts = t.expand(latents.shape[0]).to(latents.dtype)
    return pipe.transformer(hidden_states=latents, timestep=ts / 1000, guidance=st["guidance"],
                            pooled_projections=ppe, encoder_hidden_states=pe,
                            txt_ids=tid, img_ids=st["ids"],
                            joint_attention_kwargs=None, return_dict=False)[0]


def _spatial(pipe, packed, size):
    """Unpack a packed FLUX latent/velocity [1,seq,64] -> spatial [C,H,W] numpy float32."""
    lat = pipe._unpack_latents(packed, size, size, pipe.vae_scale_factor)
    return lat[0].float().cpu().numpy()


@torch.no_grad()
def decode(pipe, latents, size):
    lat = pipe._unpack_latents(latents, size, size, pipe.vae_scale_factor)
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type="pil")[0]


@torch.no_grad()
def record_reference(pipe, st, emb_src, emb_tgt):
    """Full-compute reference trajectory (driven by v_edit). At each step also evaluate v_src
    at the SAME x_t. Returns final image + flattened packed signals + spatial signals."""
    latents = st["latents"].clone()
    T = st["timesteps"]; size = st["size"]
    V_edit, V_src, SP_edit, SP_delta = [], [], [], []
    for t in T:
        vsrc = _fwd(pipe, latents, t, st, emb_src)
        vedit = _fwd(pipe, latents, t, st, emb_tgt)
        V_edit.append(vedit.flatten().float().cpu().numpy())
        V_src.append(vsrc.flatten().float().cpu().numpy())
        SP_edit.append(_spatial(pipe, vedit, size))
        SP_delta.append(_spatial(pipe, vedit - vsrc, size))
        latents = pipe.scheduler.step(vedit, t, latents, return_dict=False)[0]
    img = decode(pipe, latents, size)
    return img, np.array(V_edit), np.array(V_src), np.stack(SP_edit), np.stack(SP_delta)


@torch.no_grad()
def denoise_cached(pipe, st, emb_src, emb_tgt, skip_steps, signal):
    """Closed-loop denoise reusing a stale cached signal on `skip_steps`.
      signal='full'  : cache v_edit; on skip reuse it directly (no forwards that step).
      signal='delta' : always compute base v_src; on skip reuse v_used = v_src + stale delta.
    Returns final image + a forward-count dict for honest compute accounting."""
    latents = st["latents"].clone()
    T = st["timesteps"]; n = len(T); size = st["size"]
    cache_vedit = cache_delta = None
    fwd_edit = fwd_src = 0
    for i, t in enumerate(T):
        do_skip = (i in skip_steps) and (0 < i < n - 1)
        if signal == "delta":
            vsrc = _fwd(pipe, latents, t, st, emb_src); fwd_src += 1
            if do_skip and cache_delta is not None:
                vused = vsrc + cache_delta
            else:
                vedit = _fwd(pipe, latents, t, st, emb_tgt); fwd_edit += 1
                cache_delta = vedit - vsrc
                vused = vedit
        else:
            if do_skip and cache_vedit is not None:
                vused = cache_vedit
            else:
                vused = _fwd(pipe, latents, t, st, emb_tgt); fwd_edit += 1
                cache_vedit = vused
        latents = pipe.scheduler.step(vused, t, latents, return_dict=False)[0]
    img = decode(pipe, latents, size)
    return img, dict(fwd_edit=fwd_edit, fwd_src=fwd_src, n_steps=n)
