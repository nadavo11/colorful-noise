"""E16 fidelity baselines: training-free inference-time methods on Flux-dev.

Each wrapper returns (pil_image, unpacked fp32 cpu latent) -- the SAME contract as
e7_flux_phase.flux_generate / e8_psd_clamp.gen_with_cb -- so the E16 driver caches
and scores every condition uniformly. All run on the stock load_flux() pipe and
share the seeded initial latent with the cfg / SBN conditions (Flux derives the
initial noise from torch.Generator("cuda").manual_seed(seed) in prepare_latents,
independent of guidance), so paired per-seed deltas are valid.

Methods (all training-free, no retraining):
  gen_cfgzero  -- CFG-Zero* (arXiv 2503.18886): true-CFG with (a) per-sample
                  optimal scale on the unconditional term and (b) zero-init of the
                  first few ODE steps. Faithful: a transformer-forward recorder
                  captures the cond/uncond velocities and a scheduler.step override
                  recomputes the guided update. Runs true-CFG (distilled guidance
                  held neutral at 1.0, true_cfg_scale=w), matching E10's protocol.
  gen_negprompt-- native two-pass true-CFG with a fidelity-oriented negative prompt
                  (diffusers' real CFG). This is the negative-guidance baseline
                  representing NAG's GOAL on Flux-dev; NAG's attention-space
                  normalization mainly helps few-step sampling, so at 28 steps the
                  native negative-prompt path is the relevant proxy (see
                  docs/methods/FIDELITY_BASELINES.md).
  gen_seg      -- SEG (Smoothed Energy Guidance, NeurIPS'24): guide away from a
                  prediction whose image-token self-attention queries are Gaussian
                  -blurred. Implemented as true-CFG where the negative branch shares
                  the prompt but runs a blurred-query attention processor;
                  true_cfg_scale = 1 + seg_scale reproduces pred + s*(pred-blurred).
                  Best-effort -- seg_available() self-tests it; the driver skips SEG
                  if the attention hook is incompatible with the installed diffusers.
"""
import math
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e7_flux_phase import SIZE

NEG_FIDELITY = ("blurry, low quality, low resolution, oversaturated, "
                "oversharpened, jpeg artifacts, deep fried, distorted, deformed")


def _grab_cb():
    captured = {}

    def cb(p, i, t, kw):
        captured["latents"] = kw["latents"]
        return {}

    return captured, cb


def _unpack(pipe, packed):
    from diffusers import FluxPipeline
    lat = FluxPipeline._unpack_latents(packed, SIZE, SIZE, pipe.vae_scale_factor)
    return lat.float().cpu()


# ---------------------------------------------------------------------------
# CFG-Zero*
# ---------------------------------------------------------------------------

def gen_cfgzero(pipe, prompt, seed, true_cfg=3.5, guidance=1.0, steps=28,
                zero_init_steps=1, neg_prompt=""):
    """CFG-Zero* generation -> (img, lat). Records cond/uncond velocities and
    overrides scheduler.step to apply the optimal-scale + zero-init update."""
    records = []
    orig_fwd = pipe.transformer.forward
    orig_step = pipe.scheduler.step
    step_i = {"n": 0}

    def fwd(*a, **k):
        out = orig_fwd(*a, **k)
        sample = out[0] if isinstance(out, (tuple, list)) else out.sample
        records.append(sample.detach())
        return out

    def step(model_output, timestep, sample, *a, **k):
        i = step_i["n"]
        step_i["n"] += 1
        pos, neg = records[-2], records[-1]          # cond called before uncond
        B = pos.shape[0]
        if i < zero_init_steps:
            mo = torch.zeros_like(model_output)       # flow: zero velocity -> no move
        else:
            p = pos.reshape(B, -1).float()
            n = neg.reshape(B, -1).float()
            alpha = (p * n).sum(1, keepdim=True) / (n.pow(2).sum(1, keepdim=True) + 1e-8)
            alpha = alpha.reshape(B, *([1] * (pos.dim() - 1))).to(pos.dtype)
            mo = neg * alpha + true_cfg * (pos - neg * alpha)
        return orig_step(mo, timestep, sample, *a, **k)

    captured, cb = _grab_cb()
    try:
        pipe.transformer.forward = fwd
        pipe.scheduler.step = step
        img = pipe(prompt=prompt, prompt_2=prompt, height=SIZE, width=SIZE,
                   guidance_scale=guidance, true_cfg_scale=true_cfg,
                   negative_prompt=neg_prompt, num_inference_steps=steps,
                   generator=torch.Generator("cuda").manual_seed(seed),
                   callback_on_step_end=cb).images[0]
    finally:
        pipe.transformer.forward = orig_fwd
        pipe.scheduler.step = orig_step
    return img, _unpack(pipe, captured["latents"])


# ---------------------------------------------------------------------------
# Negative-prompt true-CFG (NAG proxy on 28-step Flux-dev)
# ---------------------------------------------------------------------------

def gen_negprompt(pipe, prompt, seed, true_cfg=3.5, guidance=1.0, steps=28,
                  neg_prompt=NEG_FIDELITY):
    """Native two-pass true-CFG with a fidelity negative prompt -> (img, lat)."""
    captured, cb = _grab_cb()
    img = pipe(prompt=prompt, prompt_2=prompt, height=SIZE, width=SIZE,
               guidance_scale=guidance, true_cfg_scale=true_cfg,
               negative_prompt=neg_prompt, negative_prompt_2=neg_prompt,
               num_inference_steps=steps,
               generator=torch.Generator("cuda").manual_seed(seed),
               callback_on_step_end=cb).images[0]
    return img, _unpack(pipe, captured["latents"])


# ---------------------------------------------------------------------------
# SEG -- blurred-query attention processor (best-effort)
# ---------------------------------------------------------------------------

_SEG = {"perturb": False, "sigma": 3.0}  # module-level toggle read by the processor


def _gaussian_blur_tokens(x_img, side, sigma):
    """Gaussian-blur image tokens. x_img: (B, img_len, heads, dim) on a side*side
    grid -> blurred same shape."""
    B, L, Hh, D = x_img.shape
    k = max(3, int(2 * round(3 * sigma) + 1))
    r = (k - 1) // 2
    ax = torch.arange(k, device=x_img.device, dtype=torch.float32) - r
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    ker = (g[:, None] * g[None, :])[None, None]          # (1,1,k,k)
    x = x_img.permute(0, 2, 3, 1).reshape(B * Hh * D, 1, side, side)
    x = torch.nn.functional.conv2d(
        torch.nn.functional.pad(x, (r, r, r, r), mode="reflect"), ker.to(x.dtype))
    x = x.reshape(B, Hh, D, side, side).permute(0, 3, 4, 1, 2).reshape(B, L, Hh, D)
    return x


def make_seg_processor():
    """Subclass FluxAttnProcessor; blur image-token queries when _SEG['perturb']."""
    from diffusers.models.transformers import transformer_flux as TF

    class SEGProcessor(TF.FluxAttnProcessor):
        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, image_rotary_emb=None):
            q, k, v, eq, ek, ev = TF._get_qkv_projections(
                attn, hidden_states, encoder_hidden_states)
            q = attn.norm_q(q.unflatten(-1, (attn.heads, -1)))
            k = attn.norm_k(k.unflatten(-1, (attn.heads, -1)))
            v = v.unflatten(-1, (attn.heads, -1))
            enc_len = 0
            if attn.added_kv_proj_dim is not None:
                eq = attn.norm_added_q(eq.unflatten(-1, (attn.heads, -1)))
                ek = attn.norm_added_k(ek.unflatten(-1, (attn.heads, -1)))
                ev = ev.unflatten(-1, (attn.heads, -1))
                enc_len = eq.shape[1]
                q = torch.cat([eq, q], dim=1)
                k = torch.cat([ek, k], dim=1)
                v = torch.cat([ev, v], dim=1)
            if image_rotary_emb is not None:
                q = TF.apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
                k = TF.apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
            # SEG perturbation: Gaussian-blur the image-token queries on their grid
            if _SEG["perturb"]:
                q_img = q[:, enc_len:]
                side = int(round(math.sqrt(q_img.shape[1])))
                if side * side == q_img.shape[1]:
                    q = torch.cat([q[:, :enc_len],
                                   _gaussian_blur_tokens(q_img, side, _SEG["sigma"])], dim=1)
            out = TF.dispatch_attention_fn(
                q, k, v, attn_mask=attention_mask,
                backend=self._attention_backend, parallel_config=self._parallel_config)
            out = out.flatten(2, 3).to(q.dtype)
            if encoder_hidden_states is not None:
                enc, hid = out.split_with_sizes(
                    [encoder_hidden_states.shape[1], out.shape[1] - encoder_hidden_states.shape[1]],
                    dim=1)
                hid = attn.to_out[1](attn.to_out[0](hid.contiguous()))
                enc = attn.to_add_out(enc.contiguous())
                return hid, enc
            return out

    return SEGProcessor


def _set_seg_processors(pipe, enable):
    """Swap every Flux attention processor to/from the SEG subclass. Returns the
    saved original processors (pass back to restore)."""
    if enable:
        saved = pipe.transformer.attn_processors
        proc_cls = make_seg_processor()
        pipe.transformer.set_attn_processor(
            {k: proc_cls() for k in saved})
        return saved
    return None


def seg_available(pipe, steps=4):
    """Self-test SEG on a tiny run; True if it generates without error."""
    try:
        gen_seg(pipe, "a test photo", 0, steps=steps)
        return True
    except Exception as e:
        print(f"[e16] SEG unavailable: {type(e).__name__}: {e}", flush=True)
        return False


def gen_seg(pipe, prompt, seed, seg_scale=3.0, sigma=3.0, guidance=3.5, steps=28):
    """SEG generation -> (img, lat). Negative branch shares the prompt but runs the
    blurred-query processor; true_cfg=1+seg_scale gives pred + s*(pred - blurred).
    The processor blurs only when _SEG['perturb'] is set, toggled per branch by a
    transformer-forward counter (cond call -> off, uncond/perturbed call -> on)."""
    saved = _set_seg_processors(pipe, True)
    _SEG["sigma"] = sigma
    orig_fwd = pipe.transformer.forward
    call = {"n": 0}

    def fwd(*a, **k):
        # within a step the pipe calls cond first, then the negative branch
        _SEG["perturb"] = (call["n"] % 2 == 1)
        call["n"] += 1
        try:
            return orig_fwd(*a, **k)
        finally:
            _SEG["perturb"] = False

    captured, cb = _grab_cb()
    try:
        pipe.transformer.forward = fwd
        img = pipe(prompt=prompt, prompt_2=prompt, height=SIZE, width=SIZE,
                   guidance_scale=guidance, true_cfg_scale=1.0 + seg_scale,
                   negative_prompt=prompt, negative_prompt_2=prompt,
                   num_inference_steps=steps,
                   generator=torch.Generator("cuda").manual_seed(seed),
                   callback_on_step_end=cb).images[0]
    finally:
        pipe.transformer.forward = orig_fwd
        _SEG["perturb"] = False
        if saved is not None:
            pipe.transformer.set_attn_processor(saved)
    return img, _unpack(pipe, captured["latents"])
