"""SD3.5 backend for the E17 comparison: SBN (band-norm) + CFG-Zero* on Stable
Diffusion 3.5 medium, which uses TRUE classifier-free guidance (no Flux-style
guidance distillation -- the motivation for moving off Flux).

SD3.5 vs Flux, relevant differences (verified against diffusers 0.38):
  - guidance_scale is real CFG; do_cfg = guidance_scale > 1. So guidance_scale=1
    is the pure conditional flow field == the SBN reference (analogous to Flux
    cfg=1, but without distillation).
  - one BATCHED transformer forward over [uncond, cond] per step, then
    noise_pred.chunk(2); combine = uncond + w*(cond-uncond). (Flux did two
    separate forwards.) CFG-Zero* therefore captures the batched output and
    splits it, rather than recording two calls.
  - callback latents are already UNPACKED (B,16,128,128) at 1024px -- no
    pack/unpack, so the SBN PSD-clamp callback is simpler than Flux's and the
    spectral_ops (radial_psd/psd_match/band_index_map) apply directly.

Public API (mirrors e7/e8/bandnorm so the E17 driver + scoring reuse everything):
  load_sd35(mem)                          -> pipe
  record_reference_sd3(pipe, prompt, ...) -> (ref, outs)         # cfg=1 PSD ref
  gen_sd3(pipe, prompt, seed, guidance, steps, step_override, cb_obj)
                                          -> (pil, (1,16,128,128) fp32 cpu lat)
  ClampPSD3(ref, ...)  RecordPSD3(...)    # step-end callbacks
  make_cfgzero_step(guidance, ...)        # scheduler.step override for CFG-Zero*
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_ops import band_index_map, band_power, psd_match

SD35_ID = "stabilityai/stable-diffusion-3.5-medium"
SIZE = 1024
N_CH, H, W = 16, 128, 128


def load_sd35(mem="gpu_resident"):
    """SD3.5-medium (2.5B) fits a 24GB A5000 GPU-resident in bf16
    (transformer ~5GB + T5-XXL ~9.5GB + CLIPx2 + VAE). --mem offload as fallback."""
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(SD35_ID, torch_dtype=torch.bfloat16)
    if mem == "offload":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_sd35_vae(mem="gpu_resident"):
    """Just the SD3.5 VAE (16-ch), for offline latent encode/decode (E18) without
    paying for the T5/CLIP text encoders. ~150MB; the only gated download needed
    for the offline experiment."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(SD35_ID, subfolder="vae",
                                        torch_dtype=torch.bfloat16)
    return vae if mem == "offload" else vae.to("cuda")


def sd3_vae_encode(vae, pil):
    """PIL image -> generation-space latent (1,16,128,128) fp32 cpu.

    Inverts sd3_vae_decode: the pipeline works in lat = (z - shift) * sf space
    (z = raw VAE latent), same convention as e10's Flux real-image encode."""
    import numpy as np
    sf, shift = vae.config.scaling_factor, vae.config.shift_factor
    img = pil.convert("RGB").resize((SIZE, SIZE))
    x = torch.from_numpy(np.asarray(img).copy()).float() / 255.0
    x = (x.permute(2, 0, 1)[None] * 2 - 1).to(vae.dtype).to(vae.device)  # [-1,1]
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean                    # raw VAE latent
    return ((z - shift) * sf).float().cpu()


def sd3_vae_decode(vae, lat):
    """Generation-space latent (1,16,128,128) -> PIL. SD3.5 VAE uses the same
    (divide by scaling_factor THEN add shift_factor) convention as Flux."""
    from PIL import Image
    sf, shift = vae.config.scaling_factor, vae.config.shift_factor
    z = lat.to(vae.device).float() / sf + shift
    with torch.no_grad():
        img = vae.decode(z.to(vae.dtype), return_dict=False)[0]
    arr = ((img[0].float() / 2 + 0.5).clamp(0, 1) * 255).round().byte()
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())


# ---------------------------------------------------------------------------
# Step-end callbacks (SD3 latents are already unpacked -> no pack/unpack)
# ---------------------------------------------------------------------------

class RecordPSD3:
    """Record per-step per-(channel, band) mean power; no modification."""

    def __init__(self, idx_map, n_bins, steps):
        self.idx_map, self.n_bins = idx_map, n_bins
        self.band = [None] * steps
        self.total = [0.0] * steps
        self.std = [0.0] * steps
        self.last = None

    def __call__(self, p, i, t, kw):
        lat = kw["latents"].float()
        F2 = torch.fft.fft2(lat[0]).abs() ** 2
        self.band[i] = band_power(F2, self.idx_map, self.n_bins).cpu()
        self.total[i] = float((lat ** 2).sum())
        self.std[i] = float(lat.std())
        self.last = kw["latents"]
        return {}


class ClampPSD3:
    """Clamp the latent's per-(channel, band) PSD to the cfg=1 reference at the
    same step; phase untouched. Returns modified latents to the pipeline."""

    def __init__(self, ref, idx_map, n_bins, mode="band"):
        self.mode, self.idx_map, self.n_bins = mode, idx_map, n_bins
        self.ref_band = ref["band"].cuda()      # (steps, C, n_bins)
        self.ref_total = ref["total"]
        self.std = []
        self.gmin, self.gmax, self.resid = math.inf, -math.inf, 0.0
        self.last = None

    def __call__(self, p, i, t, kw):
        lat = kw["latents"]
        if self.mode == "band":
            lat, st = psd_match(lat, self.ref_band[i], self.idx_map,
                                self.n_bins, return_stats=True)
            self.gmin = min(self.gmin, st["gain_min"])
            self.gmax = max(self.gmax, st["gain_max"])
            self.resid = max(self.resid, st["imag_residue"])
        else:
            x = lat.float()
            scale = math.sqrt(float(self.ref_total[i]) /
                              max(float((x ** 2).sum()), 1e-12))
            lat = (x * scale).to(lat.dtype)
        self.std.append(float(lat.float().std()))
        self.last = lat
        return {"latents": lat.to(kw["latents"].dtype)}


# ---------------------------------------------------------------------------
# Generation (one entry point; compose a step-override and/or a callback)
# ---------------------------------------------------------------------------

def gen_sd3(pipe, prompt, seed, guidance, steps, step_override=None,
            cb_obj=None, neg_prompt=""):
    """One SD3.5 generation -> (pil, (1,16,128,128) fp32 cpu latent).

    step_override(records, model_output, sample) -> new model_output : optional
        guidance modifier (e.g. CFG-Zero*), applied inside scheduler.step using
        the captured batched transformer output.
    cb_obj(p,i,t,kw) -> {} or {"latents":...} : optional step-end callback
        (e.g. ClampPSD3 for SBN, or RecordPSD3 for the reference)."""
    captured = {}

    def cb(p, i, t, kw):
        out = cb_obj(p, i, t, kw) if cb_obj is not None else {}
        captured["latents"] = out.get("latents", kw["latents"])
        return out

    orig_fwd = pipe.transformer.forward
    orig_step = pipe.scheduler.step
    if step_override is not None:
        records = []

        def fwd(*a, **k):
            o = orig_fwd(*a, **k)
            s = o[0] if isinstance(o, (tuple, list)) else o.sample
            records.append(s.detach())
            return o

        def step(model_output, timestep, sample, *a, **k):
            mo = step_override(records, model_output, sample)
            return orig_step(mo, timestep, sample, *a, **k)

        pipe.transformer.forward = fwd
        pipe.scheduler.step = step
    try:
        img = pipe(prompt=prompt, height=SIZE, width=SIZE,
                   guidance_scale=guidance, num_inference_steps=steps,
                   negative_prompt=neg_prompt,
                   generator=torch.Generator("cuda").manual_seed(seed),
                   callback_on_step_end=cb).images[0]
    finally:
        pipe.transformer.forward = orig_fwd
        pipe.scheduler.step = orig_step
    return img, captured["latents"].float().cpu()


def make_cfgzero_step(guidance, zero_init_steps=1):
    """CFG-Zero* scheduler.step override for SD3's BATCHED [uncond,cond] output:
    per-sample optimal scale on the uncond term + zero-init of early steps."""
    state = {"i": 0}

    def override(records, model_output, sample):
        i = state["i"]
        state["i"] += 1
        if i < zero_init_steps:
            return torch.zeros_like(model_output)
        out = records[-1]                       # batched transformer output
        if out.shape[0] != 2 * sample.shape[0]:
            return model_output                 # no cfg (guidance<=1): passthrough
        uncond, cond = out.chunk(2)
        B = cond.shape[0]
        c = cond.reshape(B, -1).float()
        u = uncond.reshape(B, -1).float()
        alpha = (c * u).sum(1, keepdim=True) / (u.pow(2).sum(1, keepdim=True) + 1e-8)
        alpha = alpha.reshape(B, *([1] * (cond.dim() - 1))).to(cond.dtype)
        return uncond * alpha + guidance * (cond - uncond * alpha)

    return override


def gen_rcfgpp_sd3(pipe, prompt, seed, guidance, steps, sigma_noise=0.005,
                   cb_obj=None, neg_prompt=""):
    """Rectified-CFG++ (arXiv 2510.07631) on SD3.5 -> (pil, fp32 cpu latent).

    Faithful to the released SD3 pipeline (constant guidance, not the paper's
    alpha(t) schedule -- the authors found it made little difference). Per step:
      predictor:  x_pred = x_t + dt * v_cond(x_t)   (+ sigma_noise jitter if t>0.1)
      corrector:  v_hat  = v_cond(x_t) + w*(v_cond(x_pred) - v_uncond(x_pred))
      step:       x_next = x_t + dt * v_hat          (scheduler's Euler)
    dt = sigmas[i+1]-sigmas[i] (<0). Two batched forwards/step. Composes with an
    SBN clamp via cb_obj (callback_on_step_end runs after the Euler step)."""
    captured = {}
    cap_kw = {}
    records = []
    state = {"i": 0}
    orig_fwd = pipe.transformer.forward
    orig_step = pipe.scheduler.step
    sched = pipe.scheduler
    gnoise = torch.Generator("cuda").manual_seed(seed + 777)

    def cb(p, i, t, kw):
        out = cb_obj(p, i, t, kw) if cb_obj is not None else {}
        captured["latents"] = out.get("latents", kw["latents"])
        return out

    def fwd(*a, **k):
        cap_kw.update(k)                       # encoder_hidden_states, pooled, jak
        o = orig_fwd(*a, **k)
        records.append((o[0] if isinstance(o, (tuple, list)) else o.sample).detach())
        return o

    def step(model_output, timestep, sample, *a, **k):
        i = state["i"]
        state["i"] += 1
        pred = records[-1]                     # batched [uncond, cond] at x_t
        if pred.shape[0] != 2 * sample.shape[0]:
            return orig_step(model_output, timestep, sample, *a, **k)
        _, v_cond_t = pred.chunk(2)
        dt = (sched.sigmas[i + 1] - sched.sigmas[i]) if i < len(sched.sigmas) - 1 \
            else -sched.sigmas[i]
        x_pred = sample + dt * v_cond_t
        if sigma_noise > 0 and float(timestep) > 0.1 * sched.config.num_train_timesteps:
            x_pred = x_pred + sigma_noise * torch.randn(
                x_pred.shape, generator=gnoise, device=x_pred.device, dtype=x_pred.dtype)
        ck = dict(cap_kw)
        ck["hidden_states"] = torch.cat([x_pred] * 2)
        ck["timestep"] = (timestep + dt).expand(ck["hidden_states"].shape[0]).to(x_pred.dtype)
        corr = orig_fwd(**ck)
        corr = corr[0] if isinstance(corr, (tuple, list)) else corr.sample
        v_un_p, v_cond_p = corr.chunk(2)
        v_hat = v_cond_t + guidance * (v_cond_p - v_un_p)
        return orig_step(v_hat, timestep, sample, *a, **k)

    try:
        pipe.transformer.forward = fwd
        pipe.scheduler.step = step
        img = pipe(prompt=prompt, height=SIZE, width=SIZE,
                   guidance_scale=guidance, num_inference_steps=steps,
                   negative_prompt=neg_prompt,
                   generator=torch.Generator("cuda").manual_seed(seed),
                   callback_on_step_end=cb).images[0]
    finally:
        pipe.transformer.forward = orig_fwd
        pipe.scheduler.step = orig_step
    return img, captured["latents"].float().cpu()


def record_reference_sd3(pipe, prompt, seeds=3, steps=28, n_bins=24, guidance=1.0):
    """Record the per-step per-(channel, band) mean-power reference from `seeds`
    guidance=1 (pure conditional field) generations. Returns (ref, outs)."""
    idx_map = band_index_map(H, W, n_bins, "cuda")
    acc_band = torch.zeros(steps, N_CH, n_bins)
    acc_total = torch.zeros(steps)
    acc_std = torch.zeros(steps)
    outs = []
    for s in range(seeds):
        rec = RecordPSD3(idx_map, n_bins, steps)
        img, lat = gen_sd3(pipe, prompt, s, guidance, steps, cb_obj=rec)
        acc_band += torch.stack(rec.band)
        acc_total += torch.tensor(rec.total)
        acc_std += torch.tensor(rec.std)
        outs.append((img, lat))
    ref = {"band": acc_band / seeds, "total": acc_total / seeds,
           "std": acc_std / seeds}
    return ref, outs
