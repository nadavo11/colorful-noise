"""E45: FlowAlign on LTX-Video + our spectral phase op, for temporally-coherent video editing.

Motivation. FlowAlign (arXiv:2505.23145) edits VIDEO frame-by-frame on an IMAGE model (SD3):
the paper itself admits "temporal consistency for the edited object is limited, as no explicit
constraint is imposed." Our bet: run FlowAlign on a real video model (LTX-Video) and add the
E41/E43 low-band PHASE-keep op in the SPATIOTEMPORAL frequency domain (3D FFT over frame x H x W)
to constrain exactly that flicker. The 2D per-frame variant ~= the paper's approach (a control);
the 3D variant is the bet.

This is a one-clip FEASIBILITY probe (KEEP/KILL). Goal = a phase setting beats plain
FlowAlign-on-LTX (sbn off) on edited-object warp-error + structure-dist, holding CLIP-directional
editability within tol, on ONE LTX demo clip.

FlowAlign math is model-agnostic; this file ports the velocity/pack/VAE to LTX and reuses the
FlowAlign loop body. Paper hyperparams: zeta=0.01, CFG w in [5,7.5,10,13.5] (default 10), 33 steps.

Parts (--part): smoke (pipeline + VAE round-trip shape/const dump) ; gen ; analyze.
Run:  uv run experiments/e45_ltx_flowalign.py --part smoke --steps 8
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.5.1",
#     "diffusers==0.36.0",
#     "transformers==4.57.6",
#     "accelerate",
#     "sentencepiece",
#     "protobuf",
#     "imageio",
#     "imageio-ffmpeg",
#     "numpy",
#     "pillow",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# ///
import argparse
import os
import sys

import numpy as np
import torch

MODEL = "Lightricks/LTX-Video"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e45")


def load_ltx():
    from diffusers import LTXPipeline
    pipe = LTXPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def vae_roundtrip(pipe, frames_np):
    """frames_np: (F, H, W, 3) uint8. Encode -> normalize -> denormalize -> decode.
    Returns (recon_frames_np, latent) and prints every shape/const so the manual
    FlowAlign plumbing can be written against real numbers, not guesses."""
    vae = pipe.vae
    x = torch.from_numpy(frames_np).float() / 255.0
    x = (x.permute(3, 0, 1, 2)[None] * 2 - 1)          # (1, 3, F, H, W)
    x = x.to(vae.device, vae.dtype)
    print(f"[smoke] video tensor in: {tuple(x.shape)} dtype={x.dtype}", flush=True)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
    print(f"[smoke] raw latent: {tuple(z.shape)}  mean={z.float().mean():.3f} std={z.float().std():.3f}", flush=True)

    lm = pipe.__class__._normalize_latents
    mean = vae.latents_mean if hasattr(vae, "latents_mean") else vae.config.latents_mean
    std = vae.latents_std if hasattr(vae, "latents_std") else vae.config.latents_std
    mean = torch.tensor(mean) if not torch.is_tensor(mean) else mean
    std = torch.tensor(std) if not torch.is_tensor(std) else std
    sf = vae.config.scaling_factor
    print(f"[smoke] latents_mean shape {tuple(mean.shape)}  std shape {tuple(std.shape)}  scaling_factor={sf}", flush=True)
    zn = lm(z, mean, std, sf)
    print(f"[smoke] normalized latent: mean={zn.float().mean():.3f} std={zn.float().std():.3f}", flush=True)

    zd = pipe.__class__._denormalize_latents(zn, mean, std, sf).to(vae.dtype)
    # LTX VAE decode is timestep-conditioned; try the simple path, fall back if it needs temb.
    import inspect
    dparams = list(inspect.signature(vae.decode).parameters)
    print(f"[smoke] vae.decode params: {dparams}", flush=True)
    with torch.no_grad():
        try:
            ts = torch.zeros(zd.shape[0], device=zd.device, dtype=zd.dtype)
            out = vae.decode(zd, temb=ts, return_dict=False)[0]
        except TypeError:
            out = vae.decode(zd, return_dict=False)[0]
    print(f"[smoke] decoded tensor: {tuple(out.shape)}", flush=True)
    out = ((out.float() + 1) / 2).clamp(0, 1)[0].permute(1, 2, 3, 0).cpu().numpy()
    recon = (out * 255).round().astype(np.uint8)
    return recon, zn


def run_smoke(args):
    os.makedirs(OUT, exist_ok=True)
    pipe = load_ltx()
    print(f"[smoke] vae spatial/temporal compression: "
          f"{pipe.vae_spatial_compression_ratio}/{pipe.vae_temporal_compression_ratio}  "
          f"patch s/t: {pipe.transformer_spatial_patch_size}/{pipe.transformer_temporal_patch_size}", flush=True)

    H = W = args.size
    F = args.frames                                    # must be 8k+1
    gen = pipe(prompt="a calm sea with gentle waves at sunset",
               num_frames=F, height=H, width=W,
               num_inference_steps=args.steps,
               guidance_scale=3.0,
               generator=torch.Generator("cuda").manual_seed(0),
               output_type="np")
    frames = gen.frames[0]                              # (F, H, W, 3) float [0,1]
    frames = (np.asarray(frames) * 255).round().astype(np.uint8)
    print(f"[smoke] generated frames: {frames.shape} dtype={frames.dtype}", flush=True)
    import imageio.v3 as iio
    iio.imwrite(os.path.join(OUT, "smoke_gen.mp4"), frames, fps=8)

    recon, lat = vae_roundtrip(pipe, frames)
    n = min(len(frames), len(recon))
    err = np.abs(frames[:n].astype(float) - recon[:n].astype(float)).mean() / 255.0
    print(f"[smoke] VAE round-trip L1 (0..1): {err:.4f}  (frames out={recon.shape})", flush=True)
    iio.imwrite(os.path.join(OUT, "smoke_recon.mp4"), recon, fps=8)
    print(f"[smoke] latent packed-able shape (B,C,F,H,W)={tuple(lat.shape)}", flush=True)
    print("[smoke] DONE", flush=True)


# ---------------------------------------------------------------------------
# LTX plumbing for FlowAlign (shapes confirmed by the smoke: latent (1,128,F,H,W),
# F=(frames-1)//8+1, H=W=size//32; latents_mean/std (128,); scaling_factor=1.0).
# ---------------------------------------------------------------------------
FRAME_RATE = 25


def _latents_mean_std(vae):
    mean = vae.latents_mean if hasattr(vae, "latents_mean") else vae.config.latents_mean
    std = vae.latents_std if hasattr(vae, "latents_std") else vae.config.latents_std
    mean = mean if torch.is_tensor(mean) else torch.tensor(mean)
    std = std if torch.is_tensor(std) else torch.tensor(std)
    return mean, std


def ltx_encode(pipe, frames_np):
    """(F,H,W,3) uint8 -> normalized latent (1,128,Fl,Hl,Wl) float32 on cuda."""
    vae = pipe.vae
    x = torch.from_numpy(frames_np).float() / 255.0
    x = (x.permute(3, 0, 1, 2)[None] * 2 - 1).to(vae.device, vae.dtype)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
    mean, std = _latents_mean_std(vae)
    zn = pipe.__class__._normalize_latents(z, mean, std, vae.config.scaling_factor)
    return zn.float()


def ltx_decode(pipe, latent):
    """normalized latent (1,128,Fl,Hl,Wl) -> (F,H,W,3) uint8."""
    vae = pipe.vae
    mean, std = _latents_mean_std(vae)
    zd = pipe.__class__._denormalize_latents(latent, mean, std, vae.config.scaling_factor).to(vae.dtype)
    with torch.no_grad():
        try:
            ts = torch.zeros(zd.shape[0], device=zd.device, dtype=zd.dtype)
            out = vae.decode(zd, temb=ts, return_dict=False)[0]
        except TypeError:
            out = vae.decode(zd, return_dict=False)[0]
    out = ((out.float() + 1) / 2).clamp(0, 1)[0].permute(1, 2, 3, 0).cpu().numpy()
    return (out * 255).round().astype(np.uint8)


def ltx_conform(video_path, width, frames, height=None):
    """Real clip -> (F,H,W,3) uint8 at width x height (height defaults to width), F = nearest
    8k+1 <= frames (LTX needs it). Accepts any imageio path incl. 'imageio:cockatoo.mp4'."""
    import imageio.v3 as iio
    from PIL import Image
    height = height or width
    vid = np.asarray(iio.imread(video_path))
    if vid.ndim == 3:
        vid = vid[None]
    vid = vid[..., :3]
    n = max(((min(len(vid), frames) - 1) // 8) * 8 + 1, 9)
    idx = np.linspace(0, len(vid) - 1, n).round().astype(int)
    out = [np.asarray(Image.fromarray(vid[i]).convert("RGB").resize((width, height), Image.BICUBIC))
           for i in idx]
    return np.stack(out).astype(np.uint8)


def ltx_pack(pipe, lat):
    return pipe._pack_latents(lat, pipe.transformer_spatial_patch_size,
                              pipe.transformer_temporal_patch_size)


def ltx_unpack(pipe, packed, Fl, Hl, Wl):
    return pipe._unpack_latents(packed, Fl, Hl, Wl,
                                pipe.transformer_spatial_patch_size,
                                pipe.transformer_temporal_patch_size)


def ltx_schedule(pipe, steps, num_tokens):
    """LTX resolution-shifted sigma/timestep grid (mirrors the pipeline)."""
    from diffusers.pipelines.ltx.pipeline_ltx import retrieve_timesteps
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift
    cfg = pipe.scheduler.config
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    mu = calculate_shift(num_tokens, cfg.get("base_image_seq_len", 256),
                         cfg.get("max_image_seq_len", 4096),
                         cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
    retrieve_timesteps(pipe.scheduler, steps, "cuda", None, sigmas=sigmas, mu=mu)
    return pipe.scheduler.sigmas.float(), pipe.scheduler.timesteps.float()


def ltx_encode_prompt(pipe, prompt):
    pe, mask, _, _ = pipe.encode_prompt(prompt, do_classifier_free_guidance=False, device="cuda")
    return pe.float(), mask


@torch.no_grad()
def ltx_velocity(pipe, packed_x, t, pe, mask, Fl, Hl, Wl):
    rope = (pipe.vae_temporal_compression_ratio / FRAME_RATE,
            pipe.vae_spatial_compression_ratio, pipe.vae_spatial_compression_ratio)
    ts = t.expand(packed_x.shape[0]).to(pipe.dtype)
    v = pipe.transformer(hidden_states=packed_x.to(pipe.dtype),
                         encoder_hidden_states=pe.to(pipe.dtype), timestep=ts,
                         encoder_attention_mask=mask, num_frames=Fl, height=Hl, width=Wl,
                         rope_interpolation_scale=rope, return_dict=False)[0]
    return v.float()


# --- low-band PHASE-keep op: keep `out` magnitude, take `ref` phase inside a radial
#     band. dims=("HW") -> per-frame 2D; dims=("FHW") -> spatiotemporal 3D. ---
def _radial_norm(shape, dims, device):
    grids = []
    for d in dims:
        f = torch.fft.fftfreq(shape[d], device=device)
        view = [1] * len(shape); view[d] = shape[d]
        grids.append(f.view(view))
    r = sum(g ** 2 for g in grids).sqrt()
    return r / r.max().clamp(min=1e-8)


def band_phase_keep(out, ref, lo, hi, mode):
    """out/ref: (1,128,Fl,Hl,Wl). mode 'phase2d' -> fft over (H,W); 'phase3d' -> (F,H,W)."""
    dims = (-2, -1) if mode == "phase2d" else (-3, -2, -1)
    r = _radial_norm(out.shape, [d % out.ndim for d in dims], out.device)
    band = ((r >= lo) & (r <= hi)).float()
    Fo = torch.fft.fftn(out.float(), dim=dims)
    Fr = torch.fft.fftn(ref.float(), dim=dims)
    phi = Fr.angle() * band + Fo.angle() * (1.0 - band)
    mix = Fo.abs() * torch.exp(1j * phi)
    # keep DC (freq 0 along every transformed dim) from `out`, in the FREQUENCY domain
    idx = [slice(None)] * out.ndim
    for d in dims:
        idx[d % out.ndim] = 0
    mix[tuple(idx)] = Fo[tuple(idx)]
    return torch.fft.ifftn(mix, dim=dims).real.to(out.dtype)


def vel_sbn_video(pipe, vp, vref, mode, cut, Fl, Hl, Wl):
    """Apply the phase-keep on the CFG velocity's low band (our structure op)."""
    if mode == "off":
        return vp
    a = ltx_unpack(pipe, vp, Fl, Hl, Wl)
    r = ltx_unpack(pipe, vref, Fl, Hl, Wl)
    a = band_phase_keep(a, r, 0.0, cut, mode)
    return ltx_pack(pipe, a)


@torch.no_grad()
def flowalign_video(pipe, x0_packed, C_src, C_tar, sig, ts, seed, w, zeta,
                    Fl, Hl, Wl, sbn_mode="off", sbn_cut=0.0, n_max=None, n_avg=1):
    """FlowAlign (arXiv:2505.23145) on LTX. Defaults (sbn off) = plain FlowAlign;
    C_tar==C_src reproduces the source clip (identity gate). 3 velocity forwards/step.
    n_max: edit only the last n_max steps (skip the highest-noise early steps, as canonical
    FlowEdit does -- editing at sigma~1 is unstable). n_avg: fresh-noise averaging per step."""
    steps = len(sig) - 1
    if n_max is None:
        n_max = steps
    gen = torch.Generator("cuda").manual_seed(seed)
    xt = x0_packed.clone()
    for i in range(steps):
        if steps - i > n_max:                                   # skip high-noise early steps
            continue
        s_hi, s_lo = float(sig[i]), float(sig[i + 1])
        vp_avg = torch.zeros_like(xt)
        vq_avg = torch.zeros_like(xt)
        term_avg = torch.zeros_like(xt)
        for _ in range(n_avg):
            eps = torch.randn(x0_packed.shape, generator=gen, device="cuda").float()
            qt = (1 - s_hi) * x0_packed + s_hi * eps             # forward-diffused source
            pt = xt + qt - x0_packed                            # == x_tar
            v_pt_src = ltx_velocity(pipe, pt, ts[i], C_src[0], C_src[1], Fl, Hl, Wl)
            v_pt_tar = ltx_velocity(pipe, pt, ts[i], C_tar[0], C_tar[1], Fl, Hl, Wl)
            vp = v_pt_src + w * (v_pt_tar - v_pt_src)            # CFG, source prompt as negative
            vp = vel_sbn_video(pipe, vp, v_pt_src, sbn_mode, sbn_cut, Fl, Hl, Wl)
            vq = ltx_velocity(pipe, qt, ts[i], C_src[0], C_src[1], Fl, Hl, Wl)
            vp_avg += vp / n_avg
            vq_avg += vq / n_avg
            term_avg += ((qt - s_hi * vq) - (pt - s_hi * vp)) / n_avg   # E[q0|qt]-E[p0|pt]
        xt = xt + (s_lo - s_hi) * (vp_avg - vq_avg) + zeta * term_avg
    return xt


@torch.no_grad()
def flowedit_video(pipe, x0_packed, C_src, C_tar, C_unc, sig, ts, seed, src_gs, tar_gs,
                   Fl, Hl, Wl, n_max=None, n_min=0, n_avg=1):
    """Canonical FlowEdit (arXiv:2412.08629, fallenshock/FlowEdit) on LTX -- the reference
    baseline. Separate src/tar CFG vs the UNCONDITIONAL; fresh per-step noise + n_avg; edits
    only the [n_min, n_max] step window (skips the highest-noise steps). C_tar==C_src ~= identity."""
    steps = len(sig) - 1
    if n_max is None:
        n_max = steps
    gen = torch.Generator("cuda").manual_seed(seed)
    zt = x0_packed.clone()

    def cfg(z, t, C, gs):
        vu = ltx_velocity(pipe, z, t, C_unc[0], C_unc[1], Fl, Hl, Wl)
        vc = ltx_velocity(pipe, z, t, C[0], C[1], Fl, Hl, Wl)
        return vu + gs * (vc - vu)

    for i in range(steps):
        if steps - i > n_max:                                   # skip high-noise early steps
            continue
        s_hi, s_lo = float(sig[i]), float(sig[i + 1])
        if steps - i > n_min:                                   # editing branch
            Vd = torch.zeros_like(zt)
            for _ in range(n_avg):
                eps = torch.randn(x0_packed.shape, generator=gen, device="cuda").float()
                zt_src = (1 - s_hi) * x0_packed + s_hi * eps
                zt_tar = zt + (zt_src - x0_packed)
                Vt_src = cfg(zt_src, ts[i], C_src, src_gs)
                Vt_tar = cfg(zt_tar, ts[i], C_tar, tar_gs)
                Vd += (Vt_tar - Vt_src) / n_avg
            zt = zt + (s_lo - s_hi) * Vd
        else:                                                   # regular-sampling tail (n_min)
            zt = zt + (s_lo - s_hi) * cfg(zt, ts[i], C_tar, tar_gs)
    return zt


# (key, source prompt, edit/target prompt) -- one LTX-generated demo clip for the probe.
SCENE = ("rover", "a small toy car driving across a wooden table",
         "a small toy tank driving across a wooden table")


@torch.no_grad()
def edit_fbf(pipe, src_frames, C_src, C_tar, args, Hl, Wl):
    """The paper's method: edit each frame as an INDEPENDENT 1-frame clip (no temporal
    coupling), independent noise per frame -> realistic flicker. The temporal reference."""
    sig, ts = ltx_schedule(pipe, args.steps, Hl * Wl)      # single-frame token count
    outs = []
    for i in range(len(src_frames)):
        x0 = ltx_pack(pipe, ltx_encode(pipe, src_frames[i:i + 1]))
        xe = flowalign_video(pipe, x0, C_src, C_tar, sig, ts, args.seed + i, args.w, args.zeta,
                             1, Hl, Wl)
        outs.append(ltx_decode(pipe, ltx_unpack(pipe, xe, 1, Hl, Wl))[0])
    return np.stack(outs)


def run_gen(args):
    import imageio.v3 as iio
    os.makedirs(OUT, exist_ok=True)
    pipe = load_ltx()
    key, src, tgt = SCENE
    if args.src_caption:
        src = args.src_caption                                   # real-clip source caption
    if args.edit_prompt:
        tgt = args.edit_prompt
    H = W = args.size
    Fl = (args.frames - 1) // pipe.vae_temporal_compression_ratio + 1
    Hl = H // pipe.vae_spatial_compression_ratio
    Wl = W // pipe.vae_spatial_compression_ratio
    num_tokens = Fl * Hl * Wl
    sig, ts = ltx_schedule(pipe, args.steps, num_tokens)
    C_src = ltx_encode_prompt(pipe, src)
    C_tar = ltx_encode_prompt(pipe, tgt)

    # source clip: a real clip (--real_video, lever 2) or generated from the source prompt
    srcp = os.path.join(OUT, "source.mp4")
    if args.real_video:
        src_frames = ltx_conform(args.real_video, H, args.frames)
        print(f"[gen] real source clip {args.real_video} -> {src_frames.shape}", flush=True)
    else:
        gen = pipe(prompt=src, num_frames=args.frames, height=H, width=W,
                   num_inference_steps=args.steps, guidance_scale=3.0,
                   generator=torch.Generator("cuda").manual_seed(args.seed), output_type="np")
        src_frames = (np.asarray(gen.frames[0]) * 255).round().astype(np.uint8)
    iio.imwrite(srcp, src_frames, fps=8)
    x0 = ltx_pack(pipe, ltx_encode(pipe, src_frames))
    print(f"[gen] source clip {src_frames.shape}  latent-tokens={num_tokens}  packed={tuple(x0.shape)}", flush=True)

    def decode_save(xe, name):
        frames = ltx_decode(pipe, ltx_unpack(pipe, xe, Fl, Hl, Wl))
        iio.imwrite(os.path.join(OUT, f"{name}.mp4"), frames, fps=8)
        return frames

    # identity gate: C_tar = C_src must reproduce the source clip
    xe = flowalign_video(pipe, x0, C_src, C_src, sig, ts, args.seed, args.w, args.zeta, Fl, Hl, Wl)
    recon = decode_save(xe, "recon")
    n = min(len(src_frames), len(recon))
    recon_l1 = float(np.abs(src_frames[:n].astype(float) - recon[:n].astype(float)).mean() / 255.0)
    print(f"[gen] identity-gate recon L1 = {recon_l1:.4f}", flush=True)

    # edit conditions: plain FlowAlign baseline (sbn off) + 2D/3D phase over an sbn_cut sweep.
    # baseline = the number to beat; phase2d ~ paper's frame-by-frame control; phase3d = our bet.
    cuts = [float(c) for c in str(args.cuts).split(",") if c]
    conds = {"baseline": ("off", 0.0)}
    for cut in cuts:
        conds[f"phase2d_c{cut}"] = ("phase2d", cut)
        conds[f"phase3d_c{cut}"] = ("phase3d", cut)
    for name, (mode, cut) in conds.items():
        if os.path.exists(os.path.join(OUT, f"{name}.mp4")):
            continue
        xe = flowalign_video(pipe, x0, C_src, C_tar, sig, ts, args.seed, args.w, args.zeta,
                             Fl, Hl, Wl, sbn_mode=mode, sbn_cut=cut)
        decode_save(xe, name)
        print(f"[gen] edit {name} done", flush=True)

    cond_list = list(conds)
    if args.fbf:
        cond_list.append("fbf")
        if not os.path.exists(os.path.join(OUT, "fbf.mp4")):
            fbf = edit_fbf(pipe, src_frames, C_src, C_tar, args, Hl, Wl)
            iio.imwrite(os.path.join(OUT, "fbf.mp4"), fbf, fps=8)
            print("[gen] edit fbf (frame-by-frame = paper's method) done", flush=True)

    import json
    with open(os.path.join(OUT, "gen_report.json"), "w") as f:
        json.dump({"scene": key, "src": src, "tgt": tgt, "recon_l1": recon_l1,
                   "conds": cond_list, "params": {"steps": args.steps, "frames": args.frames,
                   "size": args.size, "w": args.w, "zeta": args.zeta, "cuts": cuts}}, f, indent=2)
    print("[gen] DONE", flush=True)


# ---------------------------------------------------------------------------
# Metric bundle: DINO structure-dist + CLIP-directional (per-frame, averaged) +
# RAFT warp-error (temporal flicker), global and edited-region-masked.
# ---------------------------------------------------------------------------
def _load_raft():
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    w = Raft_Small_Weights.DEFAULT
    return raft_small(weights=w).eval().cuda(), w.transforms()


def _warp(img, flow):
    """img (1,3,H,W), flow (1,2,H,W) -> img sampled at x+flow (align frame t+1 to t)."""
    _, _, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device),
                            indexing="ij")
    gx = (xx + flow[0, 0]) / (W - 1) * 2 - 1
    gy = (yy + flow[0, 1]) / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None]
    return torch.nn.functional.grid_sample(img, grid, align_corners=True, padding_mode="border")


@torch.no_grad()
def warp_error(raft, src_frames, edit_frames, tau=0.1):
    """Flow from the SOURCE clip (true motion); warp the EDIT frames and measure residual
    flicker. Returns (global, masked) mean-squared warp error; mask = edited region."""
    model, tf = raft
    def t(f):  # (H,W,3) uint8 -> (1,3,H,W) [0,1] cuda
        return torch.from_numpy(f).float().permute(2, 0, 1)[None].cuda() / 255.0
    n = min(len(src_frames), len(edit_frames))
    g_errs, m_errs = [], []
    for i in range(n - 1):
        s0, s1 = t(src_frames[i]), t(src_frames[i + 1])
        e0, e1 = t(edit_frames[i]), t(edit_frames[i + 1])
        a, b = tf(s0, s1)
        flow = model(a, b)[-1]                                  # forward flow s_i -> s_{i+1}
        warped = _warp(e1, flow)
        resid = (warped - e0).pow(2).mean(1, keepdim=True)      # (1,1,H,W)
        mask = ((e0 - s0).abs().mean(1, keepdim=True) > tau).float()   # edited region at frame i
        g_errs.append(float(resid.mean()))
        m_errs.append(float((resid * mask).sum() / (mask.sum() + 1e-6)))
    return float(np.mean(g_errs)), float(np.mean(m_errs))


def run_analyze(args):
    import json
    import imageio.v3 as iio
    from PIL import Image
    from struct_metrics import load_dino, load_clip, structure_distance, clip_directional
    rp = os.path.join(OUT, "gen_report.json")
    rep = json.load(open(rp))
    src, tgt = rep["src"], rep["tgt"]
    src_frames = iio.imread(os.path.join(OUT, "source.mp4"))
    dino, clip = load_dino("cuda"), load_clip()
    raft = _load_raft()

    def pil(a):
        return Image.fromarray(a).convert("RGB")
    out = {}
    for cond in rep["conds"]:
        p = os.path.join(OUT, f"{cond}.mp4")
        if not os.path.exists(p):
            continue
        ed = iio.imread(p)
        n = min(len(src_frames), len(ed))
        sd = np.mean([structure_distance(dino, pil(ed[i]), pil(src_frames[i])) for i in range(n)])
        cd = np.mean([clip_directional(clip, pil(src_frames[i]), pil(ed[i]), src, tgt) for i in range(n)])
        wg, wm = warp_error(raft, src_frames, ed)
        out[cond] = {"struct_dist": float(sd), "clip_dir": float(cd),
                     "warp_global": wg, "warp_masked": wm}
        print(f"[an] {cond:16s} struct={sd:.4f} clip={cd:+.4f} warpG={wg:.5f} warpM={wm:.5f}", flush=True)

    rep["metrics"] = out
    json.dump(rep, open(rp, "w"), indent=2)

    # GOAL: a phase variant beats baseline on struct-dist AND edited-region warp-error,
    # holding CLIP-directional within tol. warp-masked is the headline (the paper's gap).
    base = out.get("baseline", {})
    wins = []
    for cond, m in out.items():
        if cond == "baseline":
            continue
        if (m["struct_dist"] < base["struct_dist"] and m["warp_masked"] < base["warp_masked"]
                and m["clip_dir"] >= base["clip_dir"] - args.clip_tol):
            wins.append((cond, m))
    # Paper-faithful temporal comparison: does using a video model (any 'video' cond) beat the
    # paper's frame-by-frame editing on flicker? (fbf = paper's method.)
    fbf = out.get("fbf")
    if fbf is not None:
        vid = {k: v for k, v in out.items() if k != "fbf"}
        best = min(vid.items(), key=lambda kv: kv[1]["warp_masked"])
        print(f"\n[goal] TEMPORAL vs paper (frame-by-frame): fbf warpM={fbf['warp_masked']:.5f} "
              f"struct={fbf['struct_dist']:.4f} clip={fbf['clip_dir']:+.4f}", flush=True)
        print(f"[goal]   best video method '{best[0]}': warpM={best[1]['warp_masked']:.5f} "
              f"struct={best[1]['struct_dist']:.4f} clip={best[1]['clip_dir']:+.4f}", flush=True)
        ratio = fbf["warp_masked"] / max(best[1]["warp_masked"], 1e-9)
        verdict = "PASS" if best[1]["warp_masked"] < fbf["warp_masked"] else "FAIL"
        print(f"[goal]   {verdict}: video editing is {ratio:.1f}x less flicker than frame-by-frame", flush=True)

    print("\n[goal] baseline: "
          f"struct={base.get('struct_dist'):.4f} warpM={base.get('warp_masked'):.5f} "
          f"clip={base.get('clip_dir'):+.4f}", flush=True)
    if wins:
        print(f"[goal] PASS ({len(wins)} variants beat baseline on struct+masked-warp, clip held):", flush=True)
        for cond, m in wins:
            print(f"[goal]   {cond}: struct={m['struct_dist']:.4f} (Δ{m['struct_dist']-base['struct_dist']:+.4f}) "
                  f"warpM={m['warp_masked']:.5f} (Δ{m['warp_masked']-base['warp_masked']:+.5f}) "
                  f"clip={m['clip_dir']:+.4f}", flush=True)
    else:
        print("[goal] NO winner; inspect results/e45/*.mp4 + gen_report.json", flush=True)
    print("[an] DONE", flush=True)


def run_compare(args):
    """Diagnostic: canonical FlowEdit (reference) vs my FlowAlign variants at LTX-native
    resolution (default 704x480, landscape). Isolates whether distortion came from over-guidance
    (w=10), editing-all-steps (no n_max window), or square low-res. Saves clips to eyeball."""
    import imageio.v3 as iio
    os.makedirs(OUT, exist_ok=True)
    pipe = load_ltx()
    key, src, tgt = SCENE
    if args.src_caption:
        src = args.src_caption
    if args.edit_prompt:
        tgt = args.edit_prompt
    H, W = args.height, args.width
    Fl = (args.frames - 1) // pipe.vae_temporal_compression_ratio + 1
    Hl, Wl = H // pipe.vae_spatial_compression_ratio, W // pipe.vae_spatial_compression_ratio
    sig, ts = ltx_schedule(pipe, args.steps, Fl * Hl * Wl)
    C_src, C_tar = ltx_encode_prompt(pipe, src), ltx_encode_prompt(pipe, tgt)
    C_unc = ltx_encode_prompt(pipe, "")
    nmax = args.steps - max(2, round(args.skip_frac * args.steps))   # skip the first ~skip_frac steps
    print(f"[cmp] {W}x{H} {args.frames}f, steps={args.steps}, n_max={nmax} "
          f"(skip first {args.steps - nmax}); src_gs={args.src_gs} tar_gs={args.tar_gs}", flush=True)

    if args.real_video:
        src_frames = ltx_conform(args.real_video, W, args.frames, H)
    else:
        g = pipe(prompt=src, num_frames=args.frames, height=H, width=W,
                 num_inference_steps=args.steps, guidance_scale=3.0,
                 generator=torch.Generator("cuda").manual_seed(args.seed), output_type="np")
        src_frames = (np.asarray(g.frames[0]) * 255).round().astype(np.uint8)
    iio.imwrite(os.path.join(OUT, "source.mp4"), src_frames, fps=8)
    x0 = ltx_pack(pipe, ltx_encode(pipe, src_frames))
    print(f"[cmp] source {src_frames.shape} packed {tuple(x0.shape)}", flush=True)

    def dec(xe, name):
        fr = ltx_decode(pipe, ltx_unpack(pipe, xe, Fl, Hl, Wl))
        iio.imwrite(os.path.join(OUT, f"{name}.mp4"), fr, fps=8)
        print(f"[cmp] {name} done", flush=True)

    dec(flowedit_video(pipe, x0, C_src, C_src, C_unc, sig, ts, args.seed, args.src_gs, args.tar_gs,
                       Fl, Hl, Wl, n_max=nmax, n_avg=args.n_avg), "identity")
    dec(flowedit_video(pipe, x0, C_src, C_tar, C_unc, sig, ts, args.seed, args.src_gs, args.tar_gs,
                       Fl, Hl, Wl, n_max=nmax, n_avg=args.n_avg), "flowedit")
    dec(flowalign_video(pipe, x0, C_src, C_tar, sig, ts, args.seed, args.w, args.zeta,
                        Fl, Hl, Wl), "flowalign_hi_allsteps")
    dec(flowalign_video(pipe, x0, C_src, C_tar, sig, ts, args.seed, args.w, args.zeta,
                        Fl, Hl, Wl, n_max=nmax, n_avg=args.n_avg), "flowalign_hi_window")
    dec(flowalign_video(pipe, x0, C_src, C_tar, sig, ts, args.seed, args.w_lo, args.zeta,
                        Fl, Hl, Wl, n_max=nmax, n_avg=args.n_avg), "flowalign_lo_window")
    print("[cmp] DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="smoke")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w", type=float, default=10.0)        # FlowAlign CFG (paper default)
    ap.add_argument("--zeta", type=float, default=0.01)     # FlowAlign source-consistency (paper)
    ap.add_argument("--cuts", default="0.2,0.35")           # sbn_cut sweep for the phase ops
    ap.add_argument("--clip_tol", type=float, default=0.01)  # editability tolerance vs baseline
    ap.add_argument("--fbf", action="store_true")           # also run paper's frame-by-frame baseline
    ap.add_argument("--out_tag", default="")                # suffix for results dir (per-w frontier runs)
    ap.add_argument("--real_video", default="")             # lever 2: edit a real clip instead of generating
    ap.add_argument("--src_caption", default="")            # source caption (real clip / override)
    ap.add_argument("--edit_prompt", default="")            # edit prompt (override SCENE target)
    # compare-part knobs (FlowEdit baseline vs FlowAlign diagnostic)
    ap.add_argument("--width", type=int, default=704)       # LTX-native landscape default
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--src_gs", type=float, default=1.5)    # FlowEdit source guidance (vs uncond)
    ap.add_argument("--tar_gs", type=float, default=3.5)    # FlowEdit target guidance (LTX ~3)
    ap.add_argument("--w_lo", type=float, default=3.0)      # low FlowAlign guidance to test over-drive
    ap.add_argument("--skip_frac", type=float, default=0.15)  # fraction of early steps to skip (n_max)
    ap.add_argument("--n_avg", type=int, default=1)         # fresh-noise averaging per edit step
    args = ap.parse_args()
    if args.out_tag:
        global OUT
        OUT = f"{OUT}_{args.out_tag}"
    for part in args.part.split(","):
        if part == "smoke":
            run_smoke(args)
        elif part == "gen":
            run_gen(args)
        elif part == "analyze":
            run_analyze(args)
        elif part == "compare":
            run_compare(args)
        else:
            print(f"[e45] part '{part}' not implemented yet", flush=True)


if __name__ == "__main__":
    main()
