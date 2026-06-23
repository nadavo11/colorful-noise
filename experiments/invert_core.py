"""Shared RF-inversion + spectral-clamp edit core for FLUX.

Factored out of spectral_demo.py so the Gradio invert tab and the e41 calibration
harness drive the *exact same* code path. Everything is parameterized by an explicit
`pipe`; the only module state is the latent-geometry constants below.

`forward_edit` also implements the RF-inversion `eta` controller (Rout et al.): at each
step it can pull the velocity toward the field that reconstructs the source clean latent
`x0_packed`, with strength `eta` over an optional step `eta_window`. eta=0 reproduces the
plain (vanilla) inversion edit; eta=1 over all steps ~ reconstruction. This gives a
faithful RF-inversion baseline whose faithfulness<->editability knob we can sweep.
"""
import torch

FH = FW = 128                   # Flux latent dims (1024px image -> 128x128 latent)
INV_SIZE = 1024                 # px the real image is VAE-encoded at
SEQLEN = 512                    # Flux txt_ids length
N_BINS = 24                     # radial FFT bands


# ---------------------------------------------------------------------------
# Flux denoising plumbing (copied verbatim from spectral_demo's inlined helpers)
# ---------------------------------------------------------------------------
def flux_sigmas(pipe, steps, seq_len=(FH // 2) * (FW // 2)):
    """Flux's resolution-shifted sigma grid (steps+1, decreasing, [-1]=0)."""
    import numpy as np
    from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift
    cfg = pipe.scheduler.config
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    try:
        mu = calculate_shift(seq_len, cfg.get("base_image_seq_len", 256),
                             cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        retrieve_timesteps(pipe.scheduler, steps, "cuda", sigmas=sigmas, mu=mu)
    except Exception as e:
        print(f"[invert] shift schedule failed ({e}); plain set_timesteps", flush=True)
        pipe.scheduler.set_timesteps(steps, device="cuda")
    return pipe.scheduler.sigmas.float()


def gids(pipe, guidance):
    """Per-run constants: guidance embed, txt_ids (zeros), img_ids (positional)."""
    img_ids = pipe._prepare_latent_image_ids(1, FH // 2, FW // 2, "cuda", pipe.dtype)
    txt_ids = torch.zeros(SEQLEN, 3, device="cuda", dtype=pipe.dtype)
    g = torch.full([1], float(guidance), device="cuda", dtype=torch.float32)
    return g, txt_ids, img_ids


@torch.no_grad()
def flux_velocity(pipe, packed_x, sigma, pe, ppe, g):
    """Flow-matching velocity v(x, sigma | conditioning) in PACKED latent space."""
    guidance, txt_ids, img_ids = g
    t = torch.full((packed_x.shape[0],), float(sigma), device="cuda", dtype=pipe.dtype)
    v = pipe.transformer(hidden_states=packed_x.to(pipe.dtype), timestep=t,
                         guidance=guidance, pooled_projections=ppe.to(pipe.dtype),
                         encoder_hidden_states=pe.to(pipe.dtype),
                         txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
    return v.float()


def pack(pipe, lat):
    return pipe._pack_latents(lat.cuda().float(), 1, 16, FH, FW)


def unpack(pipe, packed):
    return pipe._unpack_latents(packed, INV_SIZE, INV_SIZE, pipe.vae_scale_factor)


def vae_encode(vae, pil):
    """Real image -> generation-space latent (1, 16, FH, FW)."""
    import numpy as np
    img = pil.convert("RGB").resize((INV_SIZE, INV_SIZE))
    x = torch.from_numpy(np.asarray(img).copy()).float() / 255.0
    x = (x.permute(2, 0, 1)[None] * 2 - 1).to(vae.dtype).cuda()
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
    return ((z - vae.config.shift_factor) * vae.config.scaling_factor).float().cpu()


def encode_prompt(pipe, prompt):
    """(pe, ppe) float embeddings for a prompt (used by the harness)."""
    pe, ppe, _ = pipe.encode_prompt(prompt=prompt, prompt_2=prompt, device="cuda",
                                    num_images_per_prompt=1, max_sequence_length=512)
    return pe.float(), ppe.float()


# ---------------------------------------------------------------------------
# Low-band spectral clamp (the structure-preservation operator)
# ---------------------------------------------------------------------------
def band_centers_norm(device):
    from spectral_ops import radial_bins
    rr = radial_bins(FH, FW, device)
    edges = torch.linspace(0, rr.max() + 1e-6, N_BINS + 1, device=device)
    c = 0.5 * (edges[:-1] + edges[1:])
    return c / rr.max()


def sbn_low(gen, ref, cut, strength, idx):
    from spectral_ops import band_power, psd_match
    low = band_centers_norm(gen.device) < cut
    rbp = band_power((torch.fft.fft2(ref.float()).abs() ** 2)[0], idx, N_BINS)
    cbp = band_power((torch.fft.fft2(gen.float()).abs() ** 2)[0], idx, N_BINS)
    tgt = cbp.clone()
    s = float(strength)
    tgt[:, low] = (cbp[:, low].clamp(min=1e-8) ** (1 - s)) * (rbp[:, low].clamp(min=1e-8) ** s)
    return psd_match(gen, tgt, idx, N_BINS)


def band_phase_keep(out, ref, lo, hi):
    """Keep `out`'s magnitude everywhere; take phase from `ref` (source) inside the
    radial band [lo, hi] (fractions of the corner), from `out` elsewhere. DC kept
    from `out`. The radial mask is Hermitian-symmetric, so the ifft stays real."""
    from spectral_ops import radial_bins
    rr = radial_bins(FH, FW, out.device)
    rmax = rr.max()
    band = ((rr >= lo * rmax) & (rr <= hi * rmax)).float()[None, None]
    Fo, Fr = torch.fft.fft2(out.float()), torch.fft.fft2(ref.float())
    phi = Fr.angle() * band + Fo.angle() * (1.0 - band)
    mix = Fo.abs() * torch.exp(1j * phi)
    mix[..., 0, 0] = Fo[..., 0, 0]
    return torch.fft.ifft2(mix).real.to(out.dtype)


def inv_clamp(gen, ref, mode, cut, strength, idx, M, cen_k, phase_band=(0.0, 0.25)):
    """gen, ref: unpacked (1, 16, FH, FW) cuda. Pull gen's low band toward ref."""
    if mode == "sbn":
        return sbn_low(gen, ref, cut, strength, idx)
    if mode == "phase":
        out = sbn_low(gen, ref, cut, strength, idx)
        return band_phase_keep(out, ref, phase_band[0], phase_band[1])
    if mode == "adain":
        from spectral_adain import spectral_adain
        sources = [ref if cen_k[k] < cut else gen for k in range(M.shape[0])]
        out = spectral_adain(gen, sources, M)
        s = float(strength)
        return gen * (1 - s) + out * s
    return gen


# ---------------------------------------------------------------------------
# RF inversion + edit
# ---------------------------------------------------------------------------
@torch.no_grad()
def rf_invert(pipe, pe, ppe, x0_packed, sig, g):
    """Reverse Euler sigma low->high (clean -> noise). Returns (noise, traj) where
    traj[i] = unpacked latent at sigma sig[i] (cpu)."""
    steps = len(sig) - 1
    x = x0_packed
    traj = [None] * (steps + 1)
    traj[steps] = unpack(pipe, x).float().cpu()
    for i in range(steps - 1, -1, -1):
        s_lo, s_hi = float(sig[i + 1]), float(sig[i])
        v = flux_velocity(pipe, x, s_lo, pe, ppe, g)
        x = x + (s_hi - s_lo) * v
        traj[i] = unpack(pipe, x).float().cpu()
    return x, traj


@torch.no_grad()
def forward_edit(pipe, pe, ppe, x_noise, sig, g, traj=None, mode=None, cut=0.25,
                 strength=1.0, window=None, idx=None, M=None, cen_k=None,
                 phase_band=(0.0, 0.25), x0_packed=None, eta=0.0, eta_window=None):
    """Integrate sigma high->low (noise -> clean) under conditioning (pe, ppe).

    Spectral clamp (our method): if `traj` is given, pull the low band back to
    traj[i] on the steps inside `window`.
    RF-inversion controller (the baseline knob): if `eta`>0 and `x0_packed` given,
    blend the velocity toward the source-reconstruction field v_target=(x-x0)/sigma on
    the steps inside `eta_window`. eta=0 -> plain inversion edit.

    Returns the final unpacked latent (1, 16, FH, FW). Callers decode it themselves.
    """
    steps = len(sig) - 1
    x = x_noise
    for i in range(steps):
        s_hi, s_lo = float(sig[i]), float(sig[i + 1])
        if traj is not None and (window is None or window[0] <= i <= window[1]):
            lat = inv_clamp(unpack(pipe, x).float(), traj[i].cuda(), mode, cut,
                            strength, idx, M, cen_k, phase_band)
            x = pack(pipe, lat)
        v = flux_velocity(pipe, x, s_hi, pe, ppe, g)
        if eta > 0 and x0_packed is not None and (eta_window is None or
                                                  eta_window[0] <= i <= eta_window[1]):
            v_target = (x - x0_packed) / max(s_hi, 1e-6)
            v = v + float(eta) * (v_target - v)
        x = x + (s_lo - s_hi) * v
    return unpack(pipe, x).float()


# ---------------------------------------------------------------------------
# FlowAlign (Kim et al. 2025, arXiv:2505.23145): inversion-free editing =
# FlowEdit + a source-consistency TERMINAL-POINT regularizer, with two optional
# spectral twists. Shared verbatim by the demo's FlowAlign tab and the e43 harness.
# ---------------------------------------------------------------------------
FA_SBN_MODES = ["off", "band power", "mag", "phase", "both"]


def _gauss_lowpass_lat(lat, cutoff, renorm):
    """Gaussian radial low-pass of an UNPACKED (1,16,FH,FW) latent/velocity. cutoff in [0,1]
    (normalized radius); cutoff>=1 is passthrough. Real radially-symmetric mask -> stays real."""
    if cutoff >= 1.0:
        return lat
    from latent_spectral_ops import radial_norm
    rn = radial_norm(FH, FW, lat.device)
    mask = torch.exp(-(rn / max(cutoff, 1e-3)) ** 2)
    out = torch.fft.ifft2(torch.fft.fft2(lat) * mask).real
    if renorm:
        out = out * (lat.norm() / out.norm().clamp_min(1e-8))
    return out


def gauss_lowpass(pipe, vd_packed, cutoff, renorm):
    """Packed (Flux) wrapper of _gauss_lowpass_lat."""
    if cutoff >= 1.0:
        return vd_packed
    out = _gauss_lowpass_lat(unpack(pipe, vd_packed).float(), cutoff, renorm)
    return pack(pipe, out.to(vd_packed.dtype))


def _vel_sbn_lat(a, r, mode, cut, strength):
    """Spectrally clamp UNPACKED velocity `a` toward reference `r` in the low radial band
    [0,cut]. mode=="off"/cut<=0 is passthrough. E37 velocity-SBN ops + low-band phase lock."""
    if mode == "off" or cut <= 0.0:
        return a
    import velocity_spectral_ops as VEL
    if mode in ("band power", "both"):
        a = VEL.bandpower_match_band(a, r, 0.0, cut, strength)
    elif mode == "mag":
        a = VEL.mag_transplant_band(a, r, 0.0, cut, strength)
    if mode in ("phase", "both"):
        a = band_phase_keep(a, r, 0.0, cut)
    return a


def vel_sbn(pipe, vp, v_ref, mode, cut, strength):
    """Packed (Flux) wrapper of _vel_sbn_lat: clamp `vp` toward `v_ref` (=v(pt,c_src))."""
    if mode == "off" or cut <= 0.0:
        return vp
    a = _vel_sbn_lat(unpack(pipe, vp).float(), unpack(pipe, v_ref).float(), mode, cut, strength)
    return pack(pipe, a.to(vp.dtype))


@torch.no_grad()
def flowalign(pipe, x0_packed, C_src, C_tar, sig, seed, g, w, zeta,
              sbn_mode="off", sbn_cut=0.0, sbn_strength=1.0,
              term_start_cut=1.0, term_end_cut=1.0, term_renorm=False):
    """FlowAlign on FLUX with the two spectral twists. Defaults (sbn off, term cuts=1.0)
    reproduce plain FlowAlign; C_tar==C_src reproduces the source exactly (identity gate).
    C_src/C_tar = (pe, ppe) cuda tensors. 3 velocity forwards/step. Returns PACKED latent."""
    steps = len(sig) - 1
    eps = torch.randn(x0_packed.shape, generator=torch.Generator("cuda").manual_seed(seed),
                      device="cuda").float()
    xt = x0_packed.clone()
    for i in range(steps):
        frac = i / max(steps - 1, 1)
        cutoff = term_start_cut + (term_end_cut - term_start_cut) * frac   # anneal low -> high
        s_hi, s_lo = float(sig[i]), float(sig[i + 1])
        qt = (1 - s_hi) * x0_packed + s_hi * eps                # forward-diffused source
        pt = xt + qt - x0_packed                                # == x_tar
        v_pt_src = flux_velocity(pipe, pt, s_hi, C_src[0], C_src[1], g)
        v_pt_tar = flux_velocity(pipe, pt, s_hi, C_tar[0], C_tar[1], g)
        vp = v_pt_src + w * (v_pt_tar - v_pt_src)               # CFG, source prompt as negative
        vp = vel_sbn(pipe, vp, v_pt_src, sbn_mode, sbn_cut, sbn_strength)
        vq = flux_velocity(pipe, qt, s_hi, C_src[0], C_src[1], g)
        term = (qt - s_hi * vq) - (pt - s_hi * vp)             # E[q0|qt] - E[p0|pt]
        term = gauss_lowpass(pipe, term, cutoff, term_renorm)
        xt = xt + (s_lo - s_hi) * (vp - vq) + zeta * term
    return xt


# ---------------------------------------------------------------------------
# SD3 path (StableDiffusion3Pipeline) -- FlowAlign on the paper's backbone.
# SD3 latents are UNPACKED (1,16,FH,FW) with the SAME geometry as Flux, so the
# spectral ops are reused via their `_lat` cores. Differences from Flux: no
# latent packing, no distilled guidance embed / ids, and the transformer takes
# timestep in [0, num_train_timesteps] (sigma*1000) instead of Flux's raw sigma.
# ---------------------------------------------------------------------------
def sd3_sigmas(pipe, steps):
    """SD3 sigma grid (steps+1, decreasing, [-1]=0). Shift lives in the scheduler config."""
    pipe.scheduler.set_timesteps(steps, device="cuda")
    return pipe.scheduler.sigmas.float()


@torch.no_grad()
def sd3_velocity(pipe, lat, sigma, pe, ppe):
    """Flow-matching velocity v(x, sigma | conditioning) for SD3 in UNPACKED latent space."""
    n_train = pipe.scheduler.config.get("num_train_timesteps", 1000)
    t = torch.full((lat.shape[0],), float(sigma) * n_train, device="cuda", dtype=pipe.dtype)
    v = pipe.transformer(hidden_states=lat.to(pipe.dtype), timestep=t,
                         encoder_hidden_states=pe.to(pipe.dtype),
                         pooled_projections=ppe.to(pipe.dtype), return_dict=False)[0]
    return v.float()


def sd3_encode_prompt(pipe, prompt):
    """(pe, ppe) float embeddings for a prompt (SD3 encode_prompt returns 4 values)."""
    pe, _, ppe, _ = pipe.encode_prompt(prompt=prompt, prompt_2=prompt, prompt_3=prompt,
                                       device="cuda", num_images_per_prompt=1,
                                       do_classifier_free_guidance=False)
    return pe.float(), ppe.float()


@torch.no_grad()
def sd3_flowalign(pipe, x0, C_src, C_tar, sig, seed, w, zeta,
                  sbn_mode="off", sbn_cut=0.0, sbn_strength=1.0,
                  term_start_cut=1.0, term_end_cut=1.0, term_renorm=False):
    """FlowAlign on SD3 (UNPACKED latents). Mirrors `flowalign` exactly but with no distilled
    guidance embed. x0 = unpacked (1,16,FH,FW) source latent; C_src/C_tar = (pe, ppe) cuda
    tensors. 3 velocity forwards/step. Returns an UNPACKED latent."""
    steps = len(sig) - 1
    eps = torch.randn(x0.shape, generator=torch.Generator("cuda").manual_seed(seed),
                      device="cuda").float()
    xt = x0.clone()
    for i in range(steps):
        frac = i / max(steps - 1, 1)
        cutoff = term_start_cut + (term_end_cut - term_start_cut) * frac   # anneal low -> high
        s_hi, s_lo = float(sig[i]), float(sig[i + 1])
        qt = (1 - s_hi) * x0 + s_hi * eps                      # forward-diffused source
        pt = xt + qt - x0                                      # == x_tar
        v_pt_src = sd3_velocity(pipe, pt, s_hi, C_src[0], C_src[1])
        v_pt_tar = sd3_velocity(pipe, pt, s_hi, C_tar[0], C_tar[1])
        vp = v_pt_src + w * (v_pt_tar - v_pt_src)              # CFG, source prompt as negative
        vp = _vel_sbn_lat(vp, v_pt_src, sbn_mode, sbn_cut, sbn_strength)
        vq = sd3_velocity(pipe, qt, s_hi, C_src[0], C_src[1])
        term = (qt - s_hi * vq) - (pt - s_hi * vp)            # E[q0|qt] - E[p0|pt]
        term = _gauss_lowpass_lat(term, cutoff, term_renorm)
        xt = xt + (s_lo - s_hi) * (vp - vq) + zeta * term
    return xt
