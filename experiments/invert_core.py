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
