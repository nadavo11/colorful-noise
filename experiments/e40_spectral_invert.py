"""E40: RF inversion + trajectory-matched spectral normalization (FLUX).

Real-image editing that preserves the source's STRUCTURE while an edit prompt changes
content -- even when the edit is aggressive. Three steps:

  1. RF-invert the real image to noise on FLUX: integrate the velocity ODE BACKWARDS
     (sigma 0 -> 1) under the source prompt, recording the (unpacked) latent at every
     sigma node -- i.e. the whole inversion *trajectory*.
  2. Regenerate under the edit prompt (sigma 1 -> 0). At each step, NORMALIZE the latent's
     LOW-frequency bands back to the recorded reference at the matching sigma. Low bands
     carry coarse layout/structure, so pinning them to the source trajectory keeps
     structure while high bands stay free for the edit.

This differs from the repo's BandLock (e21/e22), which clamps to a single fixed source
latent x0 at every step; here the reference is the per-step inversion trajectory, aligned
by sigma. It also doubles as a drift-correction probe: re-imposing the recorded spectrum
counteracts inversion drift (cf. e21's failed SD3.5 reconstruction).

Three normalization modes (--mode), all restricted to the low band [0, cut] and blended by
--strength:
  sbn   -- match per-(channel, radial-band) mean POWER (spectral_ops.psd_match); phase free.
  phase -- sbn on magnitude PLUS lock the low-band PHASE to the source (band_phase_swap).
  adain -- match per-band mean+std of MAGNITUDE on soft bands (spectral_adain.spectral_adain).

Parts (--part): gen (invert + edits + strips) ; analyze (CLIP/aesthetic metrics + index.html).
Reuses the FLUX plumbing from e31/e7 and the spectral primitives from spectral_ops /
spectral_adain -- the only new code is the inversion loop, the low-band clamp, and the edit loop.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid
from e7_flux_phase import load_flux, flux_vae_decode, SIZE
from e31_flowedit_freq import (flux_velocity, flux_sigmas, _gids, pack, unpack,
                               vae_encode, H, W, PACK_TOKENS)
from spectral_ops import radial_bins, band_index_map, band_power, psd_match, band_phase_swap
from spectral_adain import soft_band_masks, spectral_adain

OUT = os.path.join(RESULTS, "e40")
N_BINS = 24
ADAIN_K = 8                       # soft bands for the adain mode
EPS = 1e-8

# (key, source prompt, edit prompt)
SOURCES = [
    ("cat_dog", "a photograph of a cat sitting on a sofa",
     "a photograph of a dog sitting on a sofa"),
    ("house_storm", "a house by a lake on a sunny day",
     "a house by a lake during a dramatic thunderstorm"),
    ("street_snow", "a city street with shops in summer",
     "a city street with shops covered in deep snow"),
]


# ---------------------------------------------------------------------------
# low-band spectral clamp (the one new operator; works on unpacked (1,16,H,W))
# ---------------------------------------------------------------------------

def _band_centers_norm(device):
    """Normalized radius (0=DC, 1=corner) of each radial band center, matching
    band_index_map's binning."""
    rr = radial_bins(H, W, device)
    edges = torch.linspace(0, rr.max() + 1e-6, N_BINS + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers / rr.max()                       # (N_BINS,) in [0,1]


def _low_quantile(cut, device):
    """Fraction of FFT bins with normalized radius <= cut, i.e. the `c` that makes
    band_phase_swap's quantile mask == {radius <= cut}."""
    rr = radial_bins(H, W, device)
    return float((rr <= cut * rr.max()).float().mean())


def _sbn_low(gen, ref, cut, strength, idx):
    """Match low-band (<cut) mean power of `gen` to `ref`, blended by strength.
    High bands get gain 1 (target == current), so they pass through untouched."""
    cen = _band_centers_norm(gen.device)
    low = cen < cut
    ref_bp = band_power((torch.fft.fft2(ref.float()).abs() ** 2)[0], idx, N_BINS)
    cur_bp = band_power((torch.fft.fft2(gen.float()).abs() ** 2)[0], idx, N_BINS)
    tgt = cur_bp.clone()
    s = strength
    tgt[:, low] = (cur_bp[:, low].clamp(min=EPS) ** (1 - s)) * (ref_bp[:, low].clamp(min=EPS) ** s)
    return psd_match(gen, tgt, idx, N_BINS)


def spectral_clamp(gen, ref, mode, cut, strength, idx, masks, cen_k):
    """Pull gen's low band [0, cut] toward ref. gen, ref: (1, 16, H, W) on cuda."""
    if mode == "sbn":
        return _sbn_low(gen, ref, cut, strength, idx)
    if mode == "phase":
        out = _sbn_low(gen, ref, cut, strength, idx)
        q = _low_quantile(cut, gen.device)
        # low-band phase from ref (A), high-band phase from out (B), magnitude from out
        return band_phase_swap(ref, out, q, mag_from="B").real.to(out.dtype)
    if mode == "adain":
        sources = [ref if cen_k[k] < cut else gen for k in range(masks.shape[0])]
        out = spectral_adain(gen, sources, masks)
        return gen * (1 - strength) + out * strength
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# RF inversion + edit generation (Euler on the FLUX velocity field)
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_gen(pipe, x_packed, C, sig, gids, traj=None, mode=None, cut=0.25,
                strength=1.0, schedule=None, idx=None, masks=None, cen_k=None):
    """Integrate sigma high->low (noise -> clean). If `traj` given, clamp the low band
    to traj[i] at each scheduled step. Returns the final packed latent."""
    steps = len(sig) - 1
    pe, ppe = C
    x = x_packed
    for i in range(steps):
        s_hi, s_lo = float(sig[i]), float(sig[i + 1])
        if traj is not None and (schedule is None or i in schedule):
            lat = spectral_clamp(unpack(pipe, x), traj[i].cuda(), mode, cut,
                                 strength, idx, masks, cen_k)
            x = pack(pipe, lat)
        v = flux_velocity(pipe, x, s_hi, pe, ppe, gids)
        x = x + (s_lo - s_hi) * v
    return x


@torch.no_grad()
def rf_invert(pipe, x0_packed, C, sig, gids):
    """Reverse Euler sigma low->high (clean -> noise) under conditioning C. Returns
    (inverted-noise packed latent, traj) where traj[i] = unpacked latent at sigma sig[i]."""
    steps = len(sig) - 1
    pe, ppe = C
    x = x0_packed
    traj = [None] * (steps + 1)
    traj[steps] = unpack(pipe, x).cpu()
    for i in range(steps - 1, -1, -1):
        s_lo, s_hi = float(sig[i + 1]), float(sig[i])
        v = flux_velocity(pipe, x, s_lo, pe, ppe, gids)
        x = x + (s_hi - s_lo) * v
        traj[i] = unpack(pipe, x).cpu()
    return x, traj


# ---------------------------------------------------------------------------
# Part: gen
# ---------------------------------------------------------------------------

def _encode(pipe, prompt):
    pe, ppe, _ = pipe.encode_prompt(prompt=prompt, prompt_2=prompt, device="cuda",
                                    num_images_per_prompt=1, max_sequence_length=512)
    return pe.float(), ppe.float()


def _schedule_steps(name, steps):
    if name == "early":
        return set(range(steps // 2))
    if name == "late":
        return set(range(steps // 2, steps))
    return None                                    # all


def run_gen(args):
    pipe = load_flux(args.mem)
    sig = flux_sigmas(pipe, args.steps)
    gids_inv = _gids(pipe, args.inv_guidance)
    gids_gen = _gids(pipe, args.guidance)
    idx = band_index_map(H, W, N_BINS, "cuda")
    cen = torch.linspace(0, 1, ADAIN_K)
    masks = soft_band_masks(H, W, cen.tolist(), [1.0 / ADAIN_K] * ADAIN_K, "cuda")
    cen_k = cen.tolist()
    sched = _schedule_steps(args.schedule, args.steps)
    modes = [args.mode] if args.mode != "all" else ["sbn", "phase", "adain"]

    for key, src, edit in SOURCES[: args.num]:
        d = os.path.join(OUT, key)
        os.makedirs(d, exist_ok=True)
        C_src = _encode(pipe, src)
        C_edit = _encode(pipe, edit)

        # source latent x0: real image (VAE-encode) or generated from the source prompt
        srcp = os.path.join(d, "source.png")
        real = os.path.join(args.real_dir, f"{key}.png") if args.real_dir else ""
        if real and os.path.exists(real):
            x0 = pack(pipe, vae_encode(pipe.vae, Image.open(real)))
            Image.open(real).convert("RGB").resize((SIZE, SIZE)).save(srcp)
        else:
            gen = torch.Generator("cuda").manual_seed(args.seed)
            noise = torch.randn(1, PACK_TOKENS, 64, generator=gen, device="cuda").float()
            x0 = forward_gen(pipe, noise, C_src, sig, gids_gen)
            flux_vae_decode(pipe.vae, unpack(pipe, x0)).save(srcp)
        print(f"[e40] {key}: source ready", flush=True)

        # RF inversion under the source prompt -> noise + trajectory
        x_noise, traj = rf_invert(pipe, x0, C_src, sig, gids_inv)
        print(f"[e40] {key}: inverted (noise std={float(unpack(pipe, x_noise).std()):.3f})",
              flush=True)

        # conditions
        conds = {
            "recon": (C_src, None),                 # inversion+plumbing gate (no clamp)
            "recon_clamp": (C_src, modes[0]),       # clamp under source -> drift correction
            "edit_noclamp": (C_edit, None),         # plain RF-inversion edit baseline
        }
        for m in modes:
            conds[f"edit_{m}"] = (C_edit, m)
        for cond, (C, mode) in conds.items():
            outp = os.path.join(d, f"{cond}.png")
            if os.path.exists(outp):
                continue
            xe = forward_gen(pipe, x_noise, C, sig, gids_gen,
                             traj=(None if mode is None else traj), mode=mode,
                             cut=args.cut, strength=args.strength, schedule=sched,
                             idx=idx, masks=masks, cen_k=cen_k)
            flux_vae_decode(pipe.vae, unpack(pipe, xe)).save(outp)
            print(f"[e40] {key}/{cond} done", flush=True)

        names = ["source"] + list(conds)
        row = [Image.open(os.path.join(d, f"{n}.png")).convert("RGB") for n in names
               if os.path.exists(os.path.join(d, f"{n}.png"))]
        save_grid([row], [key], names, os.path.join(d, "strip.png"), thumb=240)
        print(f"[e40] {key} grid done", flush=True)


# ---------------------------------------------------------------------------
# Part: analyze
# ---------------------------------------------------------------------------

def run_analyze(args):
    from e9_clipt import agg, load_clip, clip_scores
    from fidelity_metrics import load_aesthetic, aesthetic_scores
    clip = load_clip(args.clip_model)
    aes = load_aesthetic()
    report = {"params": vars(args), "sources": {}}
    for key, src, edit in SOURCES[: args.num]:
        d = os.path.join(OUT, key)
        srcp = os.path.join(d, "source.png")
        if not os.path.exists(srcp):
            continue
        src_img = Image.open(srcp).convert("RGB")
        b = np.asarray(src_img).astype(np.float32) / 255
        ents = {}
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".png") or fn in ("source.png", "strip.png"):
                continue
            cond = fn[:-4]
            im = Image.open(os.path.join(d, fn)).convert("RGB")
            a = np.asarray(im).astype(np.float32) / 255
            ents[cond] = {
                "clip_edit": agg(clip_scores(*clip, edit, [im])),
                "clip_src": agg(clip_scores(*clip, src, [im])),
                "px_dist_to_src": float(np.sqrt(((a - b) ** 2).mean())),
                "aesthetic": agg(aesthetic_scores(aes, *clip, [im])),
            }
        report["sources"][key] = {"src": src, "edit": edit, "conds": ents}
        rc = ents.get("recon", {}).get("px_dist_to_src")
        rcc = ents.get("recon_clamp", {}).get("px_dist_to_src")
        print(f"[e40] {key}: recon px-dist={rc} recon_clamp={rcc}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    _site(report)
    print("[e40] wrote report.json + index.html", flush=True)


def _site(report):
    def gv(sc, k):
        v = sc.get(k)
        return v.get("mean") if isinstance(v, dict) else v

    h = ["<!doctype html><meta charset=utf-8><title>E40 — RF inversion + spectral clamp</title>",
         "<style>body{font:15px/1.6 -apple-system,Segoe UI,sans-serif;max-width:1000px;"
         "margin:24px auto;padding:0 16px}img{width:100%;border:1px solid #ccc;border-radius:4px}"
         "table{border-collapse:collapse;margin:8px 0;font-size:14px}"
         "th,td{border:1px solid #ccc;padding:4px 9px;text-align:right}"
         "td.v{text-align:left;font-weight:600}h3{margin-top:26px}</style>",
         "<h1>E40 — RF inversion + trajectory-matched spectral normalization (FLUX)</h1>",
         "<p>Invert a real image to noise, recording the latent spectrum at every step; "
         "regenerate under the edit prompt while clamping the low band [0, cut] back to the "
         "recorded trajectory. Modes: <b>sbn</b> (power), <b>phase</b> (power+low-phase), "
         "<b>adain</b> (mean+std). Gate: <code>recon</code> should reproduce the source.</p>",
         f"<p>params: cut={report['params']['cut']} strength={report['params']['strength']} "
         f"steps={report['params']['steps']} schedule={report['params']['schedule']}</p>"]
    for key, e in report["sources"].items():
        h.append(f"<h3>{key}: <code>{e['src']}</code> → <code>{e['edit']}</code></h3>")
        p = os.path.join(key, "strip.png")
        if os.path.exists(os.path.join(OUT, p)):
            h.append(f"<img src='{p}'>")
        h.append("<table><tr><th>condition</th><th>CLIP→edit ↑</th><th>CLIP→source ↑</th>"
                 "<th>px-dist→source ↓</th><th>aesthetic ↑</th></tr>")
        for c, sc in e["conds"].items():
            h.append(f"<tr><td class=v>{c}</td><td>{gv(sc,'clip_edit'):.3f}</td>"
                     f"<td>{gv(sc,'clip_src'):.3f}</td><td>{sc['px_dist_to_src']:.3f}</td>"
                     f"<td>{gv(sc,'aesthetic'):.3f}</td></tr>")
        h.append("</table>")
    with open(os.path.join(OUT, "index.html"), "w") as fh:
        fh.write("\n".join(h))


# ---------------------------------------------------------------------------
# model-free preflight
# ---------------------------------------------------------------------------

def preflight():
    from diffusers import FluxPipeline
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. pack/unpack round-trip
    x = torch.randn(1, 16, H, W, device=dev)
    packed = FluxPipeline._pack_latents(x, 1, 16, H, W)
    assert packed.shape == (1, PACK_TOKENS, 64), packed.shape
    assert torch.equal(FluxPipeline._unpack_latents(packed, SIZE, SIZE, 8), x), "pack/unpack"

    idx = band_index_map(H, W, N_BINS, dev)
    cen = torch.linspace(0, 1, ADAIN_K)
    masks = soft_band_masks(H, W, cen.tolist(), [1.0 / ADAIN_K] * ADAIN_K, dev)
    cen_k = cen.tolist()
    gen = torch.randn(1, 16, H, W, device=dev)
    ref = torch.randn(1, 16, H, W, device=dev)

    # 2. identity: clamping a latent to itself returns it unchanged (all modes)
    for mode in ("sbn", "phase", "adain"):
        out = spectral_clamp(gen, gen, mode, 0.25, 1.0, idx, masks, cen_k)
        assert (out - gen).abs().max() < 1e-3, f"identity {mode}: {float((out-gen).abs().max())}"

    # 3. low-band restriction: high bands (>cut) unchanged for sbn
    cut = 0.25
    out = _sbn_low(gen, ref, cut, 1.0, idx)
    Fg, Fo = torch.fft.fft2(gen.float()), torch.fft.fft2(out.float())
    rr = radial_bins(H, W, dev)
    hi = (rr > cut * rr.max())[None, None].expand_as(Fg)
    assert (Fg[hi].abs() - Fo[hi].abs()).abs().max() < 1e-2, "high-band magnitude changed"

    # 4. full clamp (strength 1, cut 1) makes sbn match ref band power everywhere
    out = _sbn_low(gen, ref, 1.0, 1.0, idx)
    rbp = band_power((torch.fft.fft2(ref.float()).abs() ** 2)[0], idx, N_BINS)
    obp = band_power((torch.fft.fft2(out.float()).abs() ** 2)[0], idx, N_BINS)
    assert (rbp - obp).abs().max() / rbp.abs().max() < 1e-3, "full sbn did not match ref"
    print("[e40] preflight OK (pack/unpack, clamp identity, low-band restriction, full match)")


# ---------------------------------------------------------------------------

def main(args):
    global OUT
    if args.out_tag:
        OUT = os.path.join(RESULTS, f"e40_{args.out_tag}")
    os.makedirs(OUT, exist_ok=True)
    parts = [p.strip() for p in args.part.split(",") if p.strip()]
    if "preflight" in parts:
        preflight()
    if "gen" in parts:
        run_gen(args)
    if "analyze" in parts:
        run_analyze(args)
    if "site" in parts:
        rp = os.path.join(OUT, "report.json")
        if not os.path.exists(rp):
            raise SystemExit(f"[e40] --part site needs {rp}")
        with open(rp) as f:
            _site(json.load(f))
        print(f"[e40] rebuilt index.html", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="gen,analyze")
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--mode", default="all", choices=["all", "sbn", "phase", "adain"])
    ap.add_argument("--cut", type=float, default=0.25, help="low-band cutoff (normalized radius)")
    ap.add_argument("--strength", type=float, default=1.0, help="clamp blend 0..1")
    ap.add_argument("--schedule", default="all", choices=["all", "early", "late"])
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--inv_guidance", type=float, default=1.0, help="guidance during inversion")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem", default="bnb4", choices=["bnb4", "gpu_resident", "seq_offload"])
    ap.add_argument("--real_dir", default="", help="dir of <key>.png real images (else generate)")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out_tag", default="")
    main(ap.parse_args())
