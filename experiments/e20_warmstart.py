"""E20: spectral warm-start -- "skip the beginning" of generation.

The early denoising steps fix low-frequency STRUCTURE (low-band phase), the late
steps fix power/detail (E8: power locks in late; E12: low-band phase coherence
rises first). So if we hand the model the low-frequency content up front -- a
latent whose bands are pre-set -- we can re-enter the trajectory partway and skip
the early steps. Useful for conditioning/style transfer (commit a reference's
structure) and, speculatively, for shaping plain generation.

Re-entry uses rectified flow: x_t = (1-sigma)*x0 + sigma*eps
(FlowMatchEulerDiscreteScheduler.scale_noise), via e17_sd35.gen_sd3_warmstart.

Parts (--part, comma list):
  preflight  -- model-free construction asserts + cached power-lock-in (runs now)
  profile    -- WHEN does each band lock in? S seeds x RecordTraj -> per-step
                cross-seed phase coherence R[band,t] (needs SD3.5)
  oracle     -- commit a finished run's TRUE low bands (cutoff c), re-enter at
                strength: how many steps can we skip and still recover the image?
                2-D (c, strength) sweep, vs a pure-noise re-init baseline (SD3.5)
  condition  -- commit a REFERENCE image's low bands = band-controlled SDEdit;
                baseline = full SDEdit (c=1). structure vs prompt vs steps (SD3.5)
  noiseshape -- color step-0 noise toward natural-latent spectrum (psd_match);
                colored-init vs white-init at several step counts (SD3.5)
  analyze    -- plots (lock-in curves, recovery heatmaps) + summary.md

All generation parts need the gated SD3.5 download -> cluster. preflight + the
cached-trajectory lock-in run locally.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid
from spectral_ops import band_index_map, band_power, phase_coherence
from style_ops import band_spectrum_split, color_noise, latent_band_power
from bandnorm import band_centers

OUT = os.path.join(RESULTS, "e20")
SIZE, H, W, N_CH = 1024, 128, 128, 16
PROMPTS = [
    ("fox", "a red fox sitting in a snowy forest, photograph"),
    ("market", "a busy moroccan spice market, colorful stalls, photograph"),
    ("portrait", "studio portrait of an old fisherman, dramatic lighting"),
    ("city", "a futuristic city skyline at night, neon lights"),
]


# ---------------------------------------------------------------------------
# Lock-in helpers (shared by preflight cached-data + profile)
# ---------------------------------------------------------------------------

def lockin_step(series, final, tol=0.1):
    """First step t after which series stays within tol*|final| of final."""
    for t in range(len(series)):
        if (series[t:] - final).abs().max() <= tol * abs(float(final)) + 1e-9:
            return t
    return len(series) - 1


def coherence_lockin(R_traj, frac=0.9):
    """Per-band lock-in step from a coherence trajectory R_traj (steps, n_bins):
    first t at/after which R stays >= frac * R_final."""
    final = R_traj[-1]
    out = []
    for b in range(R_traj.shape[1]):
        thr = frac * float(final[b])
        t = next((tt for tt in range(R_traj.shape[0])
                  if bool((R_traj[tt:, b] >= thr).all())), R_traj.shape[0] - 1)
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# preflight: construction math (no model) + cached power-lock-in
# ---------------------------------------------------------------------------

def preflight(args):
    print("[e20] pre-flight (model-free) ...", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    nb = args.n_bins
    idx = band_index_map(H, W, nb, dev)
    x0 = torch.randn(1, N_CH, H, W, device=dev)
    noise = torch.randn(1, N_CH, H, W, device=dev)

    # 1. warm-start band commit: low bands from x0, high from noise
    ws = band_spectrum_split(x0, noise, c=0.25)
    assert ws.shape == x0.shape
    assert (band_spectrum_split(x0, noise, 1.0) - x0).abs().max() < 1e-3, "c=1 != x0"
    assert (band_spectrum_split(x0, noise, 0.0) - noise).abs().max() < 1e-3, "c=0 != noise"

    # 2. scale_noise interpolation endpoints (the re-entry formula)
    for sig, ref, name in [(0.0, x0, "sigma=0->x0"), (1.0, noise, "sigma=1->noise")]:
        xt = (1.0 - sig) * x0 + sig * noise
        assert (xt - ref).abs().max() < 1e-5, name

    # 3. noise coloring matches a target band power and stays ~unit variance
    target = latent_band_power(torch.randn(1, N_CH, H, W, device=dev) * 2 + 0.5, idx, nb)
    cn = color_noise(noise.clone(), target, idx, nb, unit_var=True)
    cb = band_power((torch.fft.fft2(cn.float()).abs() ** 2)[0], idx, nb)
    shape_err = float((band_shape_ratio(cb) - band_shape_ratio(target)).abs().max())
    assert shape_err < 0.05, f"colored-noise spectrum shape off: {shape_err}"
    assert abs(float(cn.std()) - 1.0) < 0.05, "colored noise not unit variance"
    print(f"[e20] construction OK (band-split endpoints, scale_noise, "
          f"color shape_err={shape_err:.3f})", flush=True)

    # 4. cached power lock-in (Flux E8 trajectory) -- the motivating prior
    pth = os.path.join(RESULTS, "e8", "ref_psd.pt")
    if os.path.exists(pth):
        band = torch.load(pth, weights_only=True)["band"].float().mean(1)  # (T, nb)
        los = [lockin_step(band[:, b], band[-1, b]) for b in range(nb // 3)]
        his = [lockin_step(band[:, b], band[-1, b]) for b in range(2 * nb // 3, nb)]
        T = band.shape[0]
        print(f"[e20] cached POWER lock-in (E8, /{T}): low={sum(los)/len(los):.1f} "
              f"high={sum(his)/len(his):.1f}  -> power locks LATE for all bands; "
              "PHASE lock-in (profile part) is the signal the warm-start targets.",
              flush=True)


def band_shape_ratio(band, eps=1e-8):
    """Per-channel band power normalized to its own sum (level-invariant shape)."""
    return band.float() / band.float().sum(-1, keepdim=True).clamp(min=eps)


# ---------------------------------------------------------------------------
# Generation parts (SD3.5)
# ---------------------------------------------------------------------------

def _load():
    from e17_sd35 import load_sd35, load_sd35_img2img
    pipe = load_sd35("gpu_resident")
    return pipe, load_sd35_img2img(pipe)


def clip_img_feats(model, proc, imgs):
    from clip_sim import clip_image_features
    return clip_image_features(model, proc, imgs)


def part_profile(args):
    from e17_sd35 import gen_sd3, RecordTraj
    pipe, _ = _load()
    idx = band_index_map(H, W, args.n_bins, "cuda")
    os.makedirs(OUT, exist_ok=True)
    for pid, prompt in PROMPTS[: args.num_prompts]:
        seed_lats = []
        power = torch.zeros(args.steps, args.n_bins)
        for s in range(args.seeds):
            rec = RecordTraj(idx, args.n_bins, args.steps)
            gen_sd3(pipe, prompt, s, 1.0, args.steps, cb_obj=rec)
            seed_lats.append(rec.lats)                       # list[steps] of (1,16,128,128)
            power += torch.stack(rec.band).mean(1)           # (steps, nb)
        power /= args.seeds
        # per-step cross-seed phase coherence, averaged over channels
        R_traj = torch.zeros(args.steps, args.n_bins)
        for t in range(args.steps):
            phis = torch.stack([torch.fft.fft2(seed_lats[s][t].float().cuda()[0]).angle()
                                for s in range(args.seeds)])      # (S, C, H, W)
            _, R, r_null = phase_coherence(phis, args.n_bins)     # R: (C, nb)
            R_traj[t] = R.mean(0)
        prof = {"R_traj": R_traj, "power_traj": power, "r_null": r_null,
                "centers": band_centers(args.n_bins),
                "phase_lockin": coherence_lockin(R_traj),
                "power_lockin": [lockin_step(power[:, b], power[-1, b])
                                 for b in range(args.n_bins)]}
        torch.save(prof, f"{OUT}/profile_{pid}.pt")
        lo = sum(prof["phase_lockin"][: args.n_bins // 3]) / (args.n_bins // 3)
        hi = sum(prof["phase_lockin"][2 * args.n_bins // 3:]) / (args.n_bins
                                                                 - 2 * args.n_bins // 3)
        print(f"[e20] {pid} PHASE lock-in (/{args.steps}): low={lo:.1f} high={hi:.1f}",
              flush=True)


def _recover_metrics(model, proc, img, ref_img, lat, x0_star):
    from clip_sim import cosine
    fi, fr = clip_img_feats(model, proc, [img, ref_img])
    return {"clip_i": cosine(fi, fr),
            "latent_l2": float(((lat - x0_star) ** 2).mean().sqrt())}


def part_oracle(args):
    from e17_sd35 import gen_sd3, gen_sd3_warmstart
    from e9_clipt import load_clip
    pipe, pipe_i2i = _load()
    model, proc = load_clip(args.clip_model)
    idx = band_index_map(H, W, args.n_bins, "cuda")
    os.makedirs(f"{OUT}/oracle", exist_ok=True)
    cuts, strengths = args.cuts, args.strengths
    report = {}
    for pid, prompt in PROMPTS[: args.num_prompts]:
        ref_img, x0_star = gen_sd3(pipe, prompt, 0, args.cfg, args.steps)  # full run
        x0_star = x0_star.cuda()
        report[pid] = {"prompt": prompt, "cells": {}}
        rows, rlabels = [[ref_img]], ["full"]
        for c in cuts:
            row = []
            for st in strengths:
                noise = torch.randn(1, N_CH, H, W, device="cuda")
                x0_warm = band_spectrum_split(x0_star, noise, c)   # true low bands + noise
                img, lat = gen_sd3_warmstart(pipe_i2i, x0_warm, st, prompt, 1,
                                             args.steps, args.cfg)
                m = _recover_metrics(model, proc, img, ref_img, lat.cuda(), x0_star)
                report[pid]["cells"][f"c{c:g}_s{st:g}"] = m
                row.append(img)
            rows.append(row)
            rlabels.append(f"c={c:g}")
        # baseline: pure-noise re-init (c=0) already covered at c=cuts[0] if 0 in cuts
        save_grid(rows, rlabels, ["ref"] + [f"skip {1-st:.0%}" for st in strengths],
                  f"{OUT}/oracle/grid_{pid}.png")
        print(f"[e20] oracle {pid} done", flush=True)
    json.dump(report, open(f"{OUT}/oracle.json", "w"), indent=2)


def part_condition(args):
    from e17_sd35 import gen_sd3_warmstart, sd3_vae_encode
    from e9_clipt import load_clip, clip_scores
    from PIL import Image
    pipe, pipe_i2i = _load()
    vae = pipe.vae
    model, proc = load_clip(args.clip_model)
    idx = band_index_map(H, W, args.n_bins, "cuda")
    os.makedirs(f"{OUT}/condition", exist_ok=True)
    refs = sorted(p for p in os.listdir(args.refs)
                  if p.lower().endswith((".jpg", ".jpeg", ".png")))[: args.num_refs]
    report = {}
    for rf in refs:
        ref_pil = Image.open(os.path.join(args.refs, rf)).convert("RGB").resize((SIZE, SIZE))
        x0_ref = sd3_vae_encode(vae, ref_pil).cuda()
        for pid, prompt in PROMPTS[: args.num_prompts]:
            rows, rlabels = [[ref_pil]], ["ref"]
            tag = f"{os.path.splitext(rf)[0]}__{pid}"
            report[tag] = {"prompt": prompt, "cells": {}}
            for c in args.cuts:                       # c=1 == full SDEdit baseline
                row = []
                for st in args.strengths:
                    noise = torch.randn(1, N_CH, H, W, device="cuda")
                    x0_warm = band_spectrum_split(x0_ref, noise, c)
                    img, _ = gen_sd3_warmstart(pipe_i2i, x0_warm, st, prompt, 0,
                                               args.steps, args.cfg)
                    fi, fr = clip_img_feats(model, proc, [img, ref_pil])
                    from clip_sim import cosine
                    report[tag]["cells"][f"c{c:g}_s{st:g}"] = {
                        "struct_clip": cosine(fi, fr),
                        "prompt_clip": clip_scores(model, proc, prompt, [img])[0]}
                    row.append(img)
                rows.append(row)
                rlabels.append(f"c={c:g}" + (" (SDEdit)" if c >= 1.0 else ""))
            save_grid(rows, rlabels, ["ref"] + [f"skip {1-st:.0%}" for st in args.strengths],
                      f"{OUT}/condition/grid_{tag}.png")
            print(f"[e20] condition {tag} done", flush=True)
    json.dump(report, open(f"{OUT}/condition.json", "w"), indent=2)


def part_noiseshape(args):
    from e17_sd35 import gen_sd3, sd3_vae_encode
    from e9_clipt import load_clip, clip_scores
    from fidelity_metrics import load_aesthetic, aesthetic_scores
    from PIL import Image
    pipe, _ = _load()
    vae = pipe.vae
    model, proc = load_clip(args.clip_model)
    mlp = load_aesthetic()
    idx = band_index_map(H, W, args.n_bins, "cuda")
    os.makedirs(f"{OUT}/noiseshape", exist_ok=True)
    # natural-latent target spectrum: mean band power over SD3.5-encoded real photos
    photos = sorted(p for p in os.listdir(args.refs)
                    if p.lower().endswith((".jpg", ".jpeg", ".png")))[:8]
    target = torch.stack([latent_band_power(
        sd3_vae_encode(vae, Image.open(os.path.join(args.refs, p)).convert("RGB")
                       ).cuda(), idx, args.n_bins) for p in photos]).mean(0)
    report = {}
    for pid, prompt in PROMPTS[: args.num_prompts]:
        rows, rlabels = [], []
        report[pid] = {"prompt": prompt, "white": {}, "colored": {}}
        for nsteps in args.step_counts:
            white_imgs, col_imgs = [], []
            for s in range(args.seeds):
                g = torch.Generator("cuda").manual_seed(s)
                noise = torch.randn(1, N_CH, H, W, generator=g, device="cuda")
                col = color_noise(noise.clone(), target, idx, args.n_bins)
                wi, _ = gen_sd3(pipe, prompt, s, args.cfg, nsteps)
                ci, _ = gen_sd3(pipe, prompt, s, args.cfg, nsteps, init_latents=col)
                white_imgs.append(wi); col_imgs.append(ci)
            report[pid]["white"][nsteps] = {
                "aesthetic": float(sum(aesthetic_scores(mlp, model, proc, white_imgs))
                                   / len(white_imgs)),
                "clip_t": float(sum(clip_scores(model, proc, prompt, white_imgs))
                                / len(white_imgs))}
            report[pid]["colored"][nsteps] = {
                "aesthetic": float(sum(aesthetic_scores(mlp, model, proc, col_imgs))
                                   / len(col_imgs)),
                "clip_t": float(sum(clip_scores(model, proc, prompt, col_imgs))
                                / len(col_imgs))}
            rows += [white_imgs[: args.grid_n], col_imgs[: args.grid_n]]
            rlabels += [f"white {nsteps}st", f"colored {nsteps}st"]
        save_grid(rows, rlabels, [f"s{s}" for s in range(args.grid_n)],
                  f"{OUT}/noiseshape/grid_{pid}.png")
        print(f"[e20] noiseshape {pid} done", flush=True)
    json.dump(report, open(f"{OUT}/noiseshape.json", "w"), indent=2)


# ---------------------------------------------------------------------------
# analyze: lock-in curves + summaries
# ---------------------------------------------------------------------------

def analyze(args):
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    profs = sorted(glob.glob(f"{OUT}/profile_*.pt"))
    if profs:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        p = torch.load(profs[0], weights_only=True)
        R, P, c = p["R_traj"], p["power_traj"], p["centers"]
        for b, lab in [(1, "low"), (R.shape[1] // 2, "mid"), (R.shape[1] - 2, "high")]:
            ax[0].plot(R[:, b], label=f"{lab} f={c[b]:.2f}")
            ax[1].plot(P[:, b] / P[-1, b].clamp(min=1e-8), label=f"{lab} f={c[b]:.2f}")
        ax[0].axhline(p["r_null"], ls=":", c="gray"); ax[0].set_title("phase coherence R[band,t]")
        ax[1].set_title("power / final"); [a.legend(fontsize=8) for a in ax]
        ax[0].set_xlabel("step"); ax[1].set_xlabel("step")
        fig.tight_layout(); fig.savefig(f"{OUT}/lockin.png", dpi=110)
        print(f"[e20] wrote {OUT}/lockin.png", flush=True)
    for jf in ("oracle.json", "condition.json", "noiseshape.json"):
        path = f"{OUT}/{jf}"
        if os.path.exists(path):
            print(f"[e20] {jf}: {len(json.load(open(path)))} entries", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="preflight")
    ap.add_argument("--num_prompts", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--cuts", default="0.0,0.1,0.25,0.5,1.0",
                    type=lambda s: [float(x) for x in s.split(",")])
    ap.add_argument("--strengths", default="0.4,0.6,0.8",
                    type=lambda s: [float(x) for x in s.split(",")])
    ap.add_argument("--step_counts", default="8,16,28",
                    type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--refs", default=os.path.join(RESULTS, "e10", "real_photos"))
    ap.add_argument("--num_refs", type=int, default=2)
    ap.add_argument("--grid_n", type=int, default=4)
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    runners = {"preflight": preflight, "profile": part_profile, "oracle": part_oracle,
               "condition": part_condition, "noiseshape": part_noiseshape,
               "analyze": analyze}
    for part in filter(None, (p.strip() for p in args.part.split(","))):
        runners[part](args)


if __name__ == "__main__":
    main()
