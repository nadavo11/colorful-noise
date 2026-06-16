"""E23: real-image spectral gap -- measure it, then correct toward real (real-SBN).

SBN (E8/E9, bandnorm.py) clamps a generated latent's per-(channel, radial band)
power toward a cfg=1 PROXY reference. E10 showed CFG inflates spectral power above
the real-image level, so cfg=1 is a stand-in for "less inflated" -- not the real
spectrum. E23 swaps the target: measure the actual generated-vs-real gap as a
distribution, then clamp toward the REAL-image band power (results/e10/real_latents.pt).

Key subtlety: real_latents.pt is a CLEAN x0 reference. During denoising the working
latent is noisy, so a clean-image target is only dimensionally correct on the
final / near-clean latent. Hence the interventions:
  cfg1.0           cfg=1 baseline (also the sbn_cfg1 reference) -- ref_seeds samples
  cfg{cfg}         cfg=3.5 baseline (also the measurement pool)
  sbn_cfg1         existing SBN, clamp every step toward the cfg=1 reference
  sbn_real_off{s}  PRIMARY: offline psd_match of the FINAL cfg latent toward real
                   (restyle_latent, strength s in --strength_sweep), VAE-decoded
  sbn_real_last    on-manifold: clamp toward real on the LAST denoising step only
  sbn_real_init    exploratory: shape the INITIAL noise toward real, then denoise
                   normally (tests whether the white-noise prior undoes it)

Parts (--part): measure (gap distribution + correction curve + plots),
gen (all conditions, cached -> killed runs resume), score (spectral_dist_to_real,
aesthetic, ImageReward, CLIP-T -- heavy models loaded after diffusion is freed),
analyze (grids + summary plots). Output under results/e23/<prompt>/.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_grid, RESULTS
from spectral_ops import band_index_map, band_power, psd_match
from e7_flux_phase import (load_flux, flux_generate, flux_vae_decode,
                           SIZE, LAT_SHAPE)
from e8_psd_clamp import gen_with_cb, preflight as e8_preflight, N_CH, H, W
from bandnorm import record_reference, generate_bandnorm, band_centers
from style_ops import restyle_latent
from fidelity_metrics import (load_real_psd, spectral_dist_to_real,
                              load_aesthetic, aesthetic_scores,
                              load_imagereward, imagereward_scores)
from e9_clipt import load_clip, clip_scores
from compbench import load_bvqa, bvqa_scores
from e9_bandnorm_classes import cached_gen, CLASSES
from real_spectral import (load_real_latents, real_band_power,
                           band_power_distribution, correction_curve)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def agg(vals):
    """Mean/std/n over non-None values (None if all missing)."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return {"mean": m, "std": sd, "n": len(vals)}


# Long, compositional prompts (multiple objects + attributes + counts + spatial
# relations) -- the regime where weak guidance (cfg=1) visibly drops elements, so
# they motivate why we keep cfg=3.5 adherence and only fix its over-baking.
COMPLEX_PROMPTS = [
    ("market",
     "A bustling outdoor market at golden hour: a woman in a red headscarf weighing "
     "oranges on a brass scale, a boy chasing a small white dog past wooden crates of "
     "purple eggplants, paper lanterns strung between striped awnings, steam rising "
     "from a copper teapot in the foreground"),
    ("desk",
     "An overhead photo of a cluttered architect's desk: three rolled blueprints tied "
     "with twine, a half-empty cup of black coffee on a round coaster, a brass compass, "
     "a small potted succulent, scattered yellow sticky notes, and a silver laptop "
     "showing a blue 3D model"),
    ("alley",
     "A rainy neon-lit Tokyo alley at night: a man under a transparent umbrella reading "
     "a glowing phone, pink and blue sign reflections in the wet pavement, a tall vending "
     "machine on the left, a black cat sitting on a stack of crates on the right"),
    ("still_life",
     "A Dutch still life: a cracked pomegranate split open beside a tipped silver goblet, "
     "three green grapes, a moth resting on a folded blue cloth, a single lit candle "
     "casting long shadows across a dark wooden table"),
    ("fox_scene",
     "A children's-book illustration of a fox in a yellow raincoat riding a red bicycle "
     "through a puddle, carrying a wicker basket of mushrooms, three crows watching from "
     "a wooden fence, an old stone windmill in the misty background"),
    ("blacksmith",
     "A close-up portrait of an elderly blacksmith with a braided grey beard, a leather "
     "apron with brass rivets, soot smudged on his left cheek, holding glowing orange "
     "tongs, forge light reflecting in his round spectacles"),
]


def get_prompts(args):
    """[(key, prompt), ...] -- a single --prompt, else the chosen prompt set."""
    if args.prompt:
        return [("single", args.prompt)]
    if getattr(args, "prompt_set", "classes") == "complex":
        return COMPLEX_PROMPTS[: args.num_classes]
    return CLASSES[: args.num_classes]


def cond_list(args):
    """All condition keys (rows of the grid / score table), in display order."""
    conds = ["cfg1.0", f"cfg{args.cfg}", "sbn_cfg1"]
    conds += [f"sbn_real_off{s}" for s in args.strength_sweep]
    conds += ["sbn_real_last", "sbn_real_init"]
    return conds


def _ensure_dirs(cdir):
    os.makedirs(f"{cdir}/images", exist_ok=True)
    os.makedirs(f"{cdir}/latents", exist_ok=True)


def _lazy_flux(state, mem):
    """Load the Flux pipeline at most once per process (shared across parts)."""
    if state.get("pipe") is None:
        state["pipe"] = load_flux(mem)
    return state["pipe"]


def get_real_band(out, n_bins, idx_map):
    """Universal per-(channel, band) real-image target, cached to real_band.pt."""
    path = f"{out}/real_band.pt"
    if os.path.exists(path):
        return torch.load(path, weights_only=True)
    _, real_band = real_band_power(load_real_latents(), idx_map, n_bins)
    torch.save(real_band, path)
    return real_band


def band_logrms(lat, real_band, idx_map, n_bins, eps=1e-12):
    """Per-(channel, band) RMS of log(gen) - log(real) -- the richer companion to
    the channel-mean spectral_dist_to_real scalar (keeps per-channel structure)."""
    gb = band_power((torch.fft.fft2(lat.cuda().float()).abs() ** 2)[0],
                    idx_map, n_bins)
    lr = gb.clamp(min=eps).log() - real_band.to(gb.device).clamp(min=eps).log()
    return float((lr ** 2).mean().sqrt())


# ---------------------------------------------------------------------------
# Generation primitives specific to E23
# ---------------------------------------------------------------------------

class ClampRealLastStep:
    """Step-end callback that clamps the latent's PSD toward the real-image band
    power on the LAST denoising step only (the near-clean latent the VAE sees);
    earlier steps pass through untouched so the trajectory stays on-manifold.

    Mirrors e8_psd_clamp.ClampPSD's pack/unpack + psd_match plumbing, and exposes
    `last` (the post-clamp packed latent) so gen_with_cb can recover the final
    latent exactly as it does for ClampPSD/RecordPSD."""

    def __init__(self, real_band, idx_map, n_bins, steps):
        self.real_band = real_band.cuda()
        self.idx_map, self.n_bins, self.steps = idx_map, n_bins, steps
        self.last = None
        self.resid = 0.0

    def __call__(self, p, i, t, kw):
        packed = kw["latents"]
        if i < self.steps - 1:
            self.last = packed
            return {}
        lat = type(p)._unpack_latents(packed, SIZE, SIZE, p.vae_scale_factor)
        lat, st = psd_match(lat, self.real_band, self.idx_map, self.n_bins,
                            return_stats=True)
        self.resid = st["imag_residue"]
        new_packed = type(p)._pack_latents(lat, 1, N_CH, H, W)
        self.last = new_packed
        return {"latents": new_packed.to(packed.dtype)}


def flux_generate_initnoise(pipe, prompt, seed, guidance, steps,
                            real_band, idx_map, n_bins):
    """Generate with the INITIAL noise spectrally shaped toward real_band.

    Reproduces the seed's white noise (same shape/RNG as the stock prepare_latents),
    psd_match-es it to the real band power, packs it, and feeds it as the starting
    latent. Returns (img, final latent, init_band) -- init_band lets the caller
    verify the shaping actually landed at step 0."""
    from diffusers import FluxPipeline
    gen = torch.Generator("cuda").manual_seed(seed)
    noise = torch.randn(LAT_SHAPE, generator=gen, device="cuda", dtype=torch.float32)
    shaped = psd_match(noise, real_band.cuda(), idx_map, n_bins)
    init_band = band_power((torch.fft.fft2(shaped.float()).abs() ** 2)[0],
                           idx_map, n_bins).cpu()
    packed = FluxPipeline._pack_latents(shaped.to(torch.bfloat16), 1, N_CH, H, W)
    captured = {}

    def grab(p, i, t, kw):
        captured["latents"] = kw["latents"]
        return {}

    img = pipe(prompt=prompt, height=SIZE, width=SIZE, guidance_scale=guidance,
               num_inference_steps=steps, latents=packed,
               callback_on_step_end=grab).images[0]
    lat = FluxPipeline._unpack_latents(captured["latents"], SIZE, SIZE,
                                       pipe.vae_scale_factor)
    return img, lat.float().cpu(), init_band


def _offline_real(pipe, cfg_lat, real_band, idx_map, n_bins, strength):
    """Offline real-SBN: psd_match the final cfg latent toward real (restyle_latent
    interpolates the target in log space at `strength`), decode with the Flux VAE."""
    lat = restyle_latent(cfg_lat.cuda(), real_band, idx_map, n_bins,
                         strength=strength)
    img = flux_vae_decode(pipe.vae, lat)
    return img, lat.float().cpu()


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def preflight(args):
    """E8's numeric asserts + E23-specific checks on the real reference."""
    e8_preflight(args)  # pack/unpack, band_power==radial_psd, psd_match, Parseval
    print("[e23] extra asserts ...", flush=True)
    dev, nb = "cuda", args.n_bins
    idx_map = band_index_map(H, W, nb, dev)

    _, real_band = real_band_power(load_real_latents(), idx_map, nb)
    assert real_band.shape == (N_CH, nb), real_band.shape
    assert torch.isfinite(real_band).all() and (real_band > 0).all(), \
        "real_band must be finite and positive"

    cc = correction_curve(real_band, real_band)
    assert (cc["ratio"] - 1).abs().max() < 1e-4, "ratio(real,real) != 1"
    assert (cc["gain"] - 1).abs().max() < 1e-4, "gain(real,real) != 1"

    # restyle_latent at strength=1 == psd_match to the same target
    a = torch.randn(1, N_CH, H, W, device=dev)
    r1 = restyle_latent(a, real_band, idx_map, nb, strength=1.0)
    r2 = psd_match(a, real_band.to(dev), idx_map, nb)
    assert (r1 - r2).abs().max() < 1e-3, "restyle(strength=1) != psd_match"
    print("[e23] pre-flight OK", flush=True)


# ---------------------------------------------------------------------------
# Part: measure -- generated-vs-real gap distribution + correction curve
# ---------------------------------------------------------------------------

def run_measure(args, report, out, state):
    nb = args.n_bins
    idx_map = band_index_map(H, W, nb, "cuda")
    real = load_real_latents()
    stacked_real, real_band = real_band_power(real, idx_map, nb)
    torch.save(real_band, f"{out}/real_band.pt")

    # generated pool = the cfg baseline (cached, shared with run_gen)
    gen_lats = []
    for key, prompt in get_prompts(args):
        cdir = f"{out}/{key}"
        _ensure_dirs(cdir)
        for s in range(args.seeds):
            tag = f"cfg{args.cfg}_s{s}"
            lpath, ipath = f"{cdir}/latents/{tag}.pt", f"{cdir}/images/{tag}.png"
            if not (os.path.exists(lpath) and os.path.exists(ipath)):
                pipe = _lazy_flux(state, args.mem)
                img, lat = flux_generate(pipe, prompt, s, args.cfg, args.steps)
                img.save(ipath)
                torch.save(lat, lpath)
                print(f"[e23] measure {key} {tag} done", flush=True)
            gen_lats.append(torch.load(lpath, weights_only=True))

    stacked_gen = torch.stack([
        band_power((torch.fft.fft2(l.cuda().float()).abs() ** 2)[0], idx_map, nb).cpu()
        for l in gen_lats])
    gen_band = stacked_gen.mean(0)

    real_dist = band_power_distribution(stacked_real)
    gen_dist = band_power_distribution(stacked_gen)
    cc = correction_curve(real_band, gen_band)

    real_ref = load_real_psd(nb, drop_dc=True)
    sdists = [spectral_dist_to_real(l, real_ref, nb) for l in gen_lats]

    report["measure"] = {
        "n_real": int(stacked_real.shape[0]),
        "n_gen": int(stacked_gen.shape[0]),
        "real_band_cmean": real_band.mean(0).tolist(),
        "gen_band_cmean": gen_band.mean(0).tolist(),
        "ratio_cmean": cc["ratio_cmean"].tolist(),
        "spectral_dist_to_real_cfg": agg(sdists),
    }
    _measure_plots(out, nb, real_dist, gen_dist, cc)
    print(f"[e23] measure: n_real={stacked_real.shape[0]} n_gen={stacked_gen.shape[0]} "
          f"scalar_gap(cfg)={report['measure']['spectral_dist_to_real_cfg']}",
          flush=True)


def _measure_plots(out, n_bins, real_dist, gen_dist, cc):
    os.makedirs(f"{out}/plots", exist_ok=True)
    centers = band_centers(n_bins).numpy()
    sl = slice(1, None)  # drop DC for log plots
    x = centers[sl]

    # 1. radial band power real vs generated, channel mean + p10-p90 spread
    def cm(d, k):
        return d[k].mean(0).numpy()[sl]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, cm(real_dist, "mean"), color="C0", label="real")
    ax.fill_between(x, cm(real_dist, "p10"), cm(real_dist, "p90"),
                    color="C0", alpha=0.2)
    ax.plot(x, cm(gen_dist, "mean"), color="C1", label="generated (cfg)")
    ax.fill_between(x, cm(gen_dist, "p10"), cm(gen_dist, "p90"),
                    color="C1", alpha=0.2)
    ax.set(xlabel="radial freq", ylabel="band power", yscale="log",
           title="real vs generated radial band power (channel mean, p10-p90)")
    ax.legend()
    fig.savefig(f"{out}/plots/psd_real_vs_gen.png", dpi=120, bbox_inches="tight")

    # 2. correction curve: per-band ratio real/gen
    fig, ax = plt.subplots(figsize=(7, 4))
    ratio = cc["ratio"].numpy()
    for c in range(ratio.shape[0]):
        ax.plot(x, ratio[c][sl], color="gray", alpha=0.25, lw=0.6)
    ax.plot(x, cc["ratio_cmean"].numpy()[sl], color="C3", lw=2,
            label="channel mean")
    ax.axhline(1.0, ls="--", c="k", lw=0.8)
    ax.set(xlabel="radial freq", ylabel="ratio real / gen", yscale="log",
           title="correction curve (real / generated band power)")
    ax.legend()
    fig.savefig(f"{out}/plots/correction_curve.png", dpi=120, bbox_inches="tight")

    # 3. per-(channel, band) heatmap of log2(real/gen)
    fig, ax = plt.subplots(figsize=(8, 4))
    l2 = np.log2(np.clip(cc["ratio"].numpy(), 1e-6, None))
    vmax = float(np.abs(l2).max()) or 1.0
    im = ax.imshow(l2, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set(xlabel="radial band", ylabel="channel",
           title="log2(real / gen) per (channel, band)")
    fig.colorbar(im, ax=ax)
    fig.savefig(f"{out}/plots/correction_heatmap.png", dpi=120,
                bbox_inches="tight")
    plt.close("all")
    print("[e23] measure plots saved", flush=True)


# ---------------------------------------------------------------------------
# Part: gen -- all conditions (cached)
# ---------------------------------------------------------------------------

def run_gen(args, report, out, state):
    nb = args.n_bins
    idx_map = band_index_map(H, W, nb, "cuda")
    real_band = get_real_band(out, nb, idx_map)
    pipe = _lazy_flux(state, args.mem)

    for key, prompt in get_prompts(args):
        cdir = f"{out}/{key}"
        _ensure_dirs(cdir)

        # --sweep_only: just cfg baseline + offline real-SBN strengths (the cheap
        # path for mining qualitative big-gain examples). The cfg=1 reference and
        # the during-gen / init conditions (settled in the first run) are skipped.
        ref = None
        if not args.sweep_only:
            ref_path = f"{cdir}/ref_cfg1.pt"
            ref_tags = [f"cfg1.0_s{s}" for s in range(args.ref_seeds)]
            if os.path.exists(ref_path) and all(
                    os.path.exists(f"{cdir}/images/{t}.png") for t in ref_tags):
                ref = torch.load(ref_path, weights_only=True)
                print(f"[e23] {key} cfg=1 reference (cached)", flush=True)
            else:
                ref, outs = record_reference(pipe, prompt, args.ref_seeds, 1.0,
                                             args.steps, nb)
                for t, (img, lat) in zip(ref_tags, outs):
                    img.save(f"{cdir}/images/{t}.png")
                    torch.save(lat, f"{cdir}/latents/{t}.pt")
                torch.save(ref, ref_path)
                print(f"[e23] {key} cfg=1 reference recorded", flush=True)

        for s in range(args.seeds):
            # cfg baseline (== the measurement pool)
            cached_gen(cdir, f"cfg{args.cfg}_s{s}", lambda: flux_generate(
                pipe, prompt, s, args.cfg, args.steps))
            # offline real-SBN (per strength), reusing the cfg final latent
            cfg_lat = torch.load(f"{cdir}/latents/cfg{args.cfg}_s{s}.pt",
                                 weights_only=True)
            for st in args.strength_sweep:
                cached_gen(cdir, f"sbn_real_off{st}_s{s}",
                           lambda l=cfg_lat, st=st: _offline_real(
                               pipe, l, real_band, idx_map, nb, st))
            if not args.sweep_only:
                # existing SBN: clamp every step toward the cfg=1 reference
                cached_gen(cdir, f"sbn_cfg1_s{s}", lambda: generate_bandnorm(
                    pipe, prompt, s, ref, args.cfg, args.steps, nb)[:2])
                # during-gen real-SBN: clamp toward real on the last step only
                cached_gen(cdir, f"sbn_real_last_s{s}", lambda: gen_with_cb(
                    pipe, prompt, s, args.cfg, args.steps,
                    ClampRealLastStep(real_band, idx_map, nb, args.steps)))
                # exploratory: shape the initial noise toward real
                cached_gen(cdir, f"sbn_real_init_s{s}",
                           lambda: flux_generate_initnoise(
                               pipe, prompt, s, args.cfg, args.steps,
                               real_band, idx_map, nb)[:2])
        print(f"[e23] {key} generation complete", flush=True)


# ---------------------------------------------------------------------------
# Part: score -- gap (primary) + fidelity + adherence guardrail
# ---------------------------------------------------------------------------

def run_score(args, report, out, state):
    # free the diffusion model before loading the heavy scorers -- aesthetic CLIP +
    # ImageReward + Flux together contend for the 24GB A5000 (fidelity_metrics note)
    if state.get("pipe") is not None:
        import gc
        state["pipe"] = None
        gc.collect()
        torch.cuda.empty_cache()
    nb = args.n_bins
    idx_map = band_index_map(H, W, nb, "cuda")
    real_band = get_real_band(out, nb, idx_map)
    real_ref = load_real_psd(nb, drop_dc=True)
    clip_model, clip_proc = load_clip(args.clip_model)
    aesthetic = load_aesthetic()
    ir = load_imagereward()
    conds = cond_list(args)
    base_cond = f"cfg{args.cfg}"

    scores = {}
    for key, prompt in get_prompts(args):
        cdir = f"{out}/{key}"
        scores[key] = {}
        for cond in conds:
            imgs, lats, paths = [], [], []
            for s in range(args.seeds):
                ip = f"{cdir}/images/{cond}_s{s}.png"
                lp = f"{cdir}/latents/{cond}_s{s}.pt"
                if os.path.exists(ip) and os.path.exists(lp):
                    imgs.append(Image.open(ip).convert("RGB"))
                    lats.append(torch.load(lp, weights_only=True))
                    paths.append(ip)
            if not imgs:
                continue
            sdist = [spectral_dist_to_real(l, real_ref, nb) for l in lats]
            blog = [band_logrms(l, real_band, idx_map, nb) for l in lats]
            aes = aesthetic_scores(aesthetic, clip_model, clip_proc, imgs)
            irs = imagereward_scores(ir, prompt, paths)
            ct = clip_scores(clip_model, clip_proc, prompt, imgs)
            scores[key][cond] = {
                "spectral_dist": agg(sdist), "band_logrms": agg(blog),
                "aesthetic": agg(aes), "imagereward": agg(irs),
                "clip_t": agg(ct),
                "per_seed": {"spectral_dist": sdist, "aesthetic": aes,
                             "imagereward": irs, "clip_t": ct},
            }
            sd = scores[key][cond]["spectral_dist"]
            print(f"[e23] score {key}/{cond}: spectral_dist="
                  f"{sd['mean']:.4f} (n={sd['n']})" if sd else
                  f"[e23] score {key}/{cond}: no spectral_dist", flush=True)

        # paired deltas vs the cfg baseline (matched seeds)
        base = scores[key].get(base_cond, {}).get("per_seed", {})
        for cond in conds:
            ps = scores[key].get(cond, {}).get("per_seed")
            if not ps or cond == base_cond:
                continue
            dv = {}
            for metric in ("spectral_dist", "clip_t", "aesthetic"):
                b, c = base.get(metric, []), ps.get(metric, [])
                d = [c[i] - b[i] for i in range(min(len(b), len(c)))
                     if b[i] is not None and c[i] is not None]
                dv[metric] = agg(d)
            scores[key][cond]["delta_vs_cfg"] = dv

    with open(f"{out}/scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    report["score"] = scores
    print(f"[e23] scores -> {out}/scores.json", flush=True)


# ---------------------------------------------------------------------------
# Part: examples -- mine per-seed biggest gains, build side-by-side panels
# ---------------------------------------------------------------------------

def _side_by_side(left_path, right_path, out_path, left, right, thumb=512):
    """Two images stacked horizontally with a caption strip above each."""
    _montage([left_path, right_path], [left, right], out_path, thumb)


def _montage(paths, labels, out_path, thumb=512):
    """N images in a row, each with a caption strip above it."""
    from PIL import ImageDraw
    pad, lh = 8, 26
    n = len(paths)
    sheet = Image.new("RGB", (n * thumb + (n + 1) * pad, thumb + lh + pad), "white")
    d = ImageDraw.Draw(sheet)
    for i, (p, lab) in enumerate(zip(paths, labels)):
        x = pad + i * (thumb + pad)
        d.text((x, 6), lab, fill="black")
        sheet.paste(Image.open(p).convert("RGB").resize((thumb, thumb)), (x, lh))
    sheet.save(out_path)


def run_examples(args, report, out, state):
    """Rank (prompt, seed, strength) by per-seed fidelity GAIN of offline real-SBN
    over the cfg baseline, and emit side-by-side cfg-vs-real-SBN panels for the
    biggest gains -- the qualitative payoff. Self-contained (re-scores the two
    sides per seed so the pairing is exact)."""
    if state.get("pipe") is not None:
        import gc
        state["pipe"] = None
        gc.collect()
        torch.cuda.empty_cache()
    clip_model, clip_proc = load_clip(args.clip_model)
    aesthetic = load_aesthetic()
    ir = load_imagereward()
    base_cond = f"cfg{args.cfg}"
    target_conds = [f"sbn_real_off{s}" for s in args.strength_sweep]

    rows = []
    for key, prompt in get_prompts(args):
        cdir = f"{out}/{key}"
        for s in range(args.seeds):
            bip = f"{cdir}/images/{base_cond}_s{s}.png"
            if not os.path.exists(bip):
                continue
            b_img = Image.open(bip).convert("RGB")
            b_ae = aesthetic_scores(aesthetic, clip_model, clip_proc, [b_img])[0]
            b_ir = imagereward_scores(ir, prompt, [bip])[0]
            for cond in target_conds:
                tip = f"{cdir}/images/{cond}_s{s}.png"
                if not os.path.exists(tip):
                    continue
                t_img = Image.open(tip).convert("RGB")
                t_ae = aesthetic_scores(aesthetic, clip_model, clip_proc, [t_img])[0]
                t_ir = imagereward_scores(ir, prompt, [tip])[0]
                rows.append({
                    "key": key, "prompt": prompt, "seed": s, "cond": cond,
                    "strength": cond.replace("sbn_real_off", ""),
                    "base_img": bip, "sbn_img": tip,
                    "base_ae": b_ae, "sbn_ae": t_ae, "base_ir": b_ir, "sbn_ir": t_ir,
                    "d_aesthetic": (t_ae - b_ae) if (t_ae is not None and b_ae is not None) else None,
                    "d_imagereward": (t_ir - b_ir) if (t_ir is not None and b_ir is not None) else None,
                })
    print(f"[e23] examples: scored {len(rows)} (seed,strength) candidates",
          flush=True)

    exdir = f"{out}/examples"
    os.makedirs(exdir, exist_ok=True)
    manifest, built = [], {}
    for metric in ("d_aesthetic", "d_imagereward"):
        cand = sorted((r for r in rows if r[metric] is not None),
                      key=lambda r: r[metric], reverse=True)[: args.top_k]
        tag = metric.replace("d_", "")
        for rank, r in enumerate(cand):
            pid = f"{r['key']}_s{r['seed']}_{r['cond']}"
            panel = built.get(pid)
            if panel is None:
                panel = f"{exdir}/{pid}.png"
                _side_by_side(
                    r["base_img"], r["sbn_img"], panel,
                    left=f"cfg{args.cfg}  aes={r['base_ae']:.2f} ir={r['base_ir']:.2f}",
                    right=f"real-SBN s={r['strength']}  aes={r['sbn_ae']:.2f} ir={r['sbn_ir']:.2f}")
                built[pid] = panel
            manifest.append({
                "rank_by": tag, "rank": rank,
                "panel": os.path.relpath(panel, out),
                "key": r["key"], "prompt": r["prompt"], "seed": r["seed"],
                "strength": r["strength"],
                "d_aesthetic": r["d_aesthetic"], "d_imagereward": r["d_imagereward"],
                "base_ae": r["base_ae"], "sbn_ae": r["sbn_ae"],
                "base_ir": r["base_ir"], "sbn_ir": r["sbn_ir"],
            })
            print(f"[e23] top-{tag} #{rank}: {pid} "
                  f"Δaes={r['d_aesthetic']:+.2f} Δir={r['d_imagereward']:+.2f}",
                  flush=True)

    with open(f"{out}/examples.json", "w") as f:
        json.dump(manifest, f, indent=2)
    report["examples"] = {"n_candidates": len(rows), "panels": len(built)}
    print(f"[e23] examples -> {out}/examples.json ({len(built)} panels)", flush=True)


# ---------------------------------------------------------------------------
# Part: adherence -- cfg1 vs cfg vs real-SBN on LONG COMPLEX prompts
# ---------------------------------------------------------------------------
#
# Shows the motivation directly: cfg=1 looks natural but DROPS prompt elements on
# compositional prompts, while cfg=3.5 keeps them but over-bakes -- and real-SBN
# (applied to the cfg=3.5 image) keeps that adherence while looking more natural.
# Adherence is scored with B-VQA (compbench.py -- BLIP-VQA attribute binding, the
# T2I-CompBench metric; compositional, what CLIP-T misses) plus CLIP-T; emits 3-way
# cfg1|cfg|real-SBN panels.

def run_adherence(args, report, out, state):
    nb = args.n_bins
    idx_map = band_index_map(H, W, nb, "cuda")
    real_band = get_real_band(out, nb, idx_map)
    adir = os.path.join(out, "adherence")
    prompts = COMPLEX_PROMPTS[: args.num_complex]

    # --- generate cfg1, cfg, offline real-SBN @ strengths (cached) ---
    pipe = _lazy_flux(state, args.mem)
    for key, prompt in prompts:
        cdir = f"{adir}/{key}"
        _ensure_dirs(cdir)
        for s in range(args.seeds):
            cached_gen(cdir, f"cfg1.0_s{s}", lambda: flux_generate(
                pipe, prompt, s, 1.0, args.steps))
            cached_gen(cdir, f"cfg{args.cfg}_s{s}", lambda: flux_generate(
                pipe, prompt, s, args.cfg, args.steps))
            cfg_lat = torch.load(f"{cdir}/latents/cfg{args.cfg}_s{s}.pt",
                                 weights_only=True)
            for st in args.strength_sweep:
                cached_gen(cdir, f"sbn_real_off{st}_s{s}",
                           lambda l=cfg_lat, st=st: _offline_real(
                               pipe, l, real_band, idx_map, nb, st))
        print(f"[e23] adherence {key} generation complete", flush=True)

    # --- score adherence (free the diffusion model first) ---
    state["pipe"] = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    clip_model, clip_proc = load_clip(args.clip_model)
    # B-VQA (BLIP-VQA, the T2I-CompBench attribute-binding metric): per noun phrase
    # in the prompt, P(yes), multiplied -- catches the compositional drops CLIP-T misses.
    bvqa = load_bvqa() if args.vqa else None
    rec = f"sbn_real_off{args.rec_strength}"
    conds = ["cfg1.0", f"cfg{args.cfg}", rec]

    os.makedirs(f"{adir}/panels", exist_ok=True)
    scores, manifest = {}, []
    for key, prompt in prompts:
        cdir = f"{adir}/{key}"
        scores[key] = {"prompt": prompt}
        for cond in conds:
            imgs, paths = [], []
            for s in range(args.seeds):
                ip = f"{cdir}/images/{cond}_s{s}.png"
                if os.path.exists(ip):
                    imgs.append(Image.open(ip).convert("RGB"))
                    paths.append(ip)
            if not imgs:
                continue
            scores[key][cond] = {
                "clip_t": agg(clip_scores(clip_model, clip_proc, prompt, imgs)),
                "bvqa": agg(bvqa_scores(bvqa, prompt, imgs)) if bvqa else None,
            }
        # per-seed 3-way panels (cfg1 | cfg | real-SBN@rec)
        for s in range(args.seeds):
            ps = [f"{cdir}/images/{c}_s{s}.png" for c in conds]
            if not all(os.path.exists(p) for p in ps):
                continue
            panel = f"{adir}/panels/{key}_s{s}.png"
            _montage(ps, [f"cfg 1.0 (weak)", f"cfg {args.cfg}",
                          f"real-SBN s={args.rec_strength}"], panel)
            manifest.append({"key": key, "seed": s,
                             "panel": os.path.relpath(panel, out)})
        print(f"[e23] adherence {key}: scored "
              f"{[c for c in conds if c in scores[key]]}", flush=True)

    with open(f"{adir}/adherence.json", "w") as f:
        json.dump({"conds": conds, "rec_strength": args.rec_strength,
                   "scores": scores, "panels": manifest}, f, indent=2)
    report["adherence"] = {"prompts": len(prompts), "panels": len(manifest),
                           "bvqa": bool(bvqa)}
    print(f"[e23] adherence -> {adir}/adherence.json ({len(manifest)} panels, "
          f"bvqa={'on' if bvqa else 'off'})", flush=True)


# ---------------------------------------------------------------------------
# Part: analyze -- grids + summary plots
# ---------------------------------------------------------------------------

def run_analyze(args, report, out, state):
    conds = cond_list(args)
    prompts = get_prompts(args)
    ncols = min(args.seeds, 8)

    for key, _ in prompts:
        cdir = f"{out}/{key}"
        rows, labels = [], []
        for cond in conds:
            imgs = [Image.open(f"{cdir}/images/{cond}_s{s}.png")
                    for s in range(ncols)
                    if os.path.exists(f"{cdir}/images/{cond}_s{s}.png")]
            if imgs:
                rows.append(imgs)
                labels.append(cond)
        if rows:
            save_grid(rows, labels,
                      [f"seed {s}" for s in range(max(len(r) for r in rows))],
                      f"{out}/grid_{key}.png")
            print(f"[e23] grid_{key} saved", flush=True)

    spath = f"{out}/scores.json"
    if not os.path.exists(spath):
        print("[e23] no scores.json; skipping summary plots "
              "(run --part score first)", flush=True)
        return
    with open(spath) as f:
        scores = json.load(f)
    _analyze_plots(out, scores, conds, prompts)


def _analyze_plots(out, scores, conds, prompts):
    os.makedirs(f"{out}/plots", exist_ok=True)

    def cmean(cond, metric):
        vals = [scores.get(k, {}).get(cond, {}).get(metric, {}).get("mean")
                for k, _ in prompts]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    sdist = [cmean(c, "spectral_dist") for c in conds]
    aes = [cmean(c, "aesthetic") for c in conds]
    ct = [cmean(c, "clip_t") for c in conds]

    _bar(f"{out}/plots/spectral_dist.png", conds, sdist,
         "spectral_dist_to_real (lower = closer to real)")
    _bar(f"{out}/plots/clip_t.png", conds, ct,
         "CLIP-T (prompt adherence guardrail)")

    fig, ax = plt.subplots(figsize=(6, 5))
    for c, sd, a in zip(conds, sdist, aes):
        if sd is not None and a is not None:
            ax.scatter(sd, a)
            ax.annotate(c, (sd, a), fontsize=7)
    ax.set(xlabel="spectral_dist_to_real", ylabel="aesthetic",
           title="fidelity vs gap closure")
    fig.savefig(f"{out}/plots/aesthetic_vs_spectral.png", dpi=120,
                bbox_inches="tight")
    plt.close("all")
    print("[e23] analyze plots saved", flush=True)


def _bar(path, labels, vals, title):
    keep = [(l, v) for l, v in zip(labels, vals) if v is not None]
    if not keep:
        return
    ls, ys = zip(*keep)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(ys)), ys)
    ax.set_xticks(range(len(ys)))
    ax.set_xticklabels(ls, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    out = os.path.join(RESULTS, "e23")
    os.makedirs(out, exist_ok=True)
    preflight(args)

    report = {"params": vars(args)}
    state = {"pipe": None}
    runners = {"measure": run_measure, "gen": run_gen, "score": run_score,
               "examples": run_examples, "adherence": run_adherence,
               "analyze": run_analyze}
    for part in filter(None, (p.strip() for p in args.part.split(","))):
        runners[part](args, report, out, state)

    path = f"{out}/report.json"
    if os.path.exists(path):
        with open(path) as f:
            merged = json.load(f)
        merged.update(report)
        report = merged
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e23] report -> {path}", flush=True)


def floats(s):
    return [float(v) for v in s.split(",") if v.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="measure,gen,score,analyze")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--ref_seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--strength_sweep", type=floats, default="0.5,1.0")
    ap.add_argument("--sweep_only", action="store_true",
                    help="gen only cfg + offline real-SBN strengths (cheap path)")
    ap.add_argument("--top_k", type=int, default=8,
                    help="qualitative big-gain examples to surface per metric")
    ap.add_argument("--prompt_set", default="classes",
                    choices=["classes", "complex"],
                    help="which built-in prompt set to use when --prompt is unset")
    ap.add_argument("--num_complex", type=int, default=len(COMPLEX_PROMPTS),
                    help="how many long complex prompts for --part adherence")
    ap.add_argument("--rec_strength", type=float, default=0.25,
                    help="recommended real-SBN strength for the 3-way panels")
    ap.add_argument("--vqa", action="store_true",
                    help="also score VQAScore (compositional adherence; heavy)")
    ap.add_argument("--vqa_model", default="clip-flant5-xl")
    ap.add_argument("--mem", default="gpu_resident",
                    choices=["bnb4", "gpu_resident", "seq_offload"])
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--num_classes", type=int, default=3)
    main(ap.parse_args())
