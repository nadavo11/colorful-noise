"""E29 - Phase inheritance: does the seed's FFT phase determine the output's phase?

Prior repo work established that the FFT *phase* of a latent carries image
structure. Here we ask, for the deterministic DDIM map z_T -> z_0 in SD1.5:
how much of the OUTPUT latent's phase is inherited from the SEED's phase, as a
function of radial frequency band and of classifier-free-guidance strength?
We then *causally* confirm it by transplanting the seed's phase in a band.

Naming: the user calls the seed `z0` and the output `z1`; in diffusion
convention these are z_T (pure-noise seed) and z_0 (final denoised latent before
VAE decode). This script uses seed (z_T) / output (z_0).

Parts (run separately):
  preflight  - tiny end-to-end check (shapes, transplant variance, metric sanity)
  gen        - generate N seeds per (prompt, CFG) condition; cache z_T and z_0
  analyze    - phase-inheritance spectrum + null, magnitude control, dphi, plots
  transplant - causal arm: swap seed phase in a band, measure output follow
  all        - gen + analyze + transplant

Run:  python experiments/e29_phase_inherit.py <part> [quick]
Output: experiments/results/e29/{plots/, latents/, examples/, report.json}
"""
import json
import os
import sys

import torch

from common import RESULTS, save_grid
from spectral_ops import band_phase_swap, magnitude_only, phase_only
import e29_phase_ops as ops

# ---- config -----------------------------------------------------------------
SD15_IDS = ["sd-legacy/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"]
OUT = os.path.join(RESULTS, "e29")

PROMPTS = [
    "a photograph of a mountain lake at sunrise",
    "a portrait photo of an old fisherman",
    "a still life of a bowl of fruit on a table",
]
CFGS = [1.0, 3.0, 7.5]                       # guidance sweep (per prompt)
# CFG=1.0 with an empty prompt = the pure unconditional z_T -> z_0 map.

N = int(os.environ.get("E29_N", 64))         # seeds per condition
SIZE = int(os.environ.get("E29_SIZE", 512))  # SD1.5 native -> (1,4,64,64) latent
STEPS = int(os.environ.get("E29_STEPS", 50))
NBINS = int(os.environ.get("E29_NBINS", 24))
N_PERM = int(os.environ.get("E29_NPERM", 20))

# transplant arm
TPAIRS = int(os.environ.get("E29_TPAIRS", 16))      # number of (A,B) seed pairs
CUTS = [0.1, 0.25, 0.5]                              # lowest-c radial band swapped
T_PROMPT = PROMPTS[0]
T_CFG = 7.5

DTYPE = torch.float16 if os.environ.get("E29_DTYPE", "fp16") == "fp16" else torch.float32
LAT_SHAPE = (1, 4, SIZE // 8, SIZE // 8)


# ---- model ------------------------------------------------------------------
def load_sd15():
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    pipe = last = None
    for mid in SD15_IDS:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                mid, torch_dtype=DTYPE, safety_checker=None,
                requires_safety_checker=False)
            break
        except Exception as e:  # noqa: BLE001
            last = e
    if pipe is None:
        raise RuntimeError(f"could not load SD1.5: {last}")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)  # deterministic
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def seed_latent(seed):
    """The initial noise z_T for `seed` (fp32, on cuda)."""
    g = torch.Generator("cuda").manual_seed(int(seed))
    return torch.randn(LAT_SHAPE, generator=g, device="cuda", dtype=torch.float32)


@torch.no_grad()
def run(pipe, prompt, z_T, cfg):
    """Deterministic DDIM generation from a given seed latent z_T.
    Returns the final denoised latent z_0 (1,4,64,64) fp32 cpu, in the same
    (scaled) latent space as z_T."""
    z0 = pipe(prompt=prompt, latents=z_T.to(pipe.dtype),
              num_inference_steps=STEPS, guidance_scale=cfg,
              height=SIZE, width=SIZE, output_type="latent").images
    return z0.float().cpu()


@torch.no_grad()
def decode(pipe, lat):
    """Latent (1,4,64,64) in scaled space -> PIL image."""
    from PIL import Image
    sf = pipe.vae.config.scaling_factor
    img = pipe.vae.decode((lat.to("cuda").to(pipe.dtype) / sf)).sample
    img = (img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((img * 255).astype("uint8"))


def conditions(prompts):
    """List of (label, prompt, cfg). Empty-prompt CFG=1.0 = unconditional map."""
    conds = [("uncond", "", 1.0)]
    for pi, p in enumerate(prompts):
        for cfg in CFGS:
            conds.append((f"p{pi}_cfg{cfg}", p, cfg))
    return conds


# ---- parts ------------------------------------------------------------------
def part_gen(quick=False):
    os.makedirs(os.path.join(OUT, "latents"), exist_ok=True)
    prompts = PROMPTS[:1] if quick else PROMPTS
    n = 4 if quick else N
    pipe = load_sd15()
    for label, prompt, cfg in conditions(prompts):
        seeds, outs = [], []
        for s in range(n):
            z_T = seed_latent(s)
            z_0 = run(pipe, prompt, z_T, cfg)
            seeds.append(z_T.float().cpu())
            outs.append(z_0)
        pack = {"seed": torch.cat(seeds), "out": torch.cat(outs),
                "prompt": prompt, "cfg": cfg, "n": n}
        torch.save(pack, os.path.join(OUT, "latents", f"{label}.pt"))
        print(f"[gen] {label:14s} n={n} prompt={prompt!r} cfg={cfg}", flush=True)
    print(f"[gen] wrote {OUT}/latents/*.pt")


def part_analyze(quick=False):
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    prompts = PROMPTS[:1] if quick else PROMPTS
    g = torch.Generator("cuda").manual_seed(0)
    report = {"config": {"N": N, "size": SIZE, "steps": STEPS, "nbins": NBINS,
                         "cfgs": CFGS, "prompts": prompts}, "conditions": {}}

    spectra = {}
    for label, prompt, cfg in conditions(prompts):
        path = os.path.join(OUT, "latents", f"{label}.pt")
        if not os.path.exists(path):
            print(f"[analyze] missing {path}, skip")
            continue
        pk = torch.load(path)
        seed = pk["seed"].cuda()
        out = pk["out"].cuda()
        spec = ops.inheritance_spectrum(seed, out, NBINS, N_PERM, generator=g)
        mag = ops.magnitude_spectrum_corr(seed, out, NBINS)
        dphi = ops.dphi_resultant(seed, out, NBINS)
        sp = ops.spatial_pearson(seed, out)
        spectra[label] = {"spec": spec, "mag": mag, "dphi": dphi,
                          "cfg": cfg, "prompt": prompt}
        report["conditions"][label] = {
            "prompt": prompt, "cfg": cfg, "spatial_pearson": sp,
            "centers": spec["centers"].tolist(),
            "phase_corr_band": spec["r_band"].tolist(),
            "null_mean": spec["null_mean"].tolist(),
            "null_std": spec["null_std"].tolist(),
            "mag_corr_band": mag.tolist(),
            "dphi_resultant": dphi.tolist(),
        }
        print(f"[analyze] {label:14s} phase_corr(low..high)="
              f"{[round(x,2) for x in spec['r_band'][:6].tolist()]}... "
              f"spatial_r={sp:+.3f}", flush=True)

    _plots(spectra, prompts)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[analyze] wrote {OUT}/report.json + plots/")


def part_transplant(quick=False):
    os.makedirs(os.path.join(OUT, "examples"), exist_ok=True)
    npairs = 3 if quick else TPAIRS
    cuts = CUTS[:1] if quick else CUTS
    pipe = load_sd15()

    # base/donor outputs (generated once per seed), keyed by seed index
    seeds_used = list(range(2 * npairs))
    z_T = {s: seed_latent(s) for s in seeds_used}
    out = {s: run(pipe, T_PROMPT, z_T[s], T_CFG) for s in seeds_used}

    follow = {c: [] for c in cuts}      # per-cut list of (NBINS,) follow curves
    std_ok = []
    grid_rows, grid_labels = [], []
    for pi in range(npairs):
        sA, sB = 2 * pi, 2 * pi + 1
        for c in cuts:
            # A' = A magnitude + A phase outside, donor B phase inside lowest-c band
            Aprime = band_phase_swap(z_T[sB], z_T[sA], c, mag_from="B").real
            std_ok.append(float(Aprime.std()))
            out_Ap = run(pipe, T_PROMPT, Aprime, T_CFG)
            f = ops.follow_score(out[sA][0].cuda(), out_Ap[0].cuda(),
                                 out[sB][0].cuda(), NBINS)
            follow[c].append(f)
            if pi == 0:                 # qualitative grid for the first pair
                grid_rows.append([decode(pipe, out[sA]), decode(pipe, out[sB]),
                                  decode(pipe, out_Ap)])
                grid_labels.append(f"cut c={c}")

    save_grid(grid_rows, grid_labels, ["base A", "donor B", "A' (B-phase low band)"],
              os.path.join(OUT, "examples", "transplant_grid.png"))

    report = {"config": {"npairs": npairs, "cuts": cuts, "prompt": T_PROMPT,
                         "cfg": T_CFG, "nbins": NBINS},
              "transplant_seed_std_mean": sum(std_ok) / len(std_ok),
              "follow": {}}
    for c in cuts:
        stack = torch.stack(follow[c])          # (npairs, NBINS)
        report["follow"][str(c)] = {"mean": stack.mean(0).tolist(),
                                    "std": stack.std(0).tolist()}
        print(f"[transplant] c={c} follow(low..high)="
              f"{[round(x,2) for x in stack.mean(0)[:6].tolist()]}...", flush=True)
    print(f"[transplant] transplanted-seed std mean="
          f"{report['transplant_seed_std_mean']:.3f} (expect ~1.0)")
    _plot_follow(report, cuts)
    with open(os.path.join(OUT, "transplant.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[transplant] wrote {OUT}/transplant.json + examples/transplant_grid.png")


def part_preflight():
    """Tiny end-to-end smoke test + metric sanity checks."""
    pipe = load_sd15()
    z_T = seed_latent(0)
    z_0 = run(pipe, PROMPTS[0], z_T, 7.5)
    assert tuple(z_T.shape) == LAT_SHAPE, z_T.shape
    assert tuple(z_0.shape) == LAT_SHAPE, z_0.shape
    print(f"[preflight] z_T {tuple(z_T.shape)} std={z_T.std():.3f}  "
          f"z_0 {tuple(z_0.shape)} std={z_0.std():.3f}")

    # transplant preserves variance (still ~N(0,1) latent)
    Aprime = band_phase_swap(seed_latent(1), z_T, 0.25, mag_from="B").real
    print(f"[preflight] transplanted seed std={Aprime.std():.3f} (expect ~1.0)")
    assert abs(float(Aprime.std()) - 1.0) < 0.1, "transplant changed variance!"

    # metric sanity: corr(phase, phase) ~ 1 ; corr(phase, independent) ~ 0
    seeds = torch.stack([seed_latent(s)[0] for s in range(8)]).cuda()
    rand = torch.randn_like(seeds)
    self_r = ops.inheritance_spectrum(seeds, seeds, NBINS, n_perm=3)["r_band"]
    ind = ops.inheritance_spectrum(seeds, rand, NBINS, n_perm=3)
    print(f"[preflight] circ_corr(self)  mid-band={float(self_r[NBINS//2]):.3f} (expect ~1)")
    print(f"[preflight] circ_corr(indep) mid-band={float(ind['r_band'][NBINS//2]):+.3f} "
          f"null={float(ind['null_mean'][NBINS//2]):+.3f} (expect ~0)")
    assert self_r[NBINS // 2] > 0.9, "self circular-corr should be ~1"
    assert abs(float(ind['r_band'][NBINS // 2])) < 0.4, "independent corr should be ~0"

    # one real inheritance number on a tiny stack
    outs = torch.stack([run(pipe, PROMPTS[0], seed_latent(s), 7.5)[0]
                        for s in range(8)]).cuda()
    real = ops.inheritance_spectrum(seeds, outs, NBINS, n_perm=5)
    print(f"[preflight] REAL seed->output phase corr (low..high)="
          f"{[round(x,2) for x in real['r_band'][:6].tolist()]}...")
    print(f"[preflight] null (low..high)="
          f"{[round(x,2) for x in real['null_mean'][:6].tolist()]}...")
    print("[preflight] OK")


# ---- plotting ---------------------------------------------------------------
def _plots(spectra, prompts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # average phase_corr per CFG across prompts, + uncond, + null band
    def avg_over_prompts(key, cfg):
        vals = [v[key] if key in v else v["spec"]["r_band"]
                for lbl, v in spectra.items()
                if v["cfg"] == cfg and lbl != "uncond"]
        return None if not vals else torch.stack(vals).mean(0)

    centers = next(iter(spectra.values()))["spec"]["centers"].numpy()

    # 1) inheritance spectrum: phase corr vs frequency, per CFG + uncond + null
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for cfg in CFGS:
        vals = [v["spec"]["r_band"] for lbl, v in spectra.items()
                if v["cfg"] == cfg and lbl != "uncond"]
        if vals:
            ax.plot(centers, torch.stack(vals).mean(0).numpy(), marker="o",
                    ms=3, label=f"CFG={cfg}")
    if "uncond" in spectra:
        ax.plot(centers, spectra["uncond"]["spec"]["r_band"].numpy(), "k--",
                lw=1.5, label="unconditional")
        nm = spectra["uncond"]["spec"]["null_mean"].numpy()
        ns = spectra["uncond"]["spec"]["null_std"].numpy()
        ax.fill_between(centers, nm - 2 * ns, nm + 2 * ns, color="gray",
                        alpha=0.25, label="null (±2σ)")
    ax.set_xlabel("radial frequency  (0 = low / DC, ~0.7 = high / Nyquist)")
    ax.set_ylabel("circular corr(seed phase, output phase)")
    ax.set_title("E29: phase-inheritance spectrum (seed z_T -> output z_0)")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "inherit_spectrum.png"), dpi=120)
    plt.close(fig)

    # 2) 2D per-bin heatmap for the unconditional map
    if "uncond" in spectra:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(spectra["uncond"]["spec"]["r2d"].numpy(),
                       cmap="magma", vmin=0, vmax=1)
        ax.set_title("per-bin phase corr (uncond, fftshifted)\ncenter = DC")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "plots", "inherit_heatmap.png"), dpi=120)
        plt.close(fig)

    # 3) magnitude control vs phase (uncond)
    if "uncond" in spectra:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(centers, spectra["uncond"]["spec"]["r_band"].numpy(), "o-",
                label="PHASE circular corr")
        ax.plot(centers, spectra["uncond"]["mag"].numpy(), "s--",
                label="MAGNITUDE (log) Pearson")
        ax.set_xlabel("radial frequency")
        ax.set_ylabel("seed->output correlation")
        ax.set_title("E29: phase vs magnitude inheritance (unconditional)")
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "plots", "magnitude_control.png"), dpi=120)
        plt.close(fig)

    # 4) dphi resultant
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cfg in CFGS:
        vals = [v["dphi"] for lbl, v in spectra.items()
                if v["cfg"] == cfg and lbl != "uncond"]
        if vals:
            ax.plot(centers, torch.stack(vals).mean(0).numpy(), marker="o",
                    ms=3, label=f"CFG={cfg}")
    if "uncond" in spectra:
        ax.plot(centers, spectra["uncond"]["dphi"].numpy(), "k--", label="uncond")
    ax.set_xlabel("radial frequency")
    ax.set_ylabel("|mean exp(i Δφ)|  (phase-diff resultant)")
    ax.set_title("E29: phase-difference resultant (secondary diagnostic)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "dphi_resultant.png"), dpi=120)
    plt.close(fig)


def _plot_follow(report, cuts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    nb = report["config"]["nbins"]
    x = np.arange(nb)
    for c in cuts:
        m = np.array(report["follow"][str(c)]["mean"])
        s = np.array(report["follow"][str(c)]["std"])
        ax.plot(x, m, marker="o", ms=3, label=f"swapped band c={c}")
        ax.fill_between(x, m - s, m + s, alpha=0.15)
    ax.axhline(0.5, color="k", ls=":", label="no effect (0.5)")
    ax.set_xlabel("radial band id (0 = low / DC -> high)")
    ax.set_ylabel("follow score (1 = output phase moved to donor B)")
    ax.set_title("E29 causal: seed phase transplant -> output phase follows donor")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "follow.png"), dpi=120)
    plt.close(fig)


def main():
    args = sys.argv[1:]
    part = args[0] if args else "all"
    quick = "quick" in args
    if part == "preflight":
        part_preflight()
    elif part == "gen":
        part_gen(quick)
    elif part == "analyze":
        part_analyze(quick)
    elif part == "transplant":
        part_transplant(quick)
    elif part == "all":
        part_gen(quick); part_analyze(quick); part_transplant(quick)
    else:
        print(f"unknown part {part!r}; use preflight|gen|analyze|transplant|all")


if __name__ == "__main__":
    main()
