"""E14: parametric functions on the FFT phase of FLUX latents.

E13 swapped phase vs magnitude wholesale; E14 deforms the phase parametrically
and asks *which bands* carry identity and how the output bends. Four sweeps, all
applied one-shot to the cached E13 base latents then decoded through the Flux VAE
(no re-diffusion):

  scale   -- phi -> alpha*phi, alpha in {0, 0.5, 1, 2}      (scale_phase)
  shift   -- linear frequency ramp = spatial shift          (phase_ramp)
             vs a constant antisymmetric offset             (phase_offset)
             -- the ramp IS a translation (Fourier shift theorem); the constant
                offset is NOT. This sub-sweep demonstrates/dispels that.
  rotate  -- constant phase rotation within selected bands  (rotate_band_phase)
  noise   -- phi -> phi + eps*eta within a band, sweep eps  (add_band_phase_noise)
             low band vs high band -> localize where identity breaks.

Each output is scored with image_metrics + CLIP cosine to the unmodified decode.

Expected: low-band phase noise destroys identity at small eps; high-band phase
noise is near-free (mirrors E6 quantization); the ramp is a clean wrap-around
shift; alpha=0 collapses to a flat real-even latent; alpha=2 scrambles.

Reuses E13's latent bank (results/e13/latents). Single part (decode-only):
needs the Flux VAE (~160MB) + CLIP ViT-L (~1.7GB), no transformer.
"""
import argparse
import json
import os
import sys
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_grid, RESULTS
from spectral_ops import (band_index_map, scale_phase, phase_ramp, phase_offset,
                          rotate_band_phase, add_band_phase_noise)
from e7_flux_phase import load_flux_vae, flux_vae_decode
from e9_bandnorm_classes import CLASSES, image_metrics, METRICS
from clip_sim import load_clip, clip_image_features, cosine

SIZE = 1024
H = W = 128
C = 16
N_BINS = 24
E13_LAT = os.path.join(RESULTS, "e13", "latents")

ALPHAS = [0.0, 0.5, 1.0, 2.0]
SHIFTS = [8, 16, 32]                 # integer pixel shifts for the ramp
OFFSETS = [0.5, 1.0, 2.0]            # constant antisymmetric phase offsets (rad)
ROT_BANDS = {"low": list(range(0, 6)), "high": list(range(12, N_BINS))}
ROT_DELTAS = [0.5, 1.0]
EPS = [0.25, 0.5, 1.0, 2.0]          # phase-noise amplitudes
NOISE_BANDS = {"low": list(range(0, 6)), "high": list(range(12, N_BINS))}


def select_classes(args):
    if not args.classes:
        return CLASSES
    want = [c.strip() for c in args.classes.split(",")]
    by_name = dict(CLASSES)
    missing = [w for w in want if w not in by_name]
    assert not missing, f"unknown classes {missing}; have {list(by_name)}"
    return [(w, by_name[w]) for w in want]


def preflight(args):
    print("[e14] pre-flight asserts ...", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    fy = torch.fft.fftfreq(H, device=dev); fx = torch.fft.fftfreq(W, device=dev)
    rr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).clamp(min=1e-3)
    g = torch.randn(1, C, H, W, device=dev)
    lat = torch.fft.ifft2(torch.fft.fft2(g) * rr[None, None] ** -1.0).real
    idx = band_index_map(H, W, N_BINS, dev)

    # 1. ramp == roll for integer shift (the shift theorem, exactly)
    pr = phase_ramp(lat, 8, 16)
    roll = torch.roll(lat, shifts=(8, 16), dims=(-2, -1))
    assert (pr - roll).abs().max() < 1e-3, "ramp must equal roll"

    # 2. alpha=1 identity; alpha=0 real (zero-phase)
    assert (scale_phase(lat, 1.0) - lat).abs().max() < 1e-3, "alpha=1 identity"

    # 3. constant offset is NOT a shift (differs from every integer roll)
    off = phase_offset(lat, 1.0)
    assert (off - lat).abs().max() > 1e-2, "offset should change the latent"
    is_shift = min(float((off - torch.roll(lat, (dy, dx), (-2, -1))).abs().max())
                   for dy in range(0, H, 16) for dx in range(0, W, 16))
    assert is_shift > 1e-2, "constant offset must NOT match any roll"

    # 4. band phase noise is real + erodes low band more than high at eps=1
    def corr(a, b):
        a = a.flatten(); b = b.flatten(); a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
    lo = add_band_phase_noise(lat, idx, NOISE_BANDS["low"], 1.0,
                              generator=torch.Generator(dev).manual_seed(1))
    hi = add_band_phase_noise(lat, idx, NOISE_BANDS["high"], 1.0,
                              generator=torch.Generator(dev).manual_seed(1))
    assert corr(lat, lo) < corr(lat, hi), "low-band noise should erode more"
    print(f"[e14] pre-flight OK (ramp==roll, offset!=shift, "
          f"corr_low={corr(lat, lo):.3f} < corr_high={corr(lat, hi):.3f})",
          flush=True)


def _decode_cached(vae, lat, path):
    if os.path.exists(path):
        return Image.open(path)
    img = flux_vae_decode(vae, lat)
    img.save(path)
    return img


def run(args, classes, out):
    vae = load_flux_vae()
    model, proc = load_clip()
    idx = band_index_map(H, W, N_BINS, "cuda")
    os.makedirs(f"{out}/images", exist_ok=True)
    os.makedirs(f"{out}/plots", exist_ok=True)
    report = {}
    # for the cross-class identity-vs-eps curve
    noise_curves = {b: {n: [] for n, _ in classes} for b in NOISE_BANDS}

    for name, _ in classes:
        # one representative latent per class (seed 0) keeps the sweep readable
        lat = torch.load(f"{E13_LAT}/{name}_s0.pt", weights_only=True).cuda()
        base = _decode_cached(vae, lat, f"{out}/images/{name}_base.png")
        base_feat = clip_image_features(model, proc, [base])[0]
        cls = {}

        def score(tag, edited_lat):
            img = _decode_cached(vae, edited_lat, f"{out}/images/{name}_{tag}.png")
            feat = clip_image_features(model, proc, [img])[0]
            m = image_metrics(img)
            return img, {"clip_to_base": cosine(feat, base_feat),
                         **{k: m[k] for k in METRICS}}

        # --- scale sweep ---
        scale_imgs = []
        for a in ALPHAS:
            img, sc = score(f"scale_a{a}", scale_phase(lat, a))
            cls[f"scale/alpha{a}"] = sc
            scale_imgs.append(img)

        # --- shift (ramp) vs constant offset ---
        shift_imgs, off_imgs = [], []
        for d in SHIFTS:
            img, sc = score(f"shift_{d}", phase_ramp(lat, d, d))
            cls[f"shift/d{d}"] = sc
            shift_imgs.append(img)
        for o in OFFSETS:
            img, sc = score(f"offset_{o}", phase_offset(lat, o))
            cls[f"offset/d{o}"] = sc
            off_imgs.append(img)

        # --- per-band rotation ---
        for bname, bands in ROT_BANDS.items():
            for d in ROT_DELTAS:
                _, sc = score(f"rot_{bname}_d{d}",
                              rotate_band_phase(lat, idx, bands, d))
                cls[f"rotate/{bname}_d{d}"] = sc

        # --- graded phase noise, low vs high band ---
        noise_imgs = {b: [] for b in NOISE_BANDS}
        for bname, bands in NOISE_BANDS.items():
            for e in EPS:
                g = torch.Generator("cuda").manual_seed(
                    zlib.crc32(f"{name}_{bname}_{e}".encode()))
                img, sc = score(f"noise_{bname}_e{e}",
                                add_band_phase_noise(lat, idx, bands, e, generator=g))
                cls[f"noise/{bname}_e{e}"] = sc
                noise_curves[bname][name].append(sc["clip_to_base"])
                noise_imgs[bname].append(img)

        report[f"class/{name}"] = cls
        print(f"[e14] {name}: scale a2 clip={cls['scale/alpha2.0']['clip_to_base']:.3f} | "
              f"noise low e1={cls['noise/low_e1.0']['clip_to_base']:.3f} "
              f"high e1={cls['noise/high_e1.0']['clip_to_base']:.3f}", flush=True)

        # per-class grids
        save_grid([scale_imgs], ["scale"],
                  [f"alpha={a}" for a in ALPHAS], f"{out}/grid_{name}_scale.png")
        save_grid([shift_imgs, off_imgs], ["ramp(=shift)", "const offset"],
                  [f"d={d}" for d in SHIFTS] + [""],
                  f"{out}/grid_{name}_shift.png")
        save_grid([noise_imgs["low"], noise_imgs["high"]],
                  ["low-band noise", "high-band noise"],
                  [f"eps={e}" for e in EPS], f"{out}/grid_{name}_noise.png")
        del lat
        torch.cuda.empty_cache()

    make_noise_plot(noise_curves, classes, out)
    report["_noise_curves"] = {b: {n: v for n, v in d.items()}
                               for b, d in noise_curves.items()}
    report["_eps"] = EPS
    return report


def make_noise_plot(noise_curves, classes, out):
    """CLIP-to-base vs eps, low vs high band, averaged across classes -- the
    'where does identity live' headline."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for bname, style in (("low", "-o"), ("high", "--s")):
        per_class = [noise_curves[bname][n] for n, _ in classes]
        mean = [sum(c[i] for c in per_class) / len(per_class)
                for i in range(len(EPS))]
        ax.plot(EPS, mean, style, label=f"{bname}-band phase noise")
    ax.set(xlabel="phase-noise amplitude eps", ylabel="CLIP cosine to unmodified",
           title="identity erosion: low-band vs high-band phase noise")
    ax.legend()
    fig.savefig(f"{out}/plots/identity_vs_eps.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[e14] noise plot saved", flush=True)


def main(args):
    out = os.path.join(RESULTS, "e14")
    os.makedirs(out, exist_ok=True)
    preflight(args)
    classes = select_classes(args)
    assert os.path.isdir(E13_LAT), f"missing E13 latents at {E13_LAT}; run E13 gen first"

    report = {"params": vars(args)}
    if args.part == "run":
        report.update(run(args, classes, out))

    path = f"{out}/report.json"
    if os.path.exists(path):
        with open(path) as f:
            merged = json.load(f)
        merged.update(report)
        report = merged
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e14] report -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="run")
    ap.add_argument("--classes", default="", help="comma list; empty = all 6")
    main(ap.parse_args())
