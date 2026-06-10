"""E13: full-spectrum phase <-> magnitude swap in FLUX latent space.

The Oppenheim-Lim (1981) experiment, in the Flux output latent. E7 already
showed identity follows the *low-band* phase donor in a band-split interpolation;
E13 asks the global version: across the WHOLE spectrum, does perceived identity
track FFT phase or magnitude, and what do phase-only / magnitude-only latents
decode to?

For each image class we generate a small bank of cfg=3.5 latents, pair them
(A, B), and decode six variants through the Flux VAE only (no re-diffusion):
  baseA, baseB, A-phase+B-mag, B-phase+A-mag, phase-only(A), magnitude-only(A).
Each variant is scored with image_metrics + CLIP image cosine to source A and B.

Expected: identity (CLIP) tracks the phase donor; magnitude-only ~ textured
palette swatch (low CLIP to either layout); phase-only ~ recognizable but flat /
desaturated -- Oppenheim-Lim holds in the Flux latent.

Parts (--part, comma list):
  gen     -- generate + cache the base latent bank (needs the Flux transformer)
  analyze -- FFT swaps + VAE decode + image_metrics + CLIP + plots + report
             (needs only the Flux VAE ~160MB and CLIP ViT-L ~1.7GB)
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_grid, RESULTS
from spectral_ops import (band_index_map, radial_psd, band_phase_swap,
                          phase_only, magnitude_only, random_hermitian_phase,
                          flatness)
from e7_flux_phase import load_flux, flux_generate, load_flux_vae, flux_vae_decode
from e9_bandnorm_classes import CLASSES, image_metrics, METRICS
from clip_sim import load_clip, clip_image_features, cosine

SIZE = 1024
H = W = 128
C = 16

# the 6 decoded columns; baseA/baseB first, then the four hybrids
VARIANTS = ["baseA", "baseB", "Aphase_Bmag", "Bphase_Amag", "phaseonly_A", "magonly_A"]
VAR_LABELS = {"baseA": "A (orig)", "baseB": "B (orig)",
              "Aphase_Bmag": "A-phase + B-mag", "Bphase_Amag": "B-phase + A-mag",
              "phaseonly_A": "phase-only(A)", "magonly_A": "mag-only(A)"}


def select_classes(args):
    if not args.classes:
        return CLASSES
    want = [c.strip() for c in args.classes.split(",")]
    by_name = dict(CLASSES)
    missing = [w for w in want if w not in by_name]
    assert not missing, f"unknown classes {missing}; have {list(by_name)}"
    return [(w, by_name[w]) for w in want]


def preflight(args):
    """Numeric asserts on the swap machinery; no model download."""
    print("[e13] pre-flight asserts ...", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    # a non-white stand-in latent pair (red-ish spectrum, like a real latent)
    fy = torch.fft.fftfreq(H, device=dev); fx = torch.fft.fftfreq(W, device=dev)
    rr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).clamp(min=1e-3)
    def red():
        g = torch.randn(1, C, H, W, device=dev)
        return torch.fft.ifft2(torch.fft.fft2(g) * rr[None, None] ** -1.0).real
    A, B = red(), red()

    # 1. full-spectrum swap c=1, mag_from=A reconstructs A exactly
    recon = band_phase_swap(A, B, c=1.0, mag_from="A")
    assert recon.imag.abs().max() < 1e-3, recon.imag.abs().max()
    assert (recon.real - A).abs().max() < 1e-3, "c=1 mag=A must reconstruct A"

    # 2. phase-only preserves total power (latent std) and is real
    po = phase_only(A)
    assert abs(float(po.std()) / float(A.std()) - 1) < 0.02, "phase_only power"

    # 3. magnitude-only keeps magnitude (std) and gives a ~uniform phase marginal
    mo = magnitude_only(A, generator=torch.Generator(dev).manual_seed(1))
    assert abs(float(mo.std()) / float(A.std()) - 1) < 0.02, "mag_only power"
    f = flatness(torch.fft.fft2(mo.float())[0, 0].angle())
    assert f < 0.1, f"mag_only phase not uniform: {f:.3f}"

    print(f"[e13] pre-flight OK (recon_err<1e-3, mag_only flatness={f:.4f})",
          flush=True)


def gen(args, classes, out):
    """Generate + cache the base latent bank (Flux transformer)."""
    pipe = load_flux(args.mem)
    os.makedirs(f"{out}/latents", exist_ok=True)
    os.makedirs(f"{out}/images", exist_ok=True)
    for name, prompt in classes:
        for s in range(args.seeds):
            tag = f"{name}_s{s}"
            lat_path = f"{out}/latents/{tag}.pt"
            if os.path.exists(lat_path):
                print(f"[e13] {tag} (cached)", flush=True)
                continue
            img, lat = flux_generate(pipe, prompt, s, args.cfg, args.steps)
            torch.save(lat, lat_path)
            img.save(f"{out}/images/{tag}_gen.png")
            print(f"[e13] {tag} generated", flush=True)


def decode_variants(vae, lat_a, lat_b, gen_seed):
    """Return {variant: PIL} for one (A, B) pair, all via flux_vae_decode."""
    g = torch.Generator(lat_a.device if lat_a.is_cuda else "cpu").manual_seed(gen_seed)
    lats = {
        "baseA": lat_a,
        "baseB": lat_b,
        "Aphase_Bmag": band_phase_swap(lat_a, lat_b, c=1.0, mag_from="B").real,
        "Bphase_Amag": band_phase_swap(lat_b, lat_a, c=1.0, mag_from="B").real,
        "phaseonly_A": phase_only(lat_a),
        "magonly_A": magnitude_only(lat_a, generator=g),
    }
    return {k: flux_vae_decode(vae, v) for k, v in lats.items()}


def analyze(args, classes, out):
    vae = load_flux_vae()
    model, proc = load_clip()
    os.makedirs(f"{out}/images", exist_ok=True)
    os.makedirs(f"{out}/plots", exist_ok=True)
    report = {}

    for name, _ in classes:
        lats = [torch.load(f"{out}/latents/{name}_s{s}.pt", weights_only=True).cuda()
                for s in range(args.seeds)]
        npair = args.seeds // 2
        # per-variant accumulators
        clip_to_A = {v: [] for v in VARIANTS}
        clip_to_B = {v: [] for v in VARIANTS}
        met = {v: {m: [] for m in METRICS} for v in VARIANTS}
        grid_rows = []

        for p in range(npair):
            A, B = lats[2 * p], lats[2 * p + 1]
            imgs = {}
            for v in VARIANTS:
                ipath = f"{out}/images/{name}_p{p}_{v}.png"
                if os.path.exists(ipath):
                    imgs[v] = Image.open(ipath)
            if len(imgs) < len(VARIANTS):
                dec = decode_variants(vae, A, B, gen_seed=1000 * p + 7)
                for v, im in dec.items():
                    im.save(f"{out}/images/{name}_p{p}_{v}.png")
                imgs = dec
            print(f"[e13] {name} pair {p} decoded", flush=True)

            order = [imgs[v] for v in VARIANTS]
            feats = clip_image_features(model, proc, order)
            fA, fB = feats[0], feats[1]
            for vi, v in enumerate(VARIANTS):
                clip_to_A[v].append(cosine(feats[vi], fA))
                clip_to_B[v].append(cosine(feats[vi], fB))
                m = image_metrics(imgs[v])
                for k in METRICS:
                    met[v][k].append(m[k])
            grid_rows.append(order)

        def mean(xs):
            return float(sum(xs) / len(xs)) if xs else 0.0

        report[f"class/{name}"] = {
            v: {
                "clip_to_A": mean(clip_to_A[v]),
                "clip_to_B": mean(clip_to_B[v]),
                **{m: mean(met[v][m]) for m in METRICS},
            } for v in VARIANTS
        }
        r = report[f"class/{name}"]
        # the headline contrast: does identity follow the phase donor?
        print(f"[e13] {name}: A-phase+B-mag clip(A)={r['Aphase_Bmag']['clip_to_A']:.3f} "
              f"clip(B)={r['Aphase_Bmag']['clip_to_B']:.3f} | "
              f"mag-only clip(A)={r['magonly_A']['clip_to_A']:.3f}", flush=True)

        save_grid(grid_rows, [f"pair {p}" for p in range(npair)],
                  [VAR_LABELS[v] for v in VARIANTS], f"{out}/grid_{name}.png")
        for l in lats:
            del l
        torch.cuda.empty_cache()

    make_summary_plot(report, classes, out)
    return report


def make_summary_plot(report, classes, out):
    """Bar chart: CLIP-to-phase-donor vs CLIP-to-mag-donor for the two swaps,
    per class -- the Oppenheim-Lim headline."""
    names = [n for n, _ in classes]
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 4))
    x = range(len(names))
    # A-phase+B-mag: phase donor = A, mag donor = B
    to_phase = [report[f"class/{n}"]["Aphase_Bmag"]["clip_to_A"] for n in names]
    to_mag = [report[f"class/{n}"]["Aphase_Bmag"]["clip_to_B"] for n in names]
    ax.bar([i - 0.2 for i in x], to_phase, 0.4, label="CLIP to phase donor (A)")
    ax.bar([i + 0.2 for i in x], to_mag, 0.4, label="CLIP to mag donor (B)")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set(ylabel="CLIP image cosine",
           title="A-phase + B-mag: identity follows phase or magnitude?")
    ax.legend()
    fig.savefig(f"{out}/plots/identity_phase_vs_mag.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print("[e13] summary plot saved", flush=True)


def main(args):
    out = os.path.join(RESULTS, "e13")
    os.makedirs(out, exist_ok=True)
    preflight(args)
    classes = select_classes(args)
    parts = args.part.split(",")

    report = {"params": vars(args)}
    if "gen" in parts:
        gen(args, classes, out)
    if "analyze" in parts:
        report.update(analyze(args, classes, out))

    path = f"{out}/report.json"
    if os.path.exists(path):
        with open(path) as f:
            merged = json.load(f)
        merged.update(report)
        report = merged
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e13] report -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4, help="latents per class (even; paired)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--mem", default="gpu_resident",
                    choices=["gpu_resident", "bnb4", "seq_offload"])
    ap.add_argument("--part", default="gen,analyze")
    ap.add_argument("--classes", default="", help="comma list; empty = all 6")
    main(ap.parse_args())
