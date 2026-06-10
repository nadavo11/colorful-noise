"""E12: latent FFT phase distributions across image classes (FLUX.1-dev).

The phase question, framed. E6/E7 already showed the FFT phase *marginal* of
output latents is ~uniform on [-pi, pi] (flatness ~0.003, phase _|_ magnitude)
-- which is why phase "looks noisy / non-informative". But a uniform marginal
is the white-noise null, not evidence that phase carries nothing: image
structure lives in cross-frequency phase *relationships* (Oppenheim-Lim), and
E7's band-split swap showed latent identity follows the low-band phase donor.

E12 produces the per-[-pi, pi] phase histogram that did not yet exist -- not
just one global curve (E7) but resolved per radial band, per channel, and per
image class -- plus per-band circular stats (flatness, mean resultant length)
and cross-seed coherence. Expected signature: near-flat marginals everywhere
except the DC / lowest band, uniform across classes; the joint structure
(coherence) is what separates them.

Single part (analysis only -- no intervention). Memory flags as in E7:
--mem bnb4 (default, NF4 transformer + model offload) or --mem seq_offload.
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_grid, RESULTS
from spectral_ops import (band_index_map, radial_bins, phase_coherence,
                          phase_histogram, flatness, random_hermitian_phase)
from e7_flux_phase import load_flux, flux_generate
from e9_bandnorm_classes import CLASSES

SIZE = 1024            # pixels; latents are (1, 16, 128, 128)
H = W = 128
C = 16


def select_classes(args):
    if not args.classes:
        return CLASSES
    want = [c.strip() for c in args.classes.split(",")]
    by_name = dict(CLASSES)
    missing = [w for w in want if w not in by_name]
    assert not missing, f"unknown classes {missing}; have {list(by_name)}"
    return [(w, by_name[w]) for w in want]


def band_centers(n_bins):
    rr = radial_bins(H, W, "cpu")
    edges = torch.linspace(0, rr.max() + 1e-6, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def preflight(args):
    """Numeric asserts on the histogram machinery; no model download."""
    print("[e12] pre-flight asserts ...", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. band map: shape, range, DC in band 0 (same recipe as radial_psd)
    idx_map = band_index_map(H, W, args.n_bins, dev)
    assert idx_map.shape == (H, W), idx_map.shape
    assert int(idx_map.max()) == args.n_bins - 1
    assert int(idx_map[0, 0]) == 0, "DC must land in band 0"

    # 2. white Hermitian phase -> uniform marginal (flatness ~0)
    phi = random_hermitian_phase(
        C, H, W, device=dev,
        generator=torch.Generator(dev).manual_seed(0))[0]
    f = flatness(phi)
    assert f < 0.05, f"white phase not flat: flatness={f:.4f}"

    # 3. phase_histogram: shapes + mid/high bands sit at the uniform null
    out = phase_histogram(phi, idx_map, args.n_bins)
    assert out["hist"].shape == (C, args.n_bins, 64), out["hist"].shape
    assert out["edges"].shape == (65,)
    # each non-empty histogram row is a normalized distribution
    rs = out["hist"].sum(-1)
    assert ((rs - 1).abs() < 1e-4)[rs > 0].all(), "rows must sum to 1"
    midhigh = out["R"][:, 2:].mean()
    assert midhigh < 0.1, f"white phase not uniform: R_midhigh={midhigh:.3f}"
    print(f"[e12] pre-flight OK (white flatness={f:.4f}, "
          f"R_midhigh={float(midhigh):.4f})", flush=True)


def analyze_class(args, name, lats, idx_map, centers, report, out):
    """Per-class phase stats + a per-class band x phase heatmap.

    lats: (S, C, H, W) cuda. Returns plot-ready arrays for the cross-class
    figures (cpu tensors)."""
    phi = torch.fft.fft2(lats.float()).angle()        # (S, C, H, W)
    S = phi.shape[0]

    hs, flats, Rs = [], [], []
    for s in range(S):
        d = phase_histogram(phi[s], idx_map, args.n_bins)
        hs.append(d["hist"]); flats.append(d["flat"]); Rs.append(d["R"])
    hist = torch.stack(hs).mean(0)        # (C, n_bins, 64) seed-averaged
    flat = torch.stack(flats).mean(0)     # (C, n_bins)
    R = torch.stack(Rs).mean(0)           # (C, n_bins)

    # global marginal histogram (all seeds, channels, freqs)
    gh = torch.histc(phi.flatten().cpu(), bins=64, min=-math.pi, max=math.pi)
    gh = gh / gh.sum()

    # cross-seed phase coherence (joint structure, the contrast to marginals)
    _, coh, r_null = phase_coherence(phi)     # coh: (C, n_bins)

    flat_band = flat.mean(0)                   # (n_bins,) channel mean
    R_band = R.mean(0)
    coh_band = coh.mean(0)
    report[f"class/{name}"] = {
        "global_flatness": flatness(phi),
        "flat_per_band_chmean": flat_band.tolist(),
        "R_per_band_chmean": R_band.tolist(),
        "R_lowest_band": float(R_band[0]),
        "R_midhigh_mean": float(R_band[2:].mean()),
        "coherence_per_band_chmean": coh_band.tolist(),
        "coherence_lowest_band": float(coh_band[0]),
        "coherence_null": r_null,
        "flat_per_channel": flat.mean(1).tolist(),
    }
    print(f"[e12] {name}: global_flat={report[f'class/{name}']['global_flatness']:.4f} "
          f"R_low={float(R_band[0]):.3f} R_mid={float(R_band[2:].mean()):.3f} "
          f"coh_low={float(coh_band[0]):.3f} (null={r_null:.3f})", flush=True)

    # per-class band x phase heatmap (channel-averaged histogram)
    hc = hist.mean(0)                          # (n_bins, 64)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(hc, aspect="auto", origin="lower", cmap="magma",
                   extent=[-math.pi, math.pi, 0, float(centers[-1])])
    ax.set(xlabel="phase", ylabel="radial freq (band center)",
           title=f"phase distribution per band -- {name}")
    fig.colorbar(im, ax=ax, label="P(phase | band)")
    fig.savefig(f"{out}/plots/band_hist_{name}.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    return {"global_hist": gh, "flat_band": flat_band, "R_band": R_band,
            "coh_band": coh_band, "r_null": r_null}


def make_cross_class_plots(args, summ, centers, out):
    edges = torch.linspace(-math.pi, math.pi, 65)
    hcent = 0.5 * (edges[:-1] + edges[1:])

    # 1. global marginal histograms, all classes (the user's core ask)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, s in summ.items():
        ax.plot(hcent, s["global_hist"], label=name)
    ax.axhline(1 / 64, ls="--", c="gray", label="uniform")
    ax.set(xlabel="phase", ylabel="P(phase)",
           title="output-latent FFT phase marginal (per class)")
    ax.legend(fontsize=8)
    fig.savefig(f"{out}/plots/global_hist.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2. per-band flatness + resultant length vs radial freq
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for name, s in summ.items():
        a1.plot(centers, s["flat_band"], label=name)
        a2.plot(centers, s["R_band"], label=name)
    a1.set(xlabel="radial freq", ylabel="histogram flatness (std/mean)",
           title="per-band phase flatness (0 = uniform)")
    a2.set(xlabel="radial freq", ylabel="resultant length R",
           title="per-band phase concentration (0 = uniform)")
    a1.legend(fontsize=8); a2.legend(fontsize=8)
    fig.savefig(f"{out}/plots/flatness_band.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 3. cross-seed coherence vs radial freq (joint structure)
    fig, ax = plt.subplots(figsize=(7, 4))
    nulls = []
    for name, s in summ.items():
        ax.plot(centers, s["coh_band"], label=name)
        nulls.append(s["r_null"])
    ax.axhline(sum(nulls) / len(nulls), ls="--", c="gray",
               label=f"null (N={args.seeds})")
    ax.set(xlabel="radial freq", ylabel="cross-seed coherence R",
           title="cross-seed phase coherence (channel mean)")
    ax.legend(fontsize=8)
    fig.savefig(f"{out}/plots/coherence_radial.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print("[e12] cross-class plots saved", flush=True)


def run(args, report, out):
    pipe = load_flux(args.mem)
    os.makedirs(f"{out}/latents", exist_ok=True)
    os.makedirs(f"{out}/images", exist_ok=True)
    os.makedirs(f"{out}/plots", exist_ok=True)
    idx_map = band_index_map(H, W, args.n_bins, "cuda")
    centers = band_centers(args.n_bins)
    classes = select_classes(args)

    summ = {}
    for name, prompt in classes:
        lats = []
        for s in range(args.seeds):
            tag = f"{name}_s{s}"
            lat_path = f"{out}/latents/{tag}.pt"
            img_path = f"{out}/images/{tag}.png"
            if os.path.exists(lat_path) and os.path.exists(img_path):
                lat = torch.load(lat_path, weights_only=True)
                print(f"[e12] {tag} (cached)", flush=True)
            else:
                img, lat = flux_generate(pipe, prompt, s, args.cfg, args.steps)
                torch.save(lat, lat_path)
                img.save(img_path)
                print(f"[e12] {tag} done", flush=True)
            lats.append(lat)
        lats = torch.cat(lats).cuda()
        summ[name] = analyze_class(args, name, lats, idx_map, centers,
                                   report, out)
        del lats
        torch.cuda.empty_cache()

    make_cross_class_plots(args, summ, centers, out)

    # contact sheet of the first few seeds per class
    ncol = min(args.seeds, 4)
    rows = [[Image.open(f"{out}/images/{name}_s{s}.png") for s in range(ncol)]
            for name, _ in classes]
    save_grid(rows, [n for n, _ in classes],
              [f"seed {s}" for s in range(ncol)], f"{out}/grid_classes.png")
    print("[e12] grid_classes saved", flush=True)


def main(args):
    out = os.path.join(RESULTS, "e12")
    os.makedirs(out, exist_ok=True)
    preflight(args)

    report = {"params": vars(args)}
    run(args, report, out)

    path = f"{out}/report.json"
    if os.path.exists(path):
        with open(path) as f:
            merged = json.load(f)
        merged.update(report)
        report = merged
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e12] report -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--mem", default="bnb4", choices=["bnb4", "seq_offload"])
    ap.add_argument("--classes", default="",
                    help="comma list of class names; empty = all 6")
    main(ap.parse_args())
