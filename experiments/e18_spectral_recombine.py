"""E18: offline two-image spectral recombination -- the foundation of the
style-transfer / blending direction (does PHASE carry content and PER-BAND POWER
carry style well enough to recombine two REAL images in latent space?).

No diffusion: VAE-encode pairs of real images (A = content, B = style), recombine
their spectra, and VAE-decode. The premise (from E7-E14) is that A's phase fixes
layout while B's per-band power supplies "style" (texture-energy envelope +
palette). If it holds offline, the generation-time methods (E19-E22) stand on
solid ground; if not, we learn it cheaply here.

Variants decoded per (A, B) pair:
  baseA, baseB
  phaseA_magB   = band_phase_swap(A,B,c=1,mag_from=B)   -- phase A + B's full
                  magnitude (STRONG: also drags B structure through magnitude)
  styleA_s{p}   = restyle_latent(A, bandpower(B), strength=p)  -- phase A + A's
                  own within-band texture, per-band power -> B (the isotropic
                  "pure spectral style" op == AdaIN-in-Fourier; p in --strengths)
  hybrid_c{c}   = band_spectrum_split(A,B,c)            -- low bands (structure)
                  from A, high bands (detail) from B (Oliva 2006; c in --cuts)
  phaseonlyA, magonlyA  -- Oppenheim-Lim controls

Each variant scored with CLIP image cosine to A and B (content tracking) and a
log-radial-PSD distance to A and B (spectral/style tracking), plus image_metrics.

Parts (--part, comma list; default both):
  preflight -- numeric asserts on the recombination math (no model)
  analyze   -- encode + recombine + decode + score + grids + report
               (needs only a VAE ~160MB and CLIP ViT-L; --vae sd35|flux)

--vae flux runs immediately against the cached Flux VAE (smoke test); --vae sd35
matches the E19+ generation model (small gated download).
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_grid, RESULTS
from spectral_ops import (band_index_map, radial_psd, band_phase_swap,
                          phase_only, magnitude_only)
from style_ops import restyle_latent, latent_band_power, band_spectrum_split
from e9_bandnorm_classes import image_metrics
from clip_sim import load_clip, clip_image_features, cosine
from e17_sd35 import sd3_vae_encode, sd3_vae_decode  # VAE-convention-generic

SIZE = 1024
H = W = 128
C = 16
N_BINS = 24
OUT = os.path.join(RESULTS, "e18")


def load_vae(kind):
    if kind == "sd35":
        from e17_sd35 import load_sd35_vae
        return load_sd35_vae()
    from e7_flux_phase import load_flux_vae
    return load_flux_vae()


# ---------------------------------------------------------------------------
# Pre-flight: the recombination identities (no model download)
# ---------------------------------------------------------------------------

def preflight():
    print("[e18] pre-flight asserts ...", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    idx = band_index_map(H, W, N_BINS, dev)
    # red-ish stand-in latents (1/f spectrum, like real VAE latents)
    fy = torch.fft.fftfreq(H, device=dev); fx = torch.fft.fftfreq(W, device=dev)
    rr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).clamp(min=1e-3)
    red = lambda: torch.fft.ifft2(torch.fft.fft2(
        torch.randn(1, C, H, W, device=dev)) * rr[None, None] ** -1.0).real
    A, B = red(), red() * 1.6 + 0.2
    bandB = latent_band_power(B, idx, N_BINS)

    # 1. restyle(strength=1) re-levels A's bands to B's power, phase preserved
    R = restyle_latent(A, bandB, idx, N_BINS, strength=1.0)
    bandR = latent_band_power(R, idx, N_BINS)
    rel = float(((bandR - bandB).abs() / bandB.clamp(min=1e-8)).max())
    assert rel < 1e-3, f"restyle band power mismatch {rel}"
    fa, frr = torch.fft.fft2(A.float()), torch.fft.fft2(R.float())
    dphi = (fa.angle() - frr.angle()).abs()
    dphi = torch.minimum(dphi, 2 * torch.pi - dphi).max()
    assert float(dphi) < 1e-2, f"restyle moved phase by {float(dphi)}"

    # 2. hybrid endpoints: c=1 -> A, c=0 -> B; intermediate stays real
    assert (band_spectrum_split(A, B, 1.0) - A).abs().max() < 1e-3, "hybrid c=1 != A"
    assert (band_spectrum_split(A, B, 0.0) - B).abs().max() < 1e-3, "hybrid c=0 != B"
    mix = torch.fft.fft2(band_spectrum_split(A, B, 0.3).float())
    assert torch.fft.ifft2(mix).imag.abs().max() < 1e-3, "hybrid not real"

    # 3. phaseA_magB stays real
    pm = band_phase_swap(A, B, c=1.0, mag_from="B")
    assert pm.imag.abs().max() < 1e-3, "phaseA_magB not real"

    print(f"[e18] pre-flight OK (restyle rel_err={rel:.1e}, phase_drift={float(dphi):.1e})",
          flush=True)


# ---------------------------------------------------------------------------
# Variant construction
# ---------------------------------------------------------------------------

def make_variants(A, B, idx, strengths, cuts):
    """Dict {name: real latent (1,C,H,W)} for one (A=content, B=style) pair."""
    bandB = latent_band_power(B, idx, N_BINS)
    g = torch.Generator(A.device if A.is_cuda else "cpu").manual_seed(0)
    v = {"baseA": A, "baseB": B,
         "phaseA_magB": band_phase_swap(A, B, c=1.0, mag_from="B").real,
         "phaseonlyA": phase_only(A),
         "magonlyA": magnitude_only(A, generator=g)}
    for p in strengths:
        v[f"styleA_s{p:g}"] = restyle_latent(A, bandB, idx, N_BINS, strength=p)
    for c in cuts:
        v[f"hybrid_c{c:g}"] = band_spectrum_split(A, B, c)
    return v


def lum_psd(img):
    """Log-radial-PSD of an RGB image's luminance -> (n_bins,) the image-space
    'texture energy' descriptor used for style distance."""
    x = torch.from_numpy(__import__("numpy").asarray(img.convert("RGB"))).float() / 255
    gray = (x @ torch.tensor([0.299, 0.587, 0.114]))[None, None]   # (1,1,H,W)
    _, psd = radial_psd(gray.cuda() if torch.cuda.is_available() else gray)
    return torch.log(psd[0].clamp(min=1e-12))


def psd_dist(p, q):
    return float((p - q).pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def parse_pairs(spec, n):
    if spec:
        return [tuple(int(i) for i in p.split(":")) for p in spec.split(",")]
    # default: a few diverse, both-direction pairs over the loaded bank
    base = [(0, 1), (1, 0), (2, 3)]
    return [(a, b) for a, b in base if a < n and b < n]


def analyze(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(f"{OUT}/grids", exist_ok=True)
    idx = band_index_map(H, W, N_BINS, dev)
    vae = load_vae(args.vae)

    photos = sorted(p for p in os.listdir(args.photos)
                    if p.lower().endswith((".jpg", ".jpeg", ".png")))[:args.n]
    imgnames, dirs = list(photos), [args.photos] * len(photos)
    n_photo = len(photos)
    if args.styles:                         # cross-domain: paintings as the B set
        styles = sorted(p for p in os.listdir(args.styles)
                        if p.lower().endswith((".jpg", ".jpeg", ".png")))[:args.n_styles]
        imgnames += styles
        dirs += [args.styles] * len(styles)
    assert len(imgnames) >= 2, "need >=2 images"
    pils = [Image.open(os.path.join(d, p)).convert("RGB").resize((SIZE, SIZE))
            for d, p in zip(dirs, imgnames)]
    lats = [sd3_vae_encode(vae, im).to(dev) for im in pils]
    img_psd = [lum_psd(im) for im in pils]   # style descriptor of the originals
    print(f"[e18] encoded {len(lats)} images via {args.vae} VAE", flush=True)

    model, proc = load_clip()
    if args.pairs:
        pairs = parse_pairs(args.pairs, len(lats))
    elif args.styles:                       # content=photo x style=painting
        pairs = [(i, n_photo + j) for i in range(min(n_photo, 3))
                 for j in range(min(len(imgnames) - n_photo, 2))]
    else:
        pairs = parse_pairs("", len(lats))
    report, rows, row_labels = {}, [], []
    col_labels = None

    for (ai, bi) in pairs:
        A, B = lats[ai], lats[bi]
        variants = make_variants(A, B, idx, args.strengths, args.cuts)
        names = list(variants)
        if col_labels is None:
            col_labels = names
        decoded = {k: sd3_vae_decode(vae, lat) for k, lat in variants.items()}

        feats = clip_image_features(model, proc, [decoded[k] for k in names])
        fA, fB = feats[names.index("baseA")], feats[names.index("baseB")]
        tag = f"{ai}_{bi}"
        report[tag] = {}
        for k, f in zip(names, feats):
            vp = lum_psd(decoded[k])
            report[tag][k] = {
                "clip_to_A": cosine(f, fA), "clip_to_B": cosine(f, fB),
                "psd_to_A": psd_dist(vp, img_psd[ai]),
                "psd_to_B": psd_dist(vp, img_psd[bi]),
                **{m: image_metrics(decoded[k])[m]
                   for m in ("colorfulness", "saturation", "hf_frac")},
            }
        rows.append([decoded[k] for k in names])
        row_labels.append(f"A={imgnames[ai][:10]} B={imgnames[bi][:10]}")

    grid = save_grid(rows, row_labels, col_labels, f"{OUT}/grids/recombine_{args.vae}.png")
    with open(f"{OUT}/report_{args.vae}.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[e18] grid -> {grid}", flush=True)
    print(f"[e18] report -> {OUT}/report_{args.vae}.json", flush=True)
    _summary(report)


def _summary(report):
    """Print the key readout: does styleA keep A (content) while matching B (style)?"""
    print("\n[e18] per-variant means over pairs "
          "(clipA=content↑  psdB=style-dist↓):", flush=True)
    agg = {}
    for tag, d in report.items():
        for k, m in d.items():
            agg.setdefault(k, []).append(m)
    for k, ms in agg.items():
        mean = lambda f: sum(x[f] for x in ms) / len(ms)
        print(f"  {k:14s} clipA={mean('clip_to_A'):.3f} clipB={mean('clip_to_B'):.3f}"
              f"  psdA={mean('psd_to_A'):.3f} psdB={mean('psd_to_B'):.3f}"
              f"  colorful={mean('colorfulness'):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="preflight,analyze")
    ap.add_argument("--vae", default="sd35", choices=["sd35", "flux"])
    ap.add_argument("--photos", default=os.path.join(RESULTS, "e10", "real_photos"))
    ap.add_argument("--n", type=int, default=6, help="photos (content set) to load")
    ap.add_argument("--styles", default="", help="dir of style images (B set, e.g. paintings)")
    ap.add_argument("--n_styles", type=int, default=4, help="style images to load")
    ap.add_argument("--pairs", default="", help="content:style index pairs, e.g. 0:1,2:3")
    ap.add_argument("--strengths", default="0.5,1.0",
                    type=lambda s: [float(x) for x in s.split(",")])
    ap.add_argument("--cuts", default="0.1,0.25,0.5",
                    type=lambda s: [float(x) for x in s.split(",")])
    args = ap.parse_args()
    parts = args.part.split(",")
    if "preflight" in parts:
        preflight()
    if "analyze" in parts:
        analyze(args)


if __name__ == "__main__":
    main()
