"""E8: causal test of the E7 power finding -- per-step PSD clamping.

E7 (correlational): FLUX.1-dev cfg=1 vs cfg=3.5 output latents differ mainly
in POWER (std 0.83 vs 1.17, ~3x low-freq power, slope -1.5 vs -2.0); phase
statistics are identical. E8 intervenes: generate at cfg=3.5, but at EVERY
denoising step renormalize the latent's PSD to the cfg=1.0 reference at the
same step. If guidance's visible effect (contrast / saturation) is mediated
by the pumped power, the clamped run should look like cfg=1 despite cfg=3.5
conditioning.

Conditions (grid rows; cols = seeds, same seeds across rows -- the initial
latents are identical since guidance only changes the embedding):
  cfg=1.0      -- baseline AND the per-step PSD reference (part a)
  cfg=3.5      -- baseline
  band-norm    -- per (channel, radial band): |F| *= sqrt(ref/cur), phase kept
  global-norm  -- one scalar per step matching total power (Parseval: plain
                  latent rescale, no FFT); keeps cfg=3.5's spectral SHAPE --
                  dissociates total power from spectral shape.

The clamp runs at step END every step incl. the last (diffusers reads the
returned latents back), so the latent handed to the VAE is normalized.
Parts (--part): a (reference recording + row 1), b (rows 2-4, needs a's
ref_psd.pt). Memory as e7: --mem bnb4 (NF4 transformer + cpu offload),
bf16 latents, FFT math in fp32.
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
from spectral_ops import (radial_psd, spectral_slope, band_index_map,
                          band_power, psd_match)
from e7_flux_phase import load_flux, SIZE, LAT_SHAPE

N_CH, H, W = 16, 128, 128


# ---------------------------------------------------------------------------
# Step-end callbacks
# ---------------------------------------------------------------------------

class RecordPSD:
    """Record per-step per-(channel, band) mean power, total power and std
    of the unpacked latent; no modification (returns {})."""

    def __init__(self, idx_map, n_bins, steps):
        self.idx_map, self.n_bins = idx_map, n_bins
        self.band = [None] * steps
        self.total = [0.0] * steps
        self.std = [0.0] * steps
        self.last = None

    def __call__(self, p, i, t, kw):
        packed = kw["latents"]
        lat = type(p)._unpack_latents(packed, SIZE, SIZE,
                                      p.vae_scale_factor).float()
        F2 = torch.fft.fft2(lat[0]).abs() ** 2
        self.band[i] = band_power(F2, self.idx_map, self.n_bins).cpu()
        self.total[i] = float((lat ** 2).sum())
        self.std[i] = float(lat.std())
        self.last = packed
        return {}


class ClampPSD:
    """Clamp the latent's PSD to the cfg=1 reference at the same step and
    return the modified packed latents (the pipeline reads them back).

    mode='band':   per (channel, radial band) magnitude matching (psd_match)
    mode='global': one scalar matching total power -- exact in latent space
                   by Parseval, no FFT needed."""

    def __init__(self, mode, ref, idx_map, n_bins):
        self.mode, self.idx_map, self.n_bins = mode, idx_map, n_bins
        self.ref_band = ref["band"].cuda()        # (steps, C, n_bins)
        self.ref_total = ref["total"]             # (steps,)
        self.std = []
        self.gmin, self.gmax, self.resid = math.inf, -math.inf, 0.0
        self.last = None

    def __call__(self, p, i, t, kw):
        packed = kw["latents"]
        lat = type(p)._unpack_latents(packed, SIZE, SIZE, p.vae_scale_factor)
        if self.mode == "band":
            lat, st = psd_match(lat, self.ref_band[i], self.idx_map,
                                self.n_bins, return_stats=True)
            self.gmin = min(self.gmin, st["gain_min"])
            self.gmax = max(self.gmax, st["gain_max"])
            self.resid = max(self.resid, st["imag_residue"])
        else:
            x = lat.float()
            scale = math.sqrt(float(self.ref_total[i]) /
                              max(float((x ** 2).sum()), 1e-12))
            lat = (x * scale).to(lat.dtype)
        self.std.append(float(lat.float().std()))
        new_packed = type(p)._pack_latents(lat, 1, N_CH, H, W)
        self.last = new_packed
        return {"latents": new_packed.to(packed.dtype)}


def gen_with_cb(pipe, prompt, seed, guidance, steps, cb):
    """One generation -> (pil image, final unpacked fp32 cpu latent).
    For ClampPSD callbacks cb.last is the POST-clamp packed latent, i.e.
    exactly what the VAE decoded."""
    from diffusers import FluxPipeline
    img = pipe(prompt=prompt, height=SIZE, width=SIZE,
               guidance_scale=guidance, num_inference_steps=steps,
               generator=torch.Generator("cuda").manual_seed(seed),
               callback_on_step_end=cb).images[0]
    lat = FluxPipeline._unpack_latents(cb.last, SIZE, SIZE,
                                       pipe.vae_scale_factor)
    return img, lat.float().cpu()


def lat_stats(lat):
    centers, psd = radial_psd(lat.cuda())
    slopes = spectral_slope(centers, psd)
    return {"std": float(lat.std()),
            "spectral_slope_mean": sum(slopes) / len(slopes),
            "channel_means": lat.mean(dim=(0, 2, 3)).tolist()}


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def preflight(args):
    """Numeric asserts; run before any download or generation."""
    print("[e8] pre-flight asserts ...", flush=True)
    from diffusers import FluxPipeline
    dev = "cuda"
    nb = args.n_bins

    # 1. pack/unpack round-trip at 1024px / (16,128,128)
    x = torch.randn(*LAT_SHAPE, device=dev)
    packed = FluxPipeline._pack_latents(x, 1, N_CH, H, W)
    assert packed.shape == (1, 64 * 64, 64), packed.shape
    assert torch.equal(FluxPipeline._unpack_latents(packed, SIZE, SIZE, 8), x)

    # 2. band_index_map binning == radial_psd binning; DC in band 0
    idx_map = band_index_map(H, W, nb, dev)
    assert idx_map.shape == (H, W) and int(idx_map[0, 0]) == 0
    assert int(idx_map.min()) == 0 and int(idx_map.max()) == nb - 1
    xs = torch.randn(2, N_CH, H, W, device=dev)
    _, psd_a = radial_psd(xs, nb)
    F2_mean = (torch.fft.fft2(xs.float()).abs() ** 2).mean(0) / (H * W)
    psd_b = band_power(F2_mean, idx_map, nb).cpu()
    rel = ((psd_a - psd_b).abs() / psd_a.clamp(min=1e-12)).max()
    assert rel < 1e-4, f"band_power != radial_psd binning: rel={rel:.2e}"

    # 3. psd_match: real output, exact band match, identity when ref==self
    a = torch.randn(1, N_CH, H, W, device=dev)
    b = torch.randn(1, N_CH, H, W, device=dev)
    ref = band_power(torch.fft.fft2(b.float())[0].abs() ** 2, idx_map, nb)
    out, st = psd_match(a, ref, idx_map, nb, return_stats=True)
    assert st["imag_residue"] < 1e-4, st
    cur2 = band_power(torch.fft.fft2(out.float())[0].abs() ** 2, idx_map, nb)
    rel = ((cur2 - ref).abs() / ref.clamp(min=1e-12)).max()
    assert rel < 1e-3, f"re-measured band power != target: rel={rel:.2e}"
    own = band_power(torch.fft.fft2(a.float())[0].abs() ** 2, idx_map, nb)
    assert (psd_match(a, own, idx_map, nb) - a).abs().max() < 1e-3, \
        "psd_match must be identity when ref == self"

    # 4. Parseval: global-norm latent-space scalar == Fourier total power
    p_a = float((torch.fft.fft2(a.float()).abs() ** 2).sum())
    s = 1.7
    p_s = float((torch.fft.fft2((s * a).float()).abs() ** 2).sum())
    assert abs(p_s - s * s * p_a) / p_s < 1e-5, "Parseval scaling violated"
    p_ref = float((b ** 2).sum())
    a2 = a * math.sqrt(p_ref / float((a ** 2).sum()))
    assert abs(float((a2 ** 2).sum()) - p_ref) / p_ref < 1e-5, \
        "global-norm scalar does not match total power"

    # 5. spectral_slope of white noise ~ 0
    centers, psd = radial_psd(xs)
    slopes = spectral_slope(centers, psd)
    assert max(abs(v) for v in slopes) < 0.2, f"white slope != 0: {slopes}"
    print("[e8] pre-flight OK", flush=True)


# ---------------------------------------------------------------------------
# Part A -- cfg=1 baseline + per-step PSD reference
# ---------------------------------------------------------------------------

def run_a(args, report, out):
    pipe = load_flux(args.mem)
    idx_map = band_index_map(H, W, args.n_bins, "cuda")
    os.makedirs(f"{out}/images", exist_ok=True)
    acc_band = torch.zeros(args.steps, N_CH, args.n_bins)
    acc_total = torch.zeros(args.steps)
    acc_std = torch.zeros(args.steps)
    per_seed = []
    for s in range(args.seeds):
        tag = f"cfg{args.ref_cfg}_s{s}"
        rec = RecordPSD(idx_map, args.n_bins, args.steps)
        img, lat = gen_with_cb(pipe, args.prompt, s, args.ref_cfg,
                               args.steps, rec)
        img.save(f"{out}/images/{tag}.png")
        acc_band += torch.stack(rec.band)
        acc_total += torch.tensor(rec.total)
        acc_std += torch.tensor(rec.std)
        per_seed.append(lat_stats(lat) | {"perstep_std": rec.std})
        print(f"[e8] a {tag} final_std={per_seed[-1]['std']:.3f} done",
              flush=True)
    ref = {"band": acc_band / args.seeds,
           "total": acc_total / args.seeds,
           "std": acc_std / args.seeds}
    torch.save(ref, f"{out}/ref_psd.pt")
    report["a/reference"] = {"ref_psd_path": f"{out}/ref_psd.pt",
                             "ref_std_curve": ref["std"].tolist()}
    report[f"a/cfg{args.ref_cfg}"] = {
        "per_seed": per_seed,
        "mean_final_std": sum(p["std"] for p in per_seed) / len(per_seed),
        "mean_slope": sum(p["spectral_slope_mean"] for p in per_seed)
                      / len(per_seed),
    }
    print(f"[e8] a reference saved; ref_std[0]={ref['std'][0]:.3f} "
          f"ref_std[-1]={ref['std'][-1]:.3f}", flush=True)


# ---------------------------------------------------------------------------
# Part B -- cfg=3.5 plain / band-norm / global-norm + grid + plots
# ---------------------------------------------------------------------------

def run_b(args, report, out):
    pipe = load_flux(args.mem)
    idx_map = band_index_map(H, W, args.n_bins, "cuda")
    ref = torch.load(f"{out}/ref_psd.pt", weights_only=True)

    conditions = [(f"cfg{args.cfg}", None),
                  ("bandnorm", "band"),
                  ("globalnorm", "global")]
    rows = [[Image.open(f"{out}/images/cfg{args.ref_cfg}_s{s}.png")
             for s in range(args.seeds)]]
    row_labels = [f"cfg={args.ref_cfg}"]
    curves = {f"cfg={args.ref_cfg} (ref)": ref["std"].tolist()}

    for cond, mode in conditions:
        row, per_seed = [], []
        for s in range(args.seeds):
            cb = (RecordPSD(idx_map, args.n_bins, args.steps) if mode is None
                  else ClampPSD(mode, ref, idx_map, args.n_bins))
            img, lat = gen_with_cb(pipe, args.prompt, s, args.cfg,
                                   args.steps, cb)
            img.save(f"{out}/images/{cond}_s{s}.png")
            per_seed.append(lat_stats(lat) | {"perstep_std": list(cb.std)})
            row.append(img)
            print(f"[e8] b {cond} s{s} final_std={per_seed[-1]['std']:.3f} "
                  f"done", flush=True)
        entry = {
            "per_seed": per_seed,
            "mean_final_std": sum(p["std"] for p in per_seed) / len(per_seed),
            "mean_slope": sum(p["spectral_slope_mean"] for p in per_seed)
                          / len(per_seed),
        }
        if mode == "band":
            entry["gain_min"] = cb.gmin
            entry["gain_max"] = cb.gmax
            entry["imag_residue_max"] = cb.resid
        report[f"b/{cond}"] = entry
        rows.append(row)
        row_labels.append({None: f"cfg={args.cfg}", "band": "band-norm",
                           "global": "global-norm"}[mode])
        n = len(per_seed[0]["perstep_std"])
        curves[row_labels[-1]] = [
            sum(p["perstep_std"][i] for p in per_seed) / len(per_seed)
            for i in range(n)]

    save_grid(rows, row_labels, [f"seed {s}" for s in range(args.seeds)],
              f"{out}/grid_e8.png")
    print("[e8] grid_e8 saved", flush=True)

    os.makedirs(f"{out}/plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, ys in curves.items():
        ax.plot(range(len(ys)), ys, ls="--" if "(ref)" in label else "-",
                label=label)
    ax.set(xlabel="step", ylabel="latent std",
           title="per-step latent std (post-clamp where applicable)")
    ax.legend()
    fig.savefig(f"{out}/plots/perstep_std.png", dpi=120, bbox_inches="tight")
    plt.close("all")
    print("[e8] plots saved", flush=True)


def main(args):
    out = os.path.join(RESULTS, "e8")
    os.makedirs(out, exist_ok=True)
    preflight(args)

    report = {"params": vars(args)}
    runners = {"a": run_a, "b": run_b}
    for part in filter(None, (p.strip() for p in args.part.split(","))):
        runners[part](args, report, out)

    # merge with an existing report so partial runs (--part) accumulate
    path = f"{out}/report.json"
    if os.path.exists(path):
        with open(path) as f:
            merged = json.load(f)
        merged.update(report)
        report = merged
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e8] report -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="a,b")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--ref_cfg", type=float, default=1.0)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--mem", default="bnb4", choices=["bnb4", "seq_offload"])
    ap.add_argument("--prompt", default="A photo of cat in the park")
    main(ap.parse_args())
