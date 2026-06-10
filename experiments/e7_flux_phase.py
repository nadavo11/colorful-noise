"""E7: FLUX.1-dev at cfg=1 -- output-latent phase statistics + phase hybrids.

E0-E6 probed the *input* noise of SDXL (verdict: the model reads power, not
fine phase). E7 flips direction on Flux-dev: analyze the FFT phase of the
model's *output* latents (16ch, 128x128, pre-VAE).

Flux-dev is guidance-distilled: guidance_scale is an embedding input, not
true CFG. cfg=1 turns the distilled guidance off (known soft/desaturated
regime) -- what does that look like in phase/spectrum statistics?

Parts (--part, comma list):
  a -- generate `seeds` cats per guidance value in --cfg_sweep, saving image
       AND final latent per call (latent captured via the step-end callback,
       image decoded by the stock offload-aware path). Stats per cfg group:
       phase histogram (+flatness), phase-mag Pearson correlations,
       cross-seed phase coherence vs the N-uniform null, radial PSD +
       log-log spectral slope. Plots + report.json + image grid.
  b -- band-split phase interpolation between two generated latents A and B
       (--a_src/--b_src): phase from A in the lowest-c radial band, from B
       outside, magnitudes fixed from one source (rows mag-A / mag-B),
       cutoff c swept 0->1. Each hybrid decoded through the Flux VAE only
       (no transformer; loads the standalone VAE, ~160MB).

Memory (24GB A5000; bf16 transformer alone ~23.8GB): --mem bnb4 (default)
loads an NF4 4-bit transformer (~6.8GB) + enable_model_cpu_offload();
--mem seq_offload is the slow-but-fits-anything fallback.
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
from spectral_ops import (radial_psd, phase_coherence, band_phase_swap,
                          spectral_slope)

REPO = "black-forest-labs/FLUX.1-dev"
SIZE = 1024            # pixels; latents are (1, 16, 128, 128)
LAT_SHAPE = (1, 16, 128, 128)


# ---------------------------------------------------------------------------
# Flux helpers (common.py stays SDXL-only)
# ---------------------------------------------------------------------------

def load_flux(mem="bnb4"):
    from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig
    if mem == "bnb4":
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)
        tr = FluxTransformer2DModel.from_pretrained(
            REPO, subfolder="transformer", quantization_config=qc,
            torch_dtype=torch.bfloat16)
        pipe = FluxPipeline.from_pretrained(REPO, transformer=tr,
                                            torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
    elif mem == "gpu_resident":
        # NF4 transformer + bf16 T5, all GPU-resident (no cpu offload). Same
        # numerics as bnb4 -- only the weights' residence differs -- but keeps
        # CPU RAM low (~baseline) so it survives a RAM-contended box where
        # enable_model_cpu_offload()'s ~17GB CPU footprint gets OOM-killed.
        # Peak GPU ~17GB (NF4 transformer 6.8 + T5-XXL bf16 9.5 + VAE/CLIP),
        # fits the 24GB A5000. (cf. E12, which kept SD3.5 GPU-resident for the
        # same reason.)
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)
        tr = FluxTransformer2DModel.from_pretrained(
            REPO, subfolder="transformer", quantization_config=qc,
            torch_dtype=torch.bfloat16)
        pipe = FluxPipeline.from_pretrained(REPO, transformer=tr,
                                            torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.vae.enable_tiling()
    elif mem == "seq_offload":
        pipe = FluxPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(mem)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def flux_generate(pipe, prompt, seed, guidance, steps):
    """One generation -> (pil_image, unpacked (1,16,128,128) fp32 cpu latent).

    output_type='pil' so the stock (offload-aware) path decodes the image;
    the final packed latents are captured by the step-end callback.
    """
    from diffusers import FluxPipeline
    captured = {}

    def grab(p, i, t, kw):
        captured["latents"] = kw["latents"]
        return {}

    img = pipe(prompt=prompt, height=SIZE, width=SIZE,
               guidance_scale=guidance, num_inference_steps=steps,
               generator=torch.Generator("cuda").manual_seed(seed),
               callback_on_step_end=grab).images[0]
    lat = FluxPipeline._unpack_latents(captured["latents"], SIZE, SIZE,
                                       pipe.vae_scale_factor)
    return img, lat.float().cpu()


def load_flux_vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(
        REPO, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")


def flux_vae_decode(vae, lat):
    """Manual Flux decode (divide by scaling_factor THEN add shift_factor)."""
    z = lat.to("cuda").float() / vae.config.scaling_factor + vae.config.shift_factor
    with torch.no_grad():
        img = vae.decode(z.to(vae.dtype), return_dict=False)[0]
    arr = ((img[0].float() / 2 + 0.5).clamp(0, 1) * 255).round().byte()
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())


# ---------------------------------------------------------------------------
# Numerics (pearson/flatness copied from e6, as e6 itself does)
# ---------------------------------------------------------------------------

def pearson(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def flatness(phi, bins=32):
    """std/mean of the phase histogram (0 = perfectly uniform)."""
    h = torch.histc(phi.flatten().float().cpu(), bins=bins,
                    min=-math.pi, max=math.pi)
    return float(h.std() / h.mean())


def preflight(args):
    """Numeric asserts; run before any download or generation."""
    print("[e7] pre-flight asserts ...", flush=True)
    from diffusers import FluxPipeline
    dev = "cuda"

    # 1. pack/unpack round-trip at 1024px / (16,128,128) latents
    x = torch.randn(*LAT_SHAPE, device=dev)
    packed = FluxPipeline._pack_latents(x, 1, 16, 128, 128)
    assert packed.shape == (1, 64 * 64, 64), packed.shape
    assert torch.equal(
        FluxPipeline._unpack_latents(packed, SIZE, SIZE, 8), x), \
        "pack/unpack round-trip failed"

    # 2. band_phase_swap: identity on A==B, Hermitian residue, endpoints
    a, b = torch.randn(*LAT_SHAPE, device=dev), torch.randn(*LAT_SHAPE, device=dev)
    for c in (0.0, 0.1, 0.5, 1.0):
        z = band_phase_swap(a, a, c)
        assert (z.real - a).abs().max() < 1e-3, f"c={c}: A,A != A"
        z = band_phase_swap(a, b, c, mag_from="A")
        assert z.imag.abs().max() < 1e-4, f"c={c}: ifft not real"
    assert (band_phase_swap(a, b, 1.0, mag_from="A").real - a).abs().max() < 1e-3, \
        "c=1.0 must be pure A"
    z0 = band_phase_swap(a, b, 0.0, mag_from="B").real  # all-B spectrum == b
    assert (z0 - b).abs().max() < 1e-3, "c=0 mag=B must reproduce B"

    # 3. phase_coherence: identical fields -> 1; independent -> null
    phi1 = torch.fft.fft2(a.float()).angle().expand(10, -1, -1, -1)
    _, prof, _ = phase_coherence(phi1)
    assert (prof > 0.99).all(), "identical fields must give R=1"
    xs = torch.randn(10, 16, 128, 128, device=dev)
    _, prof, r_null = phase_coherence(torch.fft.fft2(xs).angle())
    chmean = prof.mean(0)[2:]  # skip the tiny innermost bins
    assert ((chmean - r_null).abs() / r_null < 0.3).all(), \
        f"independent fields must sit at the null {r_null:.3f}: {chmean.tolist()}"

    # 4. spectral_slope of white noise ~ 0
    centers, psd = radial_psd(xs)
    slopes = spectral_slope(centers, psd)
    assert max(abs(s) for s in slopes) < 0.15, f"white slope != 0: {slopes}"
    print("[e7] pre-flight OK", flush=True)


# ---------------------------------------------------------------------------
# Part A -- cfg study
# ---------------------------------------------------------------------------

def run_a(args, report, out):
    pipe = load_flux(args.mem)
    os.makedirs(f"{out}/latents", exist_ok=True)
    os.makedirs(f"{out}/images", exist_ok=True)
    rows, row_labels, groups = [], [], {}
    for g in args.cfg_sweep:
        row, lats = [], []
        for s in range(args.seeds):
            tag = f"cfg{g}_s{s}"
            lat_path = f"{out}/latents/{tag}.pt"
            img_path = f"{out}/images/{tag}.png"
            if os.path.exists(lat_path) and os.path.exists(img_path):
                lat = torch.load(lat_path, weights_only=True)
                img = Image.open(img_path)
                print(f"[e7] a {tag} (cached)", flush=True)
            else:
                img, lat = flux_generate(pipe, args.prompt, s, g, args.steps)
                torch.save(lat, lat_path)
                img.save(img_path)
                print(f"[e7] a {tag} done", flush=True)
            row.append(img)
            lats.append(lat)
        rows.append(row)
        row_labels.append(f"cfg={g}")
        groups[g] = torch.cat(lats).cuda()
    save_grid(rows, row_labels, [f"seed {s}" for s in range(args.seeds)],
              f"{out}/grid_partA.png")
    print("[e7] grid_partA saved", flush=True)
    analyze_a(args, groups, report, out)


def analyze_a(args, groups, report, out):
    os.makedirs(f"{out}/plots", exist_ok=True)
    fig_h, ax_h = plt.subplots(figsize=(7, 4))
    fig_c, ax_c = plt.subplots(figsize=(7, 4))
    fig_p, ax_p = plt.subplots(figsize=(7, 4))
    r_null = None
    for g, lats in groups.items():
        F = torch.fft.fft2(lats.float())
        mag, phi = F.abs(), F.angle()
        centers, R, r_null = phase_coherence(phi)
        pcent, psd = radial_psd(lats)
        slopes = spectral_slope(pcent, psd)
        report[f"a/cfg{g}"] = {
            "flatness": flatness(phi),
            "corr_phi_mag": pearson(phi, mag),
            "corr_cosphi_mag": pearson(torch.cos(phi), mag),
            "corr_sinphi_mag": pearson(torch.sin(phi), mag),
            "spectral_slope_per_ch": slopes,
            "spectral_slope_mean": sum(slopes) / len(slopes),
            "coherence_chmean_profile": R.mean(0).tolist(),
            "coherence_max": float(R.max()),
            "latent_std": float(lats.std()),
            "channel_means": lats.mean(dim=(0, 2, 3)).tolist(),
        }
        print(f"[e7] a cfg={g}: slope={report[f'a/cfg{g}']['spectral_slope_mean']:.3f} "
              f"flatness={report[f'a/cfg{g}']['flatness']:.4f} "
              f"coh_max={report[f'a/cfg{g}']['coherence_max']:.3f}", flush=True)
        h = torch.histc(phi.flatten().cpu(), bins=64, min=-math.pi, max=math.pi)
        ax_h.plot(torch.linspace(-math.pi, math.pi, 64), h / h.sum(),
                  label=f"cfg={g}")
        ax_c.plot(centers, R.mean(0), label=f"cfg={g}")
        ax_p.loglog(pcent[1:], psd.mean(0)[1:], label=f"cfg={g}")
    report["a/coherence_null"] = r_null
    ax_h.set(xlabel="phase", ylabel="freq", title="output-latent FFT phase histogram")
    ax_h.legend()
    fig_h.savefig(f"{out}/plots/phase_hist.png", dpi=120, bbox_inches="tight")
    ax_c.axhline(r_null, ls="--", c="gray", label=f"null (N={args.seeds})")
    ax_c.set(xlabel="radial freq", ylabel="R",
             title="cross-seed phase coherence (channel mean)")
    ax_c.legend()
    fig_c.savefig(f"{out}/plots/coherence_radial.png", dpi=120, bbox_inches="tight")
    ax_p.set(xlabel="radial freq", ylabel="power",
             title="output-latent radial PSD (channel mean)")
    ax_p.legend()
    fig_p.savefig(f"{out}/plots/psd_loglog.png", dpi=120, bbox_inches="tight")
    plt.close("all")
    print("[e7] plots saved", flush=True)


# ---------------------------------------------------------------------------
# Part B -- band-split phase interpolation (VAE only)
# ---------------------------------------------------------------------------

def run_b(args, report, out):
    pb = f"{out}/partB"
    os.makedirs(pb, exist_ok=True)
    A = torch.load(f"{out}/latents/{args.a_src}.pt", weights_only=True).cuda()
    B = torch.load(f"{out}/latents/{args.b_src}.pt", weights_only=True).cuda()
    vae = load_flux_vae()
    report["b/sources"] = {"A": args.a_src, "B": args.b_src}
    rows, row_labels = [], []
    for mag_from in ("A", "B"):
        ref = flux_vae_decode(vae, A if mag_from == "A" else B)
        row = [ref]
        for c in args.cutoffs:
            z = band_phase_swap(A, B, c, mag_from=mag_from)
            lat, resid = z.real, float(z.imag.abs().max())
            img = flux_vae_decode(vae, lat)
            img.save(f"{pb}/hybrid_mag{mag_from}_c{c}.png")
            report[f"b/mag{mag_from}/c{c}"] = {"std": float(lat.std()),
                                               "imag_residue": resid}
            row.append(img)
            print(f"[e7] b mag={mag_from} c={c} residue={resid:.2e} done",
                  flush=True)
        rows.append(row)
        row_labels.append(f"mag from {mag_from}")
    col_labels = (["ref (mag src)"] +
                  [f"c={c}" + (" (B phase)" if c == 0 else
                               " (A phase)" if c == 1.0 else "")
                   for c in args.cutoffs])
    save_grid(rows, row_labels, col_labels, f"{out}/grid_partB.png")
    print("[e7] grid_partB saved", flush=True)


def main(args):
    out = os.path.join(RESULTS, "e7")
    os.makedirs(out, exist_ok=True)
    preflight(args)

    report = {"params": vars(args)}
    runners = {"a": run_a, "b": run_b}
    for part in args.part.split(","):
        runners[part.strip()](args, report, out)

    # merge with an existing report so partial runs (--part) accumulate
    path = f"{out}/report.json"
    if os.path.exists(path):
        with open(path) as f:
            merged = json.load(f)
        merged.update(report)
        report = merged
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e7] report -> {path}", flush=True)


def floats(s):
    return [float(v) for v in s.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="a,b")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg_sweep", type=floats, default="1.0,3.5")
    ap.add_argument("--cutoffs", type=floats, default="0,0.05,0.1,0.2,0.4,0.7,1.0")
    ap.add_argument("--mem", default="bnb4", choices=["bnb4", "seq_offload"])
    ap.add_argument("--a_src", default="cfg3.5_s0")
    ap.add_argument("--b_src", default="cfg3.5_s1")
    ap.add_argument("--prompt", default="A photo of cat in the park")
    main(ap.parse_args())
