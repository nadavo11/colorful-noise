"""E6: What does SDXL read from the FFT *phase* of the initial latent?

E0-E5 probed the amplitude spectrum (low-freq notch, power ~ gamma^2). For
white Gaussian noise the FFT factorizes exactly into independent Rayleigh
magnitudes x uniform phases, so phase surgery with kept magnitudes leaves the
power spectrum perfectly in-distribution -- the minimal off-manifold probe.

Parts (--part, comma list):
  p0  -- sanity: phase uniformity + phase/mag independence numerics, then the
         phase-rerandomization control (keep |FFT|, fresh Hermitian uniform
         phase) which should be indistinguishable from plain noise; latent
         stats for every manipulated latent type used below.
  p2  -- image-phase transplant: condition_latent(phase='image', mag='noise',
         dc='noise') swept from the paper's p=0.015 up to p=1.0 (full
         spectrum); spectrum stays white at every p.
  p4  -- quantize the phase of fresh noise to k levels (Rayleigh magnitudes
         stay in-distribution; no image involved).
  p4b -- quantize to k=8 then omit one level pair: raw zeroing vs
         power-renormalized (attributes damage to the phase hole vs the
         power loss).
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_pipe, encode_image, generate, save_grid, INPUTS, RESULTS
from spectral_ops import (condition_latent, paper_low_mask, whiteness,
                          random_hermitian_phase, quantize_phase)

CASES = [  # (input image, prompt) -- the paper's own pairs
    ("cat_orange.png", "A photo of cat in the park"),
    ("savana.png", "A photo of giraffe and an elephant in the savanna"),
]
SHAPE = (1, 4, 128, 128)


def fresh_noise(seed):
    torch.manual_seed(seed)
    return torch.randn(*SHAPE, device="cuda")


def phase_rerand(noise):
    """Keep |FFT| of noise, swap in fresh uniform Hermitian phase.

    Returns the complex ifft so callers can check the imaginary residue;
    take .real for the latent.
    """
    F = torch.fft.fft2(noise.float())
    phi = random_hermitian_phase(*noise.shape[1:], device=noise.device)
    return torch.fft.ifft2(F.abs() * torch.exp(1j * phi))


def pearson(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def flatness(phi, bins=32):
    """std/mean of the phase histogram (0 = perfectly uniform)."""
    h = torch.histc(phi.flatten().float(), bins=bins, min=-math.pi, max=math.pi)
    return float(h.std() / h.mean())


def latent_stats(lat):
    return {"whiteness": whiteness(lat), "std": float(lat.std()),
            "channel_means": lat.mean(dim=(2, 3)).flatten().tolist()}


def omit_pairs(k):
    """Distinct conjugate level pairs for k levels: 0..k//2 (0 and k/2 are
    self-conjugate singletons for even k)."""
    return list(range(k // 2 + 1))


def preflight(args):
    """Numeric asserts, run before any GPU generation; abort on failure."""
    print("[e6] pre-flight asserts ...", flush=True)
    dev = "cuda"

    # p=1.0 mask covers every bin, corners included
    m = paper_low_mask(128, 128, 1.0, dev)
    assert m.sum().item() == 128 * 128, "p=1.0 mask must cover all bins"

    noise = fresh_noise(0)
    F = torch.fft.fft2(noise.float())

    # phase histogram flatness for randn phase and the Hermitian sampler
    assert flatness(F.angle()) < 0.1, "randn phase not uniform"
    assert flatness(random_hermitian_phase(4, 128, 128, dev)) < 0.1, \
        "random_hermitian_phase not uniform"

    # rerand: ifft realness
    z = phase_rerand(noise)
    assert z.imag.abs().max().item() < 1e-4, "phase_rerand ifft not real"

    # plain quantization: realness, std~1, DC preserved
    for k in args.k_sweep:
        lat, st = quantize_phase(noise, k, return_stats=True)
        assert st["imag_residue"] < 1e-4, f"k={k}: ifft not real"
        assert abs(float(lat.std()) - 1.0) < 0.05, f"k={k}: std off"
        dc_new = torch.fft.fft2(lat.float())[..., 0, 0]
        assert torch.allclose(dc_new, F[..., 0, 0], rtol=1e-3, atol=0.05), \
            f"k={k}: DC not preserved"

    # omission: realness, mask symmetry, std signatures
    k = args.k_omit
    lvl = torch.round(F.angle() / (2 * math.pi / k)).long() % k
    for omit in omit_pairs(k):
        drop = (lvl == omit % k) | (lvl == (-omit) % k)
        conj = torch.roll(torch.flip(drop, dims=[-2, -1]),
                          shifts=(1, 1), dims=(-2, -1))
        assert torch.equal(drop, conj), f"omit={omit}: mask not Hermitian"
        for mode in ("zero", "renorm"):
            lat, st = quantize_phase(noise, k, omit=omit, mode=mode,
                                     return_stats=True)
            assert st["imag_residue"] < 1e-4, f"omit={omit} {mode}: not real"
            dc_new = torch.fft.fft2(lat.float())[..., 0, 0]
            assert torch.allclose(dc_new, F[..., 0, 0], rtol=1e-3, atol=0.05), \
                f"omit={omit} {mode}: DC not preserved"
            if mode == "renorm":
                ch_std = lat.std(dim=(2, 3)).flatten()
                assert (ch_std - 1.0).abs().max().item() < 0.05, \
                    f"omit={omit} renorm: per-channel std off ({ch_std.tolist()})"
            else:
                expected = math.sqrt(st["kept_fraction"])
                assert abs(float(lat.std()) - expected) < 0.1 * expected, \
                    f"omit={omit} zero: std {float(lat.std()):.3f} != sqrt(kept)={expected:.3f}"
    print("[e6] pre-flight OK", flush=True)


def run_p0(args, pipe, report, out):
    # (a) numerics, no generation
    noise = fresh_noise(0)
    F = torch.fft.fft2(noise.float())
    mag, phi = F.abs(), F.angle()
    report["p0_uniformity"] = {
        "flatness_randn_phase": flatness(phi),
        "flatness_hermitian_sampler":
            flatness(random_hermitian_phase(4, 128, 128, "cuda")),
        "corr_phi_mag": pearson(phi, mag),
        "corr_cosphi_mag": pearson(torch.cos(phi), mag),
        "corr_sinphi_mag": pearson(torch.sin(phi), mag),
    }
    print(f"[e6] p0 uniformity: {report['p0_uniformity']}", flush=True)

    # (c) latent stats for every manipulated latent type used in p2/p4/p4b
    stats = report.setdefault("p0_latent_stats", {})
    stats["plain_noise"] = latent_stats(noise)
    stats["phase_rerand"] = latent_stats(phase_rerand(noise).real)
    z = encode_image(pipe, os.path.join(INPUTS, CASES[0][0])).float()
    for p in args.p_sweep:
        lat = condition_latent(noise, z, p=p,
                               phase="image", mag="noise", dc="noise")
        stats[f"p2_p{p}"] = latent_stats(lat)
    for k in args.k_sweep:
        stats[f"p4_k{k}"] = latent_stats(quantize_phase(noise, k))
    for omit in omit_pairs(args.k_omit):
        for mode in ("zero", "renorm"):
            lat, st = quantize_phase(noise, args.k_omit, omit=omit, mode=mode,
                                     return_stats=True)
            stats[f"p4b_omit{omit}_{mode}"] = {**latent_stats(lat), **st}
    print("[e6] p0 latent stats collected", flush=True)

    # (b) phase-rerandomization control: should match plain noise
    _, prompt = CASES[0]
    rows, row_labels = [], []
    for label in ("plain_noise", "phase_rerand"):
        row = []
        for seed in range(args.seeds):
            noise = fresh_noise(seed)
            lat = noise if label == "plain_noise" else phase_rerand(noise).real
            img = generate(pipe, prompt, lat, steps=args.steps)
            img.save(f"{out}/p0_{label}_s{seed}.png")
            row.append(img)
            print(f"[e6] p0 {label} seed{seed} done", flush=True)
        rows.append(row)
        row_labels.append(label)
    save_grid(rows, row_labels, [f"seed {s}" for s in range(args.seeds)],
              f"{out}/grid_p0.png")
    print("[e6] grid_p0 saved", flush=True)


def run_p2(args, pipe, report, out):
    for img_name, prompt in CASES[: args.num_cases]:
        z = encode_image(pipe, os.path.join(INPUTS, img_name)).float()
        rows, row_labels = [], []
        for p in args.p_sweep + [None]:  # None = white-noise control row
            label = "white" if p is None else f"p={p}"
            row = []
            for seed in range(args.seeds):
                noise = fresh_noise(seed)
                lat = noise if p is None else condition_latent(
                    noise, z, p=p, phase="image", mag="noise", dc="noise")
                if seed == 0:
                    report[f"p2/{img_name}/{label}"] = latent_stats(lat)
                img = generate(pipe, prompt, lat, steps=args.steps)
                img.save(f"{out}/p2_{img_name[:-4]}_{label}_s{seed}.png")
                row.append(img)
                print(f"[e6] p2 {img_name} {label} seed{seed} done", flush=True)
            rows.append(row)
            row_labels.append(label)
        save_grid(rows, row_labels, [f"seed {s}" for s in range(args.seeds)],
                  f"{out}/grid_p2_{img_name[:-4]}.png")
        print(f"[e6] grid_p2_{img_name[:-4]} saved", flush=True)


def run_p4(args, pipe, report, out):
    _, prompt = CASES[0]
    rows, row_labels = [], []
    for k in args.k_sweep + [None]:  # None = k=inf control (untouched noise)
        label = "k=inf" if k is None else f"k={k}"
        row = []
        for seed in range(args.seeds):
            lat = quantize_phase(fresh_noise(seed), k)
            if seed == 0:
                report[f"p4/{label}"] = latent_stats(lat)
            img = generate(pipe, prompt, lat, steps=args.steps)
            img.save(f"{out}/p4_{label.replace('=', '')}_s{seed}.png")
            row.append(img)
            print(f"[e6] p4 {label} seed{seed} done", flush=True)
        rows.append(row)
        row_labels.append(label)
    save_grid(rows, row_labels, [f"seed {s}" for s in range(args.seeds)],
              f"{out}/grid_p4.png")
    print("[e6] grid_p4 saved", flush=True)


def run_p4b(args, pipe, report, out):
    _, prompt = CASES[0]
    k = args.k_omit
    rows, row_labels = [], []
    for omit in omit_pairs(k):
        for mode in ("zero", "renorm"):
            label = f"omit±{omit} {mode}"
            row = []
            for seed in range(args.seeds):
                lat, st = quantize_phase(fresh_noise(seed), k, omit=omit,
                                         mode=mode, return_stats=True)
                if seed == 0:
                    report[f"p4b/{label}"] = {
                        **latent_stats(lat),
                        "kept_fraction": st["kept_fraction"],
                        "power_ratio": st["power_ratio"],
                    }
                img = generate(pipe, prompt, lat, steps=args.steps)
                img.save(f"{out}/p4b_omit{omit}_{mode}_s{seed}.png")
                row.append(img)
                print(f"[e6] p4b omit{omit} {mode} seed{seed} done", flush=True)
            rows.append(row)
            row_labels.append(label)
    save_grid(rows, row_labels, [f"seed {s}" for s in range(args.seeds)],
              f"{out}/grid_p4b.png")
    print("[e6] grid_p4b saved", flush=True)


def main(args):
    out = os.path.join(RESULTS, "e6")
    os.makedirs(out, exist_ok=True)
    preflight(args)

    pipe = load_pipe()
    report = {"params": vars(args)}
    runners = {"p0": run_p0, "p2": run_p2, "p4": run_p4, "p4b": run_p4b}
    for part in args.part.split(","):
        runners[part.strip()](args, pipe, report, out)

    # merge with an existing report so partial runs (--part) accumulate
    path = f"{out}/report.json"
    if os.path.exists(path):
        with open(path) as f:
            merged = json.load(f)
        merged.update(report)
        report = merged
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e6] report -> {path}", flush=True)


def floats(s):
    return [float(v) for v in s.split(",")]


def ints(s):
    return [int(v) for v in s.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="p0,p2,p4,p4b")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--num_cases", type=int, default=2)
    ap.add_argument("--p_sweep", type=floats, default="0.015,0.1,0.3,1.0")
    ap.add_argument("--k_sweep", type=ints, default="2,4,8,16,32")
    ap.add_argument("--k_omit", type=int, default=8)
    main(ap.parse_args())
