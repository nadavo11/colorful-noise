"""Band-normalized Flux generation -- the E8 intervention packaged as a method.

Generate at full guidance (cfg=3.5) while clamping the working latent's
radial-band PSD to a cfg=1.0 per-step reference at EVERY denoising step:
per (channel, radial band), |F| *= sqrt(ref/cur), phase untouched (see
spectral_ops.psd_match / e8_psd_clamp.ClampPSD). E8 found this keeps the
cfg=3.5 look with a mild contrast reduction; E9 studies it across prompt
classes as a generation technique.

Usage:
    pipe = load_flux()                       # e7_flux_phase
    ref, ref_outs = record_reference(pipe, prompt)        # a few cfg=1 runs
    img, lat, info = generate_bandnorm(pipe, prompt, seed, ref)

The reference is recorded at fixed (steps, n_bins) and must match the
generation call (asserted).
"""
import math
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_ops import band_index_map
from e8_psd_clamp import RecordPSD, ClampPSD, gen_with_cb, N_CH, H, W


def record_reference(pipe, prompt, seeds=3, ref_cfg=1.0, steps=28, n_bins=24):
    """Record the per-step per-(channel, band) mean-power reference from
    `seeds` cfg=ref_cfg generations of `prompt`.

    Returns (ref, outs): ref = {"band": (steps, C, n_bins), "total": (steps,),
    "std": (steps,)} averaged over seeds; outs = [(img, lat), ...] -- the
    reference generations double as cfg=1 baselines."""
    idx_map = band_index_map(H, W, n_bins, "cuda")
    acc_band = torch.zeros(steps, N_CH, n_bins)
    acc_total = torch.zeros(steps)
    acc_std = torch.zeros(steps)
    outs = []
    for s in range(seeds):
        rec = RecordPSD(idx_map, n_bins, steps)
        img, lat = gen_with_cb(pipe, prompt, s, ref_cfg, steps, rec)
        acc_band += torch.stack(rec.band)
        acc_total += torch.tensor(rec.total)
        acc_std += torch.tensor(rec.std)
        outs.append((img, lat))
    ref = {"band": acc_band / seeds, "total": acc_total / seeds,
           "std": acc_std / seeds}
    return ref, outs


def band_centers(n_bins=24):
    """Center radial frequency of each band (matches band_index_map binning)."""
    fmax = math.sqrt(0.5)  # max radial frequency of an unshifted FFT grid
    edges = torch.linspace(0, fmax + 1e-6, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def modulate_reference(ref, target, g, cut_freq=0.25, n_bins=24):
    """Return a copy of `ref` with the power target of a frequency range scaled.

    target='high' scales bands with center freq >= cut_freq; target='low' scales
    bands below it. Power is scaled by g**2 so the clamp drives band magnitude to
    g x its plain-band-norm level (stable: the clamp targets a level each step,
    it does not compound). g=1.0 reproduces plain band-norm.
    """
    centers = band_centers(n_bins)
    sel = centers >= cut_freq if target == "high" else centers < cut_freq
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in ref.items()}
    band = ref["band"].clone()                 # (steps, C, n_bins)
    band[:, :, sel] *= g ** 2
    out["band"] = band
    return out


def generate_bandnorm(pipe, prompt, seed, ref, cfg=3.5, steps=28, n_bins=24,
                      mode="band"):
    """One band-normalized generation -> (img, final fp32 cpu latent, info).

    info: per-step post-clamp std curve + gain range + max imag residue."""
    assert ref["band"].shape[0] == steps, \
        f"reference recorded at {ref['band'].shape[0]} steps, asked {steps}"
    assert ref["band"].shape[2] == n_bins, \
        f"reference recorded at {ref['band'].shape[2]} bins, asked {n_bins}"
    idx_map = band_index_map(H, W, n_bins, "cuda")
    cb = ClampPSD(mode, ref, idx_map, n_bins)
    img, lat = gen_with_cb(pipe, prompt, seed, cfg, steps, cb)
    info = {"gain_min": cb.gmin, "gain_max": cb.gmax,
            "imag_residue": cb.resid, "perstep_std": list(cb.std)}
    return img, lat, info
