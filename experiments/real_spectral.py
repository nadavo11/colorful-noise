"""Real-image spectral reference + generated-vs-real gap measurement (E23).

E23 premise: until now SBN clamped a generated latent's per-(channel, radial
band) power toward a cfg=1 PROXY reference (bandnorm / e8). cfg=1 is an arbitrary
anchor -- the honest target is the REAL-image spectrum. results/e10/real_latents.pt
already holds real photos VAE-encoded into Flux latent space (16ch, 128x128); this
module turns them into a per-(channel, band) power reference and measures the
generated-vs-real gap as a DISTRIBUTION (mean + spread across images), not just the
scalar RMS in fidelity_metrics.spectral_dist_to_real.

All spectral math reuses spectral_ops.band_power, so the reference lands on exactly
the bins psd_match / ClampPSD clamp to -- the measured gap and the applied
correction are the same operator.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_ops import band_power
from fidelity_metrics import REAL_LATENTS


def load_real_latents(path=REAL_LATENTS):
    """(N, 16, 128, 128) fp32 cpu real-photo latents, or a clear error if the
    E10 reference has not been built yet."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no {path}; build the real-photo latents first "
            "(python e10_cfg_spectral.py --part download,real)")
    return torch.load(path, weights_only=True)


def stacked_band_power(latents, idx_map, n_bins):
    """(N, C, H, W) -> (N, C, n_bins): per-image counts-weighted mean power per
    (channel, radial band). Same reduction band_power applies inside psd_match."""
    dev = idx_map.device
    out = []
    for i in range(latents.shape[0]):
        F2 = torch.fft.fft2(latents[i:i + 1].to(dev).float()).abs() ** 2  # (1,C,H,W)
        out.append(band_power(F2[0], idx_map, n_bins).cpu())
    return torch.stack(out)  # (N, C, n_bins)


def real_band_power(latents, idx_map, n_bins):
    """(N, C, H, W) real latents -> (stacked (N,C,n_bins), mean (C,n_bins)).

    The mean is the UNIVERSAL per-channel clamp target for real-SBN (the
    'whole distribution' target the experiment chases)."""
    stacked = stacked_band_power(latents, idx_map, n_bins)
    return stacked, stacked.mean(0)


def band_power_distribution(stacked):
    """(N, C, n_bins) -> per-(channel, band) summary stats across the N samples:
    {mean, std, p10, p90} (each (C, n_bins)). The spread is the 'distribution,
    not a scalar' deliverable -- it shows how tightly real / generated images
    cluster in each band."""
    s = stacked.float()
    return {
        "mean": s.mean(0),
        "std": s.std(0),
        "p10": s.quantile(0.10, dim=0),
        "p90": s.quantile(0.90, dim=0),
    }


def correction_curve(real_band, gen_band, eps=1e-8):
    """Per-(channel, band) correction from generated power toward real power.

    ratio[c,b] = real / gen; gain = sqrt(ratio) is exactly the psd_match
    magnitude gain that would re-level generated band power onto real. ratio > 1
    means the model UNDER-powers real images in that band; ratio < 1 means it
    OVER-powers (E10 predicts over-power in the low bands at high cfg).
    ratio_cmean is the channel-mean curve (ratio of channel-mean powers, so it
    matches the channel-mean PSD plot)."""
    rb = real_band.float()
    gb = gen_band.float().clamp(min=eps)
    ratio = rb / gb
    return {
        "ratio": ratio,                                       # (C, n_bins)
        "gain": ratio.clamp(min=0).sqrt(),                    # (C, n_bins)
        "ratio_cmean": rb.mean(0) / gb.mean(0).clamp(min=eps),  # (n_bins,)
    }
