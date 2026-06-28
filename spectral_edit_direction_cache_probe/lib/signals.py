"""E51 signal analysis: turn recorded reference trajectories into per-step cacheability
diagnostics and derive each variant's skip schedule.

Skip schedules are *oracle*-derived from the reference trajectory's own signal stability:
for a target skip ratio rho, skip the rho-fraction of interior steps where THAT variant's
change signal is smallest (endpoints never skipped). Every variant chooses skips from its own
signal at the SAME rho, so the comparison isolates "is this signal a better guide to where
reuse is safe?" — see report for why this is the right diagnostic framing.
"""
from __future__ import annotations
import numpy as np

import config as C
import spectral as S


def per_step(V_edit, V_src, SP_edit, SP_delta, want_amp=False):
    n = len(V_edit)
    D = V_edit - V_src                                   # packed edit direction [n, dim]

    def adj(fn, arr):
        out = [np.nan]
        for i in range(1, n):
            out.append(fn(arr[i], arr[i - 1]))
        return out

    def adj_abs(arr):                                    # absolute adjacent L2 = injected cache error
        out = [np.nan]
        for i in range(1, n):
            out.append(float(np.linalg.norm((arr[i] - arr[i - 1]).ravel())))
        return out

    lp_edit = np.stack([S.lowpass(SP_edit[i], C.LOWPASS_FRAC) for i in range(n)])
    lp_delta = np.stack([S.lowpass(SP_delta[i], C.LOWPASS_FRAC) for i in range(n)])

    diag = dict(
        # PRIMARY cacheability signal: absolute adjacent L2 (error injected by reusing a stale value;
        # v_src is recomputed exactly for delta caching, so both are in the same velocity units).
        abs_edit=adj_abs(V_edit),
        abs_delta=adj_abs(D),
        mag_edit=[float(np.linalg.norm(V_edit[i])) for i in range(n)],
        mag_delta=[float(np.linalg.norm(D[i])) for i in range(n)],
        # SECONDARY (scale-dependent, reported with a caveat): relative adjacent L2.
        raw_rel_edit=adj(S.rel_l2, V_edit),
        raw_rel_delta=adj(S.rel_l2, D),
        cos_edit=adj(S.cosine, V_edit),
        cos_delta=adj(S.cosine, D),
        # spectral (low-pass) absolute adjacent L2 — the signal the spectral caches gate on.
        spec_edit=adj_abs(lp_edit),
        spec_delta=adj_abs(lp_delta),
        band_edit=[list(S.band_energies(SP_edit[i], C.LOW, C.HIGH)) for i in range(n)],
        band_delta=[list(S.band_energies(SP_delta[i], C.LOW, C.HIGH)) for i in range(n)],
        radial_edit=[S.radial_power(SP_edit[i]).tolist() for i in range(n)],
        radial_delta=[S.radial_power(SP_delta[i]).tolist() for i in range(n)],
        n=n,
    )
    amp = None
    if want_amp:
        amp = dict(
            amp_edit=np.stack([S.amp_image(SP_edit[i]) for i in range(n)]),
            amp_delta=np.stack([S.amp_image(SP_delta[i]) for i in range(n)]),
        )
    return diag, amp


SIGNAL_KEY = {
    "raw_full_prediction_cache": "abs_edit",
    "spectral_full_prediction_cache": "spec_edit",
    "raw_edit_delta_cache": "abs_delta",
    "spectral_edit_delta_cache": "spec_delta",
}


def skip_schedule(change, ratio, n):
    """change: list length n (index 0 is nan). Skip the rho-fraction of interior steps with
    the smallest change. Returns (set_of_skip_indices, realized_ratio)."""
    interior = list(range(1, n - 1))
    k = int(round(ratio * len(interior)))
    if k <= 0:
        return set(), 0.0
    c = np.array([change[i] for i in interior], dtype=float)
    order = np.argsort(c)
    skip = set(interior[j] for j in order[:k])
    return skip, k / len(interior)
