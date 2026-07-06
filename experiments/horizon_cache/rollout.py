"""Safe-horizon label generation for HorizonCache-v1 (deck: DP optimizes the wrong,
path-independent thing — so we use *rollout* labels instead).

For each visited state (image, step) on a SeaCache trajectory we branch each candidate
action, continue with a fixed continuation policy for a short horizon H, and measure the
damage relative to the full trajectory's continuation from that same state. The label is
the cheapest (largest-horizon) action whose damage stays below tolerance — a genuine
safe-horizon label, not a local surrogate.

Continuation policy default: `full` for a short horizon H (most faithful). Falls back to
SeaCache continuation if H is large. Damage = latent L2 vs the full continuation (cheap,
no decode); optionally decode-based PSNR at the last node for a subset.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .scheduler import ACTIONS, JUMP_FACTORS, action_to_horizon, regrid_tail
from .policy import FEATURE_NAMES
from .flux_gen import _flux_node, FluxCacheState, _sea_filter_h


@torch.no_grad()
def _continue_full(pipe, tr, latents, sigmas, i, ctx, H, guidance_t):
    """Reference: continue with FULL forwards for min(H, remaining) nodes from node i."""
    device = latents.device
    end = min(i + H, len(sigmas) - 1)
    st = FluxCacheState()
    while i < end and sigmas[i] > 1e-8:
        sigma = sigmas[i]
        ts = torch.full((latents.shape[0],), sigma, device=device, dtype=latents.dtype)
        v, _h, _ = _flux_node(pipe, tr, latents, ts, guidance_t, ctx["ppe"], ctx["pe"],
                              ctx["text_ids"], ctx["image_ids"], sigma, do_full=True, state=st)
        latents = latents + (sigmas[i + 1] - sigma) * v
        i += 1
    return latents


@torch.no_grad()
def _rollout_action(pipe, tr, latent0, prev_residual, sigmas, i, ctx, action, H, guidance_t):
    """Take `action` from node i using the cached residual, then continue FULL for H nodes.
    Returns the resulting latent (to compare against the full continuation)."""
    device = latent0.device
    latents = latent0.clone()
    st = FluxCacheState()
    st.prev_residual = None if prev_residual is None else prev_residual.clone().to(device)
    sigma = sigmas[i]
    ts = torch.full((latents.shape[0],), sigma, device=device, dtype=latents.dtype)
    do_full = (action == "fresh")
    v, _h, _ = _flux_node(pipe, tr, latents, ts, guidance_t, ctx["ppe"], ctx["pe"],
                          ctx["text_ids"], ctx["image_ids"], sigma, do_full=do_full, state=st)
    local_sigmas = list(sigmas)
    if action.startswith("jump"):
        jf = JUMP_FACTORS[action]
        target = max(0.0, sigma + jf * (sigmas[i + 1] - sigma))
        latents = latents + (target - sigma) * v
        local_sigmas = regrid_tail(local_sigmas, i, target)
        ni = i + 1
    else:
        latents = latents + (sigmas[i + 1] - sigma) * v
        ni = i + 1
    # continue FULL from ni
    latents = _continue_full(pipe, tr, latents, local_sigmas, ni, ctx, H, guidance_t)
    return latents


@torch.no_grad()
def build_dataset(pipe, prompts, seeds, steps, height, width, guidance, device,
                  out_csv: Path, out_parquet: Path, H: int = 6, tol_l2: float = 0.02,
                  max_seq_len: int = 512, tau_cache: float = 0.3, L: int = 57,
                  stride: int = 1, damage_metric: str = "latent_l2",
                  psnr_floor: float = 30.0) -> dict[str, Any]:
    """Log rollout safe-horizon labels.

    `damage_metric`:
      - "latent_l2"    : damage = relative latent-L2 of the H-node continuation vs the
                         full-from-here continuation; action safe if damage <= tol_l2.
                         (E56 found this too strict on FLUX — 0 jump positives.)
      - "decoded_psnr" : decode both continuations; action safe if PSNR(cand, full) >=
                         `psnr_floor` dB. Aligns the label with what we actually care about
                         (perceptual closeness to full sampling) — Experiment B fix.
    """
    from .baselines import SeaCachePolicy
    from .flux_gen import sample_flux
    from .metrics import psnr as _psnr
    from flux_seacache_dp_shortcuts import decode_flux_latents

    def _damage(out_latent, ref_latent, ref_norm, ref_img=None):
        if damage_metric == "decoded_psnr":
            ci = decode_flux_latents(pipe, out_latent, height, width)
            ri = ref_img if ref_img is not None else decode_flux_latents(pipe, ref_latent, height, width)
            return _psnr(ri, ci)                      # higher = safer
        return float(((out_latent.float() - ref_latent.float()) ** 2).mean().sqrt().item()) / ref_norm

    def _is_safe(dmg):
        return (dmg >= psnr_floor) if damage_metric == "decoded_psnr" else (dmg <= tol_l2)

    rows: list[dict[str, Any]] = []
    for p in prompts:
        for seed in seeds:
            # generate a SeaCache trajectory, recording per-node states
            res = sample_flux(pipe, p["prompt"], seed, steps, height, width, guidance,
                              device, SeaCachePolicy(tau_cache), max_seq_len, "seacache",
                              L=L, record_states=True)
            ctx = res["inputs"]
            tr = pipe.transformer
            snaps = res["snapshots"]
            for s_idx, snap in enumerate(snaps):
                if s_idx % stride != 0:
                    continue
                i = snap["node_i"]
                if i == 0 or i >= len(res["sigmas_final"]) - 2:
                    continue
                if snap["feat"]["remaining_steps"] < 3:
                    continue
                latent0 = snap["latent"].to(device)
                prev_res = snap["prev_residual"]
                sig = res["sigmas_final"]
                # reference: full continuation from here
                ref = _rollout_action(pipe, tr, latent0, prev_res, sig, i, ctx, "fresh", H, ctx["guidance_t"])
                ref_norm = float((ref.float() ** 2).mean().sqrt().item()) + 1e-8
                ref_img = decode_flux_latents(pipe, ref, height, width) if damage_metric == "decoded_psnr" else None
                damages = {}
                for a in ACTIONS:
                    out = _rollout_action(pipe, tr, latent0, prev_res, sig, i, ctx, a, H, ctx["guidance_t"])
                    damages[a] = _damage(out, ref, ref_norm, ref_img)
                # safe-horizon label = largest-horizon action whose damage is within tolerance
                safe = "fresh"
                best_h = 0.0
                for a in ACTIONS:
                    if _is_safe(damages[a]) and action_to_horizon(a) >= best_h:
                        safe, best_h = a, action_to_horizon(a)
                row = {"prompt_id": p["id"], "seed": seed, "step_index": snap["step_index"],
                       "node_i": i, "label_action": safe, "label_horizon": best_h}
                row.update({k: float(snap["feat"].get(k, 0.0)) for k in FEATURE_NAMES})
                row.update({f"damage_{a}": damages[a] for a in ACTIONS})
                rows.append(row)
            print(f"[rollout] {p['id']} s{seed}: {len([r for r in rows if r['prompt_id']==p['id']])} states",
                  flush=True)

    if rows:
        keys = list(rows[0].keys())
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        try:
            import pandas as pd
            pd.DataFrame(rows).to_parquet(out_parquet)
        except Exception as e:
            print(f"[rollout] parquet skipped ({e})", flush=True)
    label_hist = {a: sum(1 for r in rows if r["label_action"] == a) for a in ACTIONS}
    return {"n_states": len(rows), "label_hist": label_hist,
            "csv": str(out_csv), "parquet": str(out_parquet), "H": H, "tol_l2": tol_l2,
            "damage_metric": damage_metric, "psnr_floor": psnr_floor}
