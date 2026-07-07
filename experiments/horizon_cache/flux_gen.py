"""FLUX generation sampler with a per-node action policy.

A faithful re-implementation of the SeaCache cached-residual step (mirroring
`flux_seacache_dp_shortcuts.install_seacache_forward`) as an explicit Euler flow loop, so
we can additionally *jump* — take a longer sigma stride that removes integration nodes.
First-order Euler only (deck constraint: do not apply jumps to multistep solvers).

The loop is model-agnostic in spirit but wired to FLUX here (the one generation backend
cached locally). Each node emits a causal feature dict (policy input) and the ledger
tracks achieved compute. The policy is any object with
`.act(feat, step_index, num_steps) -> action`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

# reuse the project's faithful SeaCache helpers
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flux_seacache_dp_shortcuts import (  # noqa: E402
    load_flux_pipeline,
    prepare_flux_inputs,
    decode_flux_latents,
    apply_sea_from_ab,
    ab_from_scheduler,
    rel_l1,
)

from .scheduler import ComputeLedger, JUMP_FACTORS, JumpEvent, regrid_tail
from .policy import action_code


def _sea_filter_h(modulated: torch.Tensor, img_ids: torch.Tensor, sigma: float) -> torch.Tensor:
    """Wiener-filter the modulated input exactly as SeaCache does (a,b)=(1-sigma,sigma)."""
    ids = img_ids[0] if img_ids.ndim == 3 else img_ids
    h = int(ids[:, 1].max().item() + 1)
    w = int(ids[:, 2].max().item() + 1)
    grid = modulated.reshape(modulated.shape[0], h, w, modulated.shape[-1])
    a, b = 1.0 - sigma, sigma
    return apply_sea_from_ab(grid, a, b, dims=(-2, -3), norm_mode="mean").reshape(
        modulated.shape[0], -1, modulated.shape[-1]
    )


class FluxCacheState:
    """Mutable cached-residual + SeaCache accumulator state for one trajectory."""

    def __init__(self):
        self.prev_residual: torch.Tensor | None = None
        self.prev_h_filt: torch.Tensor | None = None
        self.prev_h_raw: torch.Tensor | None = None   # unfiltered modulated (TeaCache family)
        self.prev_h_norm: float | None = None
        self.acc: float = 0.0        # accumulated Wiener-filtered relL1 (SeaCache)
        self.acc_raw: float = 0.0    # accumulated raw relL1 (TeaCache-family baseline)
        self.refresh_distance: int = 0
        self.prev_action: str = "fresh"


@torch.no_grad()
def _flux_node(pipe, tr, latents, timestep, guidance, ppe, pe, text_ids, image_ids,
               sigma: float, do_full: bool, state: FluxCacheState):
    """Run one FLUX forward. Returns (velocity, filtered_h, ran_full)."""
    hs = tr.x_embedder(latents)
    ts = timestep.to(hs.dtype) * 1000
    guidance_in = guidance.to(hs.dtype) * 1000 if guidance is not None else None
    temb = tr.time_text_embed(ts, ppe) if guidance_in is None else tr.time_text_embed(ts, guidance_in, ppe)
    enc = tr.context_embedder(pe)
    txt = text_ids[0] if text_ids is not None and text_ids.ndim == 3 else text_ids
    img = image_ids[0] if image_ids is not None and image_ids.ndim == 3 else image_ids
    rotary = tr.pos_embed(torch.cat((txt, img), dim=0))

    modulated, *_ = tr.transformer_blocks[0].norm1(hs, emb=temb)
    h_filt = _sea_filter_h(modulated, image_ids, sigma)

    if (not do_full) and state.prev_residual is not None:
        hs2 = hs + state.prev_residual
        ran_full = False
    else:
        ori = hs
        cur = hs
        for block in tr.transformer_blocks:
            enc, cur = block(hidden_states=cur, encoder_hidden_states=enc, temb=temb,
                             image_rotary_emb=rotary, joint_attention_kwargs=None)
        for block in tr.single_transformer_blocks:
            enc, cur = block(hidden_states=cur, encoder_hidden_states=enc, temb=temb,
                             image_rotary_emb=rotary, joint_attention_kwargs=None)
        state.prev_residual = (cur - ori).detach()
        hs2 = cur
        ran_full = True

    hs2 = tr.norm_out(hs2, temb)
    v = tr.proj_out(hs2)
    return v, h_filt, ran_full


def _features(sigma, sigma_next, step_index, num_steps, nodes_left, state: FluxCacheState,
              raw, h_filt) -> dict[str, float]:
    h_norm = float(h_filt.abs().mean().item())
    if state.prev_h_filt is not None:
        a = h_filt.flatten().float()
        b = state.prev_h_filt.flatten().float()
        cos = float((a @ b) / (a.norm() * b.norm() + 1e-12))
        h_cos_drift = 1.0 - cos
        h_norm_drift = abs(h_norm - state.prev_h_norm) / (state.prev_h_norm + 1e-12)
    else:
        h_cos_drift = 0.0
        h_norm_drift = 0.0
    return {
        "sigma": float(sigma),
        "delta_sigma": float(sigma_next - sigma),
        "progress": step_index / float(num_steps),
        "remaining_steps": float(nodes_left),
        "raw_rel_l1": float(raw),
        "acc_rel_l1": float(state.acc),
        "h_norm": h_norm,
        "h_norm_drift": h_norm_drift,
        "h_cos_drift": h_cos_drift,
        "refresh_distance": float(state.refresh_distance),
        "prev_action_code": float(action_code(state.prev_action)),
    }


@torch.no_grad()
def sample_flux(pipe, prompt: str, seed: int, steps: int, height: int, width: int,
                guidance: float, device: str, policy, max_seq_len: int = 512,
                action_source: str = "v0", L: int = 57, record_states: bool = False):
    """Run one HorizonCache trajectory. Returns a dict with final latent, per-step trace,
    jump events, and the ComputeLedger. If `record_states`, also returns per-node latent
    snapshots + inputs so a rollout can branch candidate actions from each state."""
    pe, ppe, text_ids, latents, image_ids, timesteps, guidance_t = prepare_flux_inputs(
        pipe, prompt, seed, steps, height, width, guidance, device, max_seq_len
    )
    tr = pipe.transformer
    sigmas = [float(x) for x in pipe.scheduler.sigmas.detach().cpu()]  # len steps+1, ends 0
    state = FluxCacheState()
    ledger = ComputeLedger(L=L, baseline_nodes=steps)
    traces: list[dict[str, Any]] = []
    jumps: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    i = 0
    step_index = 0
    t0 = time.perf_counter()
    # guard the executed-node budget so a pathological all-jump run still terminates
    while i < len(sigmas) - 1 and sigmas[i] > 1e-8 and step_index < steps:
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        timestep = torch.full((latents.shape[0],), sigma, device=device, dtype=latents.dtype)
        nodes_left = (len(sigmas) - 1) - i   # actual integration nodes still ahead

        # --- SeaCache signal is computed on a *provisional* cache pass to get filtered h.
        # We must know the action before committing full/cached; so peek h with a cheap
        # partial forward (block0 norm only) via one _flux_node call. To avoid double block
        # work we compute h first, decide, then run the (full|cached) forward once.
        # Cheap h peek:
        hs = tr.x_embedder(latents)
        ts = timestep.to(hs.dtype) * 1000
        guidance_in = guidance_t.to(hs.dtype) * 1000 if guidance_t is not None else None
        temb = tr.time_text_embed(ts, ppe) if guidance_in is None else tr.time_text_embed(ts, guidance_in, ppe)
        modulated, *_ = tr.transformer_blocks[0].norm1(hs, emb=temb)
        # At sigma≈1 (step 0) the Wiener coefficient a=1-sigma→0 zeros the filtered h;
        # SeaCache stores the UNFILTERED modulated at the boundary so step 1's relL1 is
        # meaningful. Mirror that: use raw modulated as the "previous" baseline at step 0.
        h_filt = _sea_filter_h(modulated, image_ids, sigma)
        h_prev_store = modulated.detach() if step_index == 0 else h_filt.detach()
        mod_raw = modulated.detach()
        if state.prev_h_filt is not None and step_index not in (0, steps - 1):
            raw = rel_l1(h_filt, state.prev_h_filt)
        else:
            raw = 0.0
        # unfiltered relL1 for the TeaCache-family baseline (no Wiener filter)
        if state.prev_h_raw is not None and step_index not in (0, steps - 1):
            raw_unfilt = rel_l1(mod_raw, state.prev_h_raw)
        else:
            raw_unfilt = 0.0
        # accumulate BEFORE deciding (SeaCache semantics)
        if step_index in (0, steps - 1):
            state.acc = 0.0
            state.acc_raw = 0.0
        else:
            state.acc += raw
            state.acc_raw += raw_unfilt
        feat = _features(sigma, sigma_next, step_index, steps, nodes_left, state, raw, h_filt)
        feat["raw_unfilt_rel_l1"] = float(raw_unfilt)
        feat["acc_raw_rel_l1"] = float(state.acc_raw)

        action = policy.act(feat, step_index, steps)
        # normalize a continuous adaptive jump: ("jump", jf) -> name + factor
        adaptive_jf = None
        if isinstance(action, tuple):
            adaptive_jf = float(action[1])
            action = "jump_adaptive"
        # if fresh, reset both accumulators (refresh)
        if action == "fresh":
            state.acc = 0.0
            state.acc_raw = 0.0
            feat["acc_rel_l1"] = 0.0
            feat["acc_raw_rel_l1"] = 0.0

        if record_states:
            snapshots.append({
                "step_index": step_index, "node_i": i, "sigma": sigma,
                "latent": latents.detach().cpu().clone(),
                "prev_residual": None if state.prev_residual is None else state.prev_residual.detach().cpu().clone(),
                "feat": feat.copy(),
            })

        # --- execute the chosen forward exactly once ---
        do_full = (action == "fresh")
        v, _hf, ran_full = _flux_node(pipe, tr, latents, timestep, guidance_t, ppe, pe,
                                      text_ids, image_ids, sigma, do_full, state)

        # --- scheduler update ---
        skipped = 0
        regridded = False
        target_sigma = sigma_next
        cfg = getattr(policy, "cfg", None)
        pc_on = bool(getattr(cfg, "pc_enabled", False))
        if action.startswith("jump"):
            jf = adaptive_jf if adaptive_jf is not None else JUMP_FACTORS[action]
            drop_mode = (adaptive_jf is None and cfg is not None
                         and getattr(cfg, "jump_mode", "regrid") == "drop")
            sigma_next_orig = sigma_next
            if drop_mode:
                j = min(i + 2, len(sigmas) - 1)
                target_sigma = sigmas[j]
            else:
                target_sigma = max(0.0, sigma + jf * (sigma_next - sigma))
            dsigma = target_sigma - sigma

            # ---- HorizonCache-PC: cached-endpoint predictor-corrector (E57) ----
            v_eff = v
            pc = dict(alpha=None, headroom=feat.get("acc_rel_l1"), curv_l1=None, curv_l2=None,
                      pc_norm=None, v0n=float(v.float().abs().sum().item()), ven=None,
                      cancelled=False, shrunk=False)
            if pc_on:
                alpha = float(getattr(cfg, "pc_alpha", 0.5))
                mode = getattr(cfg, "pc_curvature_mode", "none")
                kappa = float(getattr(cfg, "pc_curvature_kappa", 0.06))

                def _endpoint_v(x_end, sig_end):
                    tstep = torch.full((latents.shape[0],), sig_end, device=device, dtype=latents.dtype)
                    vv, _, _ = _flux_node(pipe, tr, x_end, tstep, guidance_t, ppe, pe,
                                          text_ids, image_ids, sig_end, False, state)
                    ledger.record_pc_endpoint()
                    return vv

                x_pred = latents + dsigma * v
                v_pred = _endpoint_v(x_pred, target_sigma)
                diff = (v_pred - v).float()
                vabs = float(v.float().abs().sum().item())
                pc["curv_l1"] = float(diff.abs().sum().item()) / (vabs + 1e-8)
                pc["curv_l2"] = float(diff.norm().item()) / (float(v.float().norm().item()) + 1e-8)
                pc["ven"] = float(v_pred.float().abs().sum().item())
                pc["alpha"] = alpha

                if mode == "cancel" and pc["curv_l1"] > kappa:
                    # curvature too high → abandon the jump, take a normal cached step
                    ledger.num_cancelled_jumps += 1
                    pc["cancelled"] = True
                    action = "cache"
                elif mode == "shrink" and pc["curv_l1"] > kappa:
                    # shrink the stride, then re-predict the endpoint at the smaller target
                    shrink = float(getattr(cfg, "pc_shrink_factor", 0.5))
                    jf = 1.0 + shrink * (jf - 1.0)
                    target_sigma = max(0.0, sigma + jf * (sigma_next - sigma))
                    dsigma = target_sigma - sigma
                    x_pred = latents + dsigma * v
                    v_pred = _endpoint_v(x_pred, target_sigma)
                    ledger.num_shrunk_jumps += 1
                    pc["shrunk"] = True
                    v_eff = (1.0 - alpha) * v + alpha * v_pred
                    pc["pc_norm"] = float((dsigma * (v_eff - v)).float().abs().sum().item())
                else:
                    v_eff = (1.0 - alpha) * v + alpha * v_pred
                    pc["pc_norm"] = float((dsigma * (v_eff - v)).float().abs().sum().item())

            # ---- integrate the (possibly corrected / cancelled) step ----
            if action == "cache":              # cancelled jump → normal cached step
                latents = latents + (sigma_next - sigma) * v
                i += 1
            elif drop_mode:
                latents = latents + dsigma * v_eff
                skipped = j - (i + 1)
                i = j
            else:
                latents = latents + dsigma * v_eff
                sigmas = regrid_tail(sigmas, i, target_sigma)
                regridded = True
                skipped = 1
                i += 1
            jumps.append(JumpEvent(
                step_index, ("jump_cancelled" if pc["cancelled"] else action), jf, sigma,
                target_sigma, skipped, regridded, action_source,
                sigma_next_original=sigma_next_orig, alpha=pc["alpha"], headroom=pc["headroom"],
                accumulated_score=feat.get("acc_rel_l1"), v_start_norm=pc["v0n"],
                v_endpoint_norm=pc["ven"], curvature_l1=pc["curv_l1"], curvature_l2=pc["curv_l2"],
                pc_correction_norm=pc["pc_norm"], was_cancelled=pc["cancelled"],
                was_shrunk=pc["shrunk"], effective_skipped_nodes=skipped).as_dict())
        else:
            latents = latents + (sigma_next - sigma) * v
            i += 1

        ledger.record(action, skipped=skipped)
        # update SeaCache/cache bookkeeping
        state.prev_h_filt = h_prev_store
        state.prev_h_raw = mod_raw
        state.prev_h_norm = feat["h_norm"]
        state.refresh_distance = 0 if action == "fresh" else state.refresh_distance + 1
        state.prev_action = action

        trace = {"step_index": step_index, "node_i_after": i, "action": action,
                 "sigma": sigma, "sigma_next_target": target_sigma, "ran_full": ran_full,
                 **feat}
        traces.append(trace)
        step_index += 1

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    ledger.measured_wall_time = time.perf_counter() - t0
    return {
        "final_latent": latents, "traces": traces, "jumps": jumps,
        "ledger": ledger, "sigmas_final": sigmas, "snapshots": snapshots,
        "inputs": {"pe": pe, "ppe": ppe, "text_ids": text_ids, "image_ids": image_ids,
                   "guidance_t": guidance_t},
    }
