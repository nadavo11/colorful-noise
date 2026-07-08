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
import torch.nn.functional as Fnn

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
        # --- E58 Residual Motion Cache: fresh-residual history for secant extrapolation ---
        # prev_residual above IS r_anchor (most recent fresh block residual). We also keep the
        # penultimate fresh residual r_prev + the sigmas / filtered-h / step-age at both anchors
        # so a cached step can predict r_pred = r_anchor + β·λ·P(r_anchor - r_prev).
        self.r_prev: torch.Tensor | None = None
        self.sigma_anchor: float | None = None
        self.sigma_prev: float | None = None
        self.h_anchor_filt: torch.Tensor | None = None
        self.h_prev_filt: torch.Tensor | None = None
        self.steps_since_anchor: int = 0
        self.steps_prev_to_anchor: int = 1
        # --- E59 second-order residual hold: third fresh anchor r_{a-2} + its sigma, and the
        # anchor-triple curvature diagnostics (scalars, refreshed on every fresh forward when a
        # second-order mode is enabled). None for first-order RM — E58 path untouched.
        self.r_prev2: torch.Tensor | None = None
        self.sigma_prev2: float | None = None
        self.so_diag: dict | None = None
        # --- E60 closed-loop residual motion: exponentially-forgetting LS accumulators for the
        # online secant gain β̂, updated at every fresh anchor (scalars; zero extra forwards).
        self.cl_num: float = 0.0     # Σ w^age · λ_j⟨Δr_j, y_j⟩/‖Δr_j‖²   (normalized obs)
        self.cl_den: float = 0.0     # Σ w^age · λ_j²
        self.cl_n_obs: int = 0
        self.cl_beta_hat: float | None = None   # UNclamped posterior mean (clamped at use site)
        self.cl_diag: dict | None = None         # last-refresh innovation diagnostics


# ------------------------- E58 residual-motion primitives -------------------------

def _grid_hw(image_ids: torch.Tensor) -> tuple[int, int]:
    ids = image_ids[0] if image_ids.ndim == 3 else image_ids
    return int(ids[:, 1].max().item() + 1), int(ids[:, 2].max().item() + 1)


def _project_residual(dr: torch.Tensor, image_ids: torch.Tensor, sigma: float, cfg,
                      proj: str | None = None) -> torch.Tensor:
    """Stable-subspace projection P(Δr) of the residual secant.
    raw = identity · lowpass = avg-pool+upsample over token grid · topk = keep most energetic
    channels · sea = SeaCache Wiener filter (a,b)=(1-σ,σ) on the residual grid.
    `proj` overrides cfg.rm_projection (E59 uses it for the P2 curvature projection)."""
    proj = proj if proj is not None else getattr(cfg, "rm_projection", "raw")
    if proj == "raw":
        return dr
    if proj == "topk":
        energy = dr.abs().float().mean(dim=(0, 1))               # [C]
        k = max(1, int(round(float(getattr(cfg, "rm_topk_frac", 0.25)) * energy.numel())))
        thresh = torch.topk(energy, k).values.min()
        mask = (energy >= thresh).to(dr.dtype)
        return dr * mask.view(1, 1, -1)
    h, w = _grid_hw(image_ids)
    grid = dr.reshape(dr.shape[0], h, w, dr.shape[-1])
    if proj == "lowpass":
        pool = max(1, int(getattr(cfg, "rm_lowpass_pool", 2)))
        x = grid.permute(0, 3, 1, 2).float()                     # [B,C,H,W]
        xp = Fnn.avg_pool2d(x, kernel_size=pool, ceil_mode=True)
        xl = Fnn.interpolate(xp, size=(h, w), mode="nearest")     # low-pass = pool then upsample
        return xl.permute(0, 2, 3, 1).reshape(dr.shape).to(dr.dtype)
    if proj == "sea":
        a, b = 1.0 - sigma, sigma
        return apply_sea_from_ab(grid, a, b, dims=(-2, -3), norm_mode="mean").reshape(dr.shape).to(dr.dtype)
    return dr


def _rm_lambda(state: "FluxCacheState", h_filt: torch.Tensor, sigma: float, cfg):
    """Progress coefficient λ(t) since the anchor. Returns (lambda_used, λ_sigma, λ_age, λ_h)."""
    eps = 1e-8
    ds = (state.sigma_anchor - state.sigma_prev) if (state.sigma_anchor is not None and state.sigma_prev is not None) else 0.0
    lam_sigma = 0.0 if ds == 0 else (sigma - state.sigma_anchor) / (ds - eps if ds < 0 else ds + eps)
    lam_age = state.steps_since_anchor / float(max(1, state.steps_prev_to_anchor))
    lam_h = 0.0
    if state.h_anchor_filt is not None and state.h_prev_filt is not None:
        num = rel_l1(h_filt, state.h_anchor_filt)
        den = rel_l1(state.h_anchor_filt, state.h_prev_filt) + eps
        lam_h = num / den
    mode = getattr(cfg, "rm_lambda_mode", "sigma")
    lam = {"sigma": lam_sigma, "age": lam_age, "h": lam_h}.get(mode, lam_sigma)
    lam = float(min(max(lam, 0.0), float(getattr(cfg, "rm_lambda_max", 1.5))))
    return lam, float(lam_sigma), float(lam_age), float(lam_h)


def _anchor_triple_diag(r_a: torch.Tensor, r_am1: torch.Tensor, r_am2: torch.Tensor) -> dict:
    """E59 curvature diagnostics on three fresh residual anchors (scalars, computed once per
    refresh). ρ2 = ‖Δ²r‖₁/‖Δr_a‖₁ (curvature-to-velocity), cosΔ = cos(Δr_a, Δr_{a-1})
    (directional stability), ρ_anchor = ‖Δ²r‖₁/‖r_a‖₁."""
    eps = 1e-8
    dr_a = (r_a - r_am1).float()
    dr_am1 = (r_am1 - r_am2).float()
    d2r = dr_a - dr_am1
    dr_l1 = float(dr_a.abs().sum().item())
    d2r_l1 = float(d2r.abs().sum().item())
    cos = float((dr_a.flatten() @ dr_am1.flatten()).item()
                / (float(dr_a.norm().item()) * float(dr_am1.norm().item()) + eps))
    return {
        "so_rho2": d2r_l1 / (dr_l1 + eps),
        "so_cos_delta": cos,
        "so_rho_anchor": d2r_l1 / (float(r_a.abs().float().sum().item()) + eps),
        "so_dr_l1": dr_l1,
        "so_d2r_l1": d2r_l1,
    }


def _cl_beta_used(state: "FluxCacheState", cfg) -> float:
    """The gain a closed-loop cached step would use right now: clamped posterior β̂ (prior when
    no innovation has been observed yet — the μ-regularized estimate starts at β_prior)."""
    prior = float(getattr(cfg, "rm_cl_prior", 0.5))
    bh = state.cl_beta_hat if state.cl_beta_hat is not None else prior
    return float(min(max(bh, 0.0), float(getattr(cfg, "rm_cl_beta_max", 1.0))))


def _cl_update(state: "FluxCacheState", new_res: torch.Tensor, h_filt: torch.Tensor,
               sigma: float, cfg) -> None:
    """E60: one innovation observation at a fresh anchor, BEFORE the history shift.
    Observation model  y_k = r_k − r_{k−1} ≈ β·λ_k·Δr  with Δr = r_{k−1} − r_{k−2};
    λ_k is the same clamped progress coefficient the predictor would have used at σ_k.
    Each observation is normalized by ‖Δr‖² (so μ is dimensionless, weight ∝ λ²):
      num += λ⟨Δr,y⟩/‖Δr‖²,  den += λ²,  β̂ = (μ·β_prior + num)/(μ + den)."""
    beta_used = _cl_beta_used(state, cfg)      # innovation of the predictor actually in force
    lam_k = _rm_lambda(state, h_filt, sigma, cfg)[0]
    dr = (state.prev_residual - state.r_prev).float()
    y = (new_res.float() - state.prev_residual.float())
    dr2 = float((dr * dr).sum().item()) + 1e-12
    beta_obs = float((dr.flatten() @ y.flatten()).item()) / dr2   # per-anchor LS β
    w = float(getattr(cfg, "rm_cl_forget", 0.85))
    state.cl_num = w * state.cl_num + lam_k * beta_obs
    state.cl_den = w * state.cl_den + lam_k * lam_k
    state.cl_n_obs += 1
    mu = float(getattr(cfg, "rm_cl_mu", 1.0))
    prior = float(getattr(cfg, "rm_cl_prior", 0.5))
    state.cl_beta_hat = (mu * prior + state.cl_num) / (mu + state.cl_den)
    e = y - (beta_used * lam_k) * dr
    rk_l1 = float(new_res.abs().float().sum().item()) + 1e-8
    state.cl_diag = {
        "cl_beta_obs": beta_obs, "cl_beta_hat": float(state.cl_beta_hat),
        "cl_beta_used_pre": beta_used, "cl_lambda_obs": float(lam_k),
        "cl_n_obs": state.cl_n_obs,
        "cl_innov_rel": float(e.abs().sum().item()) / rk_l1,
        "cl_frozen_rel": float(y.abs().sum().item()) / rk_l1,   # innovation of β=0 (ZOH)
    }


def _residual_motion(state: "FluxCacheState", h_filt: torch.Tensor, sigma: float,
                     image_ids: torch.Tensor, cfg, sigma_target: float | None = None):
    """Predict the moved residual and return (r_pred, diag).

    First order (E58):        r_pred = r_anchor + β1·λ·P1(Δr_a)
    Second order  (E59 'uniform', Newton backward on 3 anchors, uniform spacing):
                              r_pred = r_anchor + β1·λ·P1(Δr_a) + β2·λ(λ+1)/2·P2(Δ²r_a)
    Quadratic     (E59 'quad', exact nonuniform Lagrange through (σ_k, r_k), damped):
                              r_pred = r_anchor + β_quad·(r_quad(σ) − r_anchor)
    β2 = 0 (or missing anchors) keeps the E58 first-order path bit-identical."""
    r_anchor = state.prev_residual
    dr = r_anchor - state.r_prev
    P = _project_residual(dr, image_ids, sigma, cfg)
    lam_pt, lam_s, lam_a, lam_h = _rm_lambda(state, h_filt, sigma, cfg)
    # E60 midpoint-λ: evaluate the progress coefficient at the stride midpoint so the cached
    # step's velocity is a midpoint-rule quadrature of the moving residual path r(σ).
    lambda_eval = getattr(cfg, "rm_lambda_eval", "point")
    if lambda_eval == "mid" and sigma_target is not None:
        lam = _rm_lambda(state, h_filt, 0.5 * (sigma + float(sigma_target)), cfg)[0]
    else:
        lam = lam_pt
    # E60 closed-loop gain: β̂ from the online innovation fit instead of the fixed rm_beta
    beta_mode = getattr(cfg, "rm_beta_mode", "fixed")
    beta = _cl_beta_used(state, cfg) if beta_mode == "cl" else float(getattr(cfg, "rm_beta", 0.5))
    denom = float(r_anchor.abs().float().sum().item()) + 1e-8
    secant_norm = float(P.abs().float().sum().item()) / denom
    motion = beta * lam * P
    # ---- E59 second-order residual hold ----
    so_mode = getattr(cfg, "rm_so_mode", "none")
    beta2 = float(getattr(cfg, "rm_beta2", 0.0))
    so_coeff = lam * (lam + 1.0) / 2.0
    so_used = False
    so_gated_off = False
    so_term_ratio = 0.0
    if so_mode != "none" and state.r_prev2 is not None:
        d = state.so_diag or {}
        gamma = getattr(cfg, "rm_so_gate_gamma", None)
        rho_max = getattr(cfg, "rm_so_gate_rho_max", None)
        gate_ok = ((gamma is None or d.get("so_cos_delta", 1.0) > float(gamma)) and
                   (rho_max is None or d.get("so_rho2", 0.0) < float(rho_max)))
        if not gate_ok:
            so_gated_off = True
        elif so_mode == "uniform" and beta2 != 0.0:
            d2r = r_anchor - 2.0 * state.r_prev + state.r_prev2
            P2 = _project_residual(d2r, image_ids, sigma, cfg,
                                   proj=getattr(cfg, "rm_projection2", "raw"))
            so_term = (beta2 * so_coeff) * P2
            so_term_ratio = float(so_term.abs().float().sum().item()) / denom
            motion = motion + so_term
            so_used = True
        elif so_mode == "quad":
            # Lagrange quadratic in σ through the three fresh anchors, evaluated at σ_i;
            # replaces the first-order term entirely (damped toward r_anchor by β_quad).
            s0, s1, s2 = float(state.sigma_prev2), float(state.sigma_prev), float(state.sigma_anchor)
            si = float(sigma)
            eps = 1e-12
            L0 = ((si - s1) * (si - s2)) / ((s0 - s1) * (s0 - s2) + eps)
            L1 = ((si - s0) * (si - s2)) / ((s1 - s0) * (s1 - s2) + eps)
            L2 = ((si - s0) * (si - s1)) / ((s2 - s0) * (s2 - s1) + eps)
            r_quad = L0 * state.r_prev2 + L1 * state.r_prev + L2 * r_anchor
            bq = float(getattr(cfg, "rm_beta_quad", 0.5))
            motion = bq * (r_quad - r_anchor)
            so_term_ratio = float(motion.abs().float().sum().item()) / denom
            so_used = True
    extrap_ratio = float(motion.abs().float().sum().item()) / denom
    cancelled = False
    rho = float(getattr(cfg, "rm_gate_rho", 0.0) or 0.0)
    if rho > 0 and extrap_ratio > rho:
        motion = torch.zeros_like(motion)
        extrap_ratio = 0.0
        cancelled = True
    r_pred = r_anchor + motion
    diag = {
        "rm_used": True, "rm_was_cancelled": cancelled, "rm_beta": beta,
        "rm_projection": getattr(cfg, "rm_projection", "raw"), "rm_lambda": lam,
        "rm_lambda_sigma": lam_s, "rm_lambda_age": lam_a, "rm_lambda_h": lam_h,
        "rm_residual_secant_norm": secant_norm, "rm_residual_extrapolation_ratio": extrap_ratio,
    }
    if beta_mode == "cl" or lambda_eval != "point":   # E60 fields
        diag.update({
            "rm_beta_mode": beta_mode, "rm_lambda_eval": lambda_eval,
            "rm_lambda_point": lam_pt,
            "rm_cl_beta_hat": (None if state.cl_beta_hat is None else float(state.cl_beta_hat)),
            "rm_cl_n_obs": state.cl_n_obs,
        })
    if so_mode != "none":
        diag.update({
            "rm_so_mode": so_mode, "rm_beta2": beta2, "rm_so_used": so_used,
            "rm_so_gated_off": so_gated_off, "rm_so_coeff": so_coeff,
            "rm_so_term_ratio": so_term_ratio,
        })
        if state.so_diag:
            diag.update(state.so_diag)
    return r_pred, diag


def _rm_event_kv(rm_diag, feat, cfg) -> dict[str, Any]:
    """Map a residual-motion diagnostic dict onto the JumpEvent's rm_* fields."""
    if not rm_diag:
        return {}
    head = cfg.headroom(feat.get("acc_rel_l1")) if cfg is not None else None
    er = rm_diag["rm_residual_extrapolation_ratio"]
    kv = {
        "rm_used": True, "rm_was_cancelled": rm_diag["rm_was_cancelled"],
        "rm_beta": rm_diag["rm_beta"], "rm_projection": rm_diag["rm_projection"],
        "rm_lambda": rm_diag["rm_lambda"], "rm_lambda_sigma": rm_diag["rm_lambda_sigma"],
        "rm_lambda_age": rm_diag["rm_lambda_age"], "rm_lambda_h": rm_diag["rm_lambda_h"],
        "rm_residual_secant_norm": rm_diag["rm_residual_secant_norm"],
        "rm_residual_extrapolation_ratio": er,
        "rm_motion_per_headroom": (er / (head + 1e-8)) if head is not None else None,
    }
    if "rm_so_mode" in rm_diag:   # E59 second-order fields
        kv.update({k: rm_diag.get(k) for k in
                   ("rm_so_mode", "rm_beta2", "rm_so_used", "rm_so_gated_off", "rm_so_coeff",
                    "rm_so_term_ratio", "so_rho2", "so_cos_delta", "so_rho_anchor")})
    if "rm_beta_mode" in rm_diag:   # E60 closed-loop fields
        kv.update({k: rm_diag.get(k) for k in
                   ("rm_beta_mode", "rm_lambda_eval", "rm_lambda_point",
                    "rm_cl_beta_hat", "rm_cl_n_obs")})
    return kv


@torch.no_grad()
def _flux_node(pipe, tr, latents, timestep, guidance, ppe, pe, text_ids, image_ids,
               sigma: float, do_full: bool, state: FluxCacheState,
               rm_cfg=None, ledger=None, sigma_target: float | None = None):
    """Run one FLUX forward. Returns (velocity, filtered_h, ran_full, rm_diag).

    rm_cfg (E58): when set with rm_enabled, a cached forward uses a predicted (moved) block
    residual r_pred instead of the frozen r_anchor. Pure tensor arithmetic — no extra block
    stack. On a fresh forward it also shifts the residual/sigma/h history (r_prev←r_anchor).
    Passing rm_cfg=None keeps the E56/E57 path byte-identical."""
    rm_on = bool(rm_cfg is not None and getattr(rm_cfg, "rm_enabled", False))
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

    def _full_stack(x0):
        """Run the whole block stack from x0 and return the residual (cur - x0)."""
        cur = x0
        e = enc
        for block in tr.transformer_blocks:
            e, cur = block(hidden_states=cur, encoder_hidden_states=e, temb=temb,
                           image_rotary_emb=rotary, joint_attention_kwargs=None)
        for block in tr.single_transformer_blocks:
            e, cur = block(hidden_states=cur, encoder_hidden_states=e, temb=temb,
                           image_rotary_emb=rotary, joint_attention_kwargs=None)
        return cur

    rm_diag = None
    if (not do_full) and state.prev_residual is not None:
        if rm_on and state.r_prev is not None:
            r_pred, rm_diag = _residual_motion(state, h_filt, sigma, image_ids, rm_cfg,
                                               sigma_target=sigma_target)
            if ledger is not None:
                ledger.record_resmotion(cancelled=rm_diag["rm_was_cancelled"],
                                        so_used=rm_diag.get("rm_so_used", False),
                                        so_gated_off=rm_diag.get("rm_so_gated_off", False))
            # optional oracle: score r_pred against the TRUE residual at this cached state
            if getattr(rm_cfg, "rm_oracle_diag", False):
                r_true = (_full_stack(hs) - hs).detach()
                if ledger is not None:
                    ledger.record_rm_oracle()
                dn = float(r_true.abs().float().sum().item()) + 1e-8
                rm_diag["rm_oracle_frozen_err"] = float((state.prev_residual - r_true).abs().float().sum().item()) / dn
                rm_diag["rm_oracle_motion_err"] = float((r_pred - r_true).abs().float().sum().item()) / dn
            hs2 = hs + r_pred
        else:
            hs2 = hs + state.prev_residual
        ran_full = False
    else:
        ori = hs
        cur = _full_stack(hs)
        new_res = (cur - ori).detach()
        if rm_on and state.prev_residual is not None:
            # E60 closed-loop: absorb this anchor's innovation into β̂ BEFORE the history shift
            # (needs the outgoing pair r_{k-1}, r_{k-2} plus the new residual r_k)
            if (getattr(rm_cfg, "rm_beta_mode", "fixed") == "cl"
                    and state.r_prev is not None):
                _cl_update(state, new_res, h_filt, sigma, rm_cfg)
            # shift fresh-residual history for the secant (penultimate ← previous anchor);
            # E59 second-order additionally keeps the third anchor r_{a-2}
            if getattr(rm_cfg, "rm_so_mode", "none") != "none":
                state.r_prev2 = state.r_prev
                state.sigma_prev2 = state.sigma_prev
            state.r_prev = state.prev_residual
            state.sigma_prev = state.sigma_anchor
            state.h_prev_filt = state.h_anchor_filt
            state.steps_prev_to_anchor = max(1, state.steps_since_anchor)
        state.prev_residual = new_res
        if rm_on:
            state.sigma_anchor = float(sigma)
            state.h_anchor_filt = h_filt.detach()
            state.steps_since_anchor = 0
            # E59: refresh the anchor-triple curvature diagnostics at every new anchor
            if (getattr(rm_cfg, "rm_so_mode", "none") != "none"
                    and state.r_prev is not None and state.r_prev2 is not None):
                state.so_diag = _anchor_triple_diag(new_res, state.r_prev, state.r_prev2)
        hs2 = cur
        ran_full = True

    hs2 = tr.norm_out(hs2, temb)
    v = tr.proj_out(hs2)
    return v, h_filt, ran_full, rm_diag


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
        cfg = getattr(policy, "cfg", None)
        do_full = (action == "fresh")
        # E60 midpoint-λ needs the step's intended endpoint BEFORE the forward; mirror the
        # scheduler's target computation below (a PC 'shrink' would move it afterwards, but
        # PC and RM are never combined in any variant).
        prov_target = sigma_next
        if action.startswith("jump"):
            jf_prov = adaptive_jf if adaptive_jf is not None else JUMP_FACTORS[action]
            if (adaptive_jf is None and cfg is not None
                    and getattr(cfg, "jump_mode", "regrid") == "drop"):
                prov_target = sigmas[min(i + 2, len(sigmas) - 1)]
            else:
                prov_target = max(0.0, sigma + jf_prov * (sigma_next - sigma))
        v, _hf, ran_full, rm_diag = _flux_node(pipe, tr, latents, timestep, guidance_t, ppe, pe,
                                               text_ids, image_ids, sigma, do_full, state,
                                               rm_cfg=cfg, ledger=ledger,
                                               sigma_target=prov_target)

        # --- scheduler update ---
        skipped = 0
        regridded = False
        target_sigma = sigma_next
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

                oracle = bool(getattr(cfg, "pc_endpoint_fresh", False))

                def _endpoint_v(x_end, sig_end):
                    tstep = torch.full((latents.shape[0],), sig_end, device=device, dtype=latents.dtype)
                    if oracle:
                        # fresh endpoint = a real full forward; do NOT let it overwrite the
                        # cache anchor (save/restore prev_residual). Accounted as a full forward.
                        saved = state.prev_residual
                        vv, _, _, _ = _flux_node(pipe, tr, x_end, tstep, guidance_t, ppe, pe,
                                              text_ids, image_ids, sig_end, True, state)
                        state.prev_residual = saved
                        ledger.record_pc_oracle()
                    else:
                        vv, _, _, _ = _flux_node(pipe, tr, x_end, tstep, guidance_t, ppe, pe,
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
            rm_kv = _rm_event_kv(rm_diag, feat, cfg)
            jumps.append(JumpEvent(
                step_index, ("jump_cancelled" if pc["cancelled"] else action), jf, sigma,
                target_sigma, skipped, regridded, action_source,
                sigma_next_original=sigma_next_orig, alpha=pc["alpha"], headroom=pc["headroom"],
                accumulated_score=feat.get("acc_rel_l1"), v_start_norm=pc["v0n"],
                v_endpoint_norm=pc["ven"], curvature_l1=pc["curv_l1"], curvature_l2=pc["curv_l2"],
                pc_correction_norm=pc["pc_norm"], was_cancelled=pc["cancelled"],
                was_shrunk=pc["shrunk"], effective_skipped_nodes=skipped, **rm_kv).as_dict())
        else:
            latents = latents + (sigma_next - sigma) * v
            i += 1

        ledger.record(action, skipped=skipped)
        # update SeaCache/cache bookkeeping
        state.prev_h_filt = h_prev_store
        state.prev_h_raw = mod_raw
        state.prev_h_norm = feat["h_norm"]
        state.refresh_distance = 0 if action == "fresh" else state.refresh_distance + 1
        state.steps_since_anchor += 1     # E58 step-age (reset to 0 inside a fresh forward)
        state.prev_action = action

        trace = {"step_index": step_index, "node_i_after": i, "action": action,
                 "sigma": sigma, "sigma_next_target": target_sigma, "ran_full": ran_full,
                 **feat}
        if rm_diag is not None:           # E58 per-step residual-motion diagnostics
            hr = feat.get("acc_rel_l1")
            head = cfg.headroom(hr) if cfg is not None else None
            trace.update(rm_diag)
            trace["rm_motion_per_headroom"] = (
                rm_diag["rm_residual_extrapolation_ratio"] / (head + 1e-8) if head is not None else None)
        elif (ran_full and cfg is not None and getattr(cfg, "rm_so_mode", "none") != "none"
              and state.so_diag is not None):
            # E59: anchor-triple curvature diagnostics logged at the fresh anchor that formed them
            trace.update(state.so_diag)
        if (ran_full and cfg is not None and getattr(cfg, "rm_beta_mode", "fixed") == "cl"
                and state.cl_diag is not None):
            # E60: β̂/innovation observation logged at the fresh anchor that produced it
            trace.update(state.cl_diag)
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
