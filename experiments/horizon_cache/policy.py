"""HorizonCache policies — v0 (hand-designed causal rule) and v1 (learned tabular).

Both consume the SAME causal feature dict emitted per node by the sampler and return an
action in scheduler.ACTIONS. The feature set is intentionally cheap: everything is read
before/around the block stack (sigma geometry + the SeaCache modulated-input signal),
never from a decoded image or a future state.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .scheduler import ACTIONS, JUMP_FACTORS

# Ordered causal feature names — the contract between sampler, policy, and v1 dataset.
FEATURE_NAMES = [
    "sigma",                 # current sigma
    "delta_sigma",           # sigma_next - sigma (negative; magnitude = stride)
    "progress",              # step_index / num_steps
    "remaining_steps",       # nodes left on the schedule
    "raw_rel_l1",            # SeaCache filtered rel-L1 of h vs previous
    "acc_rel_l1",            # accumulated SeaCache score since last refresh
    "h_norm",                # mean |filtered h|
    "h_norm_drift",          # |h_norm - prev_h_norm| / prev_h_norm
    "h_cos_drift",           # 1 - cos(h_t, h_{t-1})  (0 = perfectly aligned)
    "refresh_distance",      # nodes since last fresh eval
    "prev_action_code",      # 0 fresh,1 cache,2 jump
    # editing-only (0.0 in generation)
    "src_rel_l1",
    "tar_rel_l1",
    "cos_hsrc_htar",
    "rel_l1_src_tar",
    "branch_asymmetry",
    "dv_norm_proxy",
]

_ACTION_CODE = {"fresh": 0, "cache": 1}


def action_code(action: str) -> int:
    return _ACTION_CODE.get(action, 2)  # any jump -> 2


def feature_vector(feat: dict[str, float]) -> list[float]:
    return [float(feat.get(k, 0.0)) for k in FEATURE_NAMES]


@dataclass
class HorizonV0Config:
    """Sweepable thresholds. Disabling jumps (enable_jump=False) reduces v0 to SeaCache
    exactly — same fresh/cache boundary — which is the fair-by-identity baseline.

    Jump aggressiveness is driven by *headroom* = 1 - acc/tau_cache (deck adaptive-jump:
    jf = 1+(jf_max-1)(1-acc/tau)) — the field is flattest and safest to over-step right
    after a refresh, when the accumulated staleness budget is nearly full. This is the
    correct signal: the raw per-step relL1 (~0.1-0.25) makes an absolute acc ceiling
    essentially never fire."""
    tau_cache: float = 0.30          # SeaCache refresh threshold (acc >= tau -> fresh)
    enable_jump: bool = True
    allow_jump2: bool = False        # jump_2.0 is an ablation, off in the primary frontier
    jf_max_frontier: float = 1.5     # cap for the conservative frontier (discrete mode)
    adaptive: bool = False           # deck adaptive-jump: continuous stride, not a discrete pick
    jf_max: float = 1.25             # adaptive cap: jf = 1 + (jf_max-1)*headroom
    # headroom needed to authorize each jump factor (larger factor needs more headroom)
    jump_headroom: dict = field(default_factory=lambda: {"jump_1.25": 0.45, "jump_1.5": 0.65, "jump_2.0": 0.82})
    raw_rel_l1_max: float = 0.18     # instantaneous staleness must be small to over-step
    sigma_jump_lo: float = 0.10      # never jump in the very last (fine-detail) sigmas
    sigma_jump_hi: float = 0.90      # never jump in the volatile early sigmas
    cos_drift_max: float = 0.02      # need h well-aligned to prev (small drift) to jump
    min_remaining: int = 3           # keep enough tail nodes to regrid into
    jump_mode: str = "regrid"        # "regrid" | "drop"
    # editing safety: block a jump right after a branch-alignment drop
    cos_branch_min: float = 0.30     # require cos(h_src,h_tar) above this to jump
    branch_drop_block: float = 0.10  # if cos dropped by more than this vs prev, no jump
    # --- HorizonCache-PC (E57): cached-endpoint predictor-corrector jump ---
    # x_corr = x_i + Δσ·[(1-α)·v_i + α·v_pred], v_pred = cached velocity at (x_pred, σ_target)
    # reusing the SAME cached residual (one extra cached forward ≈ 1/L, accounted separately).
    pc_enabled: bool = False
    pc_alpha: float = 0.5            # 0 → plain Euler, 0.5 → trapezoid, 1.0 → endpoint-only
    pc_curvature_mode: str = "none"  # "none" | "cancel" | "shrink"
    pc_curvature_kappa: float = 0.06  # curvature (relL1 of v_pred vs v_i) gate threshold
    pc_shrink_factor: float = 0.5    # shrink rule: jf ← 1 + shrink_factor·(jf-1)
    pc_endpoint_fresh: bool = False  # ORACLE ablation: endpoint velocity is a FULL forward
    #                                  (costs a full block-stack call, accounted as such) —
    #                                  isolates whether the cached endpoint's staleness is the cause.
    # --- E58 Residual Motion Cache ---
    # On cached/jump steps, predict the slow motion of the block residual instead of freezing it:
    #   r_pred = r_anchor + β·λ(t)·P(r_anchor - r_prev);  v = head(front(x_t,σ_t) + r_pred).
    # Pure tensor arithmetic — NO extra block-stack forward (unlike E57 PC). β=0 ≡ plain HorizonCache.
    rm_enabled: bool = False
    rm_beta: float = 0.5             # shrink factor on the extrapolation (0 → frozen residual)
    rm_lambda_mode: str = "sigma"    # "sigma" | "age" | "h" — progress coefficient λ
    rm_lambda_max: float = 1.5       # clamp λ ∈ [0, rm_lambda_max]
    rm_projection: str = "raw"       # "raw" | "lowpass" | "topk" | "sea" — P(Δr)
    rm_topk_frac: float = 0.25       # topk: keep this fraction of the most energetic channels
    rm_lowpass_pool: int = 2         # lowpass: avg-pool kernel over the token grid
    rm_gate_rho: float = 0.0         # optional safety gate: cancel motion if extrap ratio > rho (0 = off)
    rm_oracle_diag: bool = False     # diagnostic only: also run the TRUE block residual to score r_pred
    # --- E59 second-order residual hold ---
    # Three fresh anchors r_{a-2}, r_{a-1}, r_a. 'uniform' adds a Newton-backward curvature term:
    #   r_pred = r_a + β1·λ·P1(Δr_a) + β2·λ(λ+1)/2·P2(Δ²r_a),  Δ²r_a = r_a − 2r_{a-1} + r_{a-2}
    # 'quad' fits the exact σ-nonuniform Lagrange quadratic through the three anchors and damps:
    #   r_pred = r_a + β_quad·(r_quad(σ) − r_a)   (replaces the first-order term)
    # rm_so_mode='none' or β2=0 (uniform) recovers E58 first-order RM exactly. Still NO extra forward.
    rm_so_mode: str = "none"         # "none" | "uniform" | "quad"
    rm_beta2: float = 0.0            # curvature shrink (uniform mode); 0 ≡ first-order RM
    rm_beta_quad: float = 0.5        # damping toward r_anchor (quad mode)
    rm_projection2: str = "raw"      # P2 applied to Δ²r (same options as rm_projection)
    rm_so_gate_gamma: float | None = None    # enable SO only if cos(Δr_a, Δr_{a-1}) > γ
    rm_so_gate_rho_max: float | None = None  # ... and ρ2 = ‖Δ²r‖₁/‖Δr_a‖₁ < ρ_max

    def headroom(self, acc: float) -> float:
        return max(0.0, 1.0 - acc / max(1e-9, self.tau_cache))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class HorizonCacheV0:
    """Deterministic causal rule. Always fresh at the first and last node."""

    def __init__(self, cfg: HorizonV0Config, editing: bool = False):
        self.cfg = cfg
        self.editing = editing

    def _jump_allowed(self, feat: dict[str, float]) -> bool:
        """Hard safety gate (independent of which factor). Never jump when the field is
        volatile or too close to the schedule endpoints."""
        c = self.cfg
        if not c.enable_jump:
            return False
        s = feat["sigma"]
        if not (c.sigma_jump_lo <= s <= c.sigma_jump_hi):
            return False
        if feat["remaining_steps"] < c.min_remaining:
            return False
        if feat["raw_rel_l1"] > c.raw_rel_l1_max:
            return False
        if feat["h_cos_drift"] > c.cos_drift_max:
            return False
        if self.editing:
            if feat.get("cos_hsrc_htar", 1.0) < c.cos_branch_min:
                return False
            if feat.get("branch_cos_drop", 0.0) > c.branch_drop_block:
                return False
        return True

    def act(self, feat: dict[str, float], step_index: int, num_steps: int) -> str:
        # forced-fresh anchors (matches SeaCache: first + last always fresh)
        if step_index == 0 or step_index >= num_steps - 1:
            return "fresh"
        c = self.cfg
        acc = feat["acc_rel_l1"]
        if acc >= c.tau_cache:
            return "fresh"
        # reuse territory: pick the largest jump the headroom authorizes, else cache
        if self._jump_allowed(feat):
            head = c.headroom(acc)
            if c.adaptive:
                # deck adaptive-jump: continuous stride scaled by flatness, capped at jf_max
                jf = 1.0 + (c.jf_max - 1.0) * head
                return ("jump", jf) if jf > 1.01 else "cache"
            candidates = ["jump_1.25", "jump_1.5"]
            if c.allow_jump2:
                candidates.append("jump_2.0")
            best = "cache"
            for a in candidates:  # ascending factor -> keep the largest that passes
                if (not c.allow_jump2) and JUMP_FACTORS[a] > c.jf_max_frontier:
                    continue
                if head >= c.jump_headroom.get(a, 1.0):
                    best = a
            return best
        return "cache"


class HorizonCacheV1:
    """Learned tabular policy. Loaded from a joblib bundle produced by train_v1.py.
    Applies the same hard safety gate as v0 on top of the model's prediction so a
    mis-prediction can never take a jump the safety rule forbids (asymmetric cost:
    a false jump is far more damaging than a false cache)."""

    def __init__(self, bundle: dict[str, Any], safety_cfg: HorizonV0Config, editing: bool = False):
        self.model = bundle["model"]
        self.classes = list(bundle["classes"])
        self.mode = bundle.get("mode", "classify")
        self.safety = HorizonCacheV0(safety_cfg, editing=editing)
        self.editing = editing

    def act(self, feat: dict[str, float], step_index: int, num_steps: int) -> str:
        if step_index == 0 or step_index >= num_steps - 1:
            return "fresh"
        import numpy as np

        x = np.array([[float(feat.get(k, 0.0)) for k in FEATURE_NAMES]], dtype=np.float64)
        pred = self.model.predict(x)[0]
        if self.mode == "ordinal":
            from .scheduler import horizon_to_action
            action = horizon_to_action(float(pred))
        else:
            action = str(pred)
        if action not in ACTIONS:
            action = "cache"
        # hard safety gate: never take a forbidden jump; downgrade to cache
        if action.startswith("jump") and not self.safety._jump_allowed(feat):
            action = "cache"
        # refresh discipline: if SeaCache score already says refresh, obey it
        if feat["acc_rel_l1"] >= self.safety.cfg.tau_cache:
            action = "fresh"
        return action
