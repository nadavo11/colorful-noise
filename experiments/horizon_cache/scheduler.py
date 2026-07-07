"""Jump scheduler mechanics + compute accounting for HorizonCache.

Two jump modes (Euler / first-order flow sampler only — see deck slide "Two ways to
take the bigger step"):

  drop   : on-grid jump. Skip an existing later node and land back on the original
           schedule (sigma_i -> sigma_{i+2}).
  regrid : off-grid jump. sigma_target = sigma_i + jf*(sigma_{i+1} - sigma_i); because
           sigma decreases this lands farther toward 0. The remaining tail is then
           re-spaced from sigma_target to 0, preserving the endpoint (net -1 node).

Everything that touches the sigma schedule is logged so the report can reconstruct
exactly what was skipped (deck fair-comparison rule: report *achieved* speedup, never
nominal only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ----- action set (deck: "Three actions, two errors" + the jump-factor sweep) -----
ACTIONS = ["fresh", "cache", "jump_1.25", "jump_1.5", "jump_2.0"]
JUMP_FACTORS = {"jump_1.25": 1.25, "jump_1.5": 1.5, "jump_2.0": 2.0}
# ordinal safe-horizon labels (0 == must refresh)
HORIZON_LEVELS = [0.0, 1.0, 1.25, 1.5, 2.0]


def action_to_horizon(action: str) -> float:
    if action == "fresh":
        return 0.0
    if action == "cache":
        return 1.0
    return JUMP_FACTORS[action]


def horizon_to_action(h: float) -> str:
    # nearest ordinal level
    lvl = min(HORIZON_LEVELS, key=lambda x: abs(x - h))
    if lvl == 0.0:
        return "fresh"
    if lvl == 1.0:
        return "cache"
    return {1.25: "jump_1.25", 1.5: "jump_1.5", 2.0: "jump_2.0"}[lvl]


@dataclass
class ComputeLedger:
    """Achieved-compute accounting. `L` is the #transformer blocks whose stack is the
    ~entire forward FLOPs (FLUX 57, SD3 24); a cached/jump node pays ~1/L of a fresh one.
    """
    L: int = 57
    num_full_forwards: int = 0
    num_cached_forwards: int = 0
    num_jump_actions: int = 0
    num_skipped_nodes: int = 0
    num_executed_nodes: int = 0
    baseline_nodes: int = 0          # nodes a full (no-cache) run would execute
    measured_wall_time: float = 0.0
    baseline_wall_time: float = 0.0
    # HorizonCache-PC (E57): a cached endpoint velocity call is an EXTRA cached forward
    # (~1/L) — accounted separately so PC never claims free speedup.
    num_pc_endpoint_cached_forwards: int = 0
    num_pc_oracle_full_forwards: int = 0   # oracle ablation: fresh endpoint = a full forward
    num_cancelled_jumps: int = 0
    num_shrunk_jumps: int = 0
    # E58 Residual Motion Cache: r_pred extrapolation is pure tensor arithmetic (NO extra
    # forward), so it does not change block_stack_equiv_cost — we only COUNT applications
    # and (for the oracle diagnostic) any true-residual full forwards, tracked separately.
    num_residual_motion_applications: int = 0
    num_residual_motion_cancelled: int = 0
    num_rm_oracle_full_forwards: int = 0    # diagnostic-only true-residual evals (not in deploy cost)

    def record(self, action: str, skipped: int = 0) -> None:
        self.num_executed_nodes += 1
        if action == "fresh":
            self.num_full_forwards += 1
        else:
            self.num_cached_forwards += 1
        if action.startswith("jump"):
            self.num_jump_actions += 1
        self.num_skipped_nodes += skipped

    def record_pc_endpoint(self, n: int = 1) -> None:
        """One extra cached forward for the predictor-corrector endpoint velocity."""
        self.num_pc_endpoint_cached_forwards += n

    def record_pc_oracle(self, n: int = 1) -> None:
        """Oracle ablation: endpoint velocity is a FULL forward (full block stack ≈ cost 1)."""
        self.num_pc_oracle_full_forwards += n

    def record_resmotion(self, cancelled: bool = False) -> None:
        """One residual-motion application on a cached/jump step (free tensor arithmetic)."""
        self.num_residual_motion_applications += 1
        if cancelled:
            self.num_residual_motion_cancelled += 1

    def record_rm_oracle(self, n: int = 1) -> None:
        """Diagnostic true-residual full forward (measured, NOT charged to deploy cost)."""
        self.num_rm_oracle_full_forwards += n

    @property
    def block_stack_equiv_cost(self) -> float:
        """Compute in units of one fresh forward: fresh=1, cached/jump/PC-endpoint≈1/L,
        oracle fresh endpoint = a full forward (≈1)."""
        return (self.num_full_forwards + self.num_pc_oracle_full_forwards
                + (self.num_cached_forwards + self.num_pc_endpoint_cached_forwards) / float(self.L))

    @property
    def compute_speedup(self) -> float:
        """Achieved speedup by block-stack-equivalent cost vs a full baseline run."""
        c = self.block_stack_equiv_cost
        return (self.baseline_nodes / c) if c > 0 else float("nan")

    @property
    def wall_speedup(self) -> float:
        return (self.baseline_wall_time / self.measured_wall_time) if self.measured_wall_time > 0 else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "L": self.L,
            "num_full_forwards": self.num_full_forwards,
            "num_cached_forwards": self.num_cached_forwards,
            "num_jump_actions": self.num_jump_actions,
            "num_skipped_nodes": self.num_skipped_nodes,
            "num_executed_nodes": self.num_executed_nodes,
            "baseline_nodes": self.baseline_nodes,
            "block_stack_equiv_cost": round(self.block_stack_equiv_cost, 4),
            "compute_speedup": round(self.compute_speedup, 4),
            "measured_wall_time": round(self.measured_wall_time, 4),
            "baseline_wall_time": round(self.baseline_wall_time, 4),
            "wall_speedup": round(self.wall_speedup, 4) if self.measured_wall_time > 0 else None,
            "num_pc_endpoint_cached_forwards": self.num_pc_endpoint_cached_forwards,
            "num_pc_oracle_full_forwards": self.num_pc_oracle_full_forwards,
            "num_cancelled_jumps": self.num_cancelled_jumps,
            "num_shrunk_jumps": self.num_shrunk_jumps,
            "num_residual_motion_applications": self.num_residual_motion_applications,
            "num_residual_motion_cancelled": self.num_residual_motion_cancelled,
            "num_rm_oracle_full_forwards": self.num_rm_oracle_full_forwards,
        }


def regrid_tail(sigmas: list[float], i: int, target: float) -> list[float]:
    """Return a new sigma list where node i+1 is replaced by `target` and the remaining
    tail (target .. 0) is re-spaced over one *fewer* node, preserving the 0 endpoint.

    `sigmas` is the current full node list (decreasing, ends at 0.0). Node i is where we
    currently sit. The count of nodes strictly after i drops by one (that is the compute
    we actually save).
    """
    head = sigmas[: i + 1]
    n_tail = len(sigmas) - (i + 1)          # old nodes after i, incl. the terminal 0
    # new tail runs target -> 0 over ONE FEWER node (that removed node is the saving)
    n_new = n_tail - 1
    if n_new <= 1:
        # nothing left to re-space into; just land on the terminal 0
        return head + [0.0]
    new_tail = [target * (1.0 - k / (n_new - 1)) for k in range(n_new)]
    new_tail[0] = target
    new_tail[-1] = 0.0
    return head + new_tail


@dataclass
class JumpEvent:
    step: int
    action: str
    jump_factor: float
    old_sigma: float
    target_sigma: float
    skipped_equiv_nodes: int
    regridded: bool
    action_source: str  # "v0" | "v1" | "seacache" | "fixed"
    # --- HorizonCache-PC (E57) diagnostics; None on plain Euler jumps ---
    sigma_next_original: float | None = None
    alpha: float | None = None
    headroom: float | None = None
    accumulated_score: float | None = None
    v_start_norm: float | None = None
    v_endpoint_norm: float | None = None
    curvature_l1: float | None = None
    curvature_l2: float | None = None
    pc_correction_norm: float | None = None
    was_cancelled: bool = False
    was_shrunk: bool = False
    effective_skipped_nodes: int | None = None
    # --- E58 Residual Motion Cache diagnostics; None when RM disabled ---
    rm_lambda: float | None = None
    rm_lambda_sigma: float | None = None
    rm_lambda_age: float | None = None
    rm_lambda_h: float | None = None
    rm_beta: float | None = None
    rm_projection: str | None = None
    rm_residual_secant_norm: float | None = None
    rm_residual_extrapolation_ratio: float | None = None
    rm_motion_per_headroom: float | None = None
    rm_used: bool = False
    rm_was_cancelled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
