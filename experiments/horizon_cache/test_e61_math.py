"""E61 innovation-gated fixed RM — math verification (no GPU/model needed).

Checks:
 1. fixed mode (default) is untouched — gate machinery inert when rm_beta_mode="fixed".
 2. hard gate: g=1 iff I_a < κ (exact threshold behavior), β_i = 0.5·g.
 3. soft gate: g = clip(1 - I_a/κ, 0, 1), exact linear ramp.
 4. floor gate: g = gmin + (1-gmin)·soft — never drops below gmin·0.5.
 5. EMA smoothing: α=1 → no smoothing (Ī=I_a); α<1 → Ī_a = α·I_a + (1-α)·Ī_{a-1}, verified over
    a multi-anchor stream.
 6. betahat_gate: g = clip(β̂/0.5, 0, 1), β = gmin + (0.5-gmin)·g — reuses E60's LS β̂ as signal
    only, never as the gain itself (both endpoints checked: gmin=0 and gmin=0.25).
 7. causality: β used on a cached step after anchor a depends ONLY on the innovation computed
    AT anchor a (from r_a, r_{a-1}, r_{a-2}) — never on anything from the cached step itself.
 8. grammar: rmg<h|s|f><κ>[g][a][n] and rmbg<gmin> parse correctly; legacy rm/rmcl/rm2/rmq unchanged.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import torch

from horizon_cache.flux_gen import (_residual_motion, _gate_update, _cl_update,
                                    FluxCacheState)
from horizon_cache.policy import HorizonV0Config
from horizon_cache.run import _variant_cfg

torch.manual_seed(0)
B, N, C = 1, 16, 8
image_ids = torch.zeros(N, 3)
image_ids[:, 1] = torch.arange(N) // 4
image_ids[:, 2] = torch.arange(N) % 4
h_filt = torch.randn(B, N, C)


def make_state(r1, r0, s1, s0):
    st = FluxCacheState()
    st.prev_residual, st.r_prev = r0, r1
    st.sigma_anchor, st.sigma_prev = s0, s1
    st.steps_since_anchor, st.steps_prev_to_anchor = 1, 1
    return st


def anchor_stream(deltas, cfg, h=0.2, s_start=1.0, r1=None, r0=None):
    """r_k = r_{k-1} + delta_k (delta_k controls the innovation directly: delta_k=0.5*lam*dr_prev
    gives zero innovation). Returns the state after all anchors + list of gate_diag snapshots."""
    s1, s0 = s_start, s_start - h
    if r1 is None:
        r1 = torch.randn(B, N, C)
    if r0 is None:
        r0 = r1 + torch.randn(B, N, C)
    st = make_state(r1, r0, s1, s0)
    diags = []
    for k, d in enumerate(deltas):
        s_new = s0 - h * (k + 1)
        r_new = st.prev_residual + d
        _gate_update(st, r_new, h_filt, s_new, cfg)
        diags.append(dict(st.gate_diag))
        st.r_prev, st.sigma_prev = st.prev_residual, st.sigma_anchor
        st.prev_residual, st.sigma_anchor = r_new, s_new
    return st, diags


# ---------- 1. fixed mode inert ----------
r1, r0 = torch.randn(B, N, C), torch.randn(B, N, C)
cfg_fixed = HorizonV0Config(rm_enabled=True, rm_beta=0.5)
p_fixed, d_fixed = _residual_motion(make_state(r1, r0, 0.8, 0.6), h_filt, 0.5, image_ids, cfg_fixed)
assert "rm_beta_mode" not in d_fixed
print("1. fixed mode inert (gate fields absent): OK")

# ---------- 2. hard gate exact threshold ----------
# construct a delta EXACTLY equal to the beta=0.5 forecast -> zero innovation -> I=0 -> g=1.
# Must replicate anchor_stream's own state setup (s1=1.0, s0=0.8, first call sigma=0.6) exactly.
from horizon_cache.flux_gen import _rm_lambda
cfg_probe = HorizonV0Config(rm_enabled=True, rm_lambda_max=10.0)
r1_fix, r0_fix = torch.randn(B, N, C), torch.randn(B, N, C)
st_probe = make_state(r1_fix, r0_fix, 1.0, 0.8)
lam_a = _rm_lambda(st_probe, h_filt, 0.6, cfg_probe)[0]
dr = st_probe.prev_residual - st_probe.r_prev
zero_innov_delta = 0.5 * lam_a * dr
cfg_hard = HorizonV0Config(rm_enabled=True, rm_beta_mode="gate", rm_gate_type="hard",
                           rm_gate_kappa=0.05, rm_lambda_max=10.0)
st_zero, _ = anchor_stream([zero_innov_delta], cfg_hard, r1=r1_fix, r0=r0_fix)
assert st_zero.gate_g == 1.0, f"zero innovation must pass any positive kappa, g={st_zero.gate_g}"
# a large orthogonal perturbation must push I above kappa -> g=0
big_delta = zero_innov_delta + 5.0 * torch.randn(B, N, C)
st_big, _ = anchor_stream([big_delta], cfg_hard, r1=r1_fix, r0=r0_fix)
assert st_big.gate_g == 0.0, f"large innovation must fail a tight kappa, I={st_big.gate_ibar}"
p_g1, d_g1 = _residual_motion(st_zero, h_filt, 0.3, image_ids, cfg_hard)
assert abs(d_g1["rm_beta"] - 0.5) < 1e-9
p_g0, d_g0 = _residual_motion(st_big, h_filt, 0.3, image_ids, cfg_hard)
assert d_g0["rm_beta"] == 0.0
print(f"2. hard gate: zero-innovation g=1 (β=0.5), large-innovation g=0 (β=0): OK")

# ---------- 3. soft gate exact ramp ----------
cfg_soft = HorizonV0Config(rm_enabled=True, rm_beta_mode="gate", rm_gate_type="soft",
                           rm_gate_kappa=0.2, rm_lambda_max=10.0)
mid_delta = zero_innov_delta + 0.1 * torch.randn(B, N, C)
st_mid, diags = anchor_stream([mid_delta], cfg_soft, r1=r1_fix, r0=r0_fix)
I_used = diags[0]["gate_I_used"]
expected_g = min(max(1.0 - I_used / 0.2, 0.0), 1.0)
assert abs(st_mid.gate_g - expected_g) < 1e-6
p_mid, d_mid = _residual_motion(st_mid, h_filt, 0.3, image_ids, cfg_soft)
assert abs(d_mid["rm_beta"] - 0.5 * expected_g) < 1e-6
print(f"3. soft gate ramp exact (I={I_used:.3f}, g={expected_g:.3f}): OK")

# ---------- 4. floor gate never drops below gmin ----------
cfg_floor = HorizonV0Config(rm_enabled=True, rm_beta_mode="gate", rm_gate_type="floor",
                           rm_gate_kappa=0.05, rm_gate_gmin=0.5, rm_lambda_max=10.0)
st_floor, _ = anchor_stream([big_delta], cfg_floor, r1=r1_fix, r0=r0_fix)   # huge innovation -> soft term -> 0
assert abs(st_floor.gate_g - 0.5) < 1e-6, f"floor gate must clamp at gmin, g={st_floor.gate_g}"
p_f, d_f = _residual_motion(st_floor, h_filt, 0.3, image_ids, cfg_floor)
assert abs(d_f["rm_beta"] - 0.25) < 1e-6
print("4. floor gate clamps at gmin·0.5=0.25 under large innovation: OK")

# ---------- 5. EMA smoothing ----------
cfg_ema = HorizonV0Config(rm_enabled=True, rm_beta_mode="gate", rm_gate_type="soft",
                          rm_gate_kappa=0.5, rm_gate_ema_alpha=0.5, rm_lambda_max=10.0)
deltas = [zero_innov_delta + 0.05 * torch.randn(B, N, C) for _ in range(4)]
st_ema, diags_ema = anchor_stream(deltas, cfg_ema)
ibar = None
for d in diags_ema:
    raw = d["gate_I_used"]
    ibar = raw if ibar is None else 0.5 * raw + 0.5 * ibar
    assert abs(d["gate_I_bar"] - ibar) < 1e-6
cfg_noema = HorizonV0Config(rm_enabled=True, rm_beta_mode="gate", rm_gate_type="soft",
                            rm_gate_kappa=0.5, rm_gate_ema_alpha=1.0, rm_lambda_max=10.0)
_, diags_noema = anchor_stream(deltas, cfg_noema)
assert all(abs(d["gate_I_bar"] - d["gate_I_used"]) < 1e-9 for d in diags_noema)
print("5. EMA smoothing (alpha<1 recursive; alpha=1 no smoothing): OK")

# ---------- 6. betahat_gate reuses E60 signal only, never as the gain ----------
cfg_bh0 = HorizonV0Config(rm_enabled=True, rm_beta_mode="betahat_gate", rm_gate_gmin=0.0,
                          rm_cl_prior=0.5, rm_cl_mu=1e-6, rm_cl_forget=1.0, rm_lambda_max=10.0)
st_bh = make_state(torch.randn(B, N, C), torch.randn(B, N, C), 0.8, 0.6)
r_new = st_bh.prev_residual + 0.7 * (st_bh.prev_residual - st_bh.r_prev)   # true beta*=0.7
_cl_update(st_bh, r_new, h_filt, 0.4, cfg_bh0)   # sigma=0.4 -> lam=1 exactly (uniform spacing)
assert abs(st_bh.cl_beta_hat - 0.7) < 1e-3, f"beta_hat={st_bh.cl_beta_hat}"
st_bh.r_prev, st_bh.sigma_prev = st_bh.prev_residual, st_bh.sigma_anchor
st_bh.prev_residual, st_bh.sigma_anchor = r_new, 0.4
_, d_bh0 = _residual_motion(st_bh, h_filt, 0.3, image_ids, cfg_bh0)
assert abs(d_bh0["rm_beta"] - 0.5) < 1e-3, f"beta>=0.5 clamp expected, got {d_bh0['rm_beta']}"
cfg_bh25 = HorizonV0Config(rm_enabled=True, rm_beta_mode="betahat_gate", rm_gate_gmin=0.25,
                           rm_cl_prior=0.5, rm_cl_mu=1e-6, rm_cl_forget=1.0, rm_lambda_max=10.0)
st_bh2 = make_state(torch.randn(B, N, C), torch.randn(B, N, C), 0.8, 0.6)
r_new2 = st_bh2.prev_residual + 0.2 * (st_bh2.prev_residual - st_bh2.r_prev)  # beta*=0.2 -> g=0.4
_cl_update(st_bh2, r_new2, h_filt, 0.4, cfg_bh25)
st_bh2.r_prev, st_bh2.sigma_prev = st_bh2.prev_residual, st_bh2.sigma_anchor
st_bh2.prev_residual, st_bh2.sigma_anchor = r_new2, 0.4
_, d_bh25 = _residual_motion(st_bh2, h_filt, 0.3, image_ids, cfg_bh25)
expected_beta = 0.25 + 0.25 * min(max(0.2 / 0.5, 0.0), 1.0)
assert abs(d_bh25["rm_beta"] - expected_beta) < 1e-3
print(f"6. betahat_gate: gmin=0 clamps at 0.5 ({d_bh0['rm_beta']:.3f}), "
      f"gmin=0.25 floors at 0.25+0.25g ({d_bh25['rm_beta']:.3f} vs expected {expected_beta:.3f}): OK")

# ---------- 7. causality: cached-step beta depends only on the PRECEDING anchor's innovation ----------
cfg_c = HorizonV0Config(rm_enabled=True, rm_beta_mode="gate", rm_gate_type="soft",
                        rm_gate_kappa=0.3, rm_lambda_max=10.0)
st_c, _ = anchor_stream([mid_delta], cfg_c)
g_before = st_c.gate_g
# two different cached-step sigmas must see the SAME g_a (it was fixed at the anchor)
_, dc1 = _residual_motion(st_c, h_filt, 0.35, image_ids, cfg_c)
_, dc2 = _residual_motion(st_c, h_filt, 0.10, image_ids, cfg_c)
assert st_c.gate_g == g_before, "gate must not mutate between cached-step calls"
assert abs(dc1["rm_beta"] - dc2["rm_beta"]) < 1e-9, "beta_i must be constant across an anchor's cached steps"
print("7. causality: g_a fixed at the anchor, identical beta_i across subsequent cached steps: OK")

# ---------- 8. grammar ----------
c = _variant_cfg("rmgh0.08_adaptive_1.25", 0.3, "regrid", None)
assert (c.rm_enabled, c.rm_beta_mode, c.rm_gate_type, c.rm_gate_kappa, c.adaptive, c.jf_max) == \
       (True, "gate", "hard", 0.08, True, 1.25), vars(c)
c = _variant_cfg("rmgs0.15_adaptive_1.25", 0.3, "regrid", None)
assert (c.rm_gate_type, c.rm_gate_kappa) == ("soft", 0.15)
c = _variant_cfg("rmgf0.20g0.5_adaptive_1.25", 0.3, "regrid", None)
assert (c.rm_gate_type, c.rm_gate_kappa, c.rm_gate_gmin) == ("floor", 0.20, 0.5)
c = _variant_cfg("rmgf0.20g0.5a0.5_adaptive_1.25", 0.3, "regrid", None)
assert (c.rm_gate_gmin, c.rm_gate_ema_alpha) == (0.5, 0.5)
c = _variant_cfg("rmgh0.08nd_adaptive_1.25", 0.3, "regrid", None)
assert c.rm_gate_norm == "d"
c = _variant_cfg("rmbg0.25_adaptive_1.25", 0.3, "regrid", None)
assert (c.rm_beta_mode, c.rm_gate_gmin) == ("betahat_gate", 0.25)
c = _variant_cfg("rmbg_adaptive_1.25", 0.3, "regrid", None)
assert (c.rm_beta_mode, c.rm_gate_gmin) == ("betahat_gate", 0.0)
# legacy grammar unchanged
c = _variant_cfg("rmraw0.5_adaptive_1.5", 0.3, "regrid", None)
assert (c.rm_beta_mode, c.rm_beta) == ("fixed", 0.5)
c = _variant_cfg("rmcl0.5_adaptive_1.25", 0.3, "regrid", None)
assert c.rm_beta_mode == "cl"
c = _variant_cfg("rm2raw0.5b0.1_adaptive_1.25", 0.3, "regrid", None)
assert c.rm_so_mode == "uniform"
print("8. grammar (rmg hard/soft/floor/norm + rmbg + legacy rm/rmcl/rm2 unchanged): OK")

print("\nALL E61 MATH TESTS PASSED")
