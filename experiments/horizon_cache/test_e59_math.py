"""E59 second-order residual hold — math verification (no GPU/model needed).

Checks:
 1. uniform mode with beta2=0 is BIT-IDENTICAL to first-order RM (E58 identity).
 2. uniform second-order hold is EXACT on a quadratic residual sequence with uniform
    sigma-spaced anchors, beta1=beta2=1 (Newton backward difference).
 3. quad (Lagrange) mode is EXACT on a quadratic in sigma with NONuniform anchors, beta_quad=1.
 4. gates: cos_delta <= gamma or rho2 >= rho_max disables the SO term (falls back to first order).
 5. anchor-triple diagnostics: linear sequence -> rho2~0, cos_delta~1.
 6. run.py variant grammar parses rm2/rmq/gated names and beta2=0 recovers first-order config.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import torch

from horizon_cache.flux_gen import _residual_motion, _anchor_triple_diag, FluxCacheState
from horizon_cache.policy import HorizonV0Config
from horizon_cache.run import _variant_cfg

torch.manual_seed(0)
B, N, C = 1, 16, 8
image_ids = torch.zeros(N, 3)
image_ids[:, 1] = torch.arange(N) // 4
image_ids[:, 2] = torch.arange(N) % 4
h_filt = torch.randn(B, N, C)


def make_state(r2, r1, r0, s2, s1, s0, so):
    """anchors ordered oldest->newest: r2@s2 (a-2), r1@s1 (a-1), r0@s0 (a)."""
    st = FluxCacheState()
    st.prev_residual, st.r_prev, st.r_prev2 = r0, r1, r2
    st.sigma_anchor, st.sigma_prev, st.sigma_prev2 = s0, s1, s2
    st.steps_since_anchor, st.steps_prev_to_anchor = 1, 1
    if so:
        st.so_diag = _anchor_triple_diag(r0, r1, r2)
    return st


def quad_seq(a, b, c, s):
    return a + b * s + c * s * s


A = torch.randn(B, N, C)
Bq = torch.randn(B, N, C)
Cq = torch.randn(B, N, C)

# ---------- 1. beta2=0 identity ----------
r2, r1, r0 = torch.randn(B, N, C), torch.randn(B, N, C), torch.randn(B, N, C)
s2, s1, s0, si = 1.0, 0.8, 0.6, 0.5
cfg_fo = HorizonV0Config(rm_enabled=True, rm_beta=0.5)
cfg_so0 = HorizonV0Config(rm_enabled=True, rm_beta=0.5, rm_so_mode="uniform", rm_beta2=0.0)
st_a = make_state(r2, r1, r0, s2, s1, s0, so=False)
st_b = make_state(r2, r1, r0, s2, s1, s0, so=True)
pa, da = _residual_motion(st_a, h_filt, si, image_ids, cfg_fo)
pb, db = _residual_motion(st_b, h_filt, si, image_ids, cfg_so0)
assert torch.equal(pa, pb), "beta2=0 must be bit-identical to first-order RM"
assert db["rm_so_used"] is False and db["rm_so_gated_off"] is False
print("1. beta2=0 identity: OK (bit-identical)")

# ---------- 2. uniform second-order exact on quadratic (uniform spacing) ----------
h = 0.2
s0u = 0.6
s1u, s2u = s0u + h, s0u + 2 * h          # sigma decreases toward 0: older anchors larger
lam = 0.75                                # predict at si = s0 - lam*h
siu = s0u - lam * h
# quadratic in anchor index n: anchors at n=0 (r_prev2), 1 (r_prev), 2 (r_anchor); target n=2+lam
rq2, rq1, rq0 = (quad_seq(A, Bq, Cq, 0.0), quad_seq(A, Bq, Cq, 1.0), quad_seq(A, Bq, Cq, 2.0))
target = quad_seq(A, Bq, Cq, 2.0 + lam)
cfg_u = HorizonV0Config(rm_enabled=True, rm_beta=1.0, rm_so_mode="uniform", rm_beta2=1.0,
                        rm_lambda_max=10.0)
st_u = make_state(rq2, rq1, rq0, s2u, s1u, s0u, so=True)
pu, du = _residual_motion(st_u, h_filt, siu, image_ids, cfg_u)
err = (pu - target).abs().max().item()
assert err < 1e-4, f"uniform SO must be exact on quadratic, max err {err}"
assert du["rm_so_used"] is True
assert abs(du["rm_lambda"] - lam) < 1e-6
assert abs(du["rm_so_coeff"] - lam * (lam + 1) / 2) < 1e-6
print(f"2. uniform SO exact on quadratic: OK (max err {err:.2e}, lambda={du['rm_lambda']:.3f})")

# first-order alone must NOT be exact on this quadratic (sanity that the test bites)
p1, _ = _residual_motion(st_u, h_filt, siu, image_ids,
                         HorizonV0Config(rm_enabled=True, rm_beta=1.0, rm_lambda_max=10.0))
assert (p1 - target).abs().max().item() > 1e-3, "first-order should be inexact on a quadratic"
print("   (first-order alone is inexact on the same quadratic: OK)")

# ---------- 3. quad Lagrange exact on quadratic in sigma, NONuniform anchors ----------
s2n, s1n, s0n, sin_ = 1.0, 0.7, 0.55, 0.35
rn2, rn1, rn0 = quad_seq(A, Bq, Cq, s2n), quad_seq(A, Bq, Cq, s1n), quad_seq(A, Bq, Cq, s0n)
target_n = quad_seq(A, Bq, Cq, sin_)
cfg_q = HorizonV0Config(rm_enabled=True, rm_so_mode="quad", rm_beta_quad=1.0)
st_q = make_state(rn2, rn1, rn0, s2n, s1n, s0n, so=True)
pq, dq = _residual_motion(st_q, h_filt, sin_, image_ids, cfg_q)
errq = (pq - target_n).abs().max().item()
assert errq < 1e-4, f"quad mode must be exact on sigma-quadratic, max err {errq}"
assert dq["rm_so_used"] is True
print(f"3. quad Lagrange exact (nonuniform anchors): OK (max err {errq:.2e})")

# damping: beta_quad=0.5 gives the midpoint between r_anchor and r_quad
cfg_q5 = HorizonV0Config(rm_enabled=True, rm_so_mode="quad", rm_beta_quad=0.5)
pq5, _ = _residual_motion(make_state(rn2, rn1, rn0, s2n, s1n, s0n, so=True),
                          h_filt, sin_, image_ids, cfg_q5)
assert (pq5 - (rn0 + 0.5 * (target_n - rn0))).abs().max().item() < 1e-4
print("   (beta_quad=0.5 damping: OK)")

# ---------- 4. gates ----------
# reversing residual motion: r goes up then down -> cos_delta < 0 -> gamma gate blocks SO
rr2, rr1 = torch.zeros(B, N, C), torch.ones(B, N, C)
rr0 = torch.zeros(B, N, C)                      # delta_a = -1, delta_{a-1} = +1 -> cos=-1
cfg_g = HorizonV0Config(rm_enabled=True, rm_beta=0.5, rm_so_mode="uniform", rm_beta2=0.25,
                        rm_so_gate_gamma=0.0)
st_g = make_state(rr2, rr1, rr0, s2u, s1u, s0u, so=True)
pg, dg = _residual_motion(st_g, h_filt, siu, image_ids, cfg_g)
assert dg["rm_so_gated_off"] is True and dg["rm_so_used"] is False
p_fo, _ = _residual_motion(make_state(rr2, rr1, rr0, s2u, s1u, s0u, so=False),
                           h_filt, siu, image_ids,
                           HorizonV0Config(rm_enabled=True, rm_beta=0.5))
assert torch.equal(pg, p_fo), "gated-off SO must equal first-order"
print("4a. cos_delta gate blocks reversing motion, falls back to first-order: OK")

# rho_max gate: high-curvature triple blocked
rc2, rc1, rc0 = torch.zeros(B, N, C), torch.full((B, N, C), 0.01), torch.full((B, N, C), 1.0)
st_c = make_state(rc2, rc1, rc0, s2u, s1u, s0u, so=True)   # rho2 ~ |0.98|/|0.99| ~ 0.99
cfg_c = HorizonV0Config(rm_enabled=True, rm_beta=0.5, rm_so_mode="uniform", rm_beta2=0.25,
                        rm_so_gate_rho_max=0.5)
pc_, dc = _residual_motion(st_c, h_filt, siu, image_ids, cfg_c)
assert dc["rm_so_gated_off"] is True
print("4b. rho2 gate blocks high curvature: OK")

# ---------- 5. diagnostics on a linear sequence ----------
lin2, lin1, lin0 = quad_seq(A, Bq, 0 * Cq, 0.0), quad_seq(A, Bq, 0 * Cq, 1.0), quad_seq(A, Bq, 0 * Cq, 2.0)
dl = _anchor_triple_diag(lin0, lin1, lin2)
assert dl["so_rho2"] < 1e-5 and dl["so_cos_delta"] > 0.9999
print(f"5. linear anchors -> rho2={dl['so_rho2']:.2e}, cos_delta={dl['so_cos_delta']:.5f}: OK")

# ---------- 6. grammar ----------
c = _variant_cfg("rm2raw0.5b0.1_adaptive_1.25", 0.3, "regrid", {"rm_projection2": "raw"})
assert (c.rm_enabled, c.rm_so_mode, c.rm_beta, c.rm_beta2, c.adaptive, c.jf_max) == \
       (True, "uniform", 0.5, 0.1, True, 1.25), vars(c)
assert c.rm_so_gate_gamma is None and c.rm_so_gate_rho_max is None
c = _variant_cfg("rm2raw0.5b0.1g0.25r0.5_adaptive_1.5", 0.4, "regrid", None)
assert (c.rm_so_gate_gamma, c.rm_so_gate_rho_max, c.rm_beta2, c.jf_max) == (0.25, 0.5, 0.1, 1.5)
c = _variant_cfg("rmqraw0.75_adaptive_1.25", 0.3, "regrid", None)
assert (c.rm_so_mode, c.rm_beta_quad) == ("quad", 0.75)
c = _variant_cfg("rmraw0.5_adaptive_1.5", 0.3, "regrid", None)   # E58 name unchanged
assert (c.rm_so_mode, c.rm_beta, c.rm_beta2) == ("none", 0.5, 0.0)
c = _variant_cfg("rm2lowpass0.5b0.25_adaptive_1.25", 0.3, "regrid", {"rm_lowpass_pool": 2})
assert (c.rm_projection, c.rm_so_mode, c.rm_beta2) == ("lowpass", "uniform", 0.25)
print("6. grammar (rm2 / rm2 gated / rmq / rm legacy / lowpass): OK")

print("\nALL E59 MATH TESTS PASSED")
