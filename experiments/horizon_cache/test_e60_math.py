"""E60 closed-loop residual motion — math verification (no GPU/model needed).

Checks:
 1. rm_beta_mode='fixed' (default) is BIT-IDENTICAL to E58 first-order RM — new fields inert.
 2. β̂ recovery: residuals evolving with a constant true gain β* → the innovation fit converges
    to β* (μ→0), and the μ-regularized posterior interpolates prior→β*.
 3. β̂→0: anti-correlated innovation (residual reverses) clamps to 0 → plain HorizonCache;
    orthogonal innovation shrinks β̂ toward 0 as evidence accumulates.
 4. clamp: β̂ never exceeds rm_cl_beta_max; before any observation the prior is used.
 5. forgetting: w<1 weights recent anchors; a regime change (β* 0.8→0.0) is tracked faster
    with w=0.6 than w=1.0.
 6. midpoint-λ: motion uses λ((σ_i+σ_target)/2); with sigma λ-mode this equals the analytic
    midpoint value; 'point' mode and sigma_target=None reproduce E58 exactly.
 7. run.py grammar parses rmcl names (prior/w/m/x/mid) and rm...mid; E58 names unchanged.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import torch

from horizon_cache.flux_gen import (_residual_motion, _cl_update, _cl_beta_used,
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
    """two anchors oldest->newest: r1@s1 (a-1), r0@s0 (a)."""
    st = FluxCacheState()
    st.prev_residual, st.r_prev = r0, r1
    st.sigma_anchor, st.sigma_prev = s0, s1
    st.steps_since_anchor, st.steps_prev_to_anchor = 1, 1
    return st


def run_anchor_stream(betas_true, cfg, h=0.2, s_start=1.0, drift=None, noise=0.0):
    """Simulate a fresh-anchor stream r_k = r_{k-1} + β*_k·λ_k·Δr(+noise) and return the state.
    λ_k = 1 exactly (uniform anchor spacing h in sigma). drift overrides the secant direction."""
    d = drift if drift is not None else torch.randn(B, N, C)
    s1, s0 = s_start, s_start - h
    r1 = torch.randn(B, N, C)
    r0 = r1 + d
    st = make_state(r1, r0, s1, s0)
    for k, bt in enumerate(betas_true):
        s_new = s0 - h * (k + 1)
        dr = st.prev_residual - st.r_prev
        r_new = st.prev_residual + bt * dr + noise * torch.randn(B, N, C)
        _cl_update(st, r_new, h_filt, s_new, cfg)
        st.r_prev, st.sigma_prev = st.prev_residual, st.sigma_anchor
        st.prev_residual, st.sigma_anchor = r_new, s_new
    return st


# ---------- 1. fixed mode bit-identity (default config inert) ----------
r1, r0 = torch.randn(B, N, C), torch.randn(B, N, C)
s1, s0, si = 0.8, 0.6, 0.5
cfg_e58 = HorizonV0Config(rm_enabled=True, rm_beta=0.5)
pa, da = _residual_motion(make_state(r1, r0, s1, s0), h_filt, si, image_ids, cfg_e58)
pb, db = _residual_motion(make_state(r1, r0, s1, s0), h_filt, si, image_ids, cfg_e58,
                          sigma_target=0.4)   # target given but eval mode 'point': must be inert
assert torch.equal(pa, pb), "'point' mode must ignore sigma_target"
assert "rm_beta_mode" not in da, "E58 diag must not grow E60 fields in fixed/point mode"
print("1. fixed-mode bit-identity (sigma_target inert, diag unchanged): OK")

# ---------- 2. β̂ recovery of a constant true gain ----------
cfg_cl = HorizonV0Config(rm_enabled=True, rm_beta_mode="cl", rm_cl_prior=0.5,
                         rm_cl_mu=1e-6, rm_cl_forget=1.0, rm_cl_beta_max=2.0)
st = run_anchor_stream([0.7] * 6, cfg_cl)
assert abs(st.cl_beta_hat - 0.7) < 1e-4, f"β̂ must recover β*=0.7, got {st.cl_beta_hat}"
# μ-regularized posterior sits between prior and β* after one observation
cfg_mu = HorizonV0Config(rm_enabled=True, rm_beta_mode="cl", rm_cl_prior=0.5,
                         rm_cl_mu=1.0, rm_cl_forget=1.0, rm_cl_beta_max=2.0)
st1 = run_anchor_stream([0.7], cfg_mu)
assert 0.5 < st1.cl_beta_hat < 0.7, f"posterior must interpolate prior->obs, got {st1.cl_beta_hat}"
print(f"2. β̂ recovery: OK (μ→0: {st.cl_beta_hat:.5f}; μ=1 one obs: {st1.cl_beta_hat:.3f})")

# ---------- 3. β̂→0 on stale/reversing secants ----------
st_rev = run_anchor_stream([-0.6] * 6, cfg_cl)                 # residual reverses direction
assert st_rev.cl_beta_hat < 0, "anti-correlated innovation must drive raw β̂ negative"
assert _cl_beta_used(st_rev, cfg_cl) == 0.0, "clamp must recover plain HorizonCache (β=0)"
# orthogonal (pure-noise) innovation: y uncorrelated with dr -> β̂ shrinks toward 0
cfg_o = HorizonV0Config(rm_enabled=True, rm_beta_mode="cl", rm_cl_prior=0.5,
                        rm_cl_mu=1.0, rm_cl_forget=1.0)
st_o = run_anchor_stream([0.0] * 8, cfg_o, noise=1.0)
assert st_o.cl_beta_hat < 0.35, f"noise anchors must shrink β̂ from the prior, got {st_o.cl_beta_hat}"
print(f"3. stale secants: reversing → clamp 0 (raw {st_rev.cl_beta_hat:.3f}); "
      f"noise → shrink ({st_o.cl_beta_hat:.3f} < prior 0.5): OK")

# ---------- 4. clamp + prior-before-evidence ----------
cfg_x = HorizonV0Config(rm_enabled=True, rm_beta_mode="cl", rm_cl_prior=0.5,
                        rm_cl_mu=1e-6, rm_cl_forget=1.0, rm_cl_beta_max=0.75)
st_hi = run_anchor_stream([1.5] * 5, cfg_x)
assert _cl_beta_used(st_hi, cfg_x) == 0.75, "β̂ must clamp at rm_cl_beta_max"
st_empty = FluxCacheState()
assert _cl_beta_used(st_empty, cfg_x) == 0.5, "no observations → use the prior"
print("4. clamp at β_max and prior-before-evidence: OK")

# ---------- 5. forgetting tracks a regime change ----------
regime = [0.8] * 5 + [0.0] * 3
cfg_w6 = HorizonV0Config(rm_enabled=True, rm_beta_mode="cl", rm_cl_prior=0.5,
                         rm_cl_mu=1e-6, rm_cl_forget=0.6)
cfg_w10 = HorizonV0Config(rm_enabled=True, rm_beta_mode="cl", rm_cl_prior=0.5,
                          rm_cl_mu=1e-6, rm_cl_forget=1.0)
b6 = run_anchor_stream(regime, cfg_w6).cl_beta_hat
b10 = run_anchor_stream(regime, cfg_w10).cl_beta_hat
assert b6 < b10, f"forgetting must track the 0.8→0.0 change faster (w0.6 {b6:.3f} vs w1 {b10:.3f})"
print(f"5. forgetting tracks regime change: OK (w=0.6: {b6:.3f} < w=1.0: {b10:.3f})")

# ---------- 6. midpoint-λ ----------
# sigma λ-mode: λ(σ) = (σ - σ_a)/(σ_a - σ_prev)·sign fix; anchors s1=0.8, s0=0.6 → denom -0.2.
cfg_mid = HorizonV0Config(rm_enabled=True, rm_beta=1.0, rm_lambda_eval="mid", rm_lambda_max=10.0)
st_m = make_state(r1, r0, 0.8, 0.6)
si_, starget = 0.5, 0.3
p_mid, d_mid = _residual_motion(st_m, h_filt, si_, image_ids, cfg_mid, sigma_target=starget)
lam_mid_expect = (0.6 - 0.5 * (si_ + starget)) / 0.2      # λ at σ=0.4 → 1.0
lam_pt_expect = (0.6 - si_) / 0.2                          # λ at σ_i  → 0.5
assert abs(d_mid["rm_lambda"] - lam_mid_expect) < 1e-5, d_mid["rm_lambda"]
assert abs(d_mid["rm_lambda_point"] - lam_pt_expect) < 1e-5
expected = r0 + lam_mid_expect * (r0 - r1)
assert (p_mid - expected).abs().max().item() < 1e-5
# no target (e.g. fresh bookkeeping edge) falls back to the point value
p_nt, d_nt = _residual_motion(make_state(r1, r0, 0.8, 0.6), h_filt, si_, image_ids, cfg_mid)
assert abs(d_nt["rm_lambda"] - lam_pt_expect) < 1e-5
print(f"6. midpoint-λ: OK (λ_mid={d_mid['rm_lambda']:.3f}, λ_point={d_mid['rm_lambda_point']:.3f})")

# ---------- 7. grammar ----------
c = _variant_cfg("rmcl0.5_adaptive_1.25", 0.3, "regrid", None)
assert (c.rm_enabled, c.rm_beta_mode, c.rm_cl_prior, c.rm_cl_forget, c.rm_cl_mu,
        c.rm_cl_beta_max, c.rm_lambda_eval, c.adaptive, c.jf_max) == \
       (True, "cl", 0.5, 0.85, 1.0, 1.0, "point", True, 1.25), vars(c)
c = _variant_cfg("rmcl0.5mid_adaptive_1.5", 0.4, "regrid", None)
assert (c.rm_beta_mode, c.rm_lambda_eval, c.jf_max) == ("cl", "mid", 1.5)
c = _variant_cfg("rmcl0.5w1.0m0.5x0.75_adaptive_1.25", 0.3, "regrid", None)
assert (c.rm_cl_prior, c.rm_cl_forget, c.rm_cl_mu, c.rm_cl_beta_max) == (0.5, 1.0, 0.5, 0.75)
c = _variant_cfg("rmcl0.5w1.0m0.5x0.75mid_adaptive_1.25", 0.3, "regrid", None)
assert (c.rm_cl_forget, c.rm_cl_beta_max, c.rm_lambda_eval) == (1.0, 0.75, "mid")
c = _variant_cfg("rmraw0.5mid_adaptive_1.25", 0.3, "regrid", None)
assert (c.rm_beta_mode, c.rm_beta, c.rm_lambda_eval) == ("fixed", 0.5, "mid")
c = _variant_cfg("rmraw0.5_adaptive_1.5", 0.3, "regrid", None)      # E58 name unchanged
assert (c.rm_beta_mode, c.rm_beta, c.rm_lambda_eval, c.rm_so_mode) == \
       ("fixed", 0.5, "point", "none")
c = _variant_cfg("rm2raw0.5b0.1_adaptive_1.25", 0.3, "regrid", None)  # E59 name unchanged
assert (c.rm_so_mode, c.rm_beta2, c.rm_beta_mode) == ("uniform", 0.1, "fixed")
print("7. grammar (rmcl / rmcl mid / rmcl w-m-x / rm mid / E58+E59 legacy): OK")

print("\nALL E60 MATH TESTS PASSED")
