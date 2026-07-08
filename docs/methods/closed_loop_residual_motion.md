# Closed-Loop Residual Motion (proposal for E60) — self-calibrating secant gain

**Status: proposal** (written at E59 time). One upgrade to the caching line, strictly under the
E56/E58 terms: no extra forwards, same sampler, same fixture, same matched-achieved-speedup
protocol. Goal: larger PSNR margins over SeaCache, a wider positive speed band, and a method whose
knob is *derived* from a stated model instead of grid-swept.

## The diagnosis (what E56–E59 established)

1. **E57**: any cheap correction that reads through the *frozen* block residual is a no-op
   (v_pred ≈ v_i). The residual itself must move.
2. **E58**: moving it along the fresh-anchor secant with a **fixed global gain β = 0.5** beats
   plain HorizonCache in every band up to 3.4× and flips the E56 overshoot — but the oracle is
   *negative pointwise* (−5.6%): the win is accumulated-drift correction, and β = 0.5 is a
   population-average compromise (the N=50 pilot showed β = 0.75 better in-band yet overshooting
   at 3.4×). **The optimal gain is regime-dependent.**
3. **E59**: the next term in σ (curvature) is *unusable*: over fresh-anchor triples
   ρ2 = ‖Δ²r‖/‖Δr‖ has p50 ≈ 1.4 (curvature ≥ velocity, i.e. noise) and cosΔ p50 ≈ 0.08 (no
   directional stability). Higher-order extrapolation in time is dead. Meanwhile at ≥ 4.3× even
   the *first*-order nudge goes stale (RM − plain < 0): with hyper-sparse anchors the right gain
   approaches 0. **A fixed β cannot be right at both 2× and 4.5×.**

So the missing ingredient is not a better *basis* (E59 killed that); it is a better *gain* — and
the sampler already measures everything needed to set it, for free.

## The idea

**At every refresh the sampler computes the true residual r_k at the new anchor anyway.** Just
before that refresh, the RM predictor implies a forecast of exactly that quantity:
r̂(σ_k) = r_{k−1} + β·λ_k·Δr_{k−2→k−1}. The innovation e_k = r_k − r̂(σ_k) is a *causal,
per-trajectory, zero-cost* measurement of how well the secant model is doing **on this image, in
this speed regime** — and none of E56–E59 uses it.

Treat the residual as a state with linear drift and observation at anchors:

    r_k = r_{k−1} + β · λ_k · Δr_{k−2→k−1} + ε_k        (drift-rate model, scalar β)

and estimate β online by exponentially-forgetting regularized least squares over the anchors seen
so far (a 1-parameter Kalman filter; all quantities are scalars accumulated per trajectory):

    β̂_k = ( μ·β_prior + Σ_j w^{k−j} λ_j ⟨Δr_j, r_j − r_{j−1}⟩ )
          / ( μ + Σ_j w^{k−j} λ_j² ‖Δr_j‖² ),      β̂ clamped to [0, β_max]

Cached/jump steps then use r_pred = r_anchor + **β̂**·λ_mid·Δr, with two refinements:

- **λ at the stride midpoint** (λ_mid = λ((σ_i + σ_target)/2)) instead of the left endpoint:
  the cached step's velocity becomes a midpoint-rule quadrature of the moving residual path —
  second-order accurate *integration* of r(σ) without needing the curvature E59 proved is noise.
  This is the theory-correct home of E58's "the win is the integral, not the point" finding.
- **Innovation-gated stride** (optional, same zero cost): the normalized innovation
  ‖e_k‖₁/‖r_k‖₁ is a direct measure of residual-model trust; feed it into the jump policy's
  headroom so strides shrink exactly when the residual stops being predictable (the ≥ 4× regime).

## Why this should win on all three axes

- **PSNR vs SeaCache**: β̂ adapts per image *and* per regime. Where drift is coherent (2–3.4×) it
  can exceed 0.5 (the pilot says up to ~0.75 helps in-band); where the secant is stale (≥ 4.3×,
  where E59 measured RM − plain < 0) the denominator grows and β̂ → 0, recovering plain
  HorizonCache **by construction**. The frontier can only gain: RM-with-β̂ ≥ max(plain, fixed-β RM)
  pointwise in expectation.
- **Wider positive band**: the current cutoff vs SeaCache (~3.8–4.2× tie, loss beyond) is driven
  by the fixed-β penalty at high τ; removing the penalty moves the crossover up, and the
  innovation-gated stride attacks the same band from the policy side.
- **Theory correspondence**: β̂ is the MMSE drift-rate estimate under the stated state-space
  model, optimizing precisely the *measurable* anchor-consistency objective (the oracle metric at
  anchors) — resolving E58's "oracle-negative yet PSNR-positive" dissonance instead of living
  with it. The method's two pre-registerable predictions make it falsifiable:
  (1) β̂ should distribute around ~0.5 in the 2–2.7× band with no sweep, and fall toward 0 above
  ~4×; (2) per-image innovation norm should anti-correlate with the per-image RM − plain gain.

## Cost & fairness

Two scalar accumulators per trajectory (numerator/denominator), plus tensors already stored by
E58 (r_prev, Δr). Zero extra block-stack forwards; achieved speedup identical to plain
HorizonCache; protocol, fixture, and bootstrap unchanged. Grammar suggestion:
`rmcl[prior0.5]_adaptive_1.25`, run-level flags for w (forgetting), μ (prior strength), β_max.
