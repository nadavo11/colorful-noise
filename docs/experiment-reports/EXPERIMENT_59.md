# E59 — Second-Order Residual Hold + Extreme-Speed SeaCache Comparison

**Thread:** fast-edit (caching line) · **Status:** done · **Verdicts:** second-order residual hold **KILL** ·
first-order RM at extreme speed **STRONG KEEP (band-limited: RM ≤ ~3.4×, plain beyond)** ·
**Models:** FLUX.1-dev (4-bit), 512 px, 28 Euler steps; smoke N=8 → **consolidation N=100 × 2 seeds = 200
paired** (τ ∈ {0.3…1.4}, ~11 k sampler runs) + N=4 oracle stage, cluster H100 (Run:AI). Canonical-fixture v1 +
deterministic GenEval extras. Protocol identical to E56/E58 (matched achieved speedup, paired percentile bootstrap).

> **Headline 1 (second order).** Adding a curvature term to the Residual Motion secant — uniform Newton-backward
> `r_a + β₁λP₁(Δr) + β₂·λ(λ+1)/2·P₂(Δ²r)` or an exact nonuniform Lagrange quadratic — **does not work, and the
> diagnostic says why**: over 11,178 fresh-anchor triples, **ρ₂ = ‖Δ²r‖/‖Δr‖ has p50 = 1.37** (the second
> difference is *never* smaller than the first — curvature is noise at anchor spacing) and **cosΔ p50 = 0.10**
> (consecutive secants nearly orthogonal). At N=200: SO−FO is −0.08* to −0.19* in most bands (best point +0.04*,
> trivial), quad collapses at speed (−5.9* @5.2×, smoke), and a cosΔ/ρ₂ gate *never opens* — correctly. **KILL.**
>
> **Headline 2 (extreme speed).** With the SeaCache τ grid extended to 1.4, the HorizonCache family stays **above
> SeaCache in every measured band from 1.8× to 5.3×** (all 95 % CIs exclude 0) — and the comparison reveals a
> structural asymmetry: **SeaCache's own frontier tops out at 4.39×** on the swept grid (its τ→speedup mapping is
> integer-quantized: τ 1.0→1.2 moves speed by +0.02×) while the jump mechanism reaches **5.30×** at the same τ.
> Bands beyond 4.4× are therefore conservative comparisons against SeaCache's fastest attained point.
>
> **Headline 3 (the honest RM cutoff).** The paired FO−plain ablation reproduces E58 exactly through 3.4×
> (+0.96*, +0.88*, +0.54*, +0.19*, +0.43*) and then **inverts**: −0.13 @3.87× (ns), **−0.26* @4.47×, −0.42* @5.30×,
> −0.97* @τ=1.4**. The residual-motion nudge helps up to ~3.4×, is neutral at ~3.9×, and **significantly hurts
> beyond ~4.3×** — with hyper-sparse anchors the secant itself is stale. Recommended operating rule: RM on ≤3.5×,
> plain HorizonCache beyond.

## Bonus finding: E58's 3.4× margin was understated ~9×

E58 reported RM +0.26 dB vs SeaCache at 3.40×. That comparison was **clamped at the edge of E58's SeaCache grid**
(sea τ=0.65 achieves 3.00×; the per-image interpolation clamps beyond the last point — replicated here: +0.23).
With the frontier actually measured to 4.39×, SeaCache's true curve falls off a cliff between 3.0× and 3.4×
(23.06 → 21.12 dB), so the honest margin at 3.40× is **+2.29 dB [CI +1.93, +2.68]**. E58's conservative-by-design
edge handling hid most of its own win.

## Method

Implemented in `experiments/horizon_cache/` on top of E58 (β₂=0 is **bit-identical** to E58 — unit-tested in
`test_e59_math.py`, which also proves quadratic exactness of both SO forms and gate behavior):

- **Uniform second-order hold** (`rm2<proj><β1>b<β2>[g<γ>][r<ρmax>]_<base>`): Newton backward difference on the
  three most recent fresh residual anchors; exact for quadratic residual motion at uniform anchor spacing.
- **Nonuniform quadratic** (`rmq<proj><βq>_<base>`): damped Lagrange quadratic through (σ_k, r_k), exact for
  arbitrary anchor spacing.
- **Gate**: enable curvature only if cosΔ > γ and ρ₂ < ρ_max (anchor-triple statistics, computed at refresh).
- **Diagnostics** logged per refresh (ρ₂, cosΔ, ρ_anchor) and per application (β₁, β₂, λ, λ(λ+1)/2, term ratios).
- Still **no extra forward**; ledger counts SO applications separately. Analysis `so_analysis.py`, report
  `so_report.py`; E59 speed bands 1.8–2.2 … 4.8–5.2, >5.2.

## Results (N=200 paired unless noted)

**Δ PSNR vs SeaCache at matched achieved speedup, best per family (all CI exclude 0):**

| band | plain | first-order RM | second-order RM | SO−FO (paired) |
|---|---|---|---|---|
| 1.8–2.2× | +1.37 | **+2.41** | +2.33 | −0.08* |
| 2.3–2.7× | +2.00 | **+2.90** | +2.86 | −0.01 |
| 2.8–3.2× | +0.77 | **+0.86** | +0.73 | −0.13* |
| 3.3–3.7× | +1.86 | **+2.29** | +2.23 | −0.03 |
| 3.8–4.2× | **+0.34** | +0.26 | +0.29 | +0.04* |
| 4.3–4.7× | **+1.55** | +1.31 | +1.14 | −0.14* |
| 4.8–5.2× † | **+0.95** | +0.64 | — | — |
| >5.2× † | −0.20* | **+0.53** | +0.40 | −0.14* |

† beyond the swept SeaCache frontier (sea max 4.39×); conservative comparison vs its fastest attained point.
LPIPS: FO better than SeaCache at every τ (CI+ everywhere except the saturated τ=1.4 point). The zigzag between
bands is SeaCache's own cliff-and-plateau curve, not method noise.

**Oracle (N=4, pointwise residual error, negative = motion worse than frozen):** FO −13.4 % @τ0.65, −26.0 % @τ1.0;
**SO −21.1 % / −36.3 %** — the curvature term strictly worsens pointwise prediction. Consistent with E58's
mechanism: the PSNR gain is accumulated-drift correction, not per-step prediction, and there is no second-order
signal to exploit.

**Second-order assumption diagnostic:** ρ₂ p50 = 1.37 (p5 = 1.06 in smoke — never below 1), cosΔ p50 = 0.10;
corr(ρ₂, SO−FO gain) = −0.03, corr(cosΔ, gain) = +0.07 — the diagnostic has no predictive power because there is
nothing to predict. The γ=0.25/ρ_max=0.5 gate never fired once in the smoke (SO−FO ≡ 0.000).

## Verdicts

- **Second-order RM (uniform Δ²): KILL.** Loses to FO significantly in-band (−0.08*, −0.13*), at extreme speed
  (−0.14*, −0.19*), worsens LPIPS at high τ, worsens the pointwise oracle, and its enabling assumption is
  measured false. Cause (in the E59 failure-mode taxonomy): *curvature term is noise* (ρ₂ > 1 everywhere),
  *cosΔ shows unstable residual direction*, and *first-order already captures the useful drift*.
- **Quad (Lagrange): KILL** (smoke; −5.9* @5.2× — undamped quadratic through noisy anchors over-extrapolates).
- **Gated SO: KILL as a method, correct as a diagnostic** — the gate never authorizes the term.
- **First-order RM at extreme speed: STRONG KEEP, band-limited.** vs SeaCache the family is CI-positive in every
  measured band to 5.3×; but RM's own paired benefit ends at ~3.4× and *reverses* past ~4.3×, where plain
  HorizonCache is the right member of the family. β must fall to ~0 at extreme speed → E60 proposal.

## Next (E60 proposal): Closed-Loop Residual Motion

E59's diagnosis: the failure at speed is not a missing higher-order term (killed here) but the **fixed gain**.
The sampler measures the true residual at every refresh, so the innovation e_k = r_k − r̂(σ_k) is a free, causal,
per-trajectory signal nothing uses. Estimate β online (exponentially-forgetting RLS over anchors — a 1-parameter
Kalman filter on residual drift), evaluate λ at the stride midpoint (midpoint-rule quadrature of the moving
residual). Where drift is coherent β̂ rises; where the secant is stale (≥4.3×) β̂→0 and the method recovers plain
HorizonCache **by construction** — exactly the band-limited rule E59 measured, without hand-tuning. Zero extra
forwards. Falsifiable: β̂ should center near 0.5 in 2–2.7× and fall toward 0 above 4×. Full note:
`docs/methods/closed_loop_residual_motion.md`.

## Artifacts

- Report: `reports/horizon_cache_second_order_extreme.html` ·
  `reports/horizon_cache_second_order_extreme_summary.{md,json}` ·
  assets `reports/horizon_cache_second_order_extreme_assets/`
- Runs: consolidation `results/horizon_so_consol/gen_20260708_143608` (N=200 paired, 11 k runs) ·
  smoke `results/horizon_so_smoke/gen_20260708_141648` (N=8, 568 runs, all images saved) ·
  oracle `results/horizon_so_oracle/gen_20260708_182803` ·
  cluster `runs/runai/20260708_170921__horizon_so_e59_smoke__0be0abc/`,
  `runs/runai/20260708_173401__horizon_so_e59_consol__b9a72b5/`
- Code: `experiments/horizon_cache/{flux_gen,scheduler,policy,run,so_analysis,so_report,test_e59_math}.py` ·
  manifest `experiments/manifests/E59.json`
