# HorizonCache-PC (E57) — KILL

**Commit** `43e46c7` · FLUX 512px/28 · smoke N=8 · cluster H100

**Headline.** PC (α=0.5) does not beat plain HorizonCache in any band (best PC−plain = -0.00 dB @ 2.48× on adaptive_1.5), and its endpoint call slightly lowers achieved speedup. Verdict: KILL.

**Bounded claim.** On FLUX, a cached-endpoint predictor-corrector is a near-no-op: because the cached endpoint velocity reuses the frozen block-stack residual, v_pred ≈ v_i (curvature ~1.4–2.6%), so the correction cannot reduce the Euler truncation error that causes the overshoot.

## PC − plain HorizonCache (paired)

| family | τ | PC speedup | plain speedup | PC−plain ΔPSNR | 95% CI | win |
|---|---|---|---|---|---|---|
| adaptive_1.5 | 0.4 | 2.48× | 2.50× | -0.000 | [-0.041, +0.058] | 25% |
| adaptive_1.5 | 0.5 | 2.71× | 2.74× | -0.047 | [-0.090, -0.006] | 38% |
| adaptive_1.5 | 0.65 | 3.35× | 3.40× | -0.024 | [-0.036, -0.014] | 0% |
| adaptive_2.0 | 0.4 | 2.35× | 2.37× | -0.058 | [-0.101, -0.015] | 25% |
| adaptive_2.0 | 0.5 | 2.71× | 2.74× | -0.055 | [-0.088, -0.024] | 25% |
| adaptive_2.0 | 0.65 | 3.35× | 3.40× | -0.024 | [-0.039, -0.012] | 0% |

## Why it failed

Curvature ‖v_pred−v_i‖₁/‖v_i‖₁: p50=0.014, p95=0.026 — the cached endpoint is ~1–3% from the start velocity, so the trapezoid correction is a near-no-op. The truncation error lives in the block-stack curvature the cache freezes.

## Verdicts

- pc_alpha_0.5 (cached): **KILL**
- pc_alpha 0.25/0.75/1.0: **KILL**
- curvature cancel/shrink: **KILL**
- oracle fresh-endpoint: **PARK** (diagnostic; no speedup)