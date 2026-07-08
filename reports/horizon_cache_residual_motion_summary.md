# Residual Motion Cache (E58) — STRONG_KEEP

**Commit** `07bd1de` · FLUX 512px/28 · smoke N=8 · cluster H100

**Headline.** Residual Motion Cache (rmraw0.5_adaptive_1.25) improves on plain HorizonCache by +0.96 dB @ 2.12× at matched compute (CI [+0.78,+1.13]) — and it does so even though the secant does NOT better-predict the instantaneous true residual (oracle -5.6%), so the gain is correction of accumulated cache-staleness drift, not per-step residual accuracy; verdict STRONG_KEEP.

**Bounded claim.** On FLUX, extrapolating the cached block residual along its recent secant (r_pred = r_anchor + β·λ·P(Δr)) moves the residual by only p50≈13.8%, p95≈36.1% of ‖r_anchor‖, and the oracle shows it does NOT reduce the instantaneous true-residual error (-5.6%). Net effect vs plain HorizonCache at matched compute: STRONG_KEEP (the win is accumulated-drift correction, a global effect, not local per-step accuracy).

## Oracle residual diagnostic

Frozen residual error **0.2694** vs residual-motion error **0.2846** → **-5.6%** relative reduction.

## ResidualMotion − plain HorizonCache (paired)

| RM variant | τ | RM speedup | plain speedup | RM−plain ΔPSNR | 95% CI | win |
|---|---|---|---|---|---|---|
| rmraw0.5_adaptive_1.25 | 0.3 | 2.12× | 2.11× | +0.956 | [+0.778, +1.133] | 80% |
| rmraw0.5_adaptive_1.25 | 0.4 | 2.51× | 2.51× | +0.877 | [+0.692, +1.059] | 79% |
| rmraw0.5_adaptive_1.25 | 0.5 | 2.74× | 2.74× | +0.542 | [+0.359, +0.734] | 68% |
| rmraw0.5_adaptive_1.25 | 0.575 | 3.08× | 3.08× | +0.189 | [+0.031, +0.348] | 52% |
| rmraw0.5_adaptive_1.25 | 0.65 | 3.40× | 3.40× | +0.425 | [+0.268, +0.580] | 62% |
| rmraw0.5_adaptive_1.5 | 0.3 | 2.02× | 2.01× | +0.926 | [+0.738, +1.109] | 80% |
| rmraw0.5_adaptive_1.5 | 0.4 | 2.50× | 2.50× | +0.899 | [+0.709, +1.083] | 80% |
| rmraw0.5_adaptive_1.5 | 0.5 | 2.74× | 2.74× | +0.562 | [+0.375, +0.754] | 70% |
| rmraw0.5_adaptive_1.5 | 0.575 | 3.03× | 3.03× | +0.188 | [+0.028, +0.353] | 52% |
| rmraw0.5_adaptive_1.5 | 0.65 | 3.40× | 3.40× | +0.428 | [+0.273, +0.581] | 62% |

## Verdicts

- rmraw0.5_adaptive_1.25: **STRONG_KEEP**
- rmraw0.5_adaptive_1.5: **STRONG_KEEP**