# Residual Motion Cache (E58) — STRONG_KEEP

**Commit** `f7ba9d1` · FLUX 512px/28 · smoke N=8 · cluster H100

**Headline.** Residual Motion Cache (rmraw0.5_adaptive_1.25) improves on plain HorizonCache by +0.97 dB @ 2.12× at matched compute (CI [+0.70,+1.24]) — and it does so even though the secant does NOT better-predict the instantaneous true residual (oracle -5.6%), so the gain is correction of accumulated cache-staleness drift, not per-step residual accuracy; verdict STRONG_KEEP.

**Bounded claim.** On FLUX, extrapolating the cached block residual along its recent secant (r_pred = r_anchor + β·λ·P(Δr)) moves the residual by only p50≈13.1%, p95≈36.5% of ‖r_anchor‖, and the oracle shows it does NOT reduce the instantaneous true-residual error (-5.6%). Net effect vs plain HorizonCache at matched compute: STRONG_KEEP (the win is accumulated-drift correction, a global effect, not local per-step accuracy).

## Oracle residual diagnostic

Frozen residual error **0.2694** vs residual-motion error **0.2846** → **-5.6%** relative reduction.

## ResidualMotion − plain HorizonCache (paired)

| RM variant | τ | RM speedup | plain speedup | RM−plain ΔPSNR | 95% CI | win |
|---|---|---|---|---|---|---|
| rmlowpass0.5_adaptive_1.5 | 0.3 | 2.01× | 2.01× | +0.436 | [+0.261, +0.607] | 79% |
| rmlowpass0.5_adaptive_1.5 | 0.4 | 2.50× | 2.50× | +0.519 | [+0.333, +0.707] | 74% |
| rmlowpass0.5_adaptive_1.5 | 0.5 | 2.74× | 2.74× | +0.451 | [+0.206, +0.697] | 71% |
| rmlowpass0.5_adaptive_1.5 | 0.65 | 3.40× | 3.40× | +0.325 | [+0.134, +0.522] | 61% |
| rmlowpass0.5_adaptive_2.0 | 0.3 | 1.98× | 1.98× | +0.454 | [+0.276, +0.628] | 76% |
| rmlowpass0.5_adaptive_2.0 | 0.4 | 2.41× | 2.41× | +0.551 | [+0.363, +0.746] | 76% |
| rmlowpass0.5_adaptive_2.0 | 0.5 | 2.74× | 2.74× | +0.443 | [+0.206, +0.685] | 69% |
| rmlowpass0.5_adaptive_2.0 | 0.65 | 3.40× | 3.40× | +0.330 | [+0.136, +0.527] | 61% |
| rmraw0.5_adaptive_1.25 | 0.3 | 2.12× | 2.12× | +0.972 | [+0.700, +1.237] | 80% |
| rmraw0.5_adaptive_1.25 | 0.4 | 2.51× | 2.51× | +0.910 | [+0.669, +1.155] | 73% |
| rmraw0.5_adaptive_1.25 | 0.5 | 2.74× | 2.74× | +0.633 | [+0.332, +0.926] | 71% |
| rmraw0.5_adaptive_1.25 | 0.65 | 3.40× | 3.40× | +0.426 | [+0.184, +0.673] | 57% |
| rmraw0.5_adaptive_1.5 | 0.3 | 2.01× | 2.01× | +0.858 | [+0.572, +1.127] | 82% |
| rmraw0.5_adaptive_1.5 | 0.4 | 2.50× | 2.50× | +0.951 | [+0.697, +1.209] | 75% |
| rmraw0.5_adaptive_1.5 | 0.5 | 2.74× | 2.74× | +0.670 | [+0.361, +0.969] | 74% |
| rmraw0.5_adaptive_1.5 | 0.65 | 3.40× | 3.40× | +0.424 | [+0.184, +0.670] | 57% |
| rmraw0.5_adaptive_2.0 | 0.3 | 1.98× | 1.98× | +0.850 | [+0.563, +1.126] | 79% |
| rmraw0.5_adaptive_2.0 | 0.4 | 2.42× | 2.41× | +0.931 | [+0.681, +1.185] | 76% |
| rmraw0.5_adaptive_2.0 | 0.5 | 2.74× | 2.74× | +0.653 | [+0.351, +0.948] | 73% |
| rmraw0.5_adaptive_2.0 | 0.65 | 3.40× | 3.40× | +0.423 | [+0.183, +0.669] | 58% |
| rmraw0.75_adaptive_1.5 | 0.3 | 2.01× | 2.01× | +0.904 | [+0.548, +1.252] | 71% |
| rmraw0.75_adaptive_1.5 | 0.4 | 2.50× | 2.50× | +0.907 | [+0.583, +1.233] | 72% |
| rmraw0.75_adaptive_1.5 | 0.5 | 2.74× | 2.74× | +0.617 | [+0.231, +0.986] | 66% |
| rmraw0.75_adaptive_1.5 | 0.65 | 3.40× | 3.40× | +0.157 | [-0.125, +0.445] | 46% |

## Verdicts

- rmlowpass0.5_adaptive_1.5: **KEEP**
- rmlowpass0.5_adaptive_2.0: **KEEP**
- rmraw0.5_adaptive_1.25: **STRONG_KEEP**
- rmraw0.5_adaptive_1.5: **STRONG_KEEP**
- rmraw0.5_adaptive_2.0: **STRONG_KEEP**
- rmraw0.75_adaptive_1.5: **KEEP**