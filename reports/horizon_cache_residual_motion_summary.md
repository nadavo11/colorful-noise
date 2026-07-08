# Residual Motion Cache (E58) — STRONG_KEEP

**Commit** `8b3a5b0` · FLUX 512px/28 · smoke N=8 · cluster H100

**Headline.** Residual Motion Cache (rmraw0.5_adaptive_1.25) improves on plain HorizonCache by +0.97 dB @ 2.12× at matched compute (CI [+0.69,+1.24]) — and it does so even though the secant does NOT better-predict the instantaneous true residual (oracle -5.6%), so the gain is correction of accumulated cache-staleness drift, not per-step residual accuracy; verdict STRONG_KEEP.

**Bounded claim.** On FLUX, extrapolating the cached block residual along its recent secant (r_pred = r_anchor + β·λ·P(Δr)) moves the residual by only p50≈13.1%, p95≈36.5% of ‖r_anchor‖, and the oracle shows it does NOT reduce the instantaneous true-residual error (-5.6%). Net effect vs plain HorizonCache at matched compute: STRONG_KEEP (the win is accumulated-drift correction, a global effect, not local per-step accuracy).

## Oracle residual diagnostic

Frozen residual error **0.2694** vs residual-motion error **0.2846** → **-5.6%** relative reduction.

## ResidualMotion − plain HorizonCache (paired)

| RM variant | τ | RM speedup | plain speedup | RM−plain ΔPSNR | 95% CI | win |
|---|---|---|---|---|---|---|
| rmlowpass0.5_adaptive_1.5 | 0.3 | 2.01× | 2.01× | +0.436 | [+0.258, +0.611] | 79% |
| rmlowpass0.5_adaptive_1.5 | 0.4 | 2.50× | 2.50× | +0.519 | [+0.339, +0.709] | 74% |
| rmlowpass0.5_adaptive_1.5 | 0.5 | 2.74× | 2.74× | +0.451 | [+0.209, +0.699] | 71% |
| rmlowpass0.5_adaptive_1.5 | 0.65 | 3.40× | 3.40× | +0.325 | [+0.129, +0.514] | 61% |
| rmlowpass0.5_adaptive_2.0 | 0.3 | 1.98× | 1.98× | +0.454 | [+0.275, +0.630] | 76% |
| rmlowpass0.5_adaptive_2.0 | 0.4 | 2.41× | 2.41× | +0.551 | [+0.362, +0.750] | 76% |
| rmlowpass0.5_adaptive_2.0 | 0.5 | 2.74× | 2.74× | +0.443 | [+0.206, +0.685] | 69% |
| rmlowpass0.5_adaptive_2.0 | 0.65 | 3.40× | 3.40× | +0.330 | [+0.134, +0.520] | 61% |
| rmraw0.5_adaptive_1.25 | 0.3 | 2.12× | 2.12× | +0.972 | [+0.693, +1.237] | 80% |
| rmraw0.5_adaptive_1.25 | 0.4 | 2.51× | 2.51× | +0.910 | [+0.669, +1.151] | 73% |
| rmraw0.5_adaptive_1.25 | 0.5 | 2.74× | 2.74× | +0.633 | [+0.336, +0.931] | 71% |
| rmraw0.5_adaptive_1.25 | 0.65 | 3.40× | 3.40× | +0.426 | [+0.177, +0.665] | 57% |
| rmraw0.5_adaptive_1.5 | 0.3 | 2.01× | 2.01× | +0.858 | [+0.577, +1.129] | 82% |
| rmraw0.5_adaptive_1.5 | 0.4 | 2.50× | 2.50× | +0.951 | [+0.704, +1.209] | 75% |
| rmraw0.5_adaptive_1.5 | 0.5 | 2.74× | 2.74× | +0.670 | [+0.366, +0.971] | 74% |
| rmraw0.5_adaptive_1.5 | 0.65 | 3.40× | 3.40× | +0.424 | [+0.176, +0.663] | 57% |
| rmraw0.5_adaptive_2.0 | 0.3 | 1.98× | 1.98× | +0.850 | [+0.565, +1.117] | 79% |
| rmraw0.5_adaptive_2.0 | 0.4 | 2.42× | 2.41× | +0.931 | [+0.684, +1.184] | 76% |
| rmraw0.5_adaptive_2.0 | 0.5 | 2.74× | 2.74× | +0.653 | [+0.359, +0.950] | 73% |
| rmraw0.5_adaptive_2.0 | 0.65 | 3.40× | 3.40× | +0.423 | [+0.175, +0.663] | 58% |
| rmraw0.75_adaptive_1.5 | 0.3 | 2.01× | 2.01× | +0.904 | [+0.552, +1.253] | 71% |
| rmraw0.75_adaptive_1.5 | 0.4 | 2.50× | 2.50× | +0.907 | [+0.586, +1.242] | 72% |
| rmraw0.75_adaptive_1.5 | 0.5 | 2.74× | 2.74× | +0.617 | [+0.231, +0.996] | 66% |
| rmraw0.75_adaptive_1.5 | 0.65 | 3.40× | 3.40× | +0.157 | [-0.134, +0.426] | 46% |

## Verdicts

- rmlowpass0.5_adaptive_1.5: **KEEP**
- rmlowpass0.5_adaptive_2.0: **KEEP**
- rmraw0.5_adaptive_1.25: **STRONG_KEEP**
- rmraw0.5_adaptive_1.5: **STRONG_KEEP**
- rmraw0.5_adaptive_2.0: **STRONG_KEEP**
- rmraw0.75_adaptive_1.5: **KEEP**