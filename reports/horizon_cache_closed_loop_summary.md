# E60 — Closed-Loop Residual Motion (online β̂ + midpoint-λ)

**Commit** `2b66277` · FLUX 512px/28 · cluster H100

**Headline.** Closed-loop RM (rmcl0.2m0.25k2.5mid_adaptive_1.25) vs SeaCache at matched speed: best +2.60 dB @ 2.74× (CI [+2.21,+2.98]); highest CI-positive band >5.2x. β̂ − fixed β: best +0.39 dB @ 5.29× (CI [+0.35,+0.44]); high-speed fixed-β penalty NOT removed by the closed loop → KILL.

**Predictions.** β̂ p50 in-band 0.185 vs fast 0.022 (zero-frac 0.015) — prediction (a) FAIL; corr(innovation, RM−plain gain) = -0.44 (n=1800) — prediction (b) PASS.

## Δ vs SeaCache by speed band (best per family; * = CI excludes 0)

| band | plain | fixed-β RM | closed-loop | closed-loop+mid |
|---|---|---|---|---|
| 1.8-2.2x | +1.37* @2.12× | +2.41* @2.12× | +1.94* @2.12× | +1.95* @2.12× |
| 2.3-2.7x | +2.00* @2.50× | +2.88* @2.50× | +2.54* @2.50× | +2.50* @2.50× |
| 2.8-3.2x | +0.77* @3.08× | +0.86* @3.08× | +0.92* @3.09× | +0.88* @3.09× |
| 3.3-3.7x | +1.84* @3.40× | +2.27* @3.40× | +2.19* @3.40× | +2.12* @3.40× |
| 3.8-4.2x | +0.31* @3.86× | +0.23* @3.87× | +0.16 @3.86× | -0.06 @3.86× |
| 4.3-4.7x | +1.55* @4.47× | +1.29* @4.47× | +1.45* @4.47× | +1.21* @4.47× |
| 4.8-5.2x † | +0.95* @5.17× | — | — | — |
| >5.2x † | -0.21* @5.29× | +0.53* @5.30× | +0.82* @5.22× | +0.62* @5.23× |

## β̂ − fixed β (paired, per τ)

- rmcl0.5_adaptive_1.25 τ0.3 (2.11×): -0.587 [-0.70,-0.48] *
- rmcl0.5_adaptive_1.25 τ0.4 (2.51×): -0.554 [-0.67,-0.44] *
- rmcl0.5_adaptive_1.25 τ0.5 (2.74×): -0.261 [-0.40,-0.12] *
- rmcl0.5_adaptive_1.25 τ0.575 (3.09×): -0.049 [-0.14,+0.04]
- rmcl0.5_adaptive_1.25 τ0.65 (3.40×): -0.187 [-0.28,-0.10] *
- rmcl0.5_adaptive_1.25 τ0.8 (3.86×): -0.028 [-0.09,+0.03]
- rmcl0.5_adaptive_1.25 τ1 (4.47×): +0.166 [+0.10,+0.23] *
- rmcl0.5_adaptive_1.25 τ1.2 (5.22×): +0.284 [+0.23,+0.34] *
- rmcl0.5_adaptive_1.25 τ1.4 (5.29×): +0.394 [+0.35,+0.44] *
- rmcl0.5m0.25g0.08_adaptive_1.25 τ0.3 (2.12×): -0.403 [-0.50,-0.31] *
- rmcl0.5m0.25g0.08_adaptive_1.25 τ0.4 (2.51×): -0.337 [-0.42,-0.25] *
- rmcl0.5m0.25g0.08_adaptive_1.25 τ0.5 (2.74×): -0.163 [-0.25,-0.08] *
- rmcl0.5m0.25g0.08_adaptive_1.25 τ0.575 (3.09×): -0.135 [-0.20,-0.07] *
- rmcl0.5m0.25g0.08_adaptive_1.25 τ0.65 (3.40×): -0.078 [-0.16,+0.04]
- rmcl0.5m0.25g0.08_adaptive_1.25 τ0.8 (3.86×): -0.039 [-0.10,+0.02]
- rmcl0.5m0.25g0.08_adaptive_1.25 τ1 (4.47×): +0.089 [+0.02,+0.16] *
- rmcl0.5m0.25g0.08_adaptive_1.25 τ1.2 (5.21×): +0.191 [+0.13,+0.25] *
- rmcl0.5m0.25g0.08_adaptive_1.25 τ1.4 (5.29×): +0.308 [+0.26,+0.36] *

## Verdicts

- closed-loop β̂ as in-band gain: **KILL**
- closed-loop β̂ at extreme speed (≥4.3×): **KEEP**
- midpoint-λ: **not_run**
- prediction (a): **FAIL/n-a**
- prediction (b): **PASS**