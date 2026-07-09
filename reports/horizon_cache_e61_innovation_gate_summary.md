# E61 — Innovation-Gated Fixed Residual Motion

**Commit** `9e5771b` · FLUX 512px/28 · cluster H100

**Headline.** Best gate (rmgh0.75_adaptive_1.25, gate_hard) vs fixed RM: +0.41 dB @ 5.29× (CI [+0.35,+0.47]); vs plain -0.56 dB. Overall: hard_gate=KEEP; soft_gate=not_run; floor_gate=not_run; beta_hat_gate=not_run; ema_floor_gate=not_run.

**Bounded claim.** Innovation diagnostic corr(Ī_a, RM−plain gain) = -0.44949284964918634 (n=1800), vs E60's −0.44 baseline. Highest CI-positive-vs-SeaCache band per family: plain: 4.8-5.2x, fixed_rm: >5.2x, closed_loop: >5.2x, gate_hard: >5.2x, gate_soft: none, gate_floor: none, betahat_gate: none

## Δ vs SeaCache by speed band

| band | plain | fixed RM | closed loop | hard gate | soft gate | floor gate | β̂-as-gate |
|---|---|---|---|---|---|---|---|
| 1.8-2.2x | +1.37* | +2.41* | +1.74* | +2.41* | — | — | — |
| 2.3-2.7x | +2.00* | +2.88* | +2.32* | +2.89* | — | — | — |
| 2.8-3.2x | +0.77* | +0.86* | +0.92* | +0.94* | — | — | — |
| 3.3-3.7x | +1.84* | +2.27* | +2.08* | +2.27* | — | — | — |
| 3.8-4.2x | +0.31* | +0.23* | +0.16 | +0.25* | — | — | — |
| 4.3-4.7x | +1.55* | +1.29* | +1.45* | +1.40* | — | — | — |
| 4.8-5.2x | +0.95* | — | — | — | — | — | — |
| >5.2x | -0.21* | +0.53* | +0.82* | +0.72* | — | — | — |

## Verdicts

- hard gate: **KEEP**
- soft gate: **not_run**
- floor gate: **not_run**
- beta hat gate: **not_run**
- ema floor gate: **not_run**