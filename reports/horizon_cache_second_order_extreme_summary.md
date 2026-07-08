# E59 — Second-Order Residual Hold + Extreme-Speed Comparison

**Commit** `916c6c9` · FLUX 512px/28 · cluster H100

**Headline.** First-order RM (rmraw0.5_adaptive_1.5) vs SeaCache at matched speed: best +2.90 dB @ 2.50× (CI [+2.57,+3.23]); highest CI-positive band >5.2x. Second-order − first-order: best +0.04 dB @ 3.88× (rm2raw0.5b0.1_adaptive_1.25, CI [+0.01,+0.08]) → KILL.

**Bounded claim.** On FLUX 512px/28 the extreme-speed comparison covers ~2×–4.4×. Plain HorizonCache stays above SeaCache through 4.8-5.2x, first-order RM through >5.2x, second-order RM through >5.2x. Second-order term: ρ2 p50=1.37 (curvature ≈ velocity — second differences are mostly noise); cosΔ p50=0.10 (weakly stable); corr(ρ2, SO−FO gain)=-0.03, corr(cosΔ, gain)=+0.07 — the diagnostic does NOT predict when SO helps.

## Δ vs SeaCache by speed band (best per family; * = CI excludes 0)

| band | plain | first-order RM | second-order RM | quad RM | SO−FO |
|---|---|---|---|---|---|
| 1.8-2.2x | +1.37* @2.12× | +2.41* @2.12× | +2.33* @2.12× | — | -0.08* |
| 2.3-2.7x | +2.00* @2.50× | +2.90* @2.50× | +2.86* @2.50× | — | -0.01 |
| 2.8-3.2x | +0.77* @3.08× | +0.86* @3.08× | +0.73* @3.08× | — | -0.13* |
| 3.3-3.7x | +1.86* @3.40× | +2.29* @3.40× | +2.23* @3.40× | — | -0.04 |
| 3.8-4.2x | +0.34* @3.86× | +0.26* @3.86× | +0.29* @3.88× | — | +0.04* |
| 4.3-4.7x | +1.55* @4.47× | +1.31* @4.47× | +1.14* @4.47× | — | -0.14* |
| 4.8-5.2x | +0.95* @5.17× | +0.64* @4.98× | — | — | — |
| >5.2x | -0.20* @5.30× | +0.53* @5.30× | +0.40* @5.30× | — | -0.14* |

**Highest CI-positive band vs SeaCache:** plain: 4.8-5.2x, FO-RM: >5.2x, SO-RM: >5.2x

## Verdicts

- first-order RM at extreme speed: **STRONG_KEEP**
- second-order RM: **KILL**
- gated second-order: **not_run**

## Second-order diagnostic

ρ2 p50=1.37 (curvature ≈ velocity — second differences are mostly noise); cosΔ p50=0.10 (weakly stable); corr(ρ2, SO−FO gain)=-0.03, corr(cosΔ, gain)=+0.07 — the diagnostic does NOT predict when SO helps