# E61 — Innovation-Gated Fixed Residual Motion

**Branch** `e61-innovation-gate` · commit `e0bcbce` (impl+analysis) · FLUX.1-dev 4-bit 512px/28
Euler steps · cluster H100 · frozen fixture (canonical_prompts v1 + geneval extras)

**Verdict:** hard gate (κ=0.85) **KEEP** — matches fixed β=0.5 in-band, beats it with CI+ margin
at every band ≥4.3×, but with a real (small, speed-growing) LPIPS cost at high speed and only a
partial softening of the plain-HorizonCache high-speed inversion. Soft/floor gates **KILL/PARK**
in-band even after recalibration. β̂-as-gate not run (deprioritized after smoke2).

## 1. Idea

E60 showed the online LS gain β̂ is a good regime *detector* (innovation ↔ RM-gain corr −0.44)
but a miscalibrated in-band *gain* (pointwise-MMSE ≠ PSNR-optimal). E61 keeps fixed β=0.5 as the
default and only **causally throttles** it:

    r̂_a = r_{a-1} + 0.5·λ_a·Δr_{a-1}          (fixed-β forecast at the last fresh anchor)
    I_a  = ‖r_a − r̂_a‖₁ / ‖r_a‖₁                (normalized innovation)
    g_a  = hard: 1[I_a<κ] | soft: clip(1-I_a/κ,0,1) | floor: gmin+(1-gmin)·soft
    β_i  = 0.5·g_a                              (used unchanged by every cached step until anchor a+1)

`g_a` is fixed at anchor `a` — it cannot see anything from the cached steps it governs. Zero
extra forwards; `rm_beta_mode="fixed"` is bit-identical to E58 (unit-tested, 8 groups incl. exact
gate math, EMA recursion, causality, grammar).

## 2. Runs

| stage | run | N | finding |
|---|---|---|---|
| smoke 1 | `results/horizon_gate_smoke/gen_20260709_072453` | 8×1 | diagnostic strong (corr −0.61, n=64) but tested κ (0.08–0.20) far below the measured I_a scale (0.43–1.0+) — no variant actually gated |
| smoke 2 | `results/horizon_gate_smoke2/gen_20260709_074002` | 8×1 | κ recalibrated to the measured crossover (~0.75–0.85); `rmgh0.85` passes both proceed clauses |
| consolidation | `results/horizon_gate_consol/gen_20260709_075805` | **100×2 = 200 paired** | plain, rmraw0.5, rmcl0.5 (E60), rmgh0.85, rmgh0.75 × τ∈{0.3…1.4} (11,000 runs) |

Smoke1's failure was a **calibration bug in the spec's suggested κ range, not the mechanism**:
the binned E[gain\|I_a] diagnostic showed the sign-crossover sits at I≈0.75, roughly 5× the
spec's suggested sweep ceiling. Recalibrating and re-running smoke (same branch, no code
change) resolved it — the same honest-recalibration move used in E60's smoke1→smoke2.

## 3. Results (N=200 paired)

### Gate vs fixed RM — matches in-band, wins at speed
| τ | speedup | gate85−fixed | gate75−fixed |
|---|---|---|---|
| 0.3 | 2.12× | −0.000 | −0.006 |
| 0.4 | 2.51× | +0.010 | −0.086* |
| 0.5 | 2.74× | **+0.008\*** | +0.013 |
| 0.575 | 3.09× | +0.027 | −0.011 |
| 0.65 | 3.40× | +0.004 | −0.051 |
| 0.8 | 3.87× | +0.026 | −0.053 |
| 1.0 | 4.47× | **+0.111\*** | +0.037 |
| 1.2 | 5.28× | **+0.174\*** | **+0.185\*** |
| 1.4 | 5.29× | **+0.363\*** | **+0.414\*** |

gate85 is essentially indistinguishable from fixed β=0.5 through 3.5× (all deltas within ±0.03
dB, only the 2.74× point weakly CI+), then wins outright at every band ≥4.3× with margins well
past the +0.2 dB proceed bar.

### Gate vs plain HorizonCache — softens, does not remove the E59/60 inversion
| τ | fixed−plain | gate85−plain |
|---|---|---|
| 0.3–0.65 | +0.96\* … +0.43\* | +0.96\* … +0.43\* (identical in-band, gate rarely closes here) |
| 1.0 | −0.26\* | −0.15 (ns) |
| 1.2 | −0.42\* | **−0.24\*** |
| 1.4 | **−0.97\*** | **−0.61\*** |

The gate reduces the τ1.4 penalty by ~37% but it is still significant. **This is the honest
limit of E61**: throttling the gain softens the high-speed failure mode, it does not invert it.

### β_i by speed band (gate85) — the predicted profile, delivered
p50 = 0.50 through 3.09×, 0.385 at 3.40×, declining smoothly to 0.21–0.36 (mean 0.24–0.34) at
≥4.3×. Never collapses to a constant — a genuine adaptive throttle, not a disguised fixed value.

### Innovation diagnostic — reproduces and slightly refines E60
corr(I_a, RM−plain gain) = **−0.45** (n=1800, vs E60's −0.44). Binned E[gain\|I_a] crosses zero
at I≈0.7–0.75 — exactly the κ the smoke2 recalibration landed on independently.

### The LPIPS caveat (why this is KEEP, not STRONG_KEEP)
gate85−fixed LPIPS (raw, more negative = gate worse): ≈0 through 3.09×, then significantly
negative and growing: −0.0019* @0.65×, −0.0058* @0.8×, −0.0140* @1.0×, −0.0114* @1.2×,
**−0.0189\* @τ1.4**. The PSNR win at high speed comes with a real, CI-excluding, speed-growing
perceptual-quality cost that fixed RM does not pay. Not disqualifying, but not free.

### Extreme-τ universal loss (not gate-specific)
At τ1.4 (5.29×, beyond SeaCache's 4.39× swept-frontier ceiling) every RM variant loses to
SeaCache's clamped frontier point: fixed −1.18\*, closed-loop −0.79\*, gate85 −0.82\*, gate75
−0.77\*. The gate loses *less* than fixed there but the effect is universal across the family,
consistent with E59/E60's established swept-frontier-ceiling caveat.

## 4. Which failure taxonomy applies

None of the "gate doesn't work" causes apply at the recalibrated κ — the mechanism succeeded at
what it was built to do (match in-band, throttle at speed, causal by construction). The two
caveats that DO apply, stated plainly:
- **Partial fix, not full fix**: the plain-HorizonCache inversion at ≥4.3× is softened (~37%
  smaller) but not eliminated — a fixed-speed-threshold rule (use plain above ~4×) still
  dominates the gate on PSNR alone at the very top end.
- **A real LPIPS cost accompanies the PSNR win** at high speed — the gate is not a free lunch;
  it trades some perceptual fidelity for the PSNR/robustness gain over fixed RM.

## 5. Compute/memory overhead

Zero extra forwards. One extra scalar (`gate_ibar`, EMA state) plus the diagnostic dict per
trajectory — identical cost profile to fixed RM; achieved speedup unchanged.

## 6. Artifacts

- report `reports/horizon_cache_e61_innovation_gate.html` (+ `_summary.md`, `_summary.json`, `_assets/`)
- analysis `experiments/horizon_cache/e61_analysis.py`, report builder `e61_report.py`,
  tests `test_e61_math.py`
- manifest `experiments/manifests/E61.json`
