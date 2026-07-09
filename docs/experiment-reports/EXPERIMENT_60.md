# E60 — Closed-Loop Residual Motion (online innovation-fit β̂ + midpoint-λ)

**Branch** `e60-closed-loop` · **commits** `43b301d` (core) → `f8ed818` (gate/scale) · FLUX.1-dev
4-bit 512px/28 Euler steps · cluster H100 · frozen fixture (canonical_prompts v1 + geneval extras)

**Verdict:** closed-loop β̂ as in-band gain **KILL** · closed-loop at extreme speed (≥4.3×) **KEEP**
· midpoint-λ **KILL** · prediction (b) innovation↔gain **PASS** (corr −0.44, n=1800) · prediction
(a) β̂ regime-dependence **shape PASS / level FAIL** — and the oracle explains exactly why.

## 1. Idea (docs/methods/closed_loop_residual_motion.md)

Every SeaCache refresh computes the true block residual r_k anyway. Just before it, the E58
predictor implies a forecast r̂_k = r_{k−1} + β·λ_k·Δr. The innovation e_k = r_k − r̂_k is a free,
causal, per-trajectory measurement of the secant model. E60 fits β online from it:

    β̂ = (μ·β_prior + Σ_j w^age λ_j⟨Δr_j, y_j⟩/‖Δr_j‖²) / (μ + Σ_j w^age λ_j²),  clamp [0, β_max]

(one normalized LS observation per fresh anchor, forgetting w, prior strength μ). β̂→0 on stale
secants recovers plain HorizonCache *by construction*. Midpoint-λ evaluates the progress
coefficient at (σ_i+σ_target)/2 — a midpoint-rule quadrature of the moving residual path. Zero
extra forwards; `rm_beta_mode="fixed"` is bit-identical to E58 (unit-tested, `test_e60_math.py`).

## 2. Runs

| stage | run | N | variants |
|---|---|---|---|
| smoke 1 | `results/horizon_cl_smoke/gen_20260708_212013` | 8×1 | raw β̂ (w0.85/w1.0), ±mid, fixed, plain |
| smoke 2 | `results/horizon_cl_smoke2/gen_20260708_214213` | 8×1 | gate g∈{0.08,0.12}, rescale κ=2.5 ±mid |
| consolidation | `results/horizon_cl_consol/gen_20260708_215716` | **100×2 = 200 paired** | plain, rmraw0.5, rmcl0.5, rmcl g0.08, rmcl κ2.5+mid × τ∈{0.3…1.4} (11,000 runs) |
| oracle | `results/horizon_cl_oracle/gen_20260709_015416` | 4×1, τ∈{0.5,1.0} | rmraw0.5 vs rmcl0.5, `--rm-oracle` |

Run:AI records: `runs/runai/20260709_001535__horizon_cl_e60_smoke__43b301d/`,
`…005158__horizon_cl_e60_smoke2__f8ed818/`, `…010719__horizon_cl_e60_consol__f8ed818/`.

## 3. Results (N=200 paired unless noted)

### β̂ is an excellent regime detector
- β̂ final p50 falls **monotonically** with achieved speedup: 0.22 @2.1× → 0.16 @2.7× → 0.12
  @3.4× → 0.07 @3.9× → 0.04 @4.5× → 0.03 @τ1.4 (prediction (a) *shape*).
- Per-image innovation ‖e‖₁/‖r‖₁ anti-correlates with the per-image fixed-RM−plain gain:
  **corr = −0.44 (n=1800)** — prediction (b) PASS. The innovation knows, per image, when RM helps.

### Closed loop vs fixed β (Q1) — the sign flips with speed
cl−fixed paired ΔPSNR (rmcl0.5, τ 0.3→1.4):
−0.59* / −0.55* / −0.26* / −0.05 / −0.19* / −0.03 / **+0.17*** / **+0.28*** / **+0.39***.
In-band the closed loop loses; beyond ~4.4× it wins exactly where fixed β inverts. The gate
(g0.08: −0.40*…−0.14* in-band) and rescale (κ2.5+mid: reintroduces −0.76* @τ1.4) variants cannot
close the in-band gap: the per-run used-β trace shows β̂ is **low mid-trajectory and rises late**
— drift coherence is σ-dependent, so no per-trajectory scalar matches fixed 0.5.

### High-speed penalty (Q2) — softened, not erased
fixed−plain: −0.26* @4.47×, −0.42* @5.3×, −0.97* @τ1.4 (E59 reproduced). cl−plain: ns at
3.9–4.5×, −0.13* @5.22×, −0.58* @τ1.4. The closed loop removes the penalty where it has ≥5–6
anchors to learn from and roughly halves it at τ1.4 (3–4 anchors — too few observations).

### vs SeaCache (matched achieved speedup, best per family)
| band | plain | fixed-β RM | closed loop |
|---|---|---|---|
| 1.8–2.2× | +1.37* | **+2.41*** | +1.94* |
| 2.3–2.7× | +2.00* | **+2.88*** | +2.54* |
| 2.8–3.2× | +0.77* | +0.86* | **+0.92*** |
| 3.3–3.7× | +1.84* | **+2.27*** | +2.19* |
| 3.8–4.2× | **+0.31*** | +0.23* | +0.16 |
| 4.3–4.7× | **+1.55*** | +1.29* | +1.45* |
| 4.8–5.2×† | **+0.95*** | — | — |
| >5.2וbra | −0.21* | +0.53* | **+0.82*** |

† beyond the swept SeaCache frontier (4.39× quantization ceiling) — conservative.
The closed loop is the **best method in the >5.2× band** and never catastrophic anywhere: a
zero-tuning "safe default" whose worst case (−0.59* in-band vs fixed) is far smaller than the
mistakes it prevents (fixed's −0.97* at τ1.4; plain's ~−1 dB in-band vs fixed).

### The oracle smoking gun (pointwise ≠ PSNR objective)
| gain | τ | frozen err | motion err | pointwise Δ |
|---|---|---|---|---|
| fixed β=0.5 | 0.5 | 0.260 | 0.283 | **−8.8%** (worse) |
| fixed β=0.5 | 1.0 | 0.499 | 0.629 | **−26.0%** (worse) |
| LS β̂ | 0.5 | 0.260 | 0.256 | +1.3% |
| LS β̂ | 1.0 | 0.468 | 0.484 | −3.4% |

The LS gain does its job perfectly — it (nearly) eliminates the pointwise degradation — and
*that is exactly why it loses PSNR in-band*: the fixed β=0.5 that wins PSNR makes the pointwise
prediction strictly worse. **The RM benefit routes through accumulated drift correction that the
anchor-consistency objective cannot see.** Attenuation bias (errors-in-variables on the noisy
secant, noise growing with τ) compounds the level gap and simultaneously explains why β̂ still
works as a *detector*.

## 4. Honest failure taxonomy

Not a tuning failure. The estimator recovers synthetic gains exactly (unit tests), adapts within
trajectories (micro), and achieves its own objective (oracle). The failure is the **objective**:
pointwise-MMSE β ≈ 0.2 ≠ PSNR-optimal β ≈ 0.5–0.75 in-band. Any self-calibrating gain driven by
anchor-consistency signals will under-correct; E60 closes the "smarter scalar gain" direction the
same way E59 closed the "better basis" direction.

## 5. What survives

1. The **operating rule** stands, now mechanistically grounded: fixed-β RM ≤3.5×, plain beyond.
2. The closed loop is a legitimate **zero-cost extreme-speed extension**: cl>fixed CI+ at ≥4.47×,
   best-in-band >5.2× vs SeaCache — use it if a single no-knob method must run at any speed.
3. The **innovation is a validated per-image regime detector** (corr −0.44) — the right home for
   it is the *jump policy* (innovation-gated stride), not the gain.

## 6. Artifacts

- report `reports/horizon_cache_closed_loop.html` (+ `_summary.md`, `_summary.json`, `_assets/`)
- analysis `experiments/horizon_cache/cl_analysis.py`, report builder `cl_report.py`,
  tests `test_e60_math.py`
- manifest `experiments/manifests/E60.json`
