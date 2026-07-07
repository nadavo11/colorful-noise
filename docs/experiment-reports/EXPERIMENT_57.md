# E57 — HorizonCache-PC: cached-endpoint predictor-corrector jumps

**Thread:** fast-edit (caching line) · **Status:** dead-end · **Verdict:** KILL (cached-endpoint PC on FLUX) ·
**Models:** FLUX.1-dev (4-bit), 512 px, 28 Euler steps; smoke N=8 × 1 seed + fresh-endpoint oracle ablation,
cluster H100 80 GB (Run:AI).

## Hypothesis

E56 found the headroom-adaptive stride overshoots at ~3.4× because long-stride **Euler truncation
error** dominates. Attack it with a cheap **Heun/trapezoid** correction that does **not** add a full
transformer recompute:

```
x_pred = x_i + Δσ·v_i
v_pred = cached_velocity(x_pred, σ_target; r_anchor)   # reuse the SAME cached residual, ≈1/L
x_corr = x_i + Δσ·[(1-α)·v_i + α·v_pred]
```

Claim to test: a cached endpoint correction reduces truncation error while preserving most of the
compute savings, and can push the safe band from ~2.7× toward 3.0–3.3×.

## Method

New PC path in `experiments/horizon_cache/` (non-PC path byte-identical to E56, fair by construction):

- **`v_pred`** is a *second cached forward* at the predicted endpoint reusing the same block-stack
  residual `r_anchor` — one extra cached forward (≈1/L), **accounted separately** (`ComputeLedger.
  num_pc_endpoint_cached_forwards`) so PC never claims free speedup.
- **α ∈ {0.25, 0.5, 0.75, 1.0}** (0 = plain Euler, 0.5 = trapezoid, 1.0 = endpoint-only).
- **Curvature score** `‖v_pred−v_i‖₁/‖v_i‖₁` with optional **cancel** / **shrink** gates.
- **Oracle ablation** `pcoracle<α>_`: `v_pred` is a **fresh full forward** (does not overwrite the cache
  anchor; accounted as a full forward) — isolates whether the cached endpoint's *staleness* is the cause.
- Same regrid scheduler, matched-achieved-speedup comparison, and percentile bootstrap as E56.

## Result (smoke N=8 × 1 seed, τ ∈ {0.4, 0.5, 0.65})

**PC is a near-no-op.** PC − plain (paired, same family/τ) is ≤ 0 in **every** band, and PC sits at
slightly *lower* achieved speedup (the endpoint cost):

| family | τ | PC speedup | plain speedup | PC − plain ΔPSNR | 95% CI |
|---|---|---|---|---|---|
| adaptive_1.5 | 0.4 | 2.48× | 2.50× | −0.00 | [−0.04, +0.06] |
| adaptive_1.5 | 0.5 | 2.71× | 2.74× | −0.05 | [−0.09, −0.01] |
| adaptive_1.5 | 0.65 | 3.35× | 3.40× | −0.02 | [−0.04, −0.01] |
| adaptive_2.0 | 0.4 | 2.35× | 2.37× | −0.06 | [−0.10, −0.02] |
| adaptive_2.0 | 0.5 | 2.71× | 2.74× | −0.06 | [−0.09, −0.02] |
| adaptive_2.0 | 0.65 | 3.35× | 3.40× | −0.02 | [−0.04, −0.01] |

In the overshoot band it does **not** help: 3.35× / 20.92 dB (PC) vs 3.40× / 20.95 dB (plain).

**Why (curvature diagnostic).** `‖v_pred−v_i‖₁/‖v_i‖₁` has **p50 = 0.014, p95 = 0.026** — the cached
endpoint is only ~1–3 % from the *start* velocity, so the trapezoid correction barely moves the state.
The correction PC needs lives in the **block-stack-driven curvature** of `v`, which the cache **freezes
by construction**; reusing `r_anchor` at `x_pred` cannot recover it.

**Mechanism proven by the oracle.** A *fresh* endpoint velocity **does** improve quality where the cached
one cannot — but at a full-forward cost per jump that collapses the speedup:

| τ | plain | PC cached | PC oracle (fresh) α0.5 | PC oracle α1.0 |
|---|---|---|---|---|
| 0.5 | 2.74× / 24.17 dB | 2.71× / 24.11 | **1.77× / 24.46** (+0.29) | 1.76× / 24.67 (+0.50) |
| 0.65 | 3.40× / 20.95 dB | 3.35× / 20.92 | 1.97× / 21.13 (+0.18) | 1.97× / 21.21 (+0.26) |

(LPIPS improves too, 0.135 → 0.114 at τ0.5.) So the failure is specifically the **staleness** of the
cached endpoint, not the correction idea — and even a fresh endpoint isn't worth it: at ~1.8–2.0× it
recovers only +0.2–0.5 dB, i.e. the compute it spends is better spent simply **not jumping so hard**
(dropping to E56's safe band).

## Verdict

**DEAD-END / KILL (cached-endpoint predictor-corrector on FLUX).** `pc_alpha_0.5` (cached) KILL — no band
beats plain HorizonCache and the endpoint call slightly lowers speedup; `pc_alpha` 0.25/0.75/1.0 KILL;
curvature cancel/shrink KILL (signal real but ~1–3 %, too small to act on). Oracle fresh-endpoint PARK
(diagnostic only — proves the mechanism, no speedup). The gate correctly stopped before the N=50–100
consolidation. **E56's headroom-adaptive stride remains the frontier.**

Failure attribution (as required): **the cached endpoint velocity is too stale** (`v_pred ≈ v_i`);
secondarily the endpoint cost slightly erases speedup; the curvature signal is real but uninformative
(too small); aggressive jumps at 3×+ are effectively unsafe on FLUX *and cannot be cheaply corrected from
cached state*.

## Next

Do **not** pursue cached-endpoint PC further on FLUX. If revisiting 3×+: (1) a **partial-stack / low-rank
endpoint refresh** (some blocks fresh) to get real curvature below full cost; (2) accept E56's ~1.7–2.7×
safe band; (3) a learned per-jump safety gate supervised by the E56 frontier-improvement label. SD3
(E56's open direction) is higher value.

## Artifacts

- Report: `reports/horizon_cache_pc.html` · `reports/horizon_cache_pc_summary.{md,json}` ·
  `reports/horizon_cache_pc_assets/` (frontier, PC−plain, curvature diagnostic, mechanism diagram)
- Runs: `results/horizon_pc_smoke/gen_20260707_140532/` (PC vs plain) ·
  `results/horizon_pc_oracle/gen_20260707_141523/` (fresh-endpoint oracle) ·
  `runs/runai/20260707_170232__horizon_pc_e57_smoke__2f53bfb/` ·
  `runs/runai/*__horizon_pc_e57_oracle__43e46c7/`
- Code: `experiments/horizon_cache/{flux_gen,scheduler,policy,run,pc_analysis,pc_report}.py` ·
  manifest `experiments/manifests/E57.json`
