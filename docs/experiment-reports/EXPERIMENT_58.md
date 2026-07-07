# E58 — HorizonCache Residual Motion Cache: move the cached block residual along its secant

**Thread:** fast-edit (caching line) · **Status:** active · **Verdict:** STRONG KEEP (raw secant, β=0.5) ·
**Models:** FLUX.1-dev (4-bit), 512 px, 28 Euler steps; smoke N=8 → **consolidation N=50 × 2 seeds = 100 paired
samples** + N=8 oracle residual diagnostic, cluster H100 80 GB (Run:AI). Canonical-fixture v1 + deterministic
GenEval extras.

> **Headline.** Residual Motion Cache beats **plain HorizonCache at matched compute** by **+0.4 to +0.95 dB across
> the entire 2.0–3.4× range** (100 paired samples, every 95 % CI excludes 0), roughly **doubles** the E56 margin
> over SeaCache in the safe band, and — the key result — **flips E56's ~3.4× overshoot**: plain adaptive HorizonCache
> *loses* to SeaCache there (−0.14 dB, CI excl 0), but RM *wins* (+0.29 dB, CI [+0.04, +0.53]). It costs **no extra
> forward**. Paper-ready report: `reports/horizon_cache_residual_motion.html`.

## Hypothesis

E56 established the headroom-adaptive stride as a STRONG KEEP in ~1.7–2.7×, overshooting at ~3.4×. E57 tried to fix
the overshoot with a cached-endpoint predictor–corrector and **failed**: re-querying the cheap output head at the
jump endpoint reuses the **frozen** block-stack residual, so `v_pred ≈ v_i` (curvature ~1–3 %) and the correction is
a near-no-op. The lesson: *any* cheap correction that reads through the frozen residual is limited the same way.

So E58 attacks the residual itself. Instead of treating the cached block residual as constant, **predict its slow
motion** from recent fresh residuals:

```
r_pred(t) = r_anchor + β · λ(t) · P(r_anchor − r_prev)
v         = head( front(x_t, σ_t) + r_pred(t) )
```

where `r_anchor` / `r_prev` are the two most recent **fresh** block residuals, `P` is an optional stable-subspace
projection, `λ(t)` is normalized progress since the anchor, and `β` a conservative shrink. Crucially this is **pure
tensor arithmetic** — it does **not** call the block stack on cached/jump steps, so the achieved speedup is identical
to plain HorizonCache.

## Method

New RM path in `experiments/horizon_cache/` (non-RM path byte-identical to E56/E57, fair by construction):

- **History.** On every fresh forward the sampler shifts `r_prev ← r_anchor` and stores σ / filtered-h / step-age at
  both anchors (`FluxCacheState`). A cached/jump step forms the secant `Δr = r_anchor − r_prev`.
- **Projection P**: `raw` (identity), `lowpass` (avg-pool + upsample over the token grid), `topk` (most energetic
  channels), `sea` (SeaCache Wiener filter (a,b)=(1−σ,σ) on the residual grid).
- **λ(t)**: σ-linear `(σ_t−σ_anchor)/(σ_anchor−σ_prev)` (default), step-age, or SeaCache-style h-drift; clamped to
  `[0, λ_max]`.
- **β** shrink ∈ {0, 0.25, 0.5, 0.75, 1.0} (0 ≡ plain HorizonCache).
- **Cost**: no extra block-stack forward. `ComputeLedger` counts residual-motion applications but does **not** change
  `block_stack_equiv_cost`; achieved speedup == plain HorizonCache. Memory: one extra residual (`r_prev`) in bf16.
- **Oracle diagnostic** (`--rm-oracle`): additionally run the **true** residual `r_true = B(h_t)` at cached steps and
  score `‖r_pred − r_true‖` vs `‖r_anchor − r_true‖` (a full forward per cached step, diagnostic only, not charged).
- Same regrid scheduler, matched-achieved-speedup comparison, and percentile bootstrap as E56.

Grammar `rm<proj><beta>_<base>` (e.g. `rmraw0.5_adaptive_1.5`). λ-mode / λ-max / gate ρ are run-level flags.

## Result — consolidation (N=50 × 2 seeds = 100 paired, τ ∈ {0.3, 0.4, 0.5, 0.65})

**1. RM beats plain HorizonCache at matched compute, everywhere, significantly.** Paired per-image (RM − plain, same
base/τ). At τ where no downstream refresh flips (the low/mid-τ regime), the action sequence *and* achieved speedup
are **byte-identical** to plain — a pure residual-value ablation:

| variant | 2.0× (τ0.3) | 2.5× (τ0.4) | 2.74× (τ0.5) | 3.4× (τ0.65) |
|---|---|---|---|---|
| rmraw0.5_adaptive_1.25 | +0.97 [0.71,1.24] | +0.91 [0.67,1.16] | +0.63 [0.33,0.93] | +0.43 [0.19,0.67] |
| rmraw0.5_adaptive_1.5 | +0.86 [0.58,1.13] | +0.95 [0.70,1.21] | +0.67 [0.37,0.98] | +0.42 [0.18,0.68] |
| rmraw0.5_adaptive_2.0 | +0.85 [0.57,1.12] | +0.93 [0.68,1.18] | +0.65 [0.36,0.95] | +0.42 [0.18,0.67] |
| rmlowpass0.5_adaptive_1.5 | +0.44 [0.26,0.61] | +0.52 [0.34,0.71] | +0.45 [0.20,0.70] | +0.33 [0.13,0.52] |
| rmraw0.75_adaptive_1.5 | +0.90 [0.55,1.26] | +0.91 [0.58,1.24] | +0.62 [0.24,1.01] | +0.16 [−0.13,0.45] |

Every raw β=0.5 CI excludes 0. **raw > lowpass** (raw keeps the full secant; lowpass discards useful residual
motion). **β = 0.5 is the sweet spot**: β = 0.75 still wins in the safe band but starts to **overshoot at 3.4×**
(+0.16, CI includes 0).

**2. vs SeaCache at matched achieved speedup (E56 fair protocol) — the frontier extension.** RM roughly **doubles**
the E56 margin in the safe band and **flips the overshoot**:

| speedup | plain adaptive vs SeaCache | **RM (raw 0.5) vs SeaCache** |
|---|---|---|
| 2.01× | +0.89 [0.64, 1.16] | **+1.87 [1.41, 2.36]** |
| 2.50× | +2.19 [1.78, 2.62] | **+3.14 [2.65, 3.63]** |
| 2.74× | +1.26 [1.01, 1.49] | **+1.93 [1.55, 2.29]** |
| **3.40×** | **−0.14 [−0.18, −0.10]** (loss) | **+0.29 [+0.04, +0.53]** (win) |

At 3.40× (≈3.0× SeaCache-equivalent) plain HorizonCache **loses** to SeaCache — this is E56's overshoot — while RM
turns it into a **significant win**. LPIPS improves everywhere too (ΔLPIPS +0.012 to +0.046, sign = better).

**3. Mechanism — the honest nuance (oracle, N=8).** The secant does **not** better-predict the instantaneous true
residual: frozen error **0.269** vs residual-motion error **0.285** → **−5.6 %** (slightly *worse* pointwise). This
is not a contradiction with the quality gain — it is the mechanism:

> The oracle measures **local** single-step residual accuracy at the (already-drifted) cached state; PSNR measures
> **global** fidelity to the full trajectory. The frozen cached residual is systematically **stale** (it lags the
> evolving true residual); nudging it forward along its recent secant reduces the **accumulated** velocity drift over
> the cached run even though it overshoots any single instantaneous residual. **The win is drift/bias correction, not
> a better per-step predictor.** Residual motion is real but small — extrapolation ratio p50 ≈ 13 %, p95 ≈ 36 % of
> ‖r_anchor‖.

## Verdict

**STRONG KEEP (raw secant, β = 0.5).** `rmraw0.5` over adaptive_1.25/1.5/2.0: beats plain HorizonCache by +0.4–0.95 dB
at matched compute (100 pairs, all CI > 0) **and** extends the positive SeaCache margin past 2.7× to ~3.4×, flipping
E56's overshoot from a measured loss into a measured win. `lowpass` secant KEEP (same sign, ~half the gain). β = 0.75
KEEP but begins to overshoot at 3.4×. `topk` / `sea` projections and the safety gate: not yet run.

**Bounded claim (do not oversell):** this is a **matched-compute quality gain plus a frontier extension to ~3.4× on
FLUX text2img**, obtained for free (no extra forward). It is **not** a better per-step residual predictor
(oracle-negative), and the mechanism is **accumulated-drift correction** — it should be stated that way. Not yet
replicated off FLUX (SD3 needs a new MMDiT harness).

## Next

1. Fold RM into the E56 consolidated frontier as the new headline — RM is free, so it **strictly dominates** plain
   HorizonCache.
2. Sweep β ∈ [0.4, 0.6] × λ-mode {σ, h}; test `topk`/`sea` projections and a per-step gate at 3.4×+ to push past the
   overshoot further.
3. Multi-seed / N=100 headline + qualitative grids.
4. SD3 (new MMDiT harness) — does residual-secant motion generalize off FLUX?
5. A learned residual-motion predictor (small MLP on the secant + σ) if the linear secant leaves gains on the table.

## Artifacts

- Report: `reports/horizon_cache_residual_motion.html` · `reports/horizon_cache_residual_motion_summary.{md,json}` ·
  `reports/horizon_cache_residual_motion_assets/` (frontier, RM−plain, method diagram, oracle residual, motion magnitude)
- Runs: `results/horizon_rm_n50/gen_20260707_164658/` (consolidation, 100 paired) ·
  `results/horizon_rm_smoke/gen_20260707_163300/` (smoke N=8, PROCEED) ·
  `results/horizon_rm_oracle_n8/gen_20260707_181928/` (oracle residual diagnostic) ·
  `runs/runai/20260707_194500__horizon_rm_e58_consol__f7ba9d1/`
- Code: `experiments/horizon_cache/{flux_gen,scheduler,policy,run,rm_analysis,rm_report}.py` ·
  manifest `experiments/manifests/E58.json`
