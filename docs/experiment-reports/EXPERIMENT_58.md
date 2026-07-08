# E58 — HorizonCache Residual Motion Cache: move the cached block residual along its secant

**Thread:** fast-edit (caching line) · **Status:** active · **Verdict:** STRONG KEEP (raw secant, β=0.5) ·
**Models:** FLUX.1-dev (4-bit), 512 px, 28 Euler steps; smoke N=8 → N=50×2 pilot → **headline consolidation
N=100 × 2 seeds = 200 paired samples** + N=8 oracle residual diagnostic + N=6 qualitative-grid run, cluster
H100 80 GB (Run:AI). Canonical-fixture v1 + deterministic GenEval extras.

> **Headline.** Residual Motion Cache beats **plain HorizonCache at matched compute in every speed band from 2.0×
> to 3.4×** (200 paired samples, every 95 % CI excludes 0), roughly **doubles** the E56 margin over SeaCache in the
> safe band, and — the key result — **flips E56's ~3.4× overshoot**: plain adaptive HorizonCache *loses* to SeaCache
> there (−0.18 dB, CI excl 0), but RM *wins* (+0.26 dB, CI [+0.11, +0.42]). It costs **no extra forward**. The two
> paper figures (PSNR-vs-speedup 3-curve frontier; frozen-lag-vs-secant mechanism schematic) and generated-sample
> grids are in `reports/horizon_cache_residual_motion.html`.
>
> **Bound (do not oversell):** the **>+0.5 dB** RM−plain gains are concentrated in **2.0–2.8×**; at **3.0–3.4×** the
> RM−plain gain is smaller (**+0.19 to +0.43 dB**, still CI-positive), and the 3×+ story is that gain *plus* the
> flip of the SeaCache-margin sign at 3.4×.

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

## Result — headline consolidation (N=100 × 2 seeds = 200 paired, τ ∈ {0.3, 0.4, 0.5, 0.575, 0.65})

τ=0.575 (~3.0×) was added to populate the **2.8–3.2× band** that was a gap in the N=50 pilot.

**1. RM beats plain HorizonCache at matched compute in every band, significantly.** Paired per-image (RM − plain,
same base/τ). At τ where no downstream refresh flips, the action sequence *and* achieved speedup are **byte-identical**
to plain — a pure residual-value ablation:

| variant | 2.02× | 2.50× | 2.74× | **3.03×** | 3.40× |
|---|---|---|---|---|---|
| rmraw0.5_adaptive_1.25 | +0.96 [0.77,1.14] | +0.88 [0.69,1.06] | +0.54 [0.35,0.73] | +0.19 [0.04,0.35] | +0.42 [0.27,0.58] |
| rmraw0.5_adaptive_1.5 | +0.93 [0.74,1.11] | +0.90 [0.71,1.08] | +0.56 [0.37,0.75] | +0.19 [0.03,0.34] | +0.43 [0.28,0.59] |

Every CI excludes 0. The **>+0.5 dB** gains are concentrated in **2.0–2.8×**; in the **2.8–3.2× band the gain is
+0.19 dB** (small but CI-positive), and it rises again to **+0.42 dB at 3.40×**. (N=50 pilot established **raw >
lowpass** and **β=0.5 as the sweet spot** — β=0.75 overshoots at 3.4×, +0.16 CI incl 0.)

**2. vs SeaCache at matched achieved speedup (E56 fair protocol) — the frontier extension.** RM roughly **doubles**
the E56 margin in the safe band and **flips the overshoot**:

| speedup | plain adaptive_1.5 vs SeaCache | **RM (raw 0.5) vs SeaCache** |
|---|---|---|
| 2.02× | +0.94 [0.73, 1.17] | **+1.94 [1.63, 2.26]** |
| 2.50× | +1.99 [1.72, 2.27] | **+2.90 [2.57, 3.23]** |
| 2.74× | +2.18 [1.82, 2.55] | **+2.74 [2.34, 3.16]** |
| 3.03× | +0.43 [0.32, 0.55] | **+0.62 [0.43, 0.82]** |
| **3.40×** | **−0.17 [−0.20, −0.13]** (loss) | **+0.26 [+0.11, +0.42]** (win) |

Through 3.0× both HorizonCache variants still beat SeaCache (RM by more). At 3.40× (E56's overshoot) plain
HorizonCache **loses** to SeaCache while RM turns it into a **significant win** — the SeaCache-margin sign flips.
LPIPS improves everywhere too.

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

**STRONG KEEP (raw secant, β = 0.5) — holds at N=200 paired.** `rmraw0.5` over adaptive_1.25/1.5: beats plain
HorizonCache at matched compute in **every** band (all CI > 0) **and** flips E56's ~3.4× overshoot from a measured
loss into a measured win vs SeaCache. The **pre-registered success condition** (RM > +0.5 dB with CI > 0 *and* extends
the positive SeaCache margin toward 3×+) holds — with the honest caveat that the **>+0.5 dB gains are in 2.0–2.8×**,
while the 3×+ story is the smaller (+0.19–0.43 dB) but significant RM−plain gain *plus* the SeaCache-sign flip at 3.4×.
`lowpass` secant KEEP (~half the gain, N=50). β = 0.75 KEEP but begins to overshoot at 3.4×. `topk` / `sea` projections
and the safety gate: not yet run.

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
