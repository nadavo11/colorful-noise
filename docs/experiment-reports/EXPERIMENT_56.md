# E56 — HorizonCache: causal safe-horizon prediction (fresh / cache / jump)

**Thread:** fast-edit (caching line) · **Status:** active · **Verdict:** KEEP (v0, narrow-band) ·
**Models:** FLUX.1-dev (4-bit transformer on a 24 GB A5000), 512 px, 28 Euler steps, N=20 prompts
(canonical-fixture v1 + deterministic GenEval extras), 1 seed. SD3 unavailable (no local weights);
editing capability-detected but not executed.

## Hypothesis

SeaCache reads one cheap score off the modulated input `h` (before the L-block stack,
Wiener-filtered relative-L1) and decides **refresh vs cache**. HorizonCache asks whether the
*same* signal supports **action selection**: predict how far the current computation stays
trustworthy — the local *safe integration horizon* — and choose one of

```
fresh   full forward                              (cost 1)
cache   freeze block residual, normal σ-stride     (cost ≈1/L)
jump    reuse v + a longer σ-stride that REMOVES an integration node (cost ≈0 + a node saved)
```

Prior caution (E53): SeaCache's rel-L1 ranks the *oracle* safe-jump length only weakly
(Spearman −0.50), so a naive "jump when accumulated score < ε" gate is expected to fail — the
policy must add features and hard safety gating. We test a hand-designed rule (**v0**) and a
learned tabular safe-horizon model (**v1**).

## Method

New module `experiments/horizon_cache/` reusing the SeaCache harness
(`flux_seacache_dp_shortcuts.py`) for the faithful Wiener-filtered-`h` signal and the
cached-residual step, re-implemented as an explicit **first-order Euler** flow loop so it can
additionally jump (multistep solvers excluded by design).

- **Scheduler.** `drop` = on-grid skip (σ_i → σ_{i+2}); `regrid` = off-grid
  `σ_target = σ_i + jf·(σ_{i+1} − σ_i)`, then re-space the remaining tail to 0 (net −1 node).
  Every jump is logged (old/target σ, jf, skipped-equivalent nodes, regridded?).
- **v0.** In cache territory (acc < τ), the jump factor is chosen by **headroom**
  `h = 1 − acc/τ` (deck adaptive-jump), gated hard on σ-band, instantaneous relL1, h-cosine
  drift, and remaining tail length. **Jumps disabled ⇒ SeaCache exactly (fair by identity).**
- **v1.** sklearn HistGradientBoosting on causal cheap features only (σ geometry + the
  SeaCache `h` signal), trained on **rollout safe-horizon labels**: from each visited state,
  branch every candidate action, continue **full** for H nodes, and label the largest-horizon
  action whose latent-L2 damage vs the full continuation is below tolerance. Asymmetric cost:
  false-jump ≫ false-cache ≫ false-fresh. The v0 safety gate is re-applied at inference.
- **Compute accounting is achieved, not nominal.** fresh = 1 forward, cache/jump ≈ 1/L
  (L = 57), jumps additionally remove executed nodes. We report block-stack-equivalent **and**
  measured wall speedup. The fair comparison is **ΔPSNR at matched achieved speedup** — the
  SeaCache PSNR(speedup) curve interpolated at each HorizonCache operating point.

Env fixes (non-obvious): patched a broken xformers `flash_attn_3/_C.so` ABI import that
otherwise breaks `from diffusers import FluxPipeline`; FLUX runs 4-bit (bitsandbytes) to fit
the 24 GB card.

## Key result (consolidation — N=20 prompts × 1 seed, 512px/28 steps)

An N=4 smoke first showed a ~+0.5 dB edge; the N=20 run is **stronger and refines the
finding**. The **surviving primitive is the conservative _adaptive_ jump**
(`jf = 1+(jf_max−1)·headroom`, capped at 1.25) — it **beats fixed `regrid_1.25`**.

SeaCache frontier (achieved speedup → PSNR): 1.48×→35.4, 2.11×→28.99, 2.48×→25.8, 3.0×→23.37 dB.

**ΔPSNR at matched achieved speedup** (deck fair rule; per-image win-rate in parens):

| variant | τ | achieved speedup | HorizonCache PSNR | SeaCache @ matched | ΔPSNR | win |
|---|---|---|---|---|---|---|
| **adaptive_1.25** | 0.2 | 1.70× | 34.18 | 33.17 | **+1.01** | 85% |
| **adaptive_1.25** | 0.35 | 2.46× | 28.30 | 25.94 | **+2.36** | 100% |
| **adaptive_1.25** | 0.5 | 2.74× | 25.65 | 24.57 | **+1.08** | 95% |
| adaptive_1.25 | 0.65 | 3.40× | 23.16 | 23.38 | −0.22 | 15% |
| regrid_1.25 | 0.35 | 2.17× | 28.86 | 28.52 | +0.34 | 85% |
| regrid_1.5 | 0.35 | 2.17× | 28.84 | 28.51 | +0.32 | 80% |

**Mechanism.** The jump lets HorizonCache keep refreshing *often* (low τ) yet still save
compute, giving a **gentler quality/speed tradeoff** than SeaCache's rare-refresh / long-cache —
exactly where SeaCache's frontier drops steepest (2.1×→2.5×). This is a **narrow-band win**
(~1.7–2.7×) that **collapses past ~3×** (τ=0.65). Naive uniform/random and TeaCache-raw collapse.
`jump_2.0` = KILL.

### The v1 label finding (the methodological crux)

The safe-horizon label is **two-dimensional** — metric × horizon — and neither current choice is
right:

- **latent-L2, 2% tol:** `{fresh:4, cache:20, jump:0}` — **0 safe jumps** (too pessimistic; v1
  degenerates to cache/fresh).
- **decoded-PSNR, H=5, floor 32 dB:** `{cache:24, jump_1.25:1, jump_2.0:11}` — **12 safe jumps,
  but 11 are `jump_2.0`**, which we *know* kills end-to-end quality. A 5-node continuation is too
  **optimistic** for big jumps: it misses compounding (the deck's DP-surrogate lesson resurfacing).

So **v1 stays PARK because the supervision is mis-specified, not because learning fails.** The fix
is a **frontier-improvement** label (does this action beat SeaCache end-to-end at matched budget)
or a full-trajectory continuation, plus a much larger dataset.

## Verdict

**KEEP (v0, narrow-band; N=20, single seed, FLUX 512px).** HorizonCache-v0 with the conservative
adaptive jump measurably improves the SeaCache frontier on FLUX in ~1.7–2.7× (+1.0 to +2.4 dB at
matched achieved speedup, 85–100% per-image win-rate) and reduces to SeaCache exactly when jumps
are disabled (fair by identity → never worse in-band). The claim is deliberately bounded: **not**
"HorizonCache beats SeaCache on FLUX" but "**edges the frontier in a conservative-jump regime;
aggressive jumps (jf ≥ 1.5, τ ≥ 0.65) overshoot.**" `adaptive_1.25` KEEP (surviving primitive) >
`regrid_1.25` KEEP; `regrid_1.5` PARK; `jump_2.0` KILL; v1 PARK (label mis-specified); editing
branch-horizon PARK (not run). Multi-seed + bootstrap needed to move KEEP → headline.

## Next

1. **SD3 generation** (download SD3.5-medium) — the deck's flat-region win should give a **wider
   band** than FLUX; highest-value transfer.
2. Multi-seed + paired bootstrap on the matched-speedup delta to make the KEEP significant.
3. **v1: replace the label** with a frontier-improvement target (or full-horizon continuation) and
   scale the rollout dataset; then test v1 > v0.
4. Editing: `cos(h_src, h_tar)` branch-**cache**-horizon on FlowEdit/PIE-Bench (features already in
   the schema) — start with cache-length prediction, not σ jumps.

## Artifacts

- Report: `reports/horizon_cache.html` · `reports/horizon_cache_summary.{md,json}` ·
  `reports/horizon_cache_assets/`
- Metrics (N=20 run): `results/horizon_cache/gen_20260707_013626/metrics.csv` · `metrics.json` ·
  `summary.json` · per-step `traces/` · `samples/`
- Action datasets (both label metrics): latent-L2 `metrics/horizon_cache/action_dataset.{csv,parquet}`
  (0 jump positives); decoded-PSNR `metrics/horizon_cache_decoded/action_dataset.{csv,parquet}` +
  `v1_diag.json` (jump positives appear). v1 bundles under `results/horizon_cache/`.
- Code: `experiments/horizon_cache/` (run.py, flux_gen.py, policy.py, scheduler.py,
  baselines.py, rollout.py, train_v1.py, figures.py, report.py, run_rollout.py, run_report.py) ·
  manifest `experiments/manifests/E56.json`
