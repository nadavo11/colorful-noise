# E56 — HorizonCache: causal safe-horizon prediction (fresh / cache / jump)

**Thread:** fast-edit (caching line) · **Status:** active · **Models:** FLUX.1-dev (4-bit
transformer on a 24 GB A5000), 512 px, 28 Euler steps, canonical-fixture v1 prompts.
SD3 unavailable (no local weights); editing capability-detected but not executed.

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

## Key result (smoke — N=4 prompts × 1 seed, directional, not powered)

| method | PSNR↑ | LPIPS↓ | achieved speedup | mean jumps |
|---|---|---|---|---|
| full | — | 0 | 1.00× | 0 |
| uniform k2 | 17.85 | 0.355 | 1.84× | 0 |
| random-k | 20.74 | 0.252 | 1.97× | 0 |
| SeaCache τ0.2 | 33.26 | 0.035 | 1.48× | 0 |
| **HorizonCache-v0 τ0.2** | 33.04 | 0.036 | **1.54×** | 1.0 |
| SeaCache τ0.3 | 28.74 | 0.102 | 1.87× | 0 |
| **HorizonCache-v0 τ0.3** | 28.57 | 0.107 | **2.12×** | 4.0 |
| SeaCache τ0.4 | 28.07 | 0.110 | 2.11× | 0 |
| **HorizonCache-v0 τ0.4** | 27.09 | 0.130 | **2.50×** | 5.0 |

**Fair frontier (ΔPSNR at matched achieved speedup):** HorizonCache-v0 edges the SeaCache
frontier by **+0.49 dB @1.54×** and **+0.51 dB @2.12×**, then **overshoots** past ~2.1×
(**−0.97 dB @2.5×**). Regrid jumps fire and stay conservative (jump_1.25 dominates). This
slightly **beats the deck's FLUX prediction of a pure tie** — the headroom-driven regrid finds
small safe strides FLUX's bending field still permits. Naive uniform/random collapse.

`jump_2.0` overshoots (KILL for the primary frontier). v1's rollout→train pipeline runs
end-to-end (safe-horizon dataset + HistGB + asymmetric cost + confusion/importance) but on a
tiny dataset — a pipeline demonstration, not yet a trained win.

## Verdict

**PARK-leaning-KEEP (directional).** HorizonCache is a strictly-more-general, fair-by-identity
SeaCache that already weakly dominates the SeaCache frontier on FLUX in the 1.5–2.1× sweet spot
(+~0.5 dB at matched achieved speedup) and never does worse there; past 2.1× jumps overshoot.
The margin is small and N=4, so not a demonstrated win. `jump_2.0` KILL; v0 KEEP-directional;
v1 PARK (pipeline only); editing branch-horizon PARK (not run).

## Next

1. **SD3 generation** (download SD3.5-medium) — the deck's flat-region jump win is the
   highest-value transfer; HorizonCache should show a larger margin there.
2. Scale N ≥ 20 + multiple seeds and paired-bootstrap the matched-speedup delta.
3. Grow the rollout dataset and retrain v1; test v1 > v0.
4. Wire the editing path: `cos(h_src, h_tar)` branch-horizon on FlowEdit/PIE-Bench (features
   already in the schema).

## Artifacts

- Report: `reports/horizon_cache.html` · `reports/horizon_cache_summary.{md,json}` ·
  `reports/horizon_cache_assets/`
- Metrics: `results/horizon_cache/gen_*/metrics.csv` · `metrics.json` · `summary.json` ·
  per-step `traces/` · `samples/`
- Action dataset: `metrics/horizon_cache/action_dataset.{csv,parquet}` · v1 bundle
  `results/horizon_cache/v1_bundle.joblib` · `metrics/horizon_cache/v1_diag.json`
- Code: `experiments/horizon_cache/` (run.py, flux_gen.py, policy.py, scheduler.py,
  baselines.py, rollout.py, train_v1.py, figures.py, report.py) · manifest
  `experiments/manifests/E56.json`
