# E53 — FLUX jump-DP skip-schedule oracle

**Thread:** fast-edit (SeaCache caching sub-line) · **Model:** FLUX.1-dev text2img, 1024px, bf16, 100 steps · **Hardware:** NVIDIA H100 NVL (remote, SSH) · **Status:** done · **Verdict:** ORACLE-ONLY / NO-GO as a schedule source.

## Why
SeaCache/TeaCache decide *refresh vs reuse* online from a relative-L1 signal. We ask the harder,
offline question: given a full vanilla 100-step no-skip FLUX trajectory, what is the **best possible**
schedule of fresh evaluations, and how far can you jump between them? We build a teacher-forced
dynamic-programming oracle over the saved trajectory and test three things:

1. Does the oracle beat naive uniform/random and SeaCache?
2. Does it survive when you can no longer peek at vanilla velocities (causal cached-residual replay)?
3. Does the SeaCache predictor also tell you *how far* you can safely jump, not just whether to refresh?

This is explicitly a diagnostic/oracle, **not** a deployable method.

## Setup
Thin orchestrator `experiments/flux_dp_jump_oracle.py` reuses the caching harness
`experiments/flux_seacache_dp_shortcuts.py` for capture / edge table / DP / replay / SeaCache
forward / decode. Prompts: 4 from the frozen canonical fixture (`fixtures.canonical_prompts()`,
`FIXTURE_VERSION=v1`), one seed each.

**Jump edge cost (sign-verified against the sampler):**

```
ẑ_i = z_k + (σ_i − σ_k)·v_k
S_jump[k,i] = ||ẑ_i − z_i||² / (||z_i||² + ε)      0 ≤ k < i ≤ 100   (4950 edges/traj)
```

One-step check (`i=k+1`) reproduces the saved vanilla next latent to worst relative error
**2.0e-3** ≤ tol, confirming the FLUX/rectified-flow convention (σ decreasing 1→0, velocity subtracted).

**DP:** `dp[b,i] = min_{k<i} ( dp[b-1,k] + S_jump[k,i] )`, `dp[0,0]=0`, backtracked to anchors;
swept over all budgets `B` and capped at `max_span ∈ {4,8,12,16}`.

**Three families, kept strictly separate:**
- **Offline surrogate** — sum of independent per-edge S_jump costs.
- **Teacher-forced jump live replay** — compounded from `z_0` reusing vanilla velocities (`replay_path`).
- **Causal baselines** — SeaCache (live, thresholds 0.2–0.6, matched by *achieved* fresh-eval budget),
  uniform, random; plus a capped **cached-residual stage-2** that refreshes the block stack at
  DP/uniform anchors and reuses the residual between them (a genuine causal replay).

Metrics vs the vanilla 100-step final: final-latent relative L2, PSNR, SSIM, LPIPS, CLIP-img/text.

## Headline numbers (mean over 4 samples)

**Teacher-forced (reuses exact vanilla velocities) — DP ≈ uniform:**

| method | saved 50 | saved 75 | saved 90 |
|---|---|---|---|
| DP jump replay (PSNR dB) | 40.3 | 39.3 | 35.4 |
| uniform jump replay (PSNR dB) | 40.2 | 38.3 | 32.9 |

When velocities are exact, *where* you place anchors barely matters — DP is within ~0.1–1 dB of uniform.

**Causal cached-residual replay of the SAME DP schedule — collapses:**

| saved | 10 | 25 | 50 | 67 | 89 |
|---|---|---|---|---|---|
| dp_cached PSNR (dB) | 40.1 | 27.0 | 19.0 | 16.5 | 12.5 |
| dp_cached LPIPS | 0.006 | 0.082 | 0.254 | 0.358 | 0.590 |

At saved 67 the causal replay is **23 dB below** the teacher-forced curve.

**Among causal methods, SeaCache wins:**

| method (≈matched budget) | saved | PSNR (dB) | LPIPS |
|---|---|---|---|
| SeaCache | 76 | 23.6 | 0.121 |
| dp_cached | 75 | 15.4 | 0.418 |
| uniform_cached | 74 | 14.3 | 0.474 |

SeaCache's adaptive online gate keeps fresh evals in the volatile early steps; DP-jump anchors
(optimised for velocity extrapolation, not residual reuse) and uniform both mis-place them.

**Predictor diagnostic (does SeaCache's signal predict jump distance?):**

| pair | n | Pearson | Spearman |
|---|---|---|---|
| SeaCache inst rel-L1 vs oracle next-anchor span | 124 | −0.19 | **−0.50** |
| SeaCache accumulated rel-L1 vs oracle span | 124 | −0.17 | −0.14 |
| SeaCache inst rel-L1 vs S_jump[k,k+1] | 392 | +0.25 | +0.17 |

The instantaneous signal ranks oracle-safe jump length only weakly-to-moderately (Spearman −0.50) —
consistent with it being tuned for refresh/no-refresh, not for *how far* to jump.

## Does the DP oracle survive live replay?
**No.** The oracle's strength is a teacher-forcing artifact: it reuses the exact vanilla velocity
captured at the true `z_k`. Under teacher forcing it is near-lossless *and barely beats uniform*.
The moment you must reuse computation causally (cached-residual), the same schedule collapses and is
beaten by SeaCache. Reported as a finding, not a failure.

## Limitations
- Non-causal / teacher-forced oracle; a diagnostic upper bound, not deployable.
- Surrogate (sum of per-edge costs) ≠ compounded replay; where they diverge, believe the replay.
- SeaCache matched by *achieved* fresh-eval budget, not nominal threshold.
- TeaCache: no implementation in the repo — marked unavailable rather than approximated.
- 4 prompts × 1 seed (a "start with 4" run); trends are consistent across samples but not a large-N claim.

## Next
Replace the teacher-forced jump cost with a **path-dependent cached-residual DP** on short spans
(≤12–16) plus SeaCache/jump-selected spans — score each edge from the *actually reached* state under
residual reuse — and compare that oracle's frontier against SeaCache. Also test whether an
early-steps-fresh prior (matching SeaCache) recovers most of SeaCache's causal frontier.

## Artifacts
Run dir: `runs/h100/20260701_160515__flux_dp_jump_oracle/`
- `report.html` (self-contained), `reports/summary.{md,json}`
- `metrics/{frontier,per_sample_metrics,per_budget_metrics,predictor_correlations}.csv`
- `figures/*.png` (method diagram, S_jump heatmaps, schedule raster, frontier curves, span histograms, predictor scatters, image grids)
- `samples/*.png`, `schedules/dp_schedules.json`, `edge_costs/*.npz`, `artifacts_manifest.json`

Code: `experiments/flux_dp_jump_oracle.py` · Manifest: `experiments/manifests/E53.json`
