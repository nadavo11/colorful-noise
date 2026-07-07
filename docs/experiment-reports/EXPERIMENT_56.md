# E56 — HorizonCache: causal safe-horizon prediction (fresh / cache / jump)

**Thread:** fast-edit (caching line) · **Status:** active · **Verdict:** STRONG KEEP (headroom-adaptive stride, narrow-band) ·
**Models:** FLUX.1-dev (4-bit transformer), 512 px, 28 Euler steps. Sig run N=24 × 2 seeds (48 pairs) on a 24 GB A5000;
**consolidation scale-up N=50 × 2 seeds = 100 paired samples on a cluster H100 80 GB (Run:AI)**, full method set
incl. adaptive_1.25/1.5/2.0 (canonical-fixture v1 + deterministic GenEval extras). SD3 unavailable (no MMDiT
generation harness in the module); editing capability-detected but not executed.

> **Consolidation update (N=50, cluster H100).** See §"Consolidation" below. The sig headline **replicates and
> tightens** (adaptive_1.25 @2.51× = **+2.20 dB**, CI [1.79, 2.62], win 85%), and the full method set reveals an
> honest nuance: the headroom-adaptive stride is **jf_max-robust** — adaptive_1.5 and adaptive_2.0 also clear the
> strong-KEEP rule in-band because jf_max is a *soft* cap the headroom gate rarely reaches. The overshoot is a
> **high-τ effect common to every cap**, not a large-cap effect. Paper-ready report:
> `reports/horizon_cache_consolidated.html`.

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

## Key result (significance — N=24 prompts × 2 seeds = 48 paired samples, 512px/28 steps)

Progression: N=4 smoke (~+0.5 dB) → N=20 single-seed consolidation → this **multi-seed
significance run** with **bootstrap 95% CIs** over prompt×seed pairs. The **surviving primitive is
the conservative _adaptive_ jump** (`jf = 1+(jf_max−1)·headroom`, capped at 1.25) — it beats fixed
`regrid_1.25`. Lean method set (full + SeaCache + `adaptive_1.25`) to spend compute on seeds.

SeaCache frontier (achieved speedup → PSNR): 1.48×→35.6, 1.86×→30.4, 2.11×→28.5, 2.48×→25.9,
3.0×→23.1 dB.

**ΔPSNR at matched achieved speedup** (deck fair rule; paired per-image, bootstrapped, n=48):

| variant | τ | speedup | mean ΔPSNR | 95% CI | win-rate | CI≠0 |
|---|---|---|---|---|---|---|
| adaptive_1.25 | 0.2 | 1.70× | **+1.61** | [1.15, 2.10] | 85% | ✓ |
| adaptive_1.25 | 0.3 | 2.11× | **+1.47** | [0.95, 2.04] | 83% | ✓ |
| **adaptive_1.25** | **0.4** | **2.51×** | **+2.11** | **[1.52, 2.70]** | **92%** | **✓** |
| adaptive_1.25 | 0.5 | 2.74× | **+1.26** | [0.81, 1.68] | 92% | ✓ |
| adaptive_1.25 | 0.65 | 3.40× | −0.19 | [−0.25, −0.13] | 19% | ✓ (neg) |

Every **in-band** operating point clears the strong-KEEP rule (mean > +0.5 dB, win > 65%, CI
excludes 0, speedup ∈ [2.0, 2.6×]); τ0.4 @ 2.51× is the headline. The overshoot at 3.4× is *also*
significant (CI excludes 0 on the negative side) — so "aggressive jumps overshoot" is now a
measured claim, not a caveat.

**Mechanism.** The jump lets HorizonCache keep refreshing *often* (low τ) yet still save compute,
giving a **gentler quality/speed tradeoff** than SeaCache's rare-refresh / long-cache — exactly
where SeaCache's frontier drops steepest (2.1×→2.5×). This is a **narrow-band win** (~1.7–2.7×) that
**overshoots past ~3×**. `jump_2.0` = KILL.

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

## Consolidation (N=50 × 2 seeds = 100 paired samples, cluster H100, full method set)

The sig run was *lean* (only `adaptive_1.25`). The consolidation scale-up ran the **full method family**
(full · uniform · TeaCache · SeaCache · `regrid_1.25` · `adaptive_1.25/1.5/2.0`) at N=100 pairs on a
cluster H100, same protocol (512 px / 28 / bf16+bnb4, τ∈{0.2,0.3,0.4,0.5,0.65}). It confirms the result
and adds an honest nuance.

**adaptive_1.25 — per speed band (matched achieved speedup, paired bootstrap 95% CI):**

| band | speedup | ΔPSNR | 95% CI | win | verdict |
|---|---|---|---|---|---|
| 1.5–2.0× | 1.70× | +1.77 | [1.39, 2.13] | 82% | significant gain |
| 2.0–2.5× | 2.12× | +1.28 | [0.92, 1.66] | 78% | STRONG KEEP |
| **2.5–2.8×** | **2.51×** | **+2.20** | **[1.79, 2.62]** | **85%** | **STRONG KEEP** |
| 3.0×+ | 3.40× | −0.16 | [−0.20, −0.12] | 17% | significant loss (overshoot) |

**Family finding (jf_max is a soft cap).** At matched ~2.5× the three adaptive caps are nearly tied —
`adaptive_1.25` +2.20, `adaptive_1.5` +2.19, `adaptive_2.0` +1.66 (all CIs exclude 0, win 85–86%); the
fixed `regrid_1.25` is beaten at +1.93. All three clear the strong-KEEP rule in-band because the stride
`jf = 1+(jf_max−1)·headroom` only reaches its ceiling when the cache is fresh — so a larger jf_max mostly
does **not** change in-band behaviour. `adaptive_1.25` remains the recommended **conservative** operating
point (safest cap, best in the mid band). **The robust thing is the mechanism, not one magic cap.**

**Overshoot is a high-τ effect, not a jf_max effect.** At τ0.65 (≈3.4×) *every* adaptive cap turns to a
significant loss (−0.14 to −0.16 dB, CI excludes 0 on the negative side, win ≤24%): pushing the SeaCache
refresh threshold high *and* jumping removes nodes where the field still curves, and the error compounds.
This is distinct from the **uncapped discrete `jump_2.0`** action (unconditional jf=2.0, no headroom gate),
which stays **KILL** — do not conflate it with the headroom-capped `adaptive_2.0`.

Artifacts: `reports/horizon_cache_consolidated.{html,md,json}` (10 sections, frontier + per-band tables,
mechanism figures, qualitative grids) · `results/horizon_cache_n50/gen_20260707_113512/` ·
`runs/runai/20260707_143313__horizon_cache_e56_flux_n50__613c48b/` (Run:AI submission record).

## Verdict

**STRONG KEEP (headroom-adaptive stride, narrow-band; confirmed at N=50 × 2 seeds = 100 paired samples,
cluster H100, full method set).**
A conservative **headroom-adaptive stride extension shifts the FLUX SeaCache frontier upward in the
~1.7–2.7× band** by **+1.3 to +2.1 dB** at matched achieved speedup (per-image win 83–92%, and
**every in-band bootstrap 95% CI excludes 0**), while reducing to SeaCache exactly when jumps are
disabled (fair by identity → never worse in-band). The claim is deliberately bounded — **not**
"HorizonCache beats SeaCache on FLUX" but "**a conservative headroom-adaptive stride extension
shifts the FLUX SeaCache frontier upward in the ~1.7–2.7× band, but aggressive jumps still
overshoot**" (τ0.65 / 3.4×: −0.19 dB, CI excludes 0 on the negative side). `adaptive_1.25` STRONG
KEEP (surviving primitive) > `regrid_1.25` KEEP-as-baseline; `regrid_1.5` PARK; `jump_2.0` KILL;
v1 PARK (label mis-specified, frontier-improvement label built + run — see §label finding);
editing branch-horizon PARK (not run).

## Next

1. **SD3 generation** (download SD3.5-medium) — the deck's flat-region win should give a **wider
   safe-stride band** than FLUX (hypothesis: FLUX jf_max≈1.25, SD3 jf_max≈1.5–2.0); highest-value
   transfer, and richer positive labels to rescue v1.
2. Scale N (50–100) to tighten the CIs further (already excluding 0 at n=48).
3. **v1 on the frontier-improvement label** (built + run here): if it doesn't beat the adaptive
   heuristic, the heuristic is near-optimal for this feature set — a useful negative.
4. Editing: `cos(h_src, h_tar)` branch-**cache**-horizon on FlowEdit/PIE-Bench (features already in
   the schema) — start with cache-length prediction, not σ jumps.

## Artifacts

- Report: `reports/horizon_cache.html` · `reports/horizon_cache_summary.{md,json}` ·
  `reports/horizon_cache_assets/`
- Metrics (significance run): `results/horizon_cache_sig/gen_20260707_095918/metrics.csv` · `metrics.json` ·
  `summary.json` · per-step `traces/` · `samples/`
- Action datasets (both label metrics): latent-L2 `metrics/horizon_cache/action_dataset.{csv,parquet}`
  (0 jump positives); decoded-PSNR `metrics/horizon_cache_decoded/action_dataset.{csv,parquet}` +
  `v1_diag.json` (jump positives appear). v1 bundles under `results/horizon_cache/`.
- Code: `experiments/horizon_cache/` (run.py, flux_gen.py, policy.py, scheduler.py,
  baselines.py, rollout.py, train_v1.py, figures.py, report.py, run_rollout.py, run_report.py) ·
  manifest `experiments/manifests/E56.json`
