# E55 — Distilled SEA-defect for dynamic SeaCache refresh

**Date:** 2026-07-02  
**Repo:** `colorful-noise`  
**Branch:** current local worktree  
**Predecessor:** E54 `runs/h100/20260701_222427__flux_runtime_jump_hints/`

---

## TL;DR

Do **not** run another jump experiment as the main branch.

E54 established:

- jump-DP is dead as a deployable method;
- curvature / AB2 jump hints are weak diagnostics, not a method;
- `SEA-defect` is the only promising causal signal;
- live `SEA-defect` is too expensive to be an accelerator because prefix probes kill wall-clock;
- SeaCache still beats jump controllers at matched fresh-call budgets.

Therefore E55 should treat `SEA-defect` as an **offline teacher** and learn a **cheap, prefix-free, causal dynamic SeaCache refresh gate**.

**One-line goal:** use SEA-defect as an offline teacher to learn a prefix-free, causal, dynamic SeaCache refresh gate; beat fixed SeaCache on wall-clock/quality.

---

## Corrected E54 context

Read these first:

- `experiments/flux_runtime_jump_hints.py`
- `runs/h100/20260701_222427__flux_runtime_jump_hints/report.html`
- `runs/h100/20260701_222427__flux_runtime_jump_hints/reports/summary.json`
- `runs/h100/20260701_222427__flux_runtime_jump_hints/metrics/per_method_budget_metrics.csv`
- `runs/h100/20260701_222427__flux_runtime_jump_hints/metrics/local_error_correlations.csv`
- `runs/h100/20260701_222427__flux_runtime_jump_hints/metrics/nearest_budget_audit.csv`
- `experiments/flux_seacache_dp_shortcuts.py`

Important corrected findings from the local patched report:

- The original E54 report had a **SeaCache budget-matching bug** in the summary table: `100-call / PSNR=inf` rows were incorrectly selected because `inf` dominated the distance penalty.
- That bug is fixed locally in `experiments/flux_runtime_jump_hints.py`.
- The corrected ~50-call comparison is:
  - best causal jump hint = `sea_defect_th0p2`, `46` calls, `33.75 dB`, `LPIPS 0.0210`
  - SeaCache = `50` calls, `36.30 dB`, `LPIPS 0.0109`
- So the broken `23.15 vs inf` line was invalid, but the corrected result still says: **SeaCache wins** at the meaningful ~50-call operating point.
- `SEA-defect` is still the most promising branch because it dominates other jump hints in quality-vs-calls terms.
- But `SEA-defect` is **wall-clock negative** in its live expensive form:
  - around 50 calls: SeaCache `7.78x` wall-speedup vs `SEA-defect` jump `0.72x`
  - around 80 calls: SeaCache `5.31x` vs `SEA-defect` jump `0.42x`

This is the reason E55 should pursue **dynamic refresh**, not **dynamic jumping**.

---

## Decision tree

Use this branch classification as locked guidance:

| branch | status |
|---|---|
| DP jump | dead as method |
| saved-velocity replay | diagnostic only |
| curvature hint | weak / not novel |
| AB2 hint | weak / not novel |
| hybrid jump | useful diagnostic, not method |
| SEA-defect jump | only promising jump branch |
| SeaCache | baseline to beat |

Interpretation:

- If `SEA-defect` is useful, it should probably improve **refresh vs reuse** decisions inside SeaCache.
- If live `SEA-defect` remains expensive, it is not itself the deployable method.

---

## Mission

Implement a new experiment titled:

**`E55 — Distilled SEA-defect for dynamic SeaCache refresh`**

Main question:

**Can we use expensive SEA-defect as an offline teacher to train or calibrate a cheap, causal, prefix-free dynamic SeaCache refresh gate that beats fixed-threshold SeaCache on wall-clock/quality?**

This is the main branch.

Do not spend the main compute budget on jump policies.

---

## Explore first

Before coding, inspect:

- `README.md`
- dependency files: `requirements*.txt`, `pyproject.toml`, `environment*.yml`
- `experiments/flux_runtime_jump_hints.py`
- `experiments/flux_dp_jump_oracle.py`
- `experiments/flux_seacache_dp_shortcuts.py`
- `experiments/cluster_flux_runtime_jump_hints.sh`
- any existing FLUX / SeaCache trajectory capture or replay code
- any report-generation helpers and manifest writers used by E53/E54
- any prompt/seed fixtures used by E53/E54
- any tests / smoke scripts for experiment entrypoints

Mirror repository patterns for:

- `argparse`
- logging
- run directory layout
- metrics CSV writing
- HTML report generation
- figure output
- call counters
- cluster submission

---

## Execution plan

1. Analyze context (read files).
2. Implement code (mirror patterns).
3. Generate docs (`README.md` if repo pattern requires, otherwise experiment doc / report notes).
4. Verify (args, explicit errors, smoke test).

---

## Core method

### Main method: distilled dynamic SeaCache refresh

Use E54-style traces or regenerated captures to build an **offline teacher-label dataset** for SeaCache refresh decisions.

For each causal step `t`, record cheap online features available without extra full transformer calls, for example:

- sigma / timestep / normalized timestep
- current SeaCache accumulated rel-L1 score `A_t`
- instantaneous rel-L1 if available
- short history of accumulated score and score deltas
- step index and scheduler metadata
- whether fixed-threshold SeaCache would refresh
- any other causal cache-state summary already available for free

Offline teacher-only signals may include:

- expensive `SEA-defect` score
- post-hoc cache error if cache were reused
- post-hoc failure label

The online predictor **must be prefix-free and causal**:

- no expensive SEA-defect at runtime
- no extra full transformer calls
- no saved vanilla future latents/velocities/features
- no leakage from offline teacher data into online decision features

### Teacher signals and labels

Log both a teacher score and actual target:

- `teacher_defect_t`: expensive SEA-defect score from prefix/modulation path
- `cache_error_t`: actual post-hoc cache-use error
- `refresh_needed_t`: binary label derived from cache failure / quality drift

You may also log a soft failure probability target for calibration diagnostics.

### Predictor family

Start with simple, auditable models:

- logistic regression
- isotonic or Platt calibration on a simple score
- thresholded moving-average / hysteresis baseline
- tiny MLP only if simpler models fail

Do not start with a heavy learned model. The method should remain cheap and interpretable.

### Dynamic policy

At runtime, the predictor should decide:

- refresh now with a fresh full transformer call, or
- reuse cached state

Compare against:

- 100-step vanilla reference
- default FLUX
- fixed-threshold SeaCache
- dynamic SeaCache refresh using distilled predictor
- optional expensive teacher/oracle diagnostic, clearly labeled non-deployable or expensive

---

## Secondary branch: rare-probe SeaCache

This is a **small ablation only**, not the main branch.

Policy:

`if A_t in [0.8 * delta, 1.2 * delta] => run SEA-defect probe`

Meaning:

- when SeaCache is clearly safe, reuse;
- when clearly unsafe, refresh;
- when uncertain near threshold, pay for an expensive SEA-defect probe.

Run this only after the main E55 result or as a minimal side branch.

Count probe calls separately and convert them to actual wall-clock cost.

If probes are frequent, state directly that the branch is not viable.

---

## Required outputs

The next run must produce:

- `report.html`
- `metrics/teacher_label_dataset.csv` or `.parquet`
- `metrics/predictor_auc_calibration.csv`
- `metrics/seacache_dynamic_frontier.csv`
- `figures/teacher_signal_vs_cache_error.png`
- `figures/predictor_calibration_curve.png`
- `figures/dynamic_vs_fixed_seacache_frontier.png`
- `figures/refresh_schedule_raster.png`
- `figures/sample_grid_dynamic_seacache.png`

Also emit repo-consistent summary and manifest files if E54 does:

- `reports/summary.md`
- `reports/summary.json`
- `artifacts_manifest.json`

Save under:

- `runs/h100/<timestamp>__flux_dynamic_seacache_refresh/`

---

## Required figures

Include at minimum:

1. `figures/teacher_signal_vs_cache_error.png`
   - SEA-defect teacher vs actual cache error or cache-failure target.

2. `figures/predictor_calibration_curve.png`
   - predictor score / probability vs empirical cache failure.
   - include calibration and AUC metrics.

3. `figures/dynamic_vs_fixed_seacache_frontier.png`
   - fixed SeaCache vs dynamic SeaCache frontier.
   - compare by actual fresh full calls and also wall-clock.

4. `figures/refresh_schedule_raster.png`
   - fixed threshold vs dynamic refresh schedule.

5. `figures/sample_grid_dynamic_seacache.png`
   - columns: 100-step vanilla, default FLUX, fixed SeaCache, best dynamic SeaCache, optional expensive teacher diagnostic.

Also include a wall-clock vs LPIPS/PSNR figure if not already integrated into the frontier.

---

## Required metrics

For every method / threshold / policy point, log:

- actual fresh full transformer calls
- cached/reused calls
- cheap auxiliary calls
- expensive probe calls if any
- wall-clock time
- wall-clock speedup vs 100-step vanilla
- latent relative L2 to 100-step vanilla
- RGB PSNR to 100-step vanilla
- SSIM
- LPIPS if available
- CLIP similarity if already available
- image SHA256

Primary comparison axes:

- quality vs actual fresh full calls
- wall-clock vs quality

Do not compare by nominal threshold alone.

---

## Required tables / audits

The report should include:

- method summary table
- corrected nearest-budget fixed-vs-dynamic SeaCache comparison table
- predictor calibration / AUC table
- call counter audit
- leakage audit
- split audit if any predictor is fit on calibration prompts

Hard audits required:

- fresh full transformer call count
- cached call count
- cheap auxiliary call count
- expensive probe count
- causal validity
- train/calibration vs evaluation split separation

Avoid the E54 summary bug:

- no `inf` PSNR identity row should ever dominate matched-budget comparisons
- use nearest achieved call count with finite metrics only

---

## Prompt / seed setup

Use the same prompt/seed set as E53/E54 if available.

Protocol:

- smoke first: `2 prompts x 1 seed`
- verify dataset rows emit
- verify predictor training / calibration runs end-to-end
- verify report renders
- verify no online leakage
- main after smoke

Main:

- at least 4 trajectories to match E54
- preferably 8 to 16 if compute allows

Model / hardware:

- FLUX.1-dev
- 1024x1024
- bf16
- H100 preferred

Reference:

- 100-step vanilla FLUX

---

## Acceptance criteria

E55 is complete only if:

- `report.html` exists and opens
- all required figures are embedded or linked correctly
- teacher-label dataset is saved
- predictor calibration / AUC CSV is saved
- dynamic SeaCache frontier CSV is saved
- comparisons use actual fresh calls and measured wall-clock
- the online policy is causal and prefix-free
- the expensive teacher is clearly marked offline-only
- the best dynamic SeaCache result is compared directly against fixed SeaCache
- any rare-probe branch reports probe counts and real wall-clock impact

If the dynamic gate fails to beat fixed SeaCache, say so directly. A negative result is acceptable if the audit is clean.

---

## Run protocol

1. Implement experiment entrypoint and cluster launcher.
2. Run local sanity checks:
   - `python -m py_compile ...`
   - `python <entrypoint> --help`
3. Submit smoke job on H100.
4. Inspect smoke artifacts and report.
5. Submit main H100 job only after smoke passes.
6. Retrieve results into the local repo without waiting for the user to remind you.

At the end, report:

- Run job names
- hardware
- start / end time
- run directory
- `report.html` path
- top-level verdict
- whether dynamic SeaCache beat fixed SeaCache
- whether rare-probe helped
- whether the distilled branch looks like a real method or a dead end

---

## Final recommendation

Run **E55 distilled dynamic SeaCache**, not **E55 jump**.

Interpret the current result this way:

- `SEA-defect` is a useful causal local-error probe,
- but FLUX acceleration is more likely to come from **better cache refresh decisions** than from **longer solver jumps**.
