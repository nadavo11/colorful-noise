# E41 — Per-image calibration vs RF-inversion (η)

Structure-preserving real-image editing on FLUX, compared fairly against **RF-inversion**
(Rout et al.) on **140 PIE-Bench images**. Metrics: **DINO self-similarity structure distance ↓**
(PIE-Bench's headline structure metric) and **CLIP-directional editability ↑**.

Artifacts (under `results/e41/`, honor `CN_RESULTS`):
`index.html` (self-contained explainer — run `python e41_site.py`), `report.md` (auto from
`--part analyze`), `aggregate_pareto.png`, `montage_*.png`, `items/*.json` (per-image: params,
metrics, all 20 trial traces, 6-point η sweep).

## The problem
RF-inversion has an **η** knob trading editability ↔ faithfulness, so a single-η comparison is
unfair. The repo had **no η controller** (the edit pass was plain reverse-Euler). We added one
and auto-calibrated our own knobs per image.

## What we built
- **RF-inversion η controller** (`invert_core.forward_edit`): `v ← v + η·(v_target − v)`,
  `v_target = (x − x₀)/σ`, over an early step window. η=0 = vanilla edit; η=1 ≈ reconstruction.
  Verified by reproducing the hand-tuned dancers run: MSE(ours, saved)=2.1e-4,
  MSE(η=1 recon, source)=6e-5.
- **Per-image calibration** (`e41_calibrate.py`): Optuna TPE active loop (~20 trials/image) over
  our knobs `mode` (sbn/phase/adain), `cut`, `strength`, `interval_end`, `phase_band`.
  **Constrained objective**: minimize DINO structure distance **s.t.** CLIP-dir ≥ vanilla's
  (preserve structure *at matched editability*). Warm-started from the dancers prior + a
  prompt-distance heuristic. Every trial saved → operating point re-selectable with no GPU.
- **Benchmark + metrics**: PIE-Bench++ loader (`piebench.py`), DINO via transformers
  (`struct_metrics.py`), CLIP-dir, LPIPS/DSSIM, CLIP-T. RF-inversion η swept {0…1} per image.
- Sharded RunAI run across A6000/H100 (`cluster_e41_job.sh`, `submit_e41.sh`).

## Results (140 images)

| method | DINO struct ↓ | LPIPS ↓ | DSSIM ↓ | CLIP-dir ↑ | CLIP-T ↑ |
|---|---|---|---|---|---|
| **ours** | **0.162** | **0.502** | **0.438** | **0.140** | **0.272** |
| vanilla RF-inv (η=0) | 0.199 | 0.595 | 0.504 | 0.123 | 0.269 |
| default RF-inv (η=0.9) | 0.067 | 0.156 | 0.118 | 0.010 | 0.228 |

- **vs vanilla RF-inversion (out-of-the-box): ours wins on every axis** — lower structure,
  lower LPIPS/DSSIM, higher editability. Per edit-type, ours < vanilla structure on **~135/140**.
- **default η=0.9 barely edits** (CLIP-dir 0.010 ≈ reconstruction) — low structure distance is an
  artifact of not editing, which is why the η *sweep* is the honest comparison.
- **Matched editability vs RF-inversion's η curve**: mean gap ≈ **−0.0003** (31/63 comparable
  wins) — essentially **on their tuned frontier, not below it**.
- On **77/140** images ours edits **beyond RF-inversion's entire η range** (more editable than
  even η=0) — a region η cannot reach (η only moves editability *down* from vanilla).
- Feasible (ours ≥ vanilla editability): **135/140**.

See `aggregate_pareto.png`: our calibrated cloud sits on RF-inversion's mean η frontier but
extends far to the right (high editability) where η has no points.

## Honest read
Two solid claims: (a) we beat **out-of-the-box** RF-inversion on structure, perceptual distance
and editability simultaneously, on a standard benchmark; (b) our knobs reach an **editability
range η cannot**. The stronger claim — beating their **tuned** η frontier on structure at matched
editability — is **not** supported here: we're roughly tied.

## Future experiments & sweeps
1. **Retarget the calibration objective** to push our cloud *below* the η curve (maximize a
   structure+editability scalar, or min-struct s.t. editability ≥ a higher target). Re-selectable
   from saved trial traces, no GPU.
2. **Widen the knob search**: `interval_start` (fixed at 0), multi-band phase locks, per-step
   schedules, guidance; more trials.
3. **Amortize calibration** (deferred Phase B): train a small predictor (CLIP image +
   prompt-distance features → knobs) on this run's calibration table.
4. **Background metrics**: recover PIE-Bench++ masks (came as strings, skipped) for background
   PSNR/LPIPS, where structure preservation should show a larger margin.
5. **Stronger RF-inversion baseline**: confirm the reference η default/window; sweep the
   controller window τ as a second axis.
6. **Scale to full PIE-Bench (700) + more seeds** once the objective is retargeted.
