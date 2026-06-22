# E46 — Seed-phase SDEdit (fast, inversion-free editing via low-band seed phase)

Idea: transplant the source image's **low-band FFT phase** into a fresh seed, then
generate toward the target prompt. Precompute is ~0 NFE (FFT only). Structure rides on
the seed's phase (Recipe B) or is added to SDEdit's injected noise (Recipe A).

Key theory settled: averaging n noised copies converges to phase(x_0) exactly (phase is
scale-invariant), so the "noise 5x and average" loop is dominated by just using phase(x_0).
The 100% pass is information-theoretically empty (q(x_T)=N(0,I) ⟂ x_0). Method collapses
to: low-band phase(x_0) -> fresh seed -> generate.

## Probe 0 — does low-band SEED phase control generated layout? (Recipe B, reconstruction)
SDXL, cut=0.2, 3 images x 4 seeds. arm white = N(0,I) seed; arm phaseB = white magnitude
+ source low-band phase. Metric: DINO structure distance to source (lower=closer layout).

Mean struct_dist (white -> phaseB):
  cat_orange     0.408 -> 0.171  (gap +0.237)
  black_panther  0.213 -> 0.161  (gap +0.052)
  savana         0.359 -> 0.245  (gap +0.115)
phaseB beats white on ALL 3 images and ALL 12 seed pairs, no reversals.

**Verdict: KEEP.** Low-band seed phase controls output layout, strongly & consistently.
Grid confirms exact layout transfer (cat pose, giraffe+rhino arrangement, panther stance).
Caveat: on flat/segmentation-like sources (cat, savana) phaseB shows OOD color artifacts
(red/cyan fringing) — white magnitude + strong cartoon phase is off-manifold. Natural photo
(panther) renders clean. -> Probe 1 must use real photos (PIE-Bench) and watch appearance.
