# E37 — Velocity spectral normalization on SD3.5, evaluated on GenEval

## TL;DR

During a Stable Diffusion 3.5 generation we edit the **CFG velocity** in frequency space:
keep its phase but pull its FFT **amplitude** toward the *same-step unconditional velocity*
`v_∅` (the "cfg=1" flow field). On GenEval (553 prompts, 1 sample/prompt, 512px, guidance
w=4.5), the effect is **entirely band-dependent**: normalizing the **bottom 25%** of radial
frequencies, or the **full** band, *hurts* prompt adherence (overall 0.561 / 0.524 vs baseline
0.644); normalizing only the **top 25%** (high-freq) slightly *beats* baseline (**0.655 vs
0.644**), with the gain concentrated in **color attribution** (0.48 → 0.54, 7 wins / 1 loss).
Low frequencies carry layout/composition (don't touch them); high frequencies are where CFG
over-amplifies magnitude without aiding composition, so normalizing them is free-to-beneficial.
Single-seed, so the +0.011 overall is within seed noise — but the per-tag pattern is coherent.

## Background (plain language)

- **Flow-matching velocity `v`** — SD3.5 is sampled by Euler integration `z_{t+1}=z_t+Δt·v`,
  where the transformer output `v` is a *velocity* (the direction the latent moves this step).
- **Classifier-free guidance (CFG)** — combines two velocities: `v_w = v_∅ + w·(v_c − v_∅)`,
  with `v_∅` unconditional, `v_c` conditional, `w` the guidance scale (here 4.5). High `w`
  improves adherence but over-amplifies certain frequency **magnitudes** (the over-contrasty
  CFG look). The **phase** of `v_w` carries layout/composition.
- **Velocity spectral normalization (this op)** — inside the band `[lo,hi]` of the 2-D radial
  FFT, replace `|V_w|` with `|V_∅|` (keep `v_w`'s phase). This pulls the magnitude toward the
  natural cfg=1 envelope without disturbing the adherence-bearing phase. Scale-correct in one
  pass because `v_∅` is the *same-step* field (see `docs/methods/VELOCITY_SPECTRAL_MATH.md`).
- **Conditions** — `baseline` (plain CFG); `mag_{full,top25,bot25}` = magnitude transplant on
  `[0,1]` / `[0.75,1]` / `[0,0.25]`. (A per-band-power `psd_match` variant exists but was dropped
  for this pass; mag-only matches the demo.)
- **GenEval** — 553 prompts over 6 tags: single_object, two_object, counting, colors, position,
  color_attr. **Score = 1** if all required objects/counts/colours/relations are detected, else 0.
  *Overall* = mean of the 6 per-tag accuracies (macro).
- **Scorer note** — we use the GenEval **protocol** (the official `evaluate()` decision logic,
  thresholds and colour templates, copied verbatim) but with a **torchvision Mask R-CNN v2**
  detector + transformers CLIP colour classifier instead of MMDetection's Mask2Former (`mmcv` is
  brittle on modern torch). Numbers **rank conditions faithfully** but are not bit-identical to
  the Mask2Former leaderboard (our baseline 0.644 vs the published ~0.71).

## Method

`experiments/e37_geneval.py` (`--part preflight,gen,score,summary,site`) generates each
condition over the 553 prompts and scores with `experiments/geneval_score.py`. Each generation
uses **seed = prompt-index, identical across conditions**, so a condition differs from baseline
only by the operator (paired). Interception = `e17_sd35.gen_sd3`'s `scheduler.step` override,
which records the batched `[v_∅, v_c]` and edits `v_w` before the Euler step (no extra forward).
Ran locally on one A5000 at ~3.5 s/img.

## Results

GenEval accuracy (per-tag and macro **Overall**); SD3.5-medium, n=1, 512px, w=4.5:

| condition | **Overall** | single | two_obj | counting | colors | position | color_attr |
|-----------|:-----------:|:------:|:-------:|:--------:|:------:|:--------:|:----------:|
| baseline (plain CFG)         | **0.644** | 0.963 | 0.879 | 0.537 | 0.766 | 0.240 | 0.480 |
| **mag_top25** `[0.75,1]`     | **0.655** | 0.963 | 0.879 | 0.550 | 0.777 | 0.220 | **0.540** |
| mag_bot25 `[0,0.25]`         | 0.561 | 0.912 | 0.808 | 0.425 | 0.681 | 0.190 | 0.350 |
| mag_full `[0,1]`             | 0.524 | 0.900 | 0.727 | 0.450 | 0.638 | 0.120 | 0.310 |

**Ranking: top-25% > baseline > bottom-25% > full** — band placement flips the sign.

- **High-freq normalization (top-25%) slightly beats baseline** (+0.011 overall), driven by
  **color_attr +0.06** (7 prompts fixed, 1 regressed; see `examples_color_attr.html`); single/
  two-object unchanged, colors/counting nudged up. On **counting** it is close to a wash
  (0.537 → 0.550, 3 wins / 2 losses; see `examples_counting.html`).
- **Low-freq normalization (bottom-25%) hurts** (−0.083) and **full-band hurts most** (−0.120),
  worst on the compositional tags (two_object, color_attr, position) — pulling the
  adherence-bearing low band toward cfg=1 erodes composition.

Interpretation: **low-frequency velocity magnitude carries adherence/composition; high-frequency
is CFG's correctable over-amplification.** Touch only the high band.

## Caveats & next

- **Single seed (n=1)** → the +0.011 overall is within seed noise; the per-tag *pattern* is the
  real signal. Confirm the top-25% win with n=4 multi-seed.
- **torchvision-detector variant**, not the Mask2Former leaderboard scorer — good for ranking,
  not absolute comparison.
- Next: sweep the high-band cut (`[0.5,1]`, `[0.85,1]`), try strength<1, a **band-amplify** (gain
  1.6) on `[0.75,1]` over late-only timestep windows, and re-run under the official scorer.

## Reproduce

```bash
# local (A5000), per condition ~3.5 s/img:
python experiments/e37_geneval.py --part gen,score,summary \
    --conditions baseline,mag_full,mag_top25,mag_bot25 --guidance 4.5 --steps 28 --size 512
# example HTML (counting / color_attr), no GPU:
python experiments/e37_geneval.py --part site --compare mag_top25 --site_tag counting
python experiments/e37_geneval.py --part site --compare mag_top25 --site_tag color_attr
# cluster: experiments/cluster_e37_geneval_job.sh (ship via kubectl cp; /storage is not git)
```
Results (gitignored) under `experiments/results/e37_geneval/` (`report.json`, `scores/*.jsonl`).
