# E16: fidelity at high CFG — SBN vs training-free guidance baselines

**Question.** `cfg=1.0` Flux already adheres to *simple* prompts and looks
realistic, but practice uses high CFG (~3.5) for *detailed* prompts, where guidance
buys composition at the cost of realism (E10: CFG inflates spectral power above the
real-image level). Does SBN + a cheap saturation postprocess (E11) give
**higher-fidelity** images than recent training-free guidance methods in that
regime, **without** losing prompt adherence?

**Design.** Fidelity is the contest; adherence is a guardrail. 8 detailed prompts,
paired seeds, all on Flux-dev with a shared seeded initial latent per seed.
Conditions: `cfg1.0` (realism anchor), `cfg3.5` (degraded baseline SBN fixes),
`bandnorm` (SBN), `bandnorm_pp` (SBN + saturation ×1.4 — **our full method**),
`cfgzero` (CFG-Zero\*), `negprompt` (true-CFG + fidelity negative prompt, NAG proxy),
`seg` (Smoothed Energy Guidance, if available). Methods + metrics are documented in
`FIDELITY_BASELINES.md`.

**Metrics.** Fidelity (primary): LAION **aesthetic**, **ImageReward**,
**spectral-distance-to-real** (vs E10 `real_latents.pt`). Adherence (guardrail):
**CLIP-T**, **VQAScore**. Plus E9 `image_metrics`.

**Files.** Driver `e16_prompt_adherence.py` (parts `gen,score,analyze`); baselines
`e16_baselines.py` (CFG-Zero\* via a transformer-forward recorder + `scheduler.step`
override; SEG via a blurred-query attention processor, self-tested by
`seg_available()`); metrics `fidelity_metrics.py` + `vqascore.py`. Outputs under
`results/e16/`: per-prompt `images/`+`latents/`, `scores.json`, `summary.md`,
`grid_<prompt>.png`, `report.json`.

**Run.**
```bash
pip install image-reward t2v-metrics              # ImageReward + VQAScore (optional)
python e10_cfg_spectral.py --part download,real   # real-image ref for spectral-dist
# smoke (always do this first — 6 steps, 2 seeds):
python e16_prompt_adherence.py --part gen   --num_prompts 1 --seeds 2 --ref_seeds 1 --steps 6
python e16_prompt_adherence.py --part score --num_prompts 1 --seeds 2 --no_vqa
python e16_prompt_adherence.py --part analyze --num_prompts 1 --seeds 2 --grid_n 2
# full sweep:
python e16_prompt_adherence.py --part gen,score,analyze --num_prompts 8 --seeds 25 --steps 28
```
Caching keys on the condition tag, **not** the step count — clear `results/e16/`
before changing `--steps`, or stale low-step images will be reused.

## Status / smoke findings (N=2, 6 steps — validation only, not results)

Pipeline validated end-to-end: all 7 conditions generate; aesthetic / ImageReward /
spectral-dist / CLIP-T all compute; VQAScore + SEG degrade gracefully when absent.
Two things to watch when reading the real sweep:

- **bandnorm_pp led on the perceptual fidelity metrics** (aesthetic, ImageReward)
  and did **not** drop CLIP-T vs `cfg=3.5` — the hoped-for pattern. Confirm at the
  full N.
- **spectral-distance-to-real does *not* favor SBN, by construction.** E10 found
  real images sit near `cfg≈3.5` spectral power, *above* the unguided `cfg=1.0`
  field; SBN clamps power *down* toward the `cfg=1.0` reference, so it moves
  **away** from real on this axis while CFG-Zero\*/SEG/negprompt stay closer. So
  spectral-dist is an honest *diagnostic* of what SBN does to the spectrum, but it
  is **not** evidence for "SBN looks more real" — that case rests on
  aesthetic/ImageReward (and a human/qualitative read). Frame accordingly.

- SEG at `seg_scale=3` looked over-perturbed in the 6-step smoke (low aesthetic);
  tune `--seg_scale` / `--seg_sigma` (or `--no_seg`) on the full run.
