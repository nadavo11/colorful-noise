# Fidelity-at-high-CFG baselines (E16)

Survey of the training-free, inference-time methods we benchmark SBN against, and
the metrics used. Built for **E16** (`e16_prompt_adherence.py`).

## Why this comparison, and what the contest is

`cfg=1.0` Flux already adheres to *simple* prompts and looks realistic; but
practice uses higher CFG (~3.5), usually with **detailed** prompts, where guidance
buys composition/adherence at the cost of realism — over-saturation, over-contrast,
the "plastic" look. E10 made this quantitative: classifier-free guidance inflates
the latent's spectral power above the real-image level, and SBN clamps it back.

So the claim E16 tests is **not** "better prompt adherence." It is: **in the
high-CFG / detailed-prompt regime, SBN + a cheap saturation postprocess (E11)
produces higher-fidelity images than competing training-free methods, while holding
prompt adherence roughly even** (not winning on it). Hence:

- **Fidelity = the contest** (primary metrics).
- **Adherence = a guardrail** — we show it does not regress vs `cfg=3.5`.

All baselines below are training-free and operate at inference time on the **same
Flux-dev** model — the same category as SBN — so the comparison is same-backbone and
fair. (Compositional/adherence specialists like InitNO/ToMe are SDXL-only and
off-axis here, so they were dropped.)

## Methods

### CFG-Zero\* — *Improved Classifier-Free Guidance for Flow Matching Models*
- **Venue / refs:** arXiv 2503.18886 (Mar 2025); repo `WeichenFan/CFG-Zero-star`;
  guider shipped in diffusers as `ClassifierFreeZeroStarGuidance`. Widely adopted
  (diffusers, ComfyUI); a top-tier acceptance was not confirmed at time of writing —
  kept because it is the closest methodological match to our problem.
- **What it does:** the most on-point competitor. Two parts: **(a) optimized scale**
  — a per-sample scalar α corrects the unconditional velocity before the CFG
  extrapolation (α = ⟨v_cond, v_uncond⟩ / ‖v_uncond‖²), and **(b) zero-init** —
  the first few ODE steps output zero velocity so the trajectory does not get pushed
  off-manifold while the flow estimate is still inaccurate. Built specifically for
  flow-matching models (tested on Flux, SD3).
- **How we run it (`e16_baselines.gen_cfgzero`):** faithful, on the stock
  `FluxPipeline`. A transformer-forward recorder captures the cond/uncond velocities
  (the pipeline calls cond then uncond per step) and a `scheduler.step` override
  recomputes the guided update with the optimal scale, zeroing the first
  `zero_init_steps`. Run as true-CFG with distilled guidance held neutral at 1.0 and
  `true_cfg_scale = 3.5` (the same protocol as E10).
- **Risk:** depends on the cond-before-uncond call order; validated in the E16
  preflight by reconstructing plain true-CFG and comparing to the stock pipe.

### SEG — *Smoothed Energy Guidance* (NeurIPS 2024)
- **Refs:** arXiv 2408.00760; repo `SusungHong/SEG-SDXL`; Flux support via
  `pamparamm/sd-perturbed-attention`.
- **What it does:** a condition-free guidance that reduces the *energy curvature* of
  self-attention. It forms a degraded prediction by Gaussian-blurring the
  self-attention (smoothing the attention energy landscape) and guides away from it:
  `pred = pred + s·(pred − pred_blurred)`. Reduces high-CFG artifacts while
  preserving fidelity; validated on FLUX.1-dev and SD3.5.
- **How we run it (`e16_baselines.gen_seg`):** a subclassed `FluxAttnProcessor`
  Gaussian-blurs the **image-token queries** on their 64×64 grid in the perturbed
  branch. We reuse the pipeline's two-pass CFG: the negative branch shares the prompt
  but runs the blurred-query processor, and `true_cfg_scale = 1 + seg_scale`
  reproduces the SEG update. Distilled guidance stays at `cfg=3.5`.
- **Risk (HIGH, droppable):** attention-processor internals are version-fragile.
  `seg_available()` self-tests it on a 4-step run; if it fails on the installed
  diffusers, the driver records `seg_available: false` and the SEG column is simply
  absent from the comparison. SEG is the optional third baseline.

### NAG — *Normalized Attention Guidance* (NeurIPS 2025)
- **Refs:** arXiv 2505.21179; repo `ChenDarYen/Normalized-Attention-Guidance`;
  diffusers + ComfyUI implementations.
- **What it does:** restores stable **negative prompting** in attention space
  (extrapolate Z⁺/Z⁻, L1-normalize, α-blend) where naive CFG diverges. Its headline
  advantage is the **few-step / distilled** regime (Flux-schnell), where ordinary
  true-CFG collapses.
- **How we run it (`e16_baselines.gen_negprompt`):** we benchmark the **native
  two-pass true-CFG with a fidelity-oriented negative prompt** as NAG's practical
  proxy on 28-step Flux-dev. This is an honest choice, not a shortcut: at 28 steps
  ordinary true-CFG negative prompting does *not* diverge, so it is the relevant
  negative-guidance baseline; NAG's attention-space normalization is what rescues the
  few-step regime we are not testing. Labeled `negprompt` in results to avoid
  overclaiming. (The full attention-space processor can be vendored later for a
  few-step study.)

## Metrics

### Fidelity (primary) — `fidelity_metrics.py`
- **LAION aesthetic** — the improved-aesthetic-predictor MLP on L2-normalized CLIP
  ViT-L/14 embeddings (the *same* CLIP `e9_clipt` loads). This is the metric
  CFG-Zero\* reports, so it is head-to-head comparable. Weights pulled once into
  `results/_models/`.
- **ImageReward** — `image-reward` pip package; learned human preference
  (fidelity + aesthetics + coherence in one score).
- **Spectral distance to real** — RMS distance in log-PSD space between a method's
  channel-mean radial PSD and the **E10 real-photo** reference
  (`results/e10/real_latents.pt`). Lower = closer to real images. This is native to
  the SBN thesis: CFG inflates spectral power above the real level, and SBN/postproc
  should sit closest.

### Adherence (guardrail) — CLIP-T + VQAScore
- **CLIP-T** (`e9_clipt.clip_scores`) — the existing cosine; weak on compositional
  detail, which is exactly why we add:
- **VQAScore** (`vqascore.py`, ECCV 2024) — a VQA model's P("yes, the image shows
  <prompt>"); correlates far better with humans on the detailed/compositional
  prompts E16 uses. Used only to confirm SBN(+pp) does **not** drop adherence vs
  `cfg=3.5`; not the contest.

All scorers degrade gracefully (missing package/weights → blank column), and the
heavy ones load only in E16's `--part score` phase, after the diffusion models are
freed, to fit the 24 GB A5000.

## Install (score phase)

```bash
pip install image-reward t2v-metrics      # ImageReward + VQAScore
# LAION aesthetic weights auto-download to results/_models/ on first use
# spectral-distance needs E10 real latents:  python e10_cfg_spectral.py --part download,real
```

`t2v-metrics`' default `clip-flant5-xxl` is ~11 GB; pass `--vqa_model clip-flant5-xl`
(or `--no_vqa`) for a lighter run. ImageReward / VQAScore are optional — the fidelity
contest (aesthetic + spectral-dist) stands without them.
