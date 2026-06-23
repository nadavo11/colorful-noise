# E17 — SD3.5 port: SBN vs CFG-Zero* vs CFG++ (true CFG)

**Status: RAN.** The full fidelity comparison ran (SD3.5-medium, 8 detailed prompts × 25
seeds, cfg=4.5, 28 steps); results live at `experiments/results/e17/{report.json, scores.json}`
plus eight `grid_<scene>.png` strips and a self-contained `index.html`. (`spectral_dist` and
`vqascore` are **null** in this run — see Caveats.) The self-contained HTML report is at
`results/e17/index.html`; this writeup tells the same story and defines the same terms.

## TL;DR

Every spectral method so far lived on **Flux**, whose *distilled* guidance makes the high-CFG
regime behave oddly. We port the methods to **Stable Diffusion 3.5-medium** — which uses *true*
two-pass classifier-free guidance (CFG) — and pit our **SBN** (clamp the generated latent's
power spectrum back to a guidance-off reference) against two published high-CFG fixes,
**CFG-Zero\*** and **Rectified-CFG++**, on image fidelity. **Headline:** the plain high-CFG
baseline (`cfg_hi`) and the two guidance fixes (`cfgzero`, `cfgpp`) win fidelity (`cfgzero` tops
ImageReward at 1.250, `cfg_hi` tops aesthetic at 6.367), while **SBN nudges every fidelity metric
slightly down** (it desaturates, lowers contrast, lifts high-frequency content), and **combining
SBN with a guidance fix does not complement** — `cfgzero_sbn` / `cfgpp_sbn` track plain
`bandnorm`, the late SBN clamp overwriting whatever the guidance produced. CLIP-T (adherence) is
flat across all eight conditions, so nothing trades adherence for the fidelity differences. On
this true-CFG model SBN is a spectral *regularizer*, not a fidelity win.

![E17 finding — on true-CFG SD3.5 the high-CFG baselines (cfg_hi/cfgzero/cfgpp) win fidelity, SBN nudges every fidelity metric slightly down, SBN+fix does not complement, and CLIP-T adherence is flat: SBN is a spectral regularizer, not a fidelity win. Numbers from results/e17/report.json.](figs/E17/finding.jpg)

## Background (plain language)

*The HTML report (`results/e17/index.html`) carries the same glossary inline and leads each
scene with its grid. Defining every term here keeps this writeup self-contained.*

**Setup terms**

- **SD3.5-medium & the latent** — a 2.5B rectified-flow text-to-image model (transformer +
  T5-XXL + 2×CLIP + a 16-channel VAE) at 1024px; it denoises a `(1, 16, 128, 128)` latent that
  the VAE decodes to the image. Fits a 24GB GPU bf16 GPU-resident (`--mem offload` fallback).
- **True CFG (vs Flux's distilled guidance)** — **classifier-free guidance** is the knob for how
  hard the prompt steers generation. SD3.5's `guidance_scale` is *real* CFG: one batched
  `[uncond, cond]` transformer pass per step, then combine `uncond + w·(cond − uncond)`. So
  `guidance=1` is the pure conditional flow field (no steering) and higher `w` pushes harder
  toward the prompt at the cost of over-saturated / over-contrasty output. Flux's guidance is a
  *distilled* embedding instead, which is why we moved here for a clean high-CFG test.
- **The cfg=1 reference** — SBN needs a target spectrum. We generate the same prompt at
  `guidance=1` (steering off) and record, per step and per (channel, radial-frequency band), the
  latent's mean power. That guidance-off trajectory is the **reference** SBN clamps toward.
- **Radial-frequency band / PSD** — 2-D Fourier transform a latent channel; group coefficients by
  distance from the centre (DC) into `n_bins=24` rings. Mean squared magnitude per ring is the
  **power spectral density (PSD)** — low bands = coarse layout / palette power, high bands = fine
  texture.

**The eight conditions (how each is computed)**

- **`cfg1`** — guidance=1, the pure conditional field. Both the SBN reference and a *realism
  anchor* (un-steered: faithful palette/contrast, but weak prompt adherence).
- **`cfg_hi`** *(the BASELINE)* — guidance=`w` (here **4.5**), the plain high-CFG generation.
  Every other condition is a treatment *on top of* this regime and is judged against it.
- **`bandnorm`** *(SBN — ours)* — **Spectral Band Normalization.** At each denoising step the
  `ClampPSD3` callback rescales the cfg=`w` latent's per-(channel, band) PSD back to the cfg=1
  reference at that step, **leaving phase untouched** (op `psd_match`, mode `band`). It pulls the
  over-amplified high-CFG power spectrum back toward guidance-off statistics.
- **`bandnorm_pp`** — `bandnorm` plus an E11 post-process: multiply image saturation by `1.4×`
  (SBN tends to desaturate; this puts colour back).
- **`cfgzero`** *(baseline fix)* — **CFG-Zero\***, a published high-CFG fix: a `scheduler.step`
  override that (1) rescales the uncond term by a per-sample *optimal* scale `α` minimizing the
  guided velocity's error, and (2) *zero-inits* the earliest step(s). Adapted to SD3.5's batched
  output (`make_cfgzero_step`).
- **`cfgzero_sbn`** — CFG-Zero\* **and** the SBN clamp together. They compose: CFG-Zero\* modifies
  the velocity inside `scheduler.step`, SBN clamps the resulting latent at step end. Tests whether
  the two *complement*.
- **`cfgpp`** *(baseline fix)* — **Rectified-CFG++** (arXiv 2510.07631): a predictor/corrector
  that evaluates guidance at a *predicted* next point (with a small corrector jitter `σ`), a
  second high-CFG fix (`gen_rcfgpp_sd3`).
- **`cfgpp_sbn`** — Rectified-CFG++ **and** the SBN clamp together (same composition as
  `cfgzero_sbn`).

**Metrics (and which direction is good)**

- **`aesthetic` (↑)** — LAION aesthetic predictor on CLIP features (learned "how nice does this
  look"). Higher = better.
- **`imagereward` (↑)** — ImageReward human-preference reward model. Higher = more preferred. The
  headline fidelity metric.
- **`clip_t` (↑)** — CLIP image↔text cosine similarity, the **adherence guardrail**: does the
  image still match the prompt? Watched so a fidelity change isn't bought by dropping the prompt.
- **image statistics** *(descriptive, no good direction)* — `sharpness`, `hf_frac` (high-frequency
  energy fraction), `rms_contrast`, `colorfulness`, `saturation`; they explain *how* a condition
  changes the image (SBN lowers saturation/contrast, raises hf_frac).
- **`spectral_dist` (↓), `vqascore` (↑)** — supported by the driver but **null in this run**:
  `spectral_dist` (distance to a real-image PSD) needs an SD3.5-VAE real reference that wasn't
  available; `vqascore` was skipped (`--no_vqa`). Defined for completeness — the story rests on
  aesthetic / ImageReward / CLIP-T.

## Method

- **Backend (`e17_sd35.py`).** `gen_sd3` is one entry point that composes an optional guidance
  `step_override` *and* an optional step-end callback; `ClampPSD3` / `RecordPSD3` (SBN clamp /
  reference recorder — SD3.5 latents are already unpacked `(B,16,128,128)`, so the spectral ops
  apply directly, the cleanest SBN path in the repo); `make_cfgzero_step`; `gen_rcfgpp_sd3`;
  `record_reference_sd3` (the cfg=1 per-step PSD reference). E20 warm-start helpers also live here.
- **What the run does.** For each of 8 detailed prompts we draw 25 seeded init latents (shared
  across conditions, so differences are the *method*, not the seed), record the cfg=1 PSD
  reference, generate all eight conditions at 28-step SD3.5 (cfg=4.5), and score every image.
- **The question each comparison answers.** (a) `bandnorm` vs `cfg_hi`: does SBN improve fidelity
  over plain high-CFG? (b) `bandnorm` vs `cfgzero`/`cfgpp`: does SBN beat the published guidance
  fixes? (c) `cfgzero_sbn` / `cfgpp_sbn` vs either alone: do the spectral clamp and the guidance
  fix *complement*?
- **Drivers.** Fidelity driver `e17_sd35_compare.py` (this writeup); the separate CompBench
  attribute-binding driver `e17_compbench.py` → `results/e17cb` answers the *compositional*
  contest (does SBN preserve binding?), which is not part of this fidelity run.

## Results

The HTML report leads **each scene with its 8-condition grid** (rows = condition, columns =
seeds) → a "what to look for" note → the per-condition numbers, then closes with the aggregate.
Below is the aggregate (mean over the 8 scenes) — the headline verdict; per-scene tables and
grids are in `results/e17/index.html`. n=25 seeds per cell; read directions, not third decimals.

### Aggregate — mean across the 8 scenes

**What to look for:** which condition wins each fidelity column, and where SBN sits relative to
`cfg_hi`. `cfg1` is the un-steered anchor (context), not a competitor.

| condition | aesthetic ↑ | imagereward ↑ | clip_t ↑ | saturation | rms_contrast | hf_frac |
|---|---|---|---|---|---|---|
| cfg1 | 6.069 | −0.172 | 0.280 | 0.320 | 0.196 | 0.034 |
| **cfg_hi** (baseline) | **6.367** | 1.194 | 0.307 | 0.578 | 0.221 | 0.028 |
| bandnorm (SBN) | 6.203 | 1.020 | 0.303 | 0.419 | 0.187 | 0.037 |
| bandnorm_pp | 6.210 | 1.056 | 0.305 | 0.535 | 0.187 | 0.036 |
| cfgzero | 6.351 | **1.250** | 0.304 | 0.506 | 0.234 | 0.029 |
| cfgzero_sbn | 6.226 | 1.052 | 0.303 | 0.421 | 0.192 | 0.037 |
| cfgpp | 6.336 | 1.245 | 0.303 | 0.557 | 0.244 | 0.028 |
| cfgpp_sbn | 6.291 | 1.071 | **0.309** | 0.422 | 0.194 | 0.040 |

*(Bold = best in column. `cfg1` excluded from the "win" since it is the un-steered anchor.)*

**Interpretation (good/bad).**

1. **Guidance owns fidelity; SBN costs fidelity.** `cfgzero` tops ImageReward (1.250) and `cfg_hi`
   tops aesthetic (6.367), all clustered tightly. **SBN (`bandnorm`) lowers every fidelity metric
   vs `cfg_hi`** (aesthetic 6.367→6.203, ImageReward 1.194→1.020): it pulls saturation
   (0.578→0.419) and contrast (0.221→0.187) down toward cfg=1 statistics and *raises* hf_frac
   (0.028→0.037). The learned-preference scorers read that flatter, less-punchy look as slightly
   worse. So **SBN does not beat the guidance fixes on fidelity here** (question a/b: no).
2. **The combinations do not complement** (question c: no). `cfgzero_sbn` (ImageReward 1.052,
   aesthetic 6.226) and `cfgpp_sbn` (1.071, 6.291) land near plain `bandnorm` (1.020, 6.203), **not
   near their guidance parents** (`cfgzero` 1.250 / `cfgpp` 1.245). The late SBN clamp overwrites
   whatever the guidance fix did to the spectrum — composing them just gives back the SBN result.
3. **`bandnorm_pp` recovers colour, not the gap.** The 1.4× saturation post-process lifts
   colourfulness/saturation (0.419→0.535) but ImageReward only 1.020→1.056 — short of `cfg_hi`.
4. **Adherence is flat.** CLIP-T sits in 0.303–0.309 across all eight conditions (`cfgpp_sbn`
   nominally highest at 0.309), so **nothing trades prompt adherence** for the fidelity
   differences — the verdict is purely about the look, not about following the prompt.

**Bottom line.** On a *true-CFG* model the spectral clamp behaves as a regularizer that pulls the
image toward guidance-off statistics (less saturated, less contrasty, more high-freq texture).
The learned fidelity scorers prefer the punchy high-CFG / guidance-fix look, so SBN loses on
fidelity and does not complement CFG-Zero\* or CFG++. SBN's intended advantage — *closeness to a
real-image spectrum* — is exactly the metric that is null here (see Caveats), so this run settles
the fidelity-by-preference question, not the closeness-to-real one.

## Caveats & next

- **`spectral_dist` and `vqascore` are null in this run.** The SD3.5-VAE real PSD reference
  (`SD35_REAL_LATENTS`) wasn't built, so the "closer to a real spectrum" claim SBN is *designed*
  for isn't measured; the fidelity verdict rests on learned-preference scores (aesthetic /
  ImageReward) that reward punchy high-CFG output. VQAScore was deferred (`--no_vqa`). Building
  the SD3.5 real ref (cf. the E23 real-target line) is the natural follow-up to test SBN on its
  home turf.
- **Fidelity ≠ the compositional contest.** Whether SBN *preserves attribute binding* is the
  separate `e17_compbench.py` → `results/e17cb` run (B-VQA on color/shape/texture), not this one.
- **SBN still pays a per-prompt cfg=1 reference cost** (extra `ref_seeds` generations per prompt);
  a universal reference is future work.
- **CFG-Zero\* / CFG++ monkey-patch diffusers internals** (`transformer.forward`,
  `scheduler.step`; restored in a `finally`) and could break on version bumps.
- **Downstream:** this SD3.5 backend (`e17_sd35.py`) is the foundation E18 (style transfer), E20
  (warm-start) and E21/E22 (editing) build on.

## Reproduce

```bash
# Cluster (runai / kubectl): full fidelity run — SD3.5-medium, cfg=4.5
python experiments/e17_sd35_compare.py --part gen,score,analyze \
    --seeds 25 --ref_seeds 3 --cfg 4.5 --steps 28 --n_bins 24 --no_vqa

# Local smoke (few prompts/seeds)
python experiments/e17_sd35_compare.py --part gen,score,analyze \
    --num_prompts 1 --seeds 4 --no_cfgpp --no_vqa

# Rebuild the HTML explainer offline (NO model load, re-templates report.json + grids)
python experiments/e17_sd35_compare.py --part site     # via the driver
python experiments/e17_site.py                          # or the standalone generator

# CompBench binding contest (separate run -> results/e17cb)
python experiments/e17_compbench.py --part gen,score,analyze \
    --categories color shape texture --per_cat 64 --seeds 4 --ref_seeds 2 --cfg 4.5
```

> Results live on `/storage` (gitignored). To rebuild locally, `kubectl cp` the `report.json` +
> the `grid_<scene>.png` files (not the heavy per-seed PNGs) from
> `mystorage-0-0:/storage/.../experiments/results/e17` into local `experiments/results/e17`, then
> run `python experiments/e17_site.py`.

Code: `experiments/e17_sd35.py` (backend), `experiments/e17_sd35_compare.py` (fidelity driver +
`--part site`), `experiments/e17_site.py` (HTML generator), `experiments/e17_compbench.py`
(CompBench), reusing `spectral_ops.py`, `fidelity_metrics.py`, `compbench.py`, and the E16
scoring helpers. Outputs: `results/e17/{report,scores}.json` + `grid_<scene>.png` +
`index.html`.
