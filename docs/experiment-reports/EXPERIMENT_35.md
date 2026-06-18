# E35 — Token-frequency operator sweep on SD1.5

**The direction.** We accumulated a large toolkit of token-axis FFT operators on the text
conditioning (E24/E30/E32) but only spot-checked pieces on Flux. E35 systematically maps the
*whole* toolkit on a fast model (SD1.5): for every operator × parameter level × prompt type,
what happens to prompt **adherence** and image **fidelity**, and how far the edit moves the
image. The output is a "what does each knob do, and to which kinds of prompts" map.

## Background (plain language)

- **Signal.** SD1.5 encodes a prompt with CLIP into a **sequence embedding** `E (1, 77, 768)`.
  Every operator FFTs the **token axis** (1-D, per channel) on the real-token span `[:L]`
  (`L = attention_mask.sum()`; BOS/…/EOS are real, padding untouched), edits the spectrum,
  inverts, and we generate from the edited `prompt_embeds` (CFG with an empty-string negative).
- **Operators (13).** baseline · low-pass · high-pass · band gain · notch · phase-only ·
  mag-only · phase band-keep · phase gain · per-object band gain · two-prompt
  band-swap/band-blend/lerp. (Same ops as the interactive demo, ported to SD1.5's CLIP-77.)
- **Frequency.** Normalised `0→1`: DC = mean across tokens; low = slow/global meaning; high =
  sharp token-to-token detail. `cut` splits low/high; `band` picks the side; `gain` scales it.

## Method

- **Model.** SD1.5 (`StableDiffusionPipeline`, fp16, DDIM, 512px, guidance 7.5, 30 steps),
  the e25 loader pattern. Generation from edited embeddings, **5 seeds per condition batched**.
- **Prompt set.** 25 prompts across 5 categories — **short / long-detailed / art-style /
  single-object / two-object** (5 each); object/two-object prompts carry an object phrase;
  6 distinct **pairs** for the two-prompt merge ops.
- **Parameter grids** (`--coverage {quick,thorough,max}`; thorough default): low/high-pass
  `cut∈{.1,.2,.3,.45,.65}`; band gain `band{low,high}×gain{.25,.5,1.5,2,3}` at cut .25;
  notch & phase band-keep over `cut`; phase gain `gain{.5,2}`; per-object `band×gain{.5,2}`;
  two-prompt swap over `cut`, blend, lerp. **≈1001 conditions × 5 = ~5005 images, ETA ~2.2h.**
- **Metrics (code refs).** `clip_scores` (CLIP-T adherence; vs A/B for pairs; vs object phrase
  for object prompts) · `aesthetic_scores` (LAION, no-ref fidelity, reuses the CLIP) ·
  `image_metrics` (sharpness/hf_frac/colorfulness) · **drift** = `1 − cosine(CLIP_img(baseline),
  CLIP_img(edit))` per (prompt, seed) via `clip_sim`. No SD1.5 PSD reference exists, so fidelity
  is no-reference (aesthetic + stats); heavy scorers (ImageReward/VQAScore/B-VQA) are skipped.
- **Outputs.** `results/e35/`: `report.json` (per (category, op, param, seed) raw + aggregates);
  per-operator **metric-vs-parameter** plots faceted by prompt category (CLIP / aesthetic /
  drift); `save_grid` contact sheets; self-contained `index.html`.

## Findings

Ran on runai (SD1.5, 1001 conditions, 5005 images). Operator means (CLIP adherence /
LAION aesthetic / baseline-drift), sorted by adherence; baseline = 0.283 / 5.83:

| operator | CLIP | aesthetic | drift |
|---|---|---|---|
| baseline | 0.283 | 5.83 | — |
| lerp | 0.275 | 5.96 | 0.13 |
| **per-object band gain** | 0.274 | 5.63 | **0.09** |
| high-pass | 0.224 | 5.07 | 0.22 |
| band-blend | 0.213 | 4.68 | 0.36 |
| band gain | 0.212 | 4.86 | 0.26 |
| band-swap | 0.204 | 4.52 | 0.37 |
| **phase-only** | **0.187** | **4.47** | 0.36 |
| low-pass | 0.174 | 4.49 | 0.38 |
| notch | 0.171 | 4.40 | 0.39 |
| phase band-keep | 0.165 | 4.09 | 0.36 |
| phase gain | 0.165 | 4.24 | 0.37 |
| **mag-only** | **0.145** | **3.85** | **0.43** |

1. **Phase ≫ magnitude carries the content — replicated on SD1.5/CLIP-77.** phase-only
   (0.187 / 4.47) beats mag-only (0.145 / 3.85) on both adherence and fidelity, and mag-only
   has the **largest drift** (0.43, moves furthest from baseline). This reproduces E30's
   Flux/T5 result on a different architecture. The phase>mag gap is **largest for long /
   compositional prompts** (mag-only CLIP drops to 0.13 on long/object/twoobj vs phase-only
   ~0.17–0.23).
2. **Localized / interpolation edits are the gentlest.** per-object band gain (drift 0.09,
   CLIP 0.274 ≈ baseline) and lerp (drift 0.13) barely perturb — per-object stays near baseline
   (object 0.281 vs 0.284), consistent with E32's small-but-real effect.
3. **High-pass > low-pass for adherence** (0.224 vs 0.174): keeping high-frequency token
   detail + DC preserves more prompt content (esp. object identity) than the low "gist" alone.
4. **Aggressive band surgery degrades both axes.** low-pass / notch / phase band-keep / phase
   gain / mag-only all fall to CLIP ~0.15–0.17 and aesthetic ~3.9–4.5 with large drift — most
   single-band surgery costs adherence *and* fidelity on SD1.5.

Per-parameter curves (faceted by prompt category) and contact sheets are in
`results/e35/index.html`.

### Follow-up: vs the BASELINE GENERATION (the right reference)

The headline/delta views score the edit against the **prompt** (CLIP-T), where the unedited
baseline is the ceiling *by construction* — so `delta.html`'s Δ-vs-baseline heatmaps can only
ever be red. That measures "how far the edit fell from the prompt", not "is the edited image a
better image than the model would have made". `e35_vs_baseline.py` re-references every edit to
its **same-seed baseline image** and splits the question in two (writes `vs_baseline.html`):

- **Directional — does the operator IMPROVE on the baseline image?** `d_imagereward`
  (prompt-aware, learned; **not** capped at baseline, so green is achievable), `d_aesthetic`,
  and signed image-stat deltas. **Result: no operator beats baseline.** Ranked by ΔImageReward
  (all-category mean): per-object −0.17, lerp −0.40, then everything −1.5…−2.6; on Δaesthetic
  only **lerp** is positive (+0.05) and per-object is ~0. The two interpolation/localized ops
  are merely the *least harmful* (near no-ops); single-band surgery is uniformly downhill.
- **Distance — how far / in what way it moved.** `clip_i2i_dist`(=drift), `lpips`, `dssim`,
  `img_psd_l2` (image-domain spectral change — the low-pass/phase-only texture axis CLIP-img
  misses), `color_l2`. Cleanly ordered: per-object/lerp gentlest (LPIPS 0.34/0.42), mag-only
  moves furthest (LPIPS 0.90, img-PSD 2.29). The operators ARE large, controllable distance
  knobs — the edits are real and big — but on average the moves lower image quality.
- **Merges still don't pull in B:** `d_imagereward_B` swap −0.07 / blend −0.07 / lerp +0.006;
  lerp's tiny B-gain is dwarfed by its A-cost (−0.40). Reproduces E24's "MERGE negative".

**Takeaway.** Measured against the correct (uncapped, prompt-aware) reference, the SD1.5
token-frequency toolkit is a clean **negative result as an image-*improvement* tool**, while
remaining a genuine **distance/diversity** lever. "Interesting by eye" is a tail/distance
effect (big LPIPS/PSD), not a mean-quality gain — both reference framings now agree.
Tier 1 of the view (Δaesthetic, image-stats, CLIP-i2i) is free from `report.json`; Tier 2
(`--with_images`: LPIPS/SSIM/PSD/color/ImageReward) re-scores the saved PNGs, no re-gen.

## Caveats & next

- **No-reference fidelity.** Aesthetic + image-stats are proxies; there's no SD1.5
  real-image PSD reference (only Flux). A follow-up could build one (encode ~500 COCO imgs).
- **CLIP-77 vs T5-512.** SD1.5 has far fewer tokens than Flux's T5; band structure is coarser,
  so absolute effects may differ from E24/E30/E32 — E35 characterises SD1.5, not Flux.
- **Per-object on CLIP** uses the token-id-subsequence fallback (slow tokenizer, no offsets);
  validated on all object prompts but can miss exotic phrasings.
- **Next:** lift the strongest operator×regime findings back to Flux for confirmation; feed the
  TI (E33) and channel-axis (E34) proposals.

## Reproduce

```bash
python experiments/e35_op_sweep.py --part preflight                 # counts/spans/ETA, no GPU
python experiments/e35_op_sweep.py --part gen,analyze --coverage quick \
    --num_prompts 2 --seeds 2 --steps 8 --out_tag smoke             # smoke
bash experiments/cluster_e35_job.sh                                 # full on runai
python experiments/e35_vs_baseline.py                              # vs-baseline view (tier 1)
python experiments/e35_vs_baseline.py --with_images               # + LPIPS/SSIM/PSD/ImageReward
bash experiments/cluster_e35_vsbase_job.sh                         # vs-baseline view on runai
```
Code: `experiments/e35_op_sweep.py` + `experiments/e35_vs_baseline.py` (vs-baseline view),
reuses `text_spectral_ops.py`, `e9_clipt.py`, `fidelity_metrics.py`, `clip_sim.py`,
`e9_bandnorm_classes.py`, `common.py`, `e27_site.py`.
Cluster: `experiments/cluster_e35_job.sh` (full sweep) and `experiments/cluster_e35_vsbase_job.sh`
(vs-baseline views, no re-gen) — ship via `kubectl cp`; `/storage` is not git.
