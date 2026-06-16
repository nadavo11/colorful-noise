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

**Run pending** (driver + preflight validated: 1001 conditions, all per-object spans resolve
on SD1.5's CLIP tokenizer; ops unit-tested). The deliverable is `results/e35/index.html`:
read the per-operator curves to see, e.g., how low-pass `cut` trades adherence for "gist",
where band/phase gain helps vs hurts fidelity, and whether any of this is prompt-type
dependent (short vs long vs style vs object).

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
```
Code: `experiments/e35_op_sweep.py`, reuses `text_spectral_ops.py`, `e9_clipt.py`,
`fidelity_metrics.py`, `clip_sim.py`, `e9_bandnorm_classes.py`, `common.py`, `e27_site.py`.
Cluster: `experiments/cluster_e35_job.sh` (ship via `kubectl cp`; `/storage` is not git).
