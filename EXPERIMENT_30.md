# E30 — Continuous text-frequency control & extraction

**Follow-up to E24** (numbered E30 — E28/E29 are unrelated seed experiments). E24 found
token-axis frequency bands of Flux's T5 **sequence** embedding are meaningful and
on-manifold, that *merging* two prompts snaps to the low-band/phase owner (and doesn't beat
a `lerp` baseline), and that high-band injection is a *style-strength knob*. E30 (a)
characterizes the bands more finely, (b) turns the manipulation into a **continuous knob**
with image-strip visualizations, and (c) asks what frequency filtering does to **long** and
**compositional** prompts.

Transform reminder (see EXPERIMENT_24.md "How the FFT works"): a **1-D DFT along the token
axis**, computed **independently per embedding channel**, on the **full T5 sequence**
embeddings `(1, L, 4096)` — not 2-D, not the pooled vector.

## Schematic

```mermaid
flowchart LR
  P["Prompt → T5"] --> E["token embeds E (L × 4096)"]
  E --> F["FFT over tokens (per channel)"]
  F --> S["spectrum: low | high"]
  S --> K["one knob: low-pass cutoff · high-band gain · A↔B swap-cut"]
  K --> I["inverse FFT → conditioning"]
  I --> FL["Flux"] --> IMG["image (morphs as the knob turns)"]
```

## Method (`experiments/e30_text_freq_control.py`, Flux)

New ops in `text_spectral_ops.py`: `band_gain_1d(E, lo, hi, gain)` (continuous band
attenuate/amplify) and `band_notch_1d(E, lo, hi)` (per-band knockout). Generation reuses
E24's `load_flux_preencoded_lens` + e10's `gen_emb` (true-CFG=1, guidance 3.5); strips via
`common.save_grid`; metrics via `e9_clipt` (CLIP-T), `e9_bandnorm_classes.image_metrics`
(sharpness / hf_frac / colorfulness), `fidelity_metrics` (aesthetic), `compbench`
(B-VQA = per-object attribute binding), `vqascore` (VQAScore = compositional entailment).

Parts (`--part`):
- **probe_deep** — one prompt: per-band **knockout**, **phase-only vs magnitude-only**
  reconstruction, low/high families. *What does each band control?* (CLIP + image stats).
- **continuous** — the headline visual: image **strips** as one knob varies —
  low-pass cutoff sweep, high-band gain sweep, and a two-prompt **A↔B morph**
  (`band_swap` cut swept), with a fine sweep around E24's one near-balanced regime.
- **concat** — *blend vs one big prompt*: compare `band_swap` / `band_blend` / `lerp`
  merges to a single concatenated prompt "A and B" (CLIP_A / CLIP_B + B-VQA: are **both**
  objects present?).
- **longprompt** — DPG-Bench (`load_dpg_prompts`, ~80-word prompts): low-pass / high-pass /
  band-knockout; **does dropping high freq drop the tail objects while low keeps the
  gist?** (VQAScore + B-VQA retention).
- **compositional** — T2I-CompBench (color/shape/texture): band filtering vs **per-object
  attribute binding** (B-VQA).
- **analyze** — `report.json`, strips/grids, self-contained `index.html` (inline-SVG
  schematic).

## What we expect to learn
- A *continuous* controllability story (the strips): how cutoff/gain/swap-cut smoothly move
  the image, and where transitions are gradual vs abrupt.
- Whether the token spectrum carries **structured semantics** (e.g. high band = specific
  objects/attributes, low band = global gist) — via which objects survive band filtering on
  long/compositional prompts.
- Whether spectral blending offers anything over simply writing "A and B".

## Run

```bash
# self-gating cluster job (smoke probe_deep -> CLIP gate -> full sweep)
runai submit --name e30-text-freq -g 1 -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
  --pvc=storage:/storage --large-shm --command -- \
  bash /storage/malnick/colorful-noise/experiments/cluster_e30_job.sh

# local / single GPU
python experiments/e30_text_freq_control.py --part continuous --num_prompts 1 --steps 8  # smoke
python experiments/e30_text_freq_control.py   # full -> results/e30/{...,index.html}
```

> Cluster note: ship code with `kubectl cp` (the `/storage` checkout is not git; the image
> has no git). Heavy scorers (VQAScore ~11 GB, B-VQA) load in `analyze` after Flux is freed.

## Status
Code complete and offline-verified (ops + builders). Cluster run pending. **Results: TBD.**
