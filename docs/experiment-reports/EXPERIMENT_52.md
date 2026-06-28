# E52 — Text-Token Modulation Autopsy

**Status: code-complete, awaiting GPU run (verdict produced by `token_analyze.py`).**

An added module to the E51 *Spectral Edit-Direction Cache Probe*. Shimon's question: before we
commit to a concrete edit-direction, understand what happens to the text/token embeddings with
respect to the input image and edit prompt — which tokens are amplified, which actually control
the edit, whether attention to individual tokens can be weighted, and whether token weighting
changes the edit direction predictably.

**Core research question.** Can we identify, visualize, and *causally* manipulate the text-token
components that drive image edits?

## Where text enters FLUX (stated explicitly)
FLUX.1-dev is an **MMDiT**, not a U-Net with cross-attention. Text reaches the image two ways:
1. the **T5-XXL token sequence** (`encoder_hidden_states`, 512 × 4096) is *concatenated* with the
   image tokens and attends **jointly** in every transformer block. In both the 19 double-stream
   (`FluxTransformerBlock`) and 38 single-stream (`FluxSingleTransformerBlock`) blocks the text
   tokens are the **leading `txt_len` key/value columns** (`cat([encoder, image], dim=1)`), so
   "image→text attention" is the image-query rows attending to those columns;
2. a **pooled CLIP-text vector** (`pooled_projections`) folded into the timestep embedding driving
   **AdaLayerNorm** modulation — a global knob, not per-token.
Token-level control therefore lives in the joint-attention columns of the T5 tokens; that is the
surface this autopsy instruments and intervenes on. We tap a depth-spanning set of double-stream
blocks (the only ones that receive `encoder_hidden_states`, so `txt_len` is unambiguous inside the
processor).

## Method
Runs alongside the E51 cache probe on FLUX.1-dev img2img / PIE-Bench (same pipeline, two-env
split). For each Pareto-subset example:

- **A. Instrumentation** (`token_attn.py`). A `RecordingFluxAttnProcessor` replaces the default
  processor on the tapped blocks. It records, per step × block, image→text attention **mass**,
  **max**, value-norm, and an additive **contribution** proxy for every token, plus per-token
  **spatial attention maps** (image-query attention reshaped to the latent grid) for the role
  tokens. Source/target prompts are tokenized and aligned (difflib) into changed/inserted/deleted
  tokens; five probe roles are assigned (edited noun / attribute / style / background / control).
- **B. Observation** (`token_analyze.py`). Which tokens dominate attention, which change most,
  which correlate with the edit, which role dominates, at which step/block edit tokens matter
  most, and which frequency band each token's effect lives in (per-token Δ_edit **ablation** —
  suppress one token's value, remeasure `v_edit`).
- **C. Causal interventions** (`token_autopsy.py`). Four *internal* mechanisms — embedding scale
  `e_i←α·e_i`, attention-logit bias `logits[:,:,i]+=β`, post-softmax reweight `A[:,:,i]*=γ`
  (renormalized), value scaling `V_i*=γ` — swept over weights {0.5, 0.75, 1, 1.25, 1.5, 2}× on
  each role token, regenerating the edit each time.
- **D. Evaluation** (`token_evaluate.py`). Edit strength (CLIP-T gain / directional), preservation
  (LPIPS / PSNR to the unmodified edit, DINO-to-source), Δ_edit change and spectral (low/mid/high
  band) change per intervention.
- **E. Cache connection** (`token_analyze.py`). Relates token-attention stability / entropy /
  peak-timestep to the E51 spectral-delta-cache quality per example.

## Deliverables
Integrated into the E51 report (`report.html` §14 "Text-Token Modulation Autopsy") plus
`outputs/spectral_edit_direction_cache_probe/token_autopsy/` (token_tables, token_heatmaps,
token_spatial_maps, token_intervention_grids, token_weight_curves, token_cache_correlation,
`token_summary.json`, `per_token_observation.csv`, `token_intervention_metrics.csv`).

## Decision framework (filled by the run)
The verdict is one of STRONG GO / GO / MIXED / NO-GO, derived from: (1) are edit tokens
identifiable attention/contribution hotspots; (2) do internal interventions causally move the
edit; (3) is attention-space weighting better than embedding-space; (4) controllability; (5)
does token weighting make Δ_edit caching easier (stable-attention ↔ better cache; can reweighting
smooth Δ_edit); (6) is the direction promising. The module is intended to decide whether the next
research step is edit-direction caching, token-attention modulation, or a combined method.

## How to run
```bash
UVPY=~/.cache/uv/environments-v2/spectral-demo-*/bin/python3
ANA=~/anaconda3/bin/python
cd spectral_edit_direction_cache_probe/lib
$UVPY run.py && $UVPY token_autopsy.py        # generation (uv env, A5000)
$ANA evaluate.py analyze.py visualize.py       # E51 downstream (or just: bash finalize.sh)
$ANA token_evaluate.py token_analyze.py token_visualize.py token_report.py
$ANA report.py                                 # integrated E51+E52 report
```
