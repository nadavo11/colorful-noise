# E51 — Spectral Edit-Direction Cache Probe

Diagnostic of the hypothesis that the **edit direction**

```
Δ_edit(t) = v_edit(t) − v_src(t) = T(x_t, target_prompt, t) − T(x_t, source_prompt, t)
```

is more temporally and spectrally **cacheable** than the full edited prediction `v_edit(t)`,
for fast image editing (a SeaCache-style idea adapted to editing).

## Pipeline
FLUX.1-dev **img2img** (SDEdit), 4-bit NF4, 512px, 24 steps (strength 0.7), guidance 2.5, seed 0,
RTX A5000. PIE-Bench is the data substrate — it is the only repo subset that ships an explicit
`source_prompt` **and** `target_prompt` per example, which is exactly what Δ_edit needs. All 24
examples (8 task types × 3) are used; an 8-example category-balanced subset gets the dense Pareto
sweep.

## Variants
| variant | cached signal | cache decision |
|---|---|---|
| `full_compute_reference` | — (recompute every step) | gold reference |
| `raw_full_prediction_cache` | `v_edit` | raw relative-L2 stability |
| `spectral_full_prediction_cache` | `v_edit` | low-pass (SEA-style) stability |
| `raw_edit_delta_cache` | `Δ_edit` (base `v_src` live) | raw relative-L2 stability |
| `spectral_edit_delta_cache` | `Δ_edit` (base `v_src` live) | low-pass (SEA-style) stability |

Skip schedules are **oracle**-derived from the reference trajectory's own signal stability, so every
variant chooses its skips from its own signal at the **same skip ratio** — isolating "is this signal a
better guide to where reuse is safe?" rather than online-estimator noise. See report §5.

## Two-environment split (hardware necessity, same as E49/E50)
- **uv env** (`~/.cache/uv/environments-v2/spectral-demo-*/bin/python3`, diffusers 0.38, torch 2.5.1):
  generation — `lib/run.py` (records v_src/v_edit trajectories + runs closed-loop caches, saves images).
- **anaconda env** (`~/anaconda3/bin/python`, torch 2.7 + lpips + matplotlib): metrics + figures + report —
  `lib/evaluate.py`, `lib/analyze.py`, `lib/visualize.py`, `lib/report.py`.

## Reproduce
```bash
UVPY=~/.cache/uv/environments-v2/spectral-demo-ef53f7caffa88925/bin/python3
ANA=~/anaconda3/bin/python
cd spectral_edit_direction_cache_probe/lib
$UVPY run.py                 # generation (uv env) -> diagnostics/generation.jsonl + samples/
$UVPY token_autopsy.py       # E52 token autopsy (uv env) -> diagnostics/token_* + token_autopsy/
$ANA  evaluate.py            # cache metrics  (anaconda) -> diagnostics/results.jsonl
$ANA  analyze.py             # cache aggregate           -> summary.json, metrics.csv, per_example_metrics.csv
$ANA  visualize.py           # cache figures             -> figures/
$ANA  token_evaluate.py      # autopsy metrics (anaconda)-> token_autopsy/token_results.jsonl
$ANA  token_analyze.py       # autopsy aggregate         -> token_autopsy/token_summary.json + csvs
$ANA  token_visualize.py     # autopsy figures           -> token_autopsy/*
$ANA  report.py              # INTEGRATED report.html + report.md (E51 cache + E52 autopsy)
```
(`finalize.sh` chains the anaconda steps once both uv-env generation passes finish.)

## E52 — Text-Token Modulation Autopsy (added module)
Runs alongside the cache probe to answer: *can we identify, visualize and causally manipulate
the text-token components that drive image edits?* FLUX is an **MMDiT** — text enters as a T5
token sequence that attends **jointly** with image tokens (no U-Net cross-attention; text tokens
are the leading key/value columns in every block) plus a pooled-CLIP AdaLN path. The autopsy:
- **instruments** image→text attention (mass / max / value-norm / contribution / spatial maps)
  on a depth-spanning set of double-stream blocks (`token_attn.py`);
- **observes** which tokens/roles dominate, at which steps/blocks, and in which frequency band;
- **intervenes** with four internal mechanisms — embedding scale, attention-logit bias,
  post-softmax reweight, value scaling — over weights {0.5,0.75,1,1.25,1.5,2}× on the edited
  noun / attribute / style / background / control tokens;
- **evaluates** edit strength, preservation, Δ_edit change and spectral change per intervention;
- **connects** token-attention stability to the spectral-delta-cache quality (§14 of the report).
Heavy by design, so it runs on the Pareto subset (intervention grid on the first
`TOK_INTERVENTION_EXAMPLES`). Verdict in `token_autopsy/token_summary.json` and report §14.

## Artifacts (`outputs/spectral_edit_direction_cache_probe/`)
`report.html` · `report.md` · `summary.json` · `metrics.csv` · `per_example_metrics.csv` ·
`figures/` · `samples/<id>/` · `diagnostics/` (per-step trajectories, generation/results manifests,
FFT spectra). Heavy pixels + base64 HTML are gitignored; the JSON/CSV mechanical record is tracked.
Manifest: `experiments/manifests/E51.json`.
