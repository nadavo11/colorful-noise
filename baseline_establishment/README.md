# Baseline Establishment (Phase 2) — spectral FLUX image editing

Training-free baseline suite for real-world **instruction editing** and **reference-based
stylisation**, run to pick the substrate for the next spectral/frequency-domain phase.
Registered as **E49** (thread `style`).

## What it does
Runs 5 no-training baselines on real benchmarks + a custom leakage diagnostic set, computes a
full content/edit/style/leakage metric suite, and emits an extensive self-contained HTML report,
image grids, representation plots, and an MP4 walkthrough.

| slot | model | env |
|---|---|---|
| weak sanity | FLUX img2img (FLUX.1-dev, 4-bit) | uv |
| weak sanity / reference | FLUX Redux (SigLIP prior + FLUX.1-dev) | uv |
| FLUX style | FLUX IP-Adapter (XLabs, InstantStyle analog) | uv |
| competent editor | **FLUX.1-Kontext-dev** (4-bit, native 1024px) | uv |
| classical control | **VGG-19 Gram (Gatys 2016)** — *not* StyleID-attention; legacy key `styleid` | anaconda |
| strong external | Qwen-Image-Edit — **not run** (20B > 25GB VRAM/disk budget) | — |

> **Verdict scope:** "PROCEED_WITH_FLUX_KONTEXT" = *FLUX.1-Kontext-dev is the strongest accessible
> no-training substrate under current compute constraints*, not a global best-editor claim
> (Qwen-Image-Edit was not run). The classical-style slot is the Gatys VGG-19 Gram control, not the
> StyleID attention-injection method.

## Environments (two, by hardware necessity)
- **uv env** `~/.cache/uv/environments-v2/spectral-demo-ef53f7caffa88925` — diffusers 0.38 /
  transformers 4.57 / torch 2.5.1+cu124. Runs the FLUX pipelines. (Do **not** `pip install`
  torchvision here — it upgrades torch to cu130 and breaks CUDA on this driver.)
- **anaconda env** `~/anaconda3` — torch 2.7+cu126 + torchvision + lpips + transformers. Runs
  StyleID generation, all metrics, all visuals, and the report.

## Reproduce
```bash
cd baseline_establishment/lib
python config.py                                   # write phase config
python build_data.py            # (anaconda) stream MagicBrush/PIE-Bench/WikiArt subsets
bash run_pilot.sh               # StyleID (anaconda) + 4 FLUX models (uv)
python evaluate.py  --phase pilot   # (anaconda) -> metrics csv + summary json
python visualize.py --phase pilot   # (anaconda) -> figures/ + videos/
python report.py    --phase pilot   # -> reports/baseline_establishment_report.html
```
Seeds/size/steps are fixed in `lib/config.py` (`PHASE_VERSION`, `SEEDS`, `GEN_SIZE`,
`INFER_STEPS`, `GUIDANCE`). Kontext runs at its native 1024 bucket with 20 steps.

## Layout
```
configs/    phase_config.json
data/       benchmark_subsets/{magicbrush,piebench}/  style_refs/  custom_leakage_set/  (manifests = jsonl)
outputs/<model>/   generated PNGs + manifest_<phase>.jsonl
metrics/    baseline_establishment_metrics.csv  baseline_establishment_summary.json
figures/    grids/  best_cases/  worst_cases/  leakage_cases/  representation_visuals/
videos/     baseline_walkthrough.mp4
reports/    baseline_establishment_report.html
logs/
```

## Hard rules honoured
No training of any kind (no LoRA/finetune/DreamBooth/learned adapters/per-image latent codes as a
baseline). Real benchmarks only (no SVG toy set). Official-ish inference settings, global (not
per-example) config, fixed seeds. Substitutions (Qwen→Kontext, StyleID→VGG-Gram, missing prior
report) are documented in the report.
