# E50 — Spectral Kontext Pilot

Focused, training-free pilot testing whether **frequency-domain interventions on FLUX.1-Kontext-dev
inputs** improve its content/style tradeoff or reduce reference semantic leakage — now that **E49**
established Kontext as the strongest accessible no-training substrate. Registered as **E50**
(thread `style`). Builds directly on `../baseline_establishment/` (E49): reuses its FluxKontext
loader, its metric suite, and the exact same source/style images.

## Interventions (all training-free, no per-example optimisation)
| exp | site | what varies | question |
|---|---|---|---|
| **C** spectral source | model input image | raw / phase-only / amplitude-only / low-band / high-band of the source | which source frequency components does Kontext need to keep identity *and* follow the instruction? |
| **A** spectral reference | model input (FFT content×style composite) | content_raw / content-phase+style-amp / style-phase+content-amp / style-high-on-content | does content-phase+style-amplitude transfer texture while suppressing reference leakage? does style-phase cause copying? |
| **B** prompt variants | text instruction | neutral / content-preserving / anti-leakage | is leakage a prompt problem or a model problem? |
| **D** latent/timestep | *deferred to E51* | — | non-invasive pilot; documented, not run |

Primary model: **FLUX.1-Kontext-dev** (4-bit NF4, same as E49). Controls (no new generation): E49
Kontext (1024px), E49 Redux (high-leak reference), E49 VGG-19 Gram/Gatys (low-leak classical).

## Subsets (exact E49 ids → one-to-one comparable)
- **Edits (Exp C):** `pie_6_0` color, `pie_7_0` material, `pie_1_0` object-replace, `pie_2_1`
  object-add, `pie_8_0` global, `pie_9_1` style.
- **Adversarial leakage pairs (Exp A/B):** `leak_{0,3,6,7,8,9}_adversarial` (photo content × WikiArt
  watercolor/abstract/oil/impressionist style).

66 Kontext generations (30 source + 24 reference + 12 prompt).

## Reproduce
```bash
cd e50_spectral_kontext_pilot/lib
python config.py                 # write configs/phase_config.json
python prepare.py                # (anaconda) build spectral input PNGs + data/manifests/e50_jobs.jsonl
python run_kontext.py            # (uv env)  run all 66 Kontext jobs -> outputs/ + e50_results.jsonl
python evaluate.py               # (anaconda) -> metrics/e50_metrics.csv + e50_summary.json
python visualize.py              # (anaconda) -> figures/ + videos/
python report.py                 # (anaconda) -> reports/e50_spectral_kontext_pilot.html
```
Two envs as in E49: FLUX in the uv env
(`~/.cache/uv/environments-v2/spectral-demo-ef53f7caffa88925`), everything else in anaconda.
Spectral operators (`spectral.py`) are pure numpy and run in either env. Seed/steps/guidance fixed
in `lib/config.py`. Note: `FluxKontextPipeline` snaps to its native 1024px bucket, so outputs match
the E49 Kontext baseline resolution; the spectral op is applied to the 512px input pre-encode.

## Traceability
Every generation in `data/manifests/e50_results.jsonl` records source id, style id, instruction,
spectral op, model, seed, inference settings, the prepared input-image path, and the output path.
Raw output PNGs are gitignored (reproducible from manifests + lib); manifests/metrics/figures/video
and the report are tracked.

## Hard rules honoured
No training (no LoRA/finetune/DreamBooth/adapters). Real images (E49 PIE-Bench + WikiArt + photo
content). Exact E49 subsets for direct comparison. Visuals-first reporting. Each method section
carries model / data / supervision / one-line insight. Exp D deferred and documented.
