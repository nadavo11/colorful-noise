# EXPERIMENT 50 — Spectral Kontext Pilot

**Thread:** `style` (Spectral style transfer & editing) · **Status:** done ·
**Verdict:** **PROCEED_INTERNAL_KONTEXT_SURGERY**

> First spectral pilot on the substrate E49 picked. **External, training-free FFT edits of
> FLUX.1-Kontext-dev's inputs/references do not beat raw Kontext's content/style/leakage tradeoff —
> but they yield a clean mechanistic map (phase = semantics, amplitude = texture, low-frequency =
> enough to edit) that says the next intervention belongs *inside* the model.**

## Why
E49 established **FLUX.1-Kontext-dev** as the strongest accessible no-training substrate. E50 asks
the first real spectral question on top of it: now that a competent editor sits underneath, do
**frequency-domain manipulations of its inputs** (source image, or a content×style reference
composite) or its **instruction** improve the content/style tradeoff, or reduce reference semantic
leakage? Focused pilot on the exact E49 subsets — comparable one-to-one to the E49 Kontext baseline.

## Setup
**66 Kontext generations** (4-bit NF4, native 1024px, 20 steps, g=2.5, seed 0), training-free, no
per-example optimization. Three interventions:

| exp | site | variants | n |
|---|---|---|---|
| **A — spectral source** | model input image | raw / phase-only / amplitude-only / low-band / high-band | 30 (6 PIE tasks ×5) |
| **B — spectral reference** | input = FFT content×style composite | content_raw / content-phase+style-amp / style-phase+content-amp / style-high-on-content | 24 (6 leak pairs ×4) |
| **C — prompt variants** | text instruction | neutral / content-preserving / anti-leakage | 12 (4 pairs ×3) |
| D — latent/timestep | *deferred to E51* | non-invasive pilot | — |

Operators: per-channel 2-D FFT (numpy), A=|F|, P=∠F, recon=Re(ifft(A·e^{iP})); radial bands
low<0.15 / mid 0.15–0.45 / high>0.45 of Nyquist. Subsets: PIE-Bench `pie_{6_0,7_0,1_0,2_1,8_0,9_1}`
(color/material/object-replace/object-add/global/style); adversarial pairs
`leak_{0,3,6,7,8,9}_adversarial` (photo content × WikiArt style). E49 metric suite reused. Controls
(no new gen): E49 Kontext (1024px), Redux (high-leak), VGG-Gram/Gatys (low-leak).

## Headline numbers
### B — spectral reference composites (the clean result)
| reference op | DINO content | CLIP-I style | DINO style (leak) | **leak_gap** (content−style) |
|---|---|---|---|---|
| content_raw (baseline) | **0.85** | 0.46 | 0.08 | **+0.77** |
| content-phase + style-amp | 0.50 | **0.58** | 0.25 | +0.24 |
| style-phase + content-amp | **0.01** | 0.77 | 0.57 | **−0.56** |
| style-high-on-content | 0.85 | 0.46 | 0.09 | +0.77 |

- **Phase carries the copy-able semantics.** `style-phase+content-amp` catastrophically leaks
  (DINO-content 0.01) — the style reference's objects fully replace the content. A face leaks into
  the architecture pair; the abstract pair imports the painting's shapes.
- **Amplitude carries texture.** `content-phase+style-amp` raises style adherence (0.46→**0.58**)
  but at a content cost (0.85→0.50, leak_gap +0.77→+0.24) — a move *along* the Pareto front, not above it.
- **`style-high-on-content` ≈ baseline** — Kontext re-renders the high-frequency graft away.

### A — spectral source decomposition (instruction edits)
| source op | DINO content | CLIP-T gain | LPIPS↓ |
|---|---|---|---|
| raw | **0.66** | +0.046 | **0.39** |
| low_band | 0.56 | **+0.049** | 0.45 |
| phase_only | 0.54 | +0.045 | 0.61 |
| high_band | 0.44 | +0.044 | 0.59 |
| amplitude_only | **0.14** | +0.033 | 0.74 |

Raw source is needed to preserve identity; **low-frequency is sufficient for (marginally better)
instruction-following** but costs content; amplitude-only destroys structure (Kontext hallucinates a
new scene).

### C — prompt formulation (leakage resistance)
| instruction | leak_gap | DINO content |
|---|---|---|
| content-preserving | **+0.78** | **0.87** |
| neutral | +0.75 | 0.83 |
| anti-leakage | +0.63 | 0.72 |

**Leakage is not mainly a prompt problem** — neutral already scores +0.75. Positive
"preserve identity/layout/shapes" wording helps slightly; the explicit **anti-leakage instruction
backfires** (+0.63), apparently priming the very objects/people/layout it tells the model to avoid.

## Verdict
**PROCEED_INTERNAL_KONTEXT_SURGERY.** No external training-free FFT edit of Kontext's
inputs/references improved its tradeoff over raw Kontext: amplitude transfer trades content for
style rather than pushing the frontier out, phase transfer is pure leakage, high-frequency grafts
are re-rendered away. But the pilot is not a dead end — it produced a clean, actionable map of
**which spectral structure controls which visual concept** (phase→semantics/leakage,
amplitude→texture, low-band→edit-following). That map localises the next intervention **inside** the
model rather than at the pixel input.

## Next (E51)
Move the same phase/amplitude/band decomposition **inside** Kontext: timestep-banded latent edits
(early low-frequency content lock, late high-frequency style injection) and/or attention-feature
interventions, probed on the adversarial leakage pairs where the phase→semantics signal is cleanest.
Scale the subset once the internal operator is wired.

## Artifacts
- Report: `e50_spectral_kontext_pilot/reports/e50_spectral_kontext_pilot.html` (19-section, self-contained)
- Metrics: `e50_spectral_kontext_pilot/metrics/e50_metrics.csv` (66 rows) + `e50_summary.json`
- Figures: `e50_spectral_kontext_pilot/figures/` — grids (source/reference/prompt), `fourier/`,
  `representation_visuals/` (heatmaps, scatters, prompt bar), `best_cases/`, `worst_cases/`, `leakage_cases/`
- Video: `e50_spectral_kontext_pilot/videos/e50_kontext_spectral_walkthrough.mp4`
- Manifest: `experiments/manifests/E50.json` · Code: `e50_spectral_kontext_pilot/lib/`
