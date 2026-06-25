# EXPERIMENT 49 — Baseline establishment for the spectral-editing substrate

**Thread:** `style` (Spectral style transfer & editing) · **Status:** done ·
**Verdict:** **PROCEED_WITH_FLUX_KONTEXT**

> **Scope of the verdict (read first).** This is **not** a global claim that Kontext beats all
> editors. It says: **FLUX.1-Kontext-dev is the strongest *accessible no-training* substrate under
> current compute constraints.** Qwen-Image-Edit (20B) — the one stronger open editor we wanted —
> was **not run** because its weights exceed the current VRAM/disk budget (single 25 GB A5000), so
> it is untested here, not beaten. The "StyleID" slot is the **classical VGG-19 Gram (Gatys 2016)
> control**, *not* the StyleID attention-injection method (Chung et al. 2024) — see Setup.

## Why
Phase 1 returned **NO_GO** for naive spectral interventions — but *on weak FLUX pipelines*
(vanilla img2img, Redux). That is a statement about the substrate, not the idea: a frequency-domain
edit needs a competent editor underneath it to have anything to act on. This phase establishes,
empirically and training-free, **which no-training baseline is actually competent** at real-world
instruction editing and reference stylisation, and on which tasks — so the next spectral phase starts
on solid ground.

## Setup
Five **training-free** baselines (no LoRA / finetune / DreamBooth / learned adapters / per-image latent
codes), all FLUX transformers 4-bit NF4 on one RTX A5000:

| slot | model | role |
|---|---|---|
| weak sanity | FLUX img2img (FLUX.1-dev) | global renoise-to-caption |
| weak sanity / reference | FLUX Redux (SigLIP prior + FLUX.1-dev) | reference remix |
| FLUX style | FLUX IP-Adapter (XLabs) | InstantStyle-on-FLUX analog |
| competent editor | **FLUX.1-Kontext-dev** (native 1024px) | in-context instruction editor |
| classical control | **VGG-19 Gram (Gatys 2016)** — *not* StyleID-attention | content-anchored texture transfer |
| strong external | Qwen-Image-Edit | **NOT RUN** — 20B > 25 GB VRAM / disk budget |

The classical-style slot is implemented as **per-image VGG-19 Gram-matrix optimization (Gatys et al.
2016)**: frozen ImageNet VGG-19, Adam over pixels, content+style losses. It is a *low-leakage
stylisation control*, **not** the diffusion-attention StyleID method (Chung et al. 2024). The
registry key `styleid` is legacy naming; treat every "StyleID" mention here as "VGG-Gram control".

**Data (real-world):** MagicBrush dev (18), PIE-Bench++ (24 across 8 task types: object
replace/add/remove, attribute, colour, material, background, style), WikiArt style bank (12), and a
custom **20-pair content×style leakage set** (aligned + adversarial). **164 generations total.**

**Metrics:** content preservation (CLIP-I, DINOv2, SigLIP, LPIPS, colour-hist); edit correctness
(CLIP-T to target, **CLIP-T gain = target − source**); style (CLIP-I/DINO to style ref, colour-hist,
Fourier-amplitude); **reference leakage** (DINO/CLIP-I to the style *image* — high ⇒ output copied the
reference's semantics). Two-env split forced by hardware: uv (diffusers 0.38) for FLUX, anaconda
(torch 2.7) for StyleID + metrics + visuals.

## Headline numbers
### Editing (MagicBrush + PIE-Bench)
| model | CLIP-I content | DINO content | CLIP-T target | **CLIP-T gain** | LPIPS↓ |
|---|---|---|---|---|---|
| FLUX img2img | 0.892 | 0.757 | 0.282 | **−0.019** | 0.253 |
| **FLUX Kontext** | 0.890 | **0.813** | 0.274 | **+0.017** | 0.301 |

img2img preserves pixels but its edit gain is **negative** — it renoises without following the
instruction. **Kontext is the only baseline that both preserves content and moves toward the target.**

### Style transfer + reference leakage (20 pairs each)
| model | CLIP-I style | DINO **style (leak)** ↓ | DINO content | **leak-resistance** (content − style) |
|---|---|---|---|---|
| FLUX Redux | **0.800** | 0.695 | 0.013 | **−0.68** |
| FLUX IP-Adapter | 0.786 | 0.518 | 0.028 | −0.49 |
| StyleID / Gram | 0.524 | 0.196 | 0.643 | +0.45 |
| **FLUX Kontext** | 0.502 | **0.097** | **0.813** | **+0.72** |

Redux has the highest raw style adherence but **the worst leakage** — it remixes the reference's
objects and destroys the content. StyleID and Kontext are the content-preserving, low-leakage options.

## Verdict
**PROCEED_WITH_FLUX_KONTEXT** — i.e. **FLUX.1-Kontext-dev is the strongest accessible no-training
substrate under current compute constraints.** Among the baselines we could actually run, Kontext is
simultaneously the best instruction editor (positive CLIP-T gain, top content preservation) and the
most leakage-resistant reference styliser — exactly the worthy substrate the Phase-1 weak baselines
were not. Redux/IP-Adapter reproduce the Phase-1 weakness (high reference-content leakage). The
**VGG-Gram (Gatys) control** is the clean low-leakage classical reference point for style work.
This is a within-budget statement: the stronger open editor (Qwen-Image-Edit, 20B) was not run, so
Kontext is "strongest accessible", not "globally best".

**Most informative subsets for the spectral phase:** PIE-Bench **colour / material / object-replace**
(clean source→target prompts give a real CLIP-T-gain signal) and the **adversarial leakage pairs**
(the cleanest probe of content-vs-style movement under an intervention).

## Next
- **P1:** run the next spectral/frequency-domain intervention **on FLUX.1-Kontext-dev**, evaluated on
  PIE-Bench colour/material/object-replace + the adversarial leakage pairs, with **StyleID as the
  low-leakage control**.
- **P2:** scale the pilot subset (50+/benchmark) on Kontext once the spectral op is wired.

## Artifacts
- Report: `baseline_establishment/reports/baseline_establishment_report.html` (16-section, self-contained)
- Metrics: `baseline_establishment/metrics/baseline_establishment_metrics.csv` (164 rows) + `…_summary.json`
- Figures: `baseline_establishment/figures/` — grids (`edit_comparison`, `style_comparison`),
  `best_cases/`, `worst_cases/`, `leakage_cases/`, `representation_visuals/` (heatmaps, scatters,
  Pareto, similarity matrix, runtime)
- Video: `baseline_establishment/videos/baseline_walkthrough.mp4`
- Manifest: `experiments/manifests/E49.json` · Code: `baseline_establishment/lib/`
