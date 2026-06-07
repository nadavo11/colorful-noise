# Spectral latent manipulation experiments

Extends [Colorful-Noise](https://github.com/Nadavc220/colorful_noise) (Cohen et al.,
SIGGRAPH 2026) — reuses their code from `/workspace/colorful_noise` unmodified.

## Files
- `spectral_ops.py` — colored noise (PSD ∝ f^β), generalized low-band conditioning
  (decouples **phase / magnitude / DC**), radial PSD + whiteness metric.
  `condition_latent(phase='image', mag='image', dc='image')` reproduces the paper's
  `fft_radial_frequency_swap` to float precision.
- `common.py` — SDXL loading, VAE encode (their `encode_img_sdxl`), grid plotting.
- `e0_diagnostics.py` — PSD of white noise vs VAE latents vs paper's mix across (α, γ).
- `e2_matrix.py` — 8 conditioning variants × seeds × inputs (the core experiment).
- `e1_colored.py` — generation from pure colored noise, β ∈ {−2,−1,0,+1,+2}.
- `e4_zero_snr.py` — repeats key conditions on Playground v2.5 (EDM, terminal SNR ≈ 0)
  to test whether SDXL's terminal-SNR leak explains the method.

## E2 condition names (phase, magnitude, DC of the lowest-α frequency band)
| name | phase | mag | DC | meaning |
|---|---|---|---|---|
| white | noise | noise | noise | control |
| paper | image | image | image | Cohen et al. defaults |
| paper_nodc | image | image | noise | paper minus channel-mean shift |
| phase_only | image | noise | noise | **whitened PSD, image layout** |
| phase_dc | image | noise | image | whitened + channel means |
| mag_only | noise | image | noise | spectral shape, no layout |
| mag_dc | noise | image | image | |
| dc_only | noise | noise | image | channel means alone |

## Findings (2026-06-07, qualitative pass; all grids under results/)

**E0** — At the paper's defaults (α=0.015, γ=0.05) the mixed latent has a
**low-frequency power notch** (~40× below white), not a natural-image peak — γ scales
amplitudes, so power × γ². The injected signal is image *phase* + small per-channel
*DC offsets* (e.g. cat_orange ch0: latent mean −2.55 → mix mean −0.127). A real peak
only appears for γ ≳ 0.35.

**E2** — The mechanism decomposes cleanly on SDXL:
**phase → layout, DC → palette, low-band amplitude → conditioning strength.**
`phase_only` (perfectly white PSD) conditions layout *more* strongly than the paper's
method, but at full white amplitude it over-conditions: outputs flip from photoreal to
flat-illustration style. `mag_only` is spatially blind (global mood only).

**E5** — Flat low-band magnitude `s·|noise|` + image phase + image DC, sweeping s:
photoreal up to s=0.5, sharp transition to flat-graphic at s=1.0. **s=0.5 beats the
paper's defaults on palette fidelity (green grass / blue sky preserved) at equal
photorealism**, with a simpler spectral footprint (uniform 1.5%-of-bins attenuation).
The natural-image magnitude profile is unnecessary.

**E1** — Full-spectrum colored noise breaks SDXL in both directions:
red/pink → prompt-ignoring color blobs (low-freq excess *rendered as signal*, nearly
identical output across different prompts at the same seed); blue/violet → flat gray
with surviving high-frequency texture. Output PSDs stay distorted — the model does not
restore a natural spectrum. SDXL only tolerates *small-budget low-band* deviations.

**E4** — On Playground v2.5 (EDM, terminal SNR ≈ 0, same architecture):
the paper's method **fails catastrophically — it renders the conditioning image
verbatim** (prompt ignored); `dc_only` → solid black; `phase_only` → no effect at all;
violet noise → perfect generations (unlike SDXL). A zero-SNR model acts as an honest
posterior: per-bin energy ≫ noise-typical is copied out literally, noise-typical
structure (coherent phase at white magnitude) is correctly destroyed.
**Colorful-Noise's photoreal-yet-conditioned regime is an exploit of SDXL's non-zero
terminal SNR** (cf. Lin et al., "Common Diffusion Noise Schedules and Sample Steps Are
Flawed") — its semantic, style-flexible reading of the low band does not transfer to
corrected schedules.
