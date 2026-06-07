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
- `e6_phase.py` — FFT-**phase** probes with kept Rayleigh magnitudes (power spectrum
  stays exactly white): phase-rerandomization control, image-phase transplant sweep
  p→1.0, phase quantization to k levels, quantize-and-omit-a-level (zero vs renorm).
- `e7_flux_phase.py` — FLUX.1-dev (guidance-distilled): phase/spectrum statistics of
  the **output** latents (16ch) at cfg=1 vs 3.5 across seeds, and band-split phase
  interpolation between two generated latents decoded through the Flux VAE.
  Needs HF auth (gated repo) + bitsandbytes (NF4 transformer fits the 24GB A5000).
- `e8_psd_clamp.py` — causal test of E7: generate at cfg=3.5 while clamping the
  latent's PSD to the cfg=1.0 per-step reference at every denoising step
  (per-band magnitude matching vs a global total-power scalar).
- `bandnorm.py` — the E8 band-normalization packaged as a reusable method
  (`record_reference()` + `generate_bandnorm()`).
- `e9_bandnorm_classes.py` — band-norm as a generation technique across 6 prompt
  classes (paired vs plain cfg=3.5 + cat-reference transfer); image-detail
  metrics + grids.

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

## Findings

See [EXPERIMENTS.md](EXPERIMENTS.md) for the per-experiment record
(motivation / setup / key results / artifacts, E0–E8), updated after every run.
