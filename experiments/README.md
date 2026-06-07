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

**E6** — Phase surgery on the white-noise FFT, magnitudes kept (PSD perfectly
in-distribution at every step). **SDXL reads almost nothing from fine phase structure;
it reads power.**
- *P0*: rerandomizing **all** phases (Hermitian uniform) is indistinguishable from
  fresh noise — confirms the exact Rayleigh × uniform factorization and calibrates the
  harness.
- *P2*: image-phase transplant at white amplitude transfers layout at every p, but
  always in the flat-illustration register — already at the paper's p=0.015 (E2's
  over-conditioning) and unchanged up to p=1.0, where the output is a near
  seed-invariant flat poster of the image layout (seeds differ only via magnitudes).
- *P4*: phase quantization is almost free: k=2 (binary phase ⇒ real, even-symmetric
  spectrum) yields mirror/film-strip tiling artifacts, but **k=4 is already photoreal
  and k≥8 is visually indistinguishable from continuous phase** — 2–3 bits of phase
  suffice for the seed→image map.
- *P4b*: zeroing one of 8 phase-level pairs (~25% of bins, broadband) gives dark foggy
  silhouettes, but renormalizing the kept power restores full photorealism — the
  **phase hole costs nothing; the power loss does all the damage** (power again,
  cf. E0/E1). Self-conjugate singleton levels (0, π → 12.5% loss) survive even
  un-renormalized.

**E7** — FLUX.1-dev *output* latents (16ch, 1024px → 128×128), "A photo of cat in the
park", 10 seeds × guidance ∈ {1.0, 3.5} (distilled guidance, no true CFG).
- *cfg=1 vs 3.5*: cfg=1 outputs are washed-out/low-contrast, and in the latents this is
  a **power story, not a slope-steepening**: latent std 0.83 vs 1.17, low-frequency
  power ~3× lower, radial slope −1.5 vs −2.0 (both converge to the same flat
  high-frequency floor ≈0.1). **Distilled guidance pumps low-frequency amplitude**
  (contrast / saturation / composition energy); per-bin phase marginals stay uniform
  (flatness ≈0.003) and phase⊥magnitude (corr ≈0.005) in both groups.
- *Cross-seed phase coherence* sits at the N=10 null (0.28) for all but the lowest
  radial bin, where R = 0.45 (cfg 3.5) vs 0.35 (cfg 1.0): seeds share only gross
  composition (centered cat, grass below, bokeh above), guidance amplifies that shared
  layout, and everything above f≈0.05 is seed-independent. Small uptick at the highest
  bin in both groups (fixed Nyquist/grid structure, likely VAE/packing related).
- *Band-split phase interpolation* (phase from A in the lowest-c band, from B outside,
  magnitudes fixed, VAE-decode only): the classic pixel-domain phase-dominance result
  holds **in Flux latent space** — c=0 (all-B phase, A magnitudes) reads as cat B with
  only a texture flavor from A's magnitudes; by **c ≈ 0.1–0.2 identity flips to A**,
  with the partner's high-band phase surviving only as ghost contours; the magnitude
  source mostly sets palette/contrast. Latent identity lives in low-band phase —
  the complement of E6's input-side result (input phase ignored, output phase is
  where the image lives).
