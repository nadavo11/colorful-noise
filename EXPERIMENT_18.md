# E18+: spectral style transfer & blending from two images ("AdaIN-in-Fourier")

**The direction.** SBN clamps a generating latent's per-(channel, radial-band)
**power** to a single reference. The new idea: drive generation with the spectrum
of **two** sources. The decomposition that makes this work is already established
(E7–E14): in latent Fourier space **PHASE (esp. low-band) = content/layout** and
**per-band MAGNITUDE/POWER = "style"** (radial energy envelope = texture slope;
DC/low bands = palette/contrast). That is the Gatys/AdaIN content–style split moved
into frequency space — re-leveling per-band power is **AdaIN on the radial power
spectrum**, and `spectral_ops.psd_match` is exactly that operator.

Lineage: this rejoins where the project began — `colorful-noise` colored the
*initial* noise from a reference; here we condition the *whole denoising
trajectory* with two references — and connects to Gatys (style = feature
correlations), AdaIN/StyleGAN (style = feature moments), and **hybrid images**
(Oliva–Torralba–Schyns 2006: low-freq of A + high-freq of B).

**Isotropy caveat (stated, then probed — not hidden).** Every operator works on
*radial* bands, so it transfers texture-energy + palette, **not** oriented/stroke
style (Gram matrices would). The experiments measure where that ceiling bites; an
anisotropic-band variant is a later extension.

Shared operators live in `experiments/style_ops.py` (built on `spectral_ops`);
SD3.5 VAE encode/decode added to `experiments/e17_sd35.py`. Model: **SD3.5-medium**
(true CFG, unpacked latents — simplest `ClampPSD3` path), consistent across
E18→E22.

---

## E18 — offline two-image spectral recombination (foundation) ✅

**Question.** Before touching generation: in the SD3.5/Flux latent, does phase
carry content and per-band power carry style well enough to recombine two **real**
images? VAE-encode pairs (A = content, B = style), recombine spectra, VAE-decode —
**no diffusion**.

**Variants** (`experiments/e18_spectral_recombine.py`, per A,B pair):
`phaseA_magB` (= `band_phase_swap`, phase A + B's full magnitude — strong),
`styleA_s{p}` (= `restyle_latent`, phase A + A's within-band texture, per-band power
→ B; the *isotropic pure-style* op, strength sweep), `hybrid_c{c}`
(= `band_spectrum_split`, low bands from A + high bands from B; Oliva 2006), plus
`phaseonlyA`/`magonlyA` Oppenheim–Lim controls. Scored with CLIP-I to A and B and a
luminance log-radial-PSD distance.

**Status: runs, premise holds.** Smoke run on Flux VAE, 6 real photos, 3 pairs
(`experiments/results/e18/grids/recombine_flux.png`, `report_flux.json`).
Per-variant means:

| variant | clip→A (content↑) | clip→B | colorfulness |
|---|---|---|---|
| baseA / baseB | 1.000 / 0.731 | 0.731 / 1.000 | 0.098 / 0.147 |
| **styleA_s0.5** | **0.973** | 0.710 | 0.101 |
| **styleA_s1** | **0.926** | 0.676 | 0.114 |
| phaseA_magB | 0.775 | 0.579 | 0.123 |
| hybrid_c0.1 → c0.5 | 0.676 → 0.835 | ~0.64 | ~0.11 |
| magonlyA | 0.516 | 0.503 | 0.161 |

- **Restyle preserves content:** re-leveling A's per-band power to B keeps A's
  layout (clip→A **0.93–0.97**) while colorfulness shifts A→B with strength — the
  palette lives in the DC/low bands, exactly as predicted.
- **Trade-off confirmed:** wholesale magnitude (`phaseA_magB`) transfers more style
  (colorfulness 0.123, lower clip→B-distance) but degrades content (clip→A 0.775).
  Isotropic band-power style is gentler on content, weaker on style.
- **Hybrid works:** clip→A rises monotonically with the cutoff `c` (0.68→0.84) —
  more low-band-from-A = more A identity; the scale-split behaves.
- Controls behave (visually): `magonlyA` = textured palette swatch (no layout),
  `phaseonlyA` = recognizable-but-flat. Oppenheim–Lim holds in this latent too.

**Caveats / next.** (1) The e10 photo bank is all *natural scenes* (A↔B baseline
CLIP 0.73), so style contrast is mild — a **painting/photo** style set (`--styles`)
will show it dramatically. (2) The luminance-PSD metric doesn't cleanly capture a
*latent* power-matching op; CLIP-I + colorfulness + the visual grid are the
reliable readouts here, and E19 measures style match in latent band-power space.
(3) Numerics verified: `restyle` band-power match rel-err 2e-6, phase drift 2e-4;
hybrid/swap imag-residue ~1e-6 (stays real).

### Cross-domain rerun (photo → painting) — the conclusion is VAE-dependent

`fetch_styles.py` pulls public-domain paintings (Van Gogh / Hokusai / Monet /
Vermeer / Bruegel); rerun pairs photos (A=content) × paintings (B=style), A↔B
baseline CLIP ≈ **0.60** (real domain gap, vs 0.73 for photo↔photo). Means over the
photo→painting pairs:

| variant | clip→A | clip→B | psd→B (↓ closer) | colorful (paint=0.17) |
|---|---|---|---|---|
| baseA (photo) | 1.000 | 0.604 | 1.92 | 0.098 |
| **styleA_s1 (Flux VAE)** | 0.920 | 0.619 | 1.80 | 0.104 |
| **styleA_s1 (SD3.5 VAE)** | 0.900 | 0.609 | **1.03** | **0.122** |
| phaseA_magB (SD3.5) | 0.835 | 0.611 | 0.57 | 0.137 |
| hybrid_c0.1 (SD3.5) | 0.677 | **0.695** | — | 0.110 |

- **It's VAE-dependent — and SD3.5 is the favourable one.** In **Flux** latent space
  isotropic band-power is nearly inert across domains (psd→B 1.92→1.80, colorful
  +0.006) — the earlier "weak style" read was partly a Flux-VAE artifact. In **SD3.5**
  latent space (the actual E19 model) it's **real**: restyling a photo toward a
  painting roughly **halves the spectral distance** (psd→B 1.92→1.03) and moves
  colorfulness ~30% toward the painting, while keeping content (clip→A 0.90).
- **But still palette/tone, not strokes.** `clip→B` barely moves (0.604→0.609) under
  restyle — CLIP (content-dominated) confirms the *layout/subject* stays photographic;
  what transfers is global palette/tone/spectral-energy, not painterly brushwork. The
  isotropy ceiling holds; the win is "spectral tone transfer," and it's stronger than
  the photo-only smoke test implied.
- **Wholesale-magnitude and hybrid remain the heavier levers** (phaseA_magB: colorful
  0.137, psd→B 0.57 but content cost; hybrid: clip→B up to 0.695). Report:
  `results/e18/{report_sd35.json, grids/recombine_sd35.png, site/index_sd35.html}`.

**Net for E19:** the AdaIN-in-Fourier clamp on **SD3.5** is worth running — frame it
honestly as **spectral tone/palette transfer** (measurable, content-safe), with
hybrid (E19 split mode) as the stronger structural blend.

**Run** (from `experiments/`):
```bash
cd experiments
python e18_spectral_recombine.py --part preflight                 # math asserts, no model
python e18_spectral_recombine.py --part analyze --vae flux --n 6  # cached Flux VAE smoke
python fetch_styles.py                                            # public-domain paintings
python e18_spectral_recombine.py --part analyze --vae sd35 \
       --styles results/e18/styles --n 3 --n_styles 3            # cross-domain (SD3.5 VAE)
```

---

## E19 — generation-time spectral style transfer (headline) — ready, needs SD3.5

**Question.** Generate a *content prompt* while clamping its spectrum toward a
*style image* during denoising: content provides phase + the per-step energy
trajectory; the style image provides the radial power envelope.

`experiments/e19_spectral_style.py`: `content_ref = record_reference_sd3(content_prompt)`;
`style_band = latent_band_power(sd3_vae_encode(style_img))`;
`ref = build_style_reference(content_ref, style_band, strength, gmax)`; generate
with `ClampPSD3(ref)` at cfg≈4.5. `strength` sweep (0 = plain SBN, 1 = full style
envelope); `gmax` clamps per-band gains for stability. Conditions: `cfg_hi`, `sbn`,
`style_{sid}_s{p}`. Metrics: **content_clip** (CLIP-I to the unstyled cfg_hi image,
paired) vs **style_clip** / **style_band_dist** (latent), with aesthetic /
ImageReward / CLIP-T guards. Model-free `preflight` passes (strength=0 ≡ SBN; gmax
clamp applied). `gen/score/analyze` need the SD3.5 download — run on the cluster:
```bash
cd experiments
python e19_spectral_style.py --part preflight                                 # passes now
python e19_spectral_style.py --part gen,score,analyze --num_prompts 2 --seeds 4 \
       --styles <paintings_dir> --num_styles 2 --strengths 0.5,1.0
```

## E19 follow-on modes — operators already in `experiments/style_ops.py`

These are thin variants of the E19 driver (same `ClampPSD3` path), **not** separate
top-level experiments — the operators already exist and are unit-tested, so they fold
into E19 as modes. (This frees the **E20** number for spectral warm-start; see
`EXPERIMENT_20.md`. Note an unrelated **E23** real-PSD-clamp thread also exists.)

- **hybrid synthesis** — `build_hybrid_reference` (low-band envelope from A, high from
  B). Generation-time controls the *energy* split only (phase from the prompt); the
  strong offline hybrid is E18's `band_spectrum_split`. Readout: low-pass(out)→A and
  high-pass(out)→B?
- **spectral morph** — `build_morph_reference` (geometric interpolation between two
  style envelopes over α) → a palette/texture morph sequence.
- **two-prompt SBN** — `blend_references` (per-band geometric mean of two cfg=1
  references) → "a cat" with the spectral signature of "a Van Gogh painting"; no
  reference image, fully generation-native, cheapest.
