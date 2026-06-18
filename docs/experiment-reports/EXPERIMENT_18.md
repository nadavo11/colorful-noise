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

*The HTML report (`results/e18/index.html`, built by `e18_site.py`) carries the same
glossary inline and leads each result with its figure. Defining every term here keeps
this writeup self-contained.*

### TL;DR
Take the 2-D Fourier transform of a diffusion **latent** and split it: the **phase**
(esp. low/coarse bands) carries the image's **content/layout**, and the
per-(channel, radial-band) **magnitude/power** carries its **"style"** (radial
texture-energy envelope + palette/contrast). Re-leveling per-band power is **AdaIN on
the radial power spectrum** (`spectral_ops.psd_match`). E18 tests this **offline, no
diffusion**: VAE-encode pairs of real images (A = content, B = style), recombine their
spectra, VAE-decode, score. **It holds — but the win is VAE-dependent and isotropic:**
restyling A toward B keeps A's layout (CLIP→A ≈ 0.90–0.97) while shifting palette/tone
toward B; the effect is nearly inert in the **Flux** latent yet **real in the SD3.5**
latent (roughly **halving** the photo→painting spectral distance), and radial bands
move **tone/palette, not oriented strokes** (Gram matrices would). The honest win is
**spectral tone transfer** — content-safe — giving the generation-time methods
(E19–E22) solid ground.

### Background (plain language)
- **Latent FFT: phase vs per-band magnitude/power.** Encode an image to a `(C,H,W)`
  latent, FFT per channel. Each frequency has a **magnitude** (energy/texture) and a
  **phase** (where ripples line up). Oppenheim–Lim: **phase carries the recognizable
  structure**.
- **Content = low-band phase; style = radial power envelope.** (From E7–E14.) Low-band
  **phase** fixes **content/layout/identity**; per-(channel, **radial band**) **power**
  is the **"style"** — texture-energy slope + palette/contrast (DC + low bands). A
  **radial band** groups Fourier coefficients into concentric rings by distance from DC
  (band 0 = coarse, high bands = fine detail).
- **psd_match / AdaIN-in-Fourier.** `psd_match` re-scales a latent so its per-band power
  hits a target envelope **leaving phase untouched** — AdaIN on the radial power
  spectrum. `strength` interpolates the target in log space between A's own power
  (0 = no change) and B's (1 = full style envelope).
- **Isotropy caveat (the stated ceiling).** Every operator works on **radial** bands
  (no orientation), so it transfers **texture-energy + palette/tone**, **not** oriented
  brush-strokes (those need anisotropic / Gram statistics). E18 measures where that bites.
- **The variants** (one decoded latent per (A, B) pair, `e18_spectral_recombine.py`):
  - `baseA` / `baseB` — the two originals (content / style).
  - `styleA_s{p}` (= `restyle_latent`) — A's phase + A's within-band texture, per-band
    power → B at strength `p`; the **isotropic pure-style** op (AdaIN-in-Fourier; sweep
    `--strengths`, default 0.5, 1.0). *Ours.*
  - `phaseA_magB` (= `band_phase_swap`, `c=1`, `mag_from=B`) — A's phase + B's **full**
    magnitude; stronger style but drags B's structure in through the magnitude.
  - `hybrid_c{c}` (= `band_spectrum_split`) — full complex spectrum (phase **and**
    magnitude) from A inside the lowest-`c` radial fraction, from B outside it; a latent
    **hybrid image** (Oliva 2006: coarse-A + fine-B; sweep `--cuts`, default 0.1/0.25/0.5).
  - `phaseonlyA` / `magonlyA` — Oppenheim–Lim controls (phase-only = recognizable-but-flat;
    mag-only = textured palette swatch, no layout).
- **The metrics.** **CLIP→A** (0–1, **↑** = content kept): CLIP-image cosine of the result
  to A. **CLIP→B**: cosine to B (style pull; content-dominated, so it moves little).
  **PSD→A / PSD→B** (**↓** = closer): luminance log-radial-PSD distance to A / B (style /
  texture-energy distance). **colorful / satur.**: colorfulness + saturation (palette readout).

### Method
- **`preflight`** (no model) — numeric asserts on the recombination algebra:
  `restyle(strength=1)` re-levels A's bands to B's power *without* moving phase
  (band-power rel-err **2e-6**, phase drift **2e-4**); hybrid endpoints exact
  (`c=1 → A`, `c=0 → B`); every recombination stays real (imag residue ~1e-6).
  *Is the operator algebra correct?*
- **`analyze`** — VAE-encode the bank, build all variants per (A, B) pair, VAE-decode,
  score (CLIP→A/→B, PSD→A/→B, colorful/satur.), save the decoded grid +
  `report_<vae>.json`. `--vae flux` = cached Flux VAE (instant smoke); `--vae sd35` =
  the E19+ generation model; `--styles` swaps a painting set in as B for the
  cross-domain photo→painting test. *Does the premise hold on real images, and how much
  does it depend on the VAE?*

### Results

> **Regeneration status: BLOCKED.** There is no `results/e18/` on the cluster or
> locally, so the figures below cannot currently be re-rendered. `e18_site.py` /
> `--part site` are wired to the report/grid filenames and will build the page once a
> run exists; numbers here are carried over from the prior E18 run.

**Part `analyze` (within-photo smoke, Flux VAE) — figure first.**
Decoded grid: `results/e18/grids/recombine_flux.png` (rows = (A, B) pairs; columns =
the variants, left→right). *What to look for:* across a row, A's scene should persist
under **restyle** while palette/tone bends toward B; **mag-only** = textured palette
swatch (no layout); **phase-only** = recognizable but flat.

Per-variant means (6 real photos, 3 pairs; A↔B baseline CLIP ≈ 0.73):

| variant | CLIP→A (content↑) | CLIP→B | colorful |
|---|---|---|---|
| baseA / baseB | 1.000 / 0.731 | 0.731 / 1.000 | 0.098 / 0.147 |
| **styleA_s0.5** *(ours)* | **0.973** | 0.710 | 0.101 |
| **styleA_s1** *(ours)* | **0.926** | 0.676 | 0.114 |
| phaseA_magB | 0.775 | 0.579 | 0.123 |
| hybrid_c0.1 → c0.5 | 0.676 → 0.835 | ~0.64 | ~0.11 |
| magonlyA | 0.516 | 0.503 | 0.161 |

*Reading:* **restyle preserves content** (CLIP→A 0.93–0.97) while colorfulness shifts
A→B with strength — palette lives in DC/low bands, as predicted. **Trade-off
confirmed:** wholesale `phaseA_magB` transfers more style but degrades content
(CLIP→A 0.775). **Hybrid works:** CLIP→A rises monotonically with the cutoff `c`
(0.68→0.84) — more low-band-from-A = more A identity. Controls behave (mag-only =
textured swatch, phase-only = recognizable-but-flat): Oppenheim–Lim holds in this
latent.

**Part `analyze` (cross-domain photo → painting) — the conclusion is VAE-dependent.**
`fetch_styles.py` pulls public-domain paintings (Van Gogh / Hokusai / Monet / Vermeer /
Bruegel); rerun pairs photos (A) × paintings (B), A↔B baseline CLIP ≈ **0.60** (real
domain gap). Decoded grid: `results/e18/grids/recombine_sd35.png`. *What to look for:*
under restyle, does the photo's palette/tone bend toward the painting while the subject
stays photographic? Means over the photo→painting pairs:

| variant | CLIP→A | CLIP→B | PSD→B (↓ closer) | colorful (paint=0.17) |
|---|---|---|---|---|
| baseA (photo) | 1.000 | 0.604 | 1.92 | 0.098 |
| **styleA_s1 (Flux VAE)** | 0.920 | 0.619 | 1.80 | 0.104 |
| **styleA_s1 (SD3.5 VAE)** | 0.900 | 0.609 | **1.03** | **0.122** |
| phaseA_magB (SD3.5) | 0.835 | 0.611 | 0.57 | 0.137 |
| hybrid_c0.1 (SD3.5) | 0.677 | **0.695** | — | 0.110 |

*Reading:* **It's VAE-dependent — SD3.5 is the favourable one.** In **Flux** latent
space isotropic band-power is nearly **inert** across domains (PSD→B 1.92→1.80,
colorful +0.006) — the earlier "weak style" read was partly a Flux-VAE artifact. In
**SD3.5** latent space (the actual E19 model) it's **real**: restyling a photo toward a
painting roughly **halves** the spectral distance (PSD→B 1.92→1.03) and moves
colorfulness ~30% toward the painting, keeping content (CLIP→A 0.90). **But still
palette/tone, not strokes:** CLIP→B barely moves (0.604→0.609) — the layout/subject
stays photographic; what transfers is global palette/tone/spectral-energy, not
painterly brushwork. The isotropy ceiling holds; the win is "spectral tone transfer."
Wholesale-magnitude and hybrid remain the heavier (content-costlier / structural)
levers (phaseA_magB colorful 0.137, PSD→B 0.57; hybrid CLIP→B up to 0.695).

### Reading of the result
The phase=content / power=style decomposition genuinely **recombines two real images in
latent space**, and the headline op (isotropic restyle = AdaIN-in-Fourier) is
**content-safe spectral tone transfer**, **real on SD3.5** (the E19 model) and nearly
inert on Flux. The stronger levers (`phaseA_magB`, hybrid) buy more style at a content
cost. **Net for E19:** the AdaIN-in-Fourier clamp on SD3.5 is worth running — framed
honestly as spectral tone/palette transfer — with the hybrid split as the stronger
structural blend.

### Caveats & next
1. **Data.** The within-photo smoke run is all natural scenes (A↔B CLIP ≈0.73), so
   style contrast is mild; the painting/photo set (`--styles`, CLIP ≈0.60) shows it
   dramatically.
2. **Isotropy ceiling.** Radial bands transfer texture-energy + palette, *not* oriented
   strokes (Gram matrices would) — the stated scope limit, probed in E19.
3. **Metric.** Luminance-PSD doesn't cleanly capture a *latent* power-matching op;
   CLIP-I + colorfulness + the visual grid are the reliable readouts; E19 measures style
   match in latent band-power space.
4. **Next:** E19 moves this into generation (content prompt + style-image envelope).

### Reproduce
```bash
cd experiments
python e18_spectral_recombine.py --part preflight                 # math asserts, no model
python e18_spectral_recombine.py --part analyze --vae flux --n 6  # cached Flux VAE smoke
python fetch_styles.py                                            # public-domain paintings
python e18_spectral_recombine.py --part analyze --vae sd35 \
       --styles results/e18/styles --n 3 --n_styles 3            # cross-domain (SD3.5 VAE)

# rebuild the self-contained HTML explainer offline (no model) from report.json + grids:
python e18_spectral_recombine.py --part site
```

> Cluster note: `/storage` is not git — ship code with `kubectl cp`. To rebuild the
> page locally, `kubectl cp` the `report_<vae>.json` + `grids/recombine_<vae>.png` into
> `experiments/results/e18`, then `python e18_spectral_recombine.py --part site`.

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
