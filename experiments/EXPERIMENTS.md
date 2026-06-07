# Experiments log

One section per experiment: **Motivation / Setup / Key results / Artifacts**.
`README.md` holds the file index; this file is the running record and is
updated after **every** experiment run. (E3 was never run; the gap is
intentional.)

---

## E0 — PSD diagnostics of the Colorful-Noise mix

**Motivation.** What does the paper's `fft_radial_frequency_swap` actually do
to the latent's power spectrum? The paper frames it as injecting a
natural-image-like low-frequency peak.

**Setup.** SDXL VAE latents (1,4,128,128). Radial PSD + per-channel DC of
white noise vs VAE latents vs the paper's mix across an (α, γ) sweep around
the defaults (α=0.015, γ=0.05). No generation; latent-level only.

**Key results.**
- At the paper's defaults the mixed latent has a **low-frequency power notch**
  (~40× below white), not a peak — γ scales amplitudes, so power ∝ γ².
- The injected signal is image *phase* + small per-channel *DC offsets*
  (e.g. cat_orange ch0: latent mean −2.55 → mix mean −0.127).
- A real spectral peak only appears for γ ≳ 0.35.

**Artifacts.** `results/e0/` (PSD plots, `summary.json`).

---

## E1 — Generation from full-spectrum colored noise

**Motivation.** If a tiny low-band tweak conditions SDXL, what does a
*global* PSD change (noise PSD ∝ f^β) do?

**Setup.** SDXL base, β ∈ {−2,−1,0,+1,+2}, several prompts × seeds,
globally normalized latents, 50 steps.

**Key results.**
- Colored noise breaks SDXL in both directions: red/pink → prompt-ignoring
  color blobs (low-freq excess *rendered as signal*; nearly identical output
  across prompts at the same seed); blue/violet → flat gray with surviving
  high-frequency texture.
- Output PSDs stay distorted — the model does not restore a natural spectrum.
  SDXL only tolerates *small-budget low-band* deviations.

**Artifacts.** `results/e1/` (grids per β, output-PSD report).

---

## E2 — 8-way phase/magnitude/DC conditioning matrix

**Motivation.** The paper's low-band swap bundles three ingredients — image
phase, image magnitude, image DC. Which one carries the conditioning?

**Setup.** SDXL base, `condition_latent()` decoupling (phase, mag, dc) each ∈
{image, noise} inside the lowest-α band (α=0.015, γ=0.05); 8 named conditions
(see README table) × 3 seeds × 2 input images, 50 steps.

**Key results.**
- The mechanism decomposes cleanly: **phase → layout, DC → palette, low-band
  amplitude → conditioning strength.**
- `phase_only` (perfectly white PSD) conditions layout *more* strongly than
  the paper's method, but at full white amplitude it over-conditions: outputs
  flip from photoreal to flat-illustration style.
- `mag_only` is spatially blind (global mood only).

**Artifacts.** `results/e2/` (`grid_*.png`, `latent_report.json`).

---

## E4 — Zero terminal SNR control (Playground v2.5)

**Motivation.** Is the paper's photoreal-yet-conditioned regime a property of
diffusion in general, or an artifact of SDXL's non-zero terminal SNR leak?

**Setup.** Playground v2.5 (EDM schedule, terminal SNR ≈ 0, same
architecture); replicates the paper's condition, `dc_only`, `phase_only`,
and violet noise.

**Key results.**
- The paper's method **fails catastrophically — it renders the conditioning
  image verbatim** (prompt ignored); `dc_only` → solid black; `phase_only` →
  no effect; violet noise → perfect generations (unlike SDXL).
- A zero-SNR model acts as an honest posterior: per-bin energy ≫ noise-typical
  is copied out literally; noise-typical structure is correctly destroyed.
- **Colorful-Noise's regime is an exploit of SDXL's non-zero terminal SNR**
  (cf. Lin et al., "Common Diffusion Noise Schedules and Sample Steps Are
  Flawed"); it does not transfer to corrected schedules.

**Artifacts.** `results/e4/`.

---

## E5 — Conditioning-strength sweep with flat low-band magnitude

**Motivation.** E2 showed amplitude = strength. Is the natural-image
magnitude *profile* needed at all, or just its scale?

**Setup.** SDXL base; latent = flat low-band magnitude `s·|noise|` + image
phase + image DC, sweeping s (via `mag_scale`), the rest as the paper.

**Key results.**
- Photoreal up to s=0.5, sharp transition to flat-graphic at s=1.0.
- **s=0.5 beats the paper's defaults on palette fidelity (green grass / blue
  sky preserved) at equal photorealism**, with a simpler spectral footprint
  (uniform attenuation of 1.5% of bins). The natural-image magnitude profile
  is unnecessary.

**Artifacts.** `results/e5/`.

---

## E6 — FFT-phase surgery on the input noise (SDXL)

**Motivation.** White Gaussian noise factorizes exactly into independent
Rayleigh magnitudes × uniform phases. Phase surgery with kept magnitudes is
the minimal off-manifold probe (PSD stays perfectly in-distribution): what
does SDXL's seed→image map read from phase?

**Setup.** SDXL base, 50 steps. Parts: P0 phase-rerandomization control +
uniformity/independence numerics; P2 image-phase transplant
(`phase='image', mag='noise', dc='noise'`) swept p ∈ {0.015, 0.1, 0.3, 1.0};
P4 phase quantization to k ∈ {2,4,8,16,32} levels; P4b quantize to k=8 and
omit one ± level pair, raw zeroing vs power-renormalized.

**Key results.** **SDXL reads almost nothing from fine phase structure; it
reads power.**
- *P0*: rerandomizing **all** phases is indistinguishable from fresh noise —
  confirms the Rayleigh × uniform factorization and calibrates the harness.
- *P2*: image-phase transplant at white amplitude transfers layout at every
  p, always in the flat-illustration register; at p=1.0 the output is a near
  seed-invariant flat poster of the layout (seeds differ only via magnitudes).
- *P4*: phase quantization is almost free — k=2 gives mirror/film-strip
  tiling (real, even-symmetric spectrum), **k=4 is already photoreal, k≥8 is
  indistinguishable** from continuous phase: 2–3 bits of phase suffice.
- *P4b*: zeroing one of 8 level pairs (~25% of bins) → dark foggy
  silhouettes; renormalizing the kept power → fully photoreal. **The phase
  hole costs nothing; the power loss does all the damage.** Self-conjugate
  singleton levels (0, π → 12.5% loss) survive even un-renormalized.

**Artifacts.** `results/e6/` (`grid_p0/p2_*/p4/p4b.png`, `report.json`).

---

## E7 — FLUX.1-dev output-latent phase & spectrum (cfg 1.0 vs 3.5)

**Motivation.** E0–E6 probed the *input* noise of SDXL. Flip direction and
model: what do the FFT statistics of Flux's *output* latents look like, and
what does distilled guidance change?

**Setup.** FLUX.1-dev (NF4 transformer + cpu offload), "A photo of cat in
the park", 10 seeds × guidance ∈ {1.0, 3.5} (guidance-distilled — an
embedding, not true CFG), 28 steps, final latents (1,16,128,128) captured
via step-end callback. Part B: band-split phase interpolation between two
generated latents, VAE-decode only.

**Key results.**
- cfg=1 outputs are washed-out, and in the latents this is a **power story,
  not a slope-steepening**: latent std 0.83 vs 1.17, low-frequency power ~3×
  lower, radial slope −1.5 vs −2.0 (same flat high-frequency floor ≈ 0.1).
  **Distilled guidance pumps low-frequency amplitude**; per-bin phase
  marginals stay uniform (flatness ≈0.003) and phase⊥magnitude (corr ≈0.005)
  in both groups.
- Cross-seed phase coherence sits at the N=10 null (0.28) except the lowest
  radial bin (R=0.45 at cfg 3.5 vs 0.35 at cfg 1.0): seeds share only gross
  composition; everything above f≈0.05 is seed-independent.
- Band-split phase interpolation: the classic phase-dominance result holds
  **in Flux latent space** — identity flips to the low-band phase donor by
  c ≈ 0.1–0.2; the magnitude source mostly sets palette/contrast. **Latent
  identity lives in low-band phase** — the complement of E6's input-side
  result.

**Artifacts.** `results/e7/` (`grid_partA/B.png`, `plots/`, `report.json`,
`latents/`).

---

## E8 — Per-step PSD clamping during generation (causal test of E7)

**Motivation.** E7 is correlational: cfg=3.5 latents carry more
(low-frequency) power than cfg=1.0. E8 intervenes: generate at cfg=3.5 but
at **every denoising step** renormalize the latent's PSD to the cfg=1.0
reference at the same step. If guidance's visible effect (contrast /
saturation) is mediated by the pumped power, the clamped run should look
like cfg=1 despite cfg=3.5 conditioning. A global-norm control (one scalar
per step, total power only) dissociates *total power* from *spectral shape*.

**Setup.** FLUX.1-dev as E7 (NF4 + offload), same prompt, 4 seeds, 28 steps.
Pass a: cfg=1.0 with a recording callback → per-step per-(channel, band)
mean power reference (24 radial bins), saved to `ref_psd.pt`; images double
as the baseline row. Pass b: cfg=3.5 plain / band-norm (per channel × band
`|F| *= sqrt(ref/cur)`, phase kept) / global-norm (Parseval scalar in latent
space). Clamp applied at step end, every step incl. the last. Same seeds
across rows (identical initial latents — guidance is just an embedding).

**Key results.** **The power difference is a correlate, not the cause, of the
guidance look — PSD clamping does NOT reproduce the cfg=1 appearance.**
- The clamp wins in latent space: band-norm lands exactly on the cfg=1
  spectrum (final std 0.825 vs ref 0.822, slope −1.53 vs −1.51); global-norm
  matches total power (0.830) while keeping cfg=3.5's shape (slope −2.04 vs
  −2.03). Per-step gains stay within [0.84, 1.12] — the model barely fights
  the clamp (it never re-pumps band power by more than ~12%/step).
- Yet **both clamped rows still look like cfg=3.5**: crisp, saturated,
  well-defined cats with only a mild contrast reduction — nothing like
  cfg=1's washed-out haze. Forcing the full second-order statistics (shape
  AND scale) onto the latent leaves the guidance look intact, so the look
  is carried by *where* the latent points (phase/content), not by its
  spectrum — consistent with E7's "identity lives in low-band phase".
- band-norm ≈ global-norm visually: once total power is set, the residual
  spectral-shape difference contributes almost nothing.
- Same seeds across rows: rows cfg=3.5 / band-norm / global-norm share
  composition per seed, but cfg=1.0 composes *differently* despite identical
  initial latents — guidance changes the trajectory's content early, and a
  spectral clamp cannot undo that.
- Dynamics: cfg=3.5's std departs from the cfg=1 trajectory only after
  step ~7 and balloons in the late steps (0.75 → 1.16 over the last third) —
  **distilled guidance pumps power late**, during detail formation, not
  during early composition.

**Artifacts.** `results/e8/` (`grid_e8.png`, `plots/perstep_std.png`,
`ref_psd.pt`, `report.json`, `images/`).

---

## E9 — Band-normalized generation as a method, across prompt classes

**Motivation.** In E8 the band-normalized cat read as *more finely detailed*
than plain cfg=3.5 (the clamp keeps the guidance look but trades punchy
contrast for texture). E9 turns the E8 intervention into a reusable method
(`bandnorm.py`) and asks whether the detail effect generalizes beyond animals
— across photo and non-photo classes — and whether one *universal* reference
suffices or each prompt needs its own (the transfer condition).

**Setup.** FLUX.1-dev (NF4 + offload), 28 steps, 6 prompt classes
(animal, portrait, landscape, urban_night, abstract, watercolor). Per class:
3 cfg=1.0 reference seeds → per-step per-(channel, 24-band) power reference;
25 plain cfg=3.5 + 25 band-norm at identical seeds (paired); 2 transfer seeds
(band-norm driven by E8's *cat* reference). Paired image metrics: Laplacian
sharpness, image-FFT high-frequency fraction `hf_frac` (contrast-invariant
detail measure), RMS contrast, Hasler–Suesstrunk colorfulness, mean
saturation; plus latent std/slope. `bandnorm.py` exposes `record_reference()`
+ `generate_bandnorm()`; driver `e9_bandnorm_classes.py` (`--part gen,analyze`,
image-level caching → resumable).

**Key results.** (Δ = band-norm − plain cfg=3.5, 25 paired seeds/class; all
6 classes, tight error bars.) **The "more detail" effect is real but
content-dependent — band-norm is a contrast/saturation tamer that *also*
shifts detail toward fine texture only for the right content.**
- Δ`hf_frac` (contrast-invariant detail): animal **+0.0041**, portrait
  **+0.0042**, landscape **+0.0024**, watercolor **+0.0012** (positive) vs
  urban_night **−0.0121**, abstract **−0.0057** (*reverses*). Band-norm adds
  fine texture when detail is broadly distributed (fur, skin, foliage) and
  removes it when detail lives in concentrated high-power structure (neon on
  dark, bold painterly strokes) that the per-band clamp smooths out.
- **Universal across all 6 classes:** band-norm reduces contrast (−0.028 to
  −0.047), colorfulness (−0.009 to −0.151) and saturation (−0.04 to −0.28).
  The taming is strongest for the most saturated classes (urban_night,
  abstract, landscape). Absolute Laplacian sharpness drops for 5/6 classes
  because it scales with contrast² — only `hf_frac` (energy *ratio*) isolates
  detail from the contrast reduction, which is why the two diverge.
- **Reference is content-specific.** cfg=1.0 reference final-std splits by
  domain: photos 0.82–0.90 (portrait 0.82, animal 0.89, urban_night 0.90) vs
  illustration/abstract 0.65–0.72 (abstract 0.65, watercolor 0.67, landscape
  0.72). So a recorded reference encodes the prompt's natural power level.
- **Transfer (E8 cat reference applied to every class) works but is not
  free.** The cat clamp forces all classes to std ≈ 0.82, fine for the photo
  classes (refdev ≤ 0.09) but a 0.12–0.21 mismatch for the low-power
  illustration classes — there transfer pulls `hf_frac` *away* from the
  class's own-reference result (landscape −0.0047, watercolor −0.0062),
  i.e. over-powering a naturally-soft prompt erodes its texture. **One
  universal reference is usable for in-domain (photo) prompts; out-of-domain
  (illustration/abstract) prompts want their own 3-seed reference.**
- **Practical recipe:** band-norm is best read as "cfg=3.5 composition with
  cfg=1.0's calmer contrast/palette, plus a texture nudge on photographic
  subjects." Use a per-prompt (or per-domain) reference; reserve it for
  broadly-textured content and skip it where the look depends on bold color
  or concentrated highlights.

**Artifacts.** `results/e9/` (`grid_<class>.png`, `plots/ref_std_curves.png`,
`plots/metrics_delta.png`, `report.json`, per-class `images/`, `latents/`,
`ref_psd.pt`).
