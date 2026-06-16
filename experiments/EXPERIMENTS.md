# Experiments log

One section per experiment: **Motivation / Setup / Key results / Artifacts**.
`README.md` holds the file index; this file is the running record and is
updated after **every** experiment run. (E3 was never run; the gap is
intentional.)

**Convention (resolution).** New experiments should generate at **512×512**
(latent 64×64), not the **1024×1024** used for the Flux work in E7–E11, to
iterate faster (~4× less compute). Scripts take resolution via the `SIZE`
constant (`height=SIZE, width=SIZE`); the radial-band machinery is
resolution-agnostic — pass the matching latent size to
`band_index_map(H, W, n_bins, ...)` (64×64 at 512px instead of 128×128).

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

## E9b — Band-norm follow-ups: CLIP-T, universal reference, frequency control, cost, explainer site

**Motivation.** Package E9's band-norm as an external method ("Spectral Band
Normalization", SBN) with an interactive explainer, and answer four open
questions: does band-norm stay on-prompt; can one general reference replace the
per-prompt pass; can we steer high/low frequencies independently; is it more
expensive than ordinary generation.

**Setup.** Backfilled cfg=1.0 to 25 seeds/class (batched gen, list-of-generators
reproduces single-image renders, corr 0.9999 — `e9_extra_cfg1.py --batch 4`).
New scripts: `e9_clipt.py` (CLIP ViT-L/14 text–image cosine), `e9_universal_ref.py`
(mean of the 6 class refs → `universal_ref.pt`; deviation stats; 5 seeds/class
`uninorm` demo), `e9_freqctrl.py` (modulate a band range of the reference by g²,
then clamp — `bandnorm.modulate_reference`; portrait+landscape × 2 seeds × {low,high}
× g∈{0.7,0.85,1,1.15,1.3}), `e9_cost.py` (per-step wall-time, plain vs clamp).
Site generator `make_e9_site.py` → `results/e9/site/index.html` (self-contained,
no internal naming).

**Key results.**
- **CLIP-T (prompt fidelity):** full guidance highest; SBN slightly below; biggest
  cost on **urban_night (−0.029)** — the same class whose texture effect reverses,
  so an independent semantic metric flags the same worst-fit case. portrait/watercolor
  Δ positive.
- **Universal reference works.** Averaging the 6 per-class references → std **0.774**;
  each prompt's own reference sits within **3.7% mean** band-power deviation (abstract
  the outlier at 5.1%). `uninorm` CLIP-T ≈ per-prompt band-norm (within ~0.01) → the
  per-prompt reference pass is largely avoidable.
- **Frequency control is asymmetric.** Low-band gain is a **clean monotonic structure
  /contrast knob** (lowband_power rises smoothly, image stays coherent in the mild
  range). High-band gain is **inverse/destructive**: amplifying high-frequency *latent*
  power does NOT add image detail — hf_frac falls monotonically (landscape
  0.023→0.002 over g 0.7→1.3) and the VAE renders the off-distribution energy as a
  granular stipple artifact that corrupts the subject. Confirms image detail is not a
  high-frequency latent dial (consistent with structure living in low bands/phase, E7/E8).
- **Cost is negligible.** Per-step overhead of the FFT clamp = **+0.77%** (1265→1275 ms
  on FLUX NF4). The only real extra cost is the reference pass, which a universal
  reference amortizes to ~0.

**Artifacts.** `results/e9/`: `clip_t.json`, `universal_ref.pt`, `universal.json`,
`freqctrl.json` + `freqctrl/<prompt>/images/`, `cost.json`, per-class `uninorm_s*`;
external explainer at `results/e9/site/index.html`.

---

## E10 — CFG inflates spectral power; where real images sit (the SBN motivation)

**Motivation.** Motivate the whole spectral-power line of work: show that the latent's
spectral power / PSD / spectral norm rises with classifier-free guidance, and locate
real-image statistics on that axis. Flux trains a velocity field by flow-matching with
**no CFG term** in the loss (`L = E‖v_θ(x_t,t,c) − (x_1−x_0)‖²`); CFG is an inference-time
extrapolation `ṽ = v_u + w(v_c − v_u)`. w=1 integrates the trained field; w>1 extrapolates.

**Setup.** `e10_cfg_spectral.py`. **True CFG** on Flux via diffusers `true_cfg_scale` +
empty `negative_prompt` (genuine two-pass `v_u + w(v_c−v_u)`), distilled `guidance_scale`
held neutral at 1.0 — isolates the cfg-equation effect, not Flux's distilled guidance.
Sweep w∈{1,1.5,2,3,4,5} × the 6 E9 prompt classes × 3 seeds = 108 final latents
(16×128×128, 28 steps). Per latent: Fourier power `mean|X|²`, latent std, radial PSD
(`radial_psd`), low-band power (centers<0.25), and the literal **spectral norm**
(mean σ_max per channel). Real-image anchor: 20 natural photos (seeded `picsum.photos`),
VAE-encoded into the generation latent space (`lat=(z−shift)·scale`). Memory: pre-encode
all prompts once then **drop the T5/CLIP text encoders** (process RSS ~1.3 GB vs ~10 GB),
so the whole sweep runs in one process on a RAM-contended shared box without OOM.

**Key results.**
- **Spectral power rises monotonically with CFG** — Fourier power 0.636 → 1.928 (w=1→5,
  **≈3.0×**); latent std 0.775→1.318; spectral norm 69.0→126.7; low-band power 21.4→77.7.
  The whole radial PSD lifts, not just the slope. Unambiguous on every measure.
- **Real photos sit at w ≈ 3, NOT at w = 1.** 20-photo anchor: power 1.229 (≈ cfg3's 1.224),
  std 1.074 (≈ cfg3 1.069), spectral norm 99.6 (≈ cfg3 100.6). So the **unguided trained
  field (w=1) is spectrally *weaker* than real data**; standard guidance (~3) is where the
  latent's power matches real-image statistics; higher w overshoots. (Refines the naive
  "cfg=1 = real" guess from a noisy 4-photo pilot — n=20 lands at cfg≈3.)
- **Takeaway / nuance.** The elevated spectral scale CFG produces is roughly where natural
  images live, so a generation *should* carry it. Band-norm operates on this same axis
  (set per-band power to a reference level): SBN's clamp toward the calm cfg=1 look is a
  deliberate stylistic choice *below* the real-image scale, not a realism restoration.

**Artifacts.** `results/e10/cfg_spectral.json` (per-cfg + real aggregates, PSD curves),
`results/e10/{class}/{images,latents}/tcfg{w}_s*`, `real_photos/`, `real_latents.pt`;
plots `results/e9/plots/cfg_power.png`, `cfg_psd.png`; surfaced as the **"Why normalize?"**
tab in the SBN explainer (`results/e9/site/index.html`).

---

## E11 — cheap color/contrast correction of band-norm outputs

**Motivation.** E9 showed band-norm (SBN) clamps cfg=3.5 latent power back to the
cfg=1.0 reference; a measured side effect is washed-out output (vs cfg=3.5: ~−23%
RMS contrast, slightly lower colorfulness) — it "lacks color." Can quick
*image-level* post-processing restore color/contrast toward the cfg=3.5 look
**without** discarding the SBN detail gain? The detail-preservation test is the
contrast-invariant high-frequency fraction `hf_frac`: a good correction lifts
colorfulness/contrast while leaving `hf_frac` ≈ SBN (i.e. doesn't re-bake).

**Setup.** Post-processing only — no regeneration, no GPU. Reprocessed the existing
SBN PNGs (6 classes × 25 seeds) under `results/e9/`. Four method families, PIL +
numpy only: `sat` (ImageEnhance.Color, **f∈{1.4,1.8}**), `contrast` (ImageEnhance.Contrast
f=1.2 + autocontrast), `lum_eq` (equalize Y channel of YCbCr → no hue shift),
`hist_match` (per-channel CDF match to the paired cfg=3.5 image — reference upper
bound). Scored every variant with E9's `image_metrics()`; deltas reported vs SBN and
vs cfg=3.5. Code: `e11_color_correct.py --sat-factors 1.4 1.8`.

**Key results** (means over 25 seeds; representative across all 6 classes).
- **Saturation boost is the practical fix for "lack of color."** `sat` raises
  colorfulness with ~zero change to contrast and **hf_frac unchanged**. **f=1.4 is
  the chosen factor**: its colorfulness lands closest to cfg=3.5 (mean |gap| 0.030 vs
  0.037 for f=1.8) and it overshoots on only 2/6 classes, whereas **f=1.8 overshoots
  on 5/6** (over-saturated/garish). Cheap, single-pass, reference-free, detail-safe.
- **`hist_match` reaches the palette target most precisely but is EXPENSIVE.** It
  matches each SBN frame to its **paired cfg=3.5 image**, so it requires a *second,
  full-guidance generation pass per image* — roughly doubling generation cost and
  needing the very cfg=3.5 output SBN is meant to avoid. It is an **oracle / upper
  bound on how close correction can get**, not a deployable correction.
- **`contrast`=1.2 lifts both** colorfulness (+0.018…+0.054) **and** RMS contrast
  (+0.03…+0.04), hf_frac preserved — a no-reference way to also recover contrast.
- **`autocontrast` is a near no-op** — SBN images already span the full 0–255 range
  (all Δ ≈ 0). **`lum_eq` overshoots contrast** (+0.08…+0.14, often past cfg=3.5),
  *lowers* colorfulness, and is the only method that perturbs `hf_frac` — avoid.
- **Takeaway / standing recommendation.** Ship a **saturation ×1.4** post-process on
  SBN outputs — it puts the color back for free. `hist_match` is only an oracle (needs
  the extra cfg=3.5 pass); reserve it for measuring the ceiling, not for production.

**Artifacts.** `results/e11/report.json` (per-class/per-method mean metrics + deltas
vs SBN and vs cfg=3.5); `results/e11/<class>/<method>/` corrected PNGs;
`results/e11/<class>/grids/<method>.png` (SBN | corrected | cfg3.5 contact sheets).
Presented in the explainer site Color-correction tab (`results/site/index.html`;
`make_e9_site.py --standalone` also writes a self-contained `index_standalone.html`
with images inlined).

---

## E12 — Latent FFT phase distributions across image classes (`e12_phase_dist.py`)

**Motivation.** The baseline that motivates the E13–E15 phase line: is the latent
FFT-phase *marginal* ever non-uniform (per band / channel / class), or is the white-noise
null the whole story — so that any phase signal must live in cross-frequency *structure*
(Oppenheim–Lim), not the marginal? (A separate SD3.5 control once occupied this slot but
was dropped; this is the renumbered former E13.)

**Setup.** FLUX.1-dev, the 6 prompt classes × seeds; per (channel, radial band) phase
histogram + circular stats (flatness, mean resultant length) and cross-seed phase
coherence, vs the `random_hermitian_phase` null. Analysis-only (no intervention).

**Key results.** (3 seeds/class, 24 radial bins.) **The phase marginal is the white-noise
null — uniform everywhere except a faint low-band elevation; the only structured signal is
joint (cross-seed coherence), concentrated in the low band.** This is the baseline that
motivates E13–E15.
- Global phase-marginal flatness ≈ **0.006** (≈ uniform); per-band resultant length R is ≈ 0
  at mid/high frequency (R_mid/high ≈ **0.029**) and only faintly elevated in the lowest band
  (R_low ≈ 0.08; watercolor the outlier at 0.17). A flat phase histogram is therefore
  *expected* and is **not** evidence that phase is uninformative.
- **Cross-seed phase coherence rises above the N=3 null (0.512) only in the lowest band**
  (coh_low ≈ **0.676** mean), ~identical across classes — seeds share gross low-frequency
  composition and nothing above it. So the phase signal lives in cross-frequency *structure*
  (Oppenheim–Lim), localized to the low band — exactly what E13 (full-spectrum swap) and E14
  (band-localized phase-noise erosion) then exploit.

**Artifacts.** `results/e12/` (`plots/band_hist_<class>.png`, `global_hist.png`,
`flatness_band.png`, `coherence_radial.png`, `grid_classes.png`, `report.json`, `latents/`).
Surfaced as section 4 of the "Phase & identity" tab in the explainer site.

---

## E13 — Full-spectrum phase ↔ magnitude swap (Oppenheim–Lim in Flux latent space)

**Motivation.** E7 found latent identity follows the *low-band* phase donor in a
band-split interpolation. E13 asks the global Oppenheim–Lim (1981) question across the
*whole* spectrum: does perceived identity track FFT phase or magnitude, and what do
phase-only / magnitude-only latents decode to?

**Setup.** FLUX.1-dev, the 6 E9 prompt classes × 4 seeds at cfg=3.5 → a base latent
bank (1,16,128,128). Decode-only (no re-diffusion) through the Flux VAE. Per class,
seeds paired (A,B); six decoded variants — baseA, baseB, A-phase+B-mag, B-phase+A-mag,
phase-only(A), magnitude-only(A) — built with new Hermitian-safe ops in `spectral_ops.py`
(`band_phase_swap` at c=1, `phase_only`, `magnitude_only`; all preserve the 4
self-conjugate bins so the ifft stays real, ~1e-6 residue). Scored with `image_metrics`
+ CLIP ViT-L image cosine to source A and B (new `clip_sim.py`). Driver
`e13_phase_mag_swap.py` (`--part gen,analyze`, image+latent cached → resumable).
Memory: new `gpu_resident` loader (NF4 transformer + bf16 T5 on GPU, no cpu-offload) —
identical numerics to `bnb4` but keeps CPU RAM ~baseline; this RAM-contended box
OOM-kills cpu-offload runs.

**Key results.** (CLIP image cosine, mean over 6 classes × 2 pairs.) **Identity follows
phase; magnitude alone carries no layout — but the full-spectrum margin is
content-graded, not absolute.**
- **The phase/magnitude asymmetry is unambiguous in the pure conditions:**
  magnitude-only ≈ **0.514** CLIP-to-source (≈ the cross-pair floor; a textured palette
  swatch, no recognizable subject) vs phase-only **0.747** (recognizable layout,
  flat/desaturated — magnitude envelope erased). Magnitude alone = no identity; phase
  alone = identity minus palette.
- **In the full-spectrum swap, identity tracks the phase donor on average but by a
  modest margin:** A-phase+B-mag → phase **0.858** vs mag **0.842**; B-phase+A-mag →
  phase **0.865** vs mag **0.833**. Softer than E7's low-band flip because CLIP image
  cosine also reads palette/contrast from the magnitude donor.
- **Content dependence (flag).** `abstract` ties/reverses: its magnitude-only retains
  real identity (0.63/0.65 vs the ~0.49 floor) and A-phase+B-mag is a dead heat
  (0.804 vs 0.802) — a structureless colourful speckle still reads as "abstract
  colourful painting" to CLIP. Phase dominance is cleanest for layout-defined photo
  classes (animal, portrait, urban_night), weakest for palette/texture-defined prompts.
- **Refines E7:** at full spectrum phase still wins, but the dominance is graded by how
  much of the prompt's identity is layout vs palette.

**Artifacts.** `results/e13/` (`grid_<class>.png` 6-variant rows,
`plots/identity_phase_vs_mag.png`, `report.json` per-class clip_to_A/B + metrics,
`images/`, `latents/`). New code: `spectral_ops.py` phase ops (`phase_only`,
`magnitude_only`, `scale_phase`, `phase_offset`, `phase_ramp`, `rotate_band_phase`,
`add_band_phase_noise`, `odd_sign_mask`), `clip_sim.py`, `gpu_resident` mode in
`e7_flux_phase.load_flux`. (Also fixed `common.py` to resolve repo paths from `__file__`
rather than a hardcoded `/workspace`, un-breaking all scripts under the current docker
mount.)

---

## E14 — Functions on phase: which bands carry identity

**Motivation.** E13 swapped phase vs magnitude wholesale. E14 deforms the phase
*parametrically* to localize identity by band and to demonstrate the Fourier shift
theorem.

**Setup.** Decode-only on E13's bank (seed 0 per class as the representative latent),
Flux VAE + CLIP. Four sweeps via the new `spectral_ops` ops (all Hermitian-preserving;
preflight asserts `phase_ramp == torch.roll` to 1e-5, constant offset ≠ any roll,
low-band noise erodes more): scale φ→αφ (α∈{0,0.5,1,2}); ramp (= spatial shift,
d∈{8,16,32}) vs constant antisymmetric offset (δ∈{0.5,1,2}); per-band rotation (low vs
high band); graded per-band phase noise φ→φ+εη (η Hermitian, low vs high band,
ε∈{0.25,0.5,1,2}). Each output scored CLIP-cosine-to-unmodified + `image_metrics`.
Driver `e14_phase_functions.py`.

**Key results.** (CLIP-to-unmodified, mean over 6 classes.) **Identity lives in low-band
phase; high-band phase edits are near-free — and a frequency-linear ramp is a
translation, a constant offset is not.**
- **Graded phase noise, low vs high band (the headline):** low-band ε=0.25→2.0 =
  **0.86 / 0.78 / 0.68 / 0.67**; high-band = **0.97 / 0.93 / 0.88 / 0.88**. Low-band
  noise erodes identity already at ε=0.25 and saturates near 0.67; high-band barely
  moves. Confirms E7 (identity in low-band phase) and complements E6 (high-band phase
  is cheap).
- **Scale φ→αφ:** α=1 → 1.000 (identity, by construction); α=0.5 → 0.81; **α=0 → 0.48
  and α=2 → 0.46 both collapse to ~chance** — zero-phase (real-even latent) and
  phase-doubling each scramble identity symmetrically.
- **Ramp vs constant offset (shift theorem, pedagogical):** the ramp = spatial shift
  stays benign (d=8/16/32 → 0.92/0.90/0.88 — a wrap-around translation CLIP largely
  ignores), while the constant antisymmetric offset *degrades monotonically*
  (δ=0.5/1/2 → 0.89/0.82/0.73). A linear phase ramp in frequency is exactly a
  translation (`phase_ramp` matches `torch.roll` to 1e-5); a constant added to every
  phase is not a shift and must be applied antisymmetrically (`odd_sign_mask`) just to
  stay real — and even then it distorts.
- **Per-band rotation:** rotating low-band phase (δ=1 → 0.83) corrupts more than
  high-band (0.94). Same low-band-carries-identity story.

**Artifacts.** `results/e14/` (`grid_<class>_{scale,shift,noise}.png`,
`plots/identity_vs_eps.png`, `report.json`).

---

## E15 — Classify outputs by phase manipulation

**Motivation.** E13/E14 produced a battery of phase-manipulated decodes. E15 asks
whether the manipulation→output relation is structured enough to "classify": do phase
edits map to consistent output groups independent of seed and image class?

**Setup.** No generation. Embed all **210** cached E13+E14 output PNGs (11 manipulation
families × 6 classes) with CLIP ViT-L + `image_metrics`; KMeans (k = #families) purity
& silhouette vs manipulation labels and vs class labels; nearest-neighbour
manipulation-consistency; per-family CLIP-centroid distance to the `orig` centroid; 2-D
PCA projection. `e15_phase_clusters.py` (sklearn if present, else numpy
PCA/KMeans/silhouette fallbacks — this box has no sklearn).

**Key results.** **The manipulation→output map is structured as a consistent
*magnitude-of-effect axis* (distance from unmodified), not as discrete
seed/class-independent clusters.**
- **Distance-to-`orig` (CLIP centroid) is monotone in how much the edit touches
  identity-bearing structure:** high-band edits sit closest to unmodified —
  rotate_high **0.17**, noise_high **0.22** (near-invisible, collapse onto the orig
  cluster) — then rotate_low 0.26, shift 0.28, phase_mag_swap 0.30, offset 0.35, then
  the identity-altering edits farthest — scale 0.46, noise_low 0.47, phase_only 0.47 —
  with **mag_only 0.76 the clear outlier** (layout stripped). Exactly the roadmap's
  expectation: low-band-touching edits move far, high-band-only edits collapse near
  unmodified.
- **But manipulations do NOT form clean clusters:** KMeans purity vs manipulation
  **0.28** (vs class **0.62**); NN-consistency vs manipulation **0.71** (vs class
  **0.94**). Raw CLIP space is dominated by image *content* — an edited fox still
  embeds near foxes — so KMeans groups by class, and a manipulation is "classifiable"
  only along the radial orig-distance axis, not as a content-independent cluster
  (mag_only is the exception that *does* cluster, since it erases content to swatches).

**Artifacts.** `results/e15/` (`report.json` purity/silhouette/NN-consistency +
per-family centroid-distance-to-orig; `plots/proj_by_manipulation.png`,
`proj_by_class.png`; `embeddings.pt` cached CLIP+metrics).

---

## E16 — SBN fidelity vs training-free guidance baselines (Flux)

**Motivation.** cfg=1 Flux is realistic+on-prompt on simple prompts, but practice uses
high CFG (~3.5) for *detailed* prompts, where guidance buys composition at the cost of
realism (E10: CFG inflates spectral power → over-saturation/contrast). Does SBN (band-norm)
+ the E11 saturation postprocess yield **higher fidelity** than recent training-free
guidance methods *without* losing adherence? Fidelity is the contest; adherence a guardrail.

**Setup.** Flux-dev, 8 detailed prompts, shared seeded init latent per seed. Conditions:
`cfg1.0` (realism anchor), `cfg3.5` (degraded baseline), `bandnorm` (SBN clamp to cfg=1
per-step reference), `bandnorm_pp` (SBN + sat×, our full method), and the training-free
baselines — `cfgzero` (CFG-Zero*: optimal-scale + zero-init true-CFG), `negprompt`
(true-CFG + fidelity negative prompt, NAG proxy at 28 steps), `seg` (Smoothed Energy
Guidance, blurred-query self-attention, best-effort). Metrics: aesthetic / ImageReward /
spectral-dist-to-real (fidelity) + CLIP-T / VQAScore (adherence guard). Baseline wrappers
in `e16_baselines.py`; driver `e16_prompt_adherence.py` (gen/score/analyze).

**Status.** Code complete; partial sweep run (`results/e16/e16_sweep.log`), scored
results not yet committed. Numbers TBD.

**Artifacts.** `experiments/e16_baselines.py`, `e16_prompt_adherence.py`, `e16_site.py`;
`results/e16/`.

---

## E17 — SD3.5 port (true CFG): SBN vs CFG-Zero* + CompBench harness

**Motivation.** E16 found Flux's *distilled* guidance makes the high-CFG regime odd. SD3.5
uses **real** classifier-free guidance (guidance=1 = pure conditional field = the SBN
reference), so it's the cleaner testbed. Is SBN better than CFG-Zero* in the high-CFG
regime, and do they **complement**? This port also adds the SD3.5 VAE encode/decode +
`gen_sd3`/`ClampPSD3`/CFG-Zero* helpers in `e17_sd35.py` that **E18–E22 reuse**.

**Setup.** SD3.5-medium (unpacked 16×128×128 latents — simplest clamp path), shared seeded
init. Conditions: `cfg1`, `cfg_hi` (4.5), `bandnorm`, `bandnorm_pp`, `cfgzero`,
`cfgzero_sbn` (compose), plus CFG++ variants. Two harnesses: `e17_sd35_compare.py`
(detailed prompts, aesthetic/ImageReward/CLIP-T) and `e17_compbench.py` (T2I-CompBench
color/shape/texture, B-VQA attribute binding — does SBN's clamp preserve or harm binding,
alone and combined with CFG-Zero*/CFG++). Spectral-dist-to-real deferred until an
SD3.5-VAE real reference exists (E23/E10 ref is Flux-VAE).

**Status.** Code complete; SD3.5 generation runs pending on the cluster. Numbers TBD. The
reused VAE/gen helpers are the load-bearing deliverable already exercised by E18–E22.

**Artifacts.** `experiments/e17_sd35.py` (backend), `e17_sd35_compare.py`,
`e17_compbench.py`; `results/e17/`, `results/e17cb/`.

---

## E18 — Offline two-image spectral recombination ("AdaIN-in-Fourier")

**Motivation.** SBN clamps a latent's per-band power to one reference; the new direction
drives generation with the spectrum of **two** sources. Premise check first, no diffusion:
in the SD3.5/Flux latent does **phase (esp. low-band) = content** and **per-band magnitude
= style** well enough to recombine two **real** images? Re-leveling per-band power is AdaIN
on the radial power spectrum (`spectral_ops.psd_match`); connects to Gatys/AdaIN and hybrid
images (Oliva 2006).

**Setup.** VAE-encode pairs (A=content, B=style), recombine spectra, VAE-decode.
`e18_spectral_recombine.py` variants: `styleA_s{p}` (= `restyle_latent`, phase A + within-band
texture, power→B; the isotropic pure-style op), `phaseA_magB` (= `band_phase_swap`), `hybrid_c`
(low from A + high from B), + `phaseonlyA`/`magonlyA` Oppenheim–Lim controls. Scored with
CLIP-I→A/B + log-radial-PSD distance. Operators in `style_ops.py`.

**Key results.** **Premise holds, and it's VAE-dependent.**
- **Restyle preserves content while shifting palette:** photo-photo smoke (Flux VAE)
  `styleA_s1` clip→A **0.926** with colorfulness moving A→B; the palette lives in DC/low
  bands as predicted. `phaseA_magB` transfers more style but degrades content (clip→A 0.775).
- **Cross-domain (photo→painting) is real on SD3.5, weak on Flux.** In the **SD3.5** latent
  (the E19 model) restyling a photo toward a painting roughly **halves** the spectral distance
  (psd→B 1.92→1.03) and moves colorfulness ~30% toward the painting at clip→A 0.90; in **Flux**
  latent it's nearly inert (psd→B 1.92→1.80). **But it's tone/palette transfer, not strokes**
  — clip→B barely moves (radial bands carry energy+palette, not oriented structure).
- Numerics verified (restyle band-power rel-err 2e-6; hybrid/swap imag-residue ~1e-6).

**Artifacts.** `results/e18/{report_flux.json, report_sd35.json, grids/, site/}`;
`experiments/e18_spectral_recombine.py`, `style_ops.py`, `fetch_styles.py`. See EXPERIMENT_18.md.

---

## E19 — Generation-time spectral style transfer

**Motivation.** The headline of the E18 thread: generate a *content prompt* while clamping
its spectrum toward a *style image* during denoising — content provides phase + the per-step
energy trajectory, the style image provides the radial power envelope.

**Setup.** `e19_spectral_style.py`: `ref = build_style_reference(content_ref, style_band,
strength, gmax)`; generate with `ClampPSD3(ref)` at cfg≈4.5. Strength sweep (0 = plain SBN,
1 = full style envelope); `gmax` clamps per-band gains. Metrics: content_clip (CLIP-I to the
unstyled image) vs style_clip / style_band_dist, with aesthetic / ImageReward / CLIP-T guards.
Follow-on modes already fold in (operators unit-tested in `style_ops.py`): hybrid synthesis,
spectral morph, two-prompt SBN (`blend_references` — "a cat" with a Van Gogh spectral
signature, reference-image-free).

**Status.** Code-complete; model-free preflight passes (strength=0 ≡ SBN, gmax clamp
applied). The gen/score/analyze parts need the SD3.5 download — pending on the cluster.
Numbers TBD.

**Artifacts.** `experiments/e19_spectral_style.py`; `results/e19/`. See EXPERIMENT_18.md.

---

## E20 — Spectral warm-start ("skip the beginning" of generation)

**Motivation.** Early denoising steps fix low-frequency **structure**, late steps fix
power/detail. Hand the model the low-frequency content up front (a band-pre-set intermediate
latent) and re-enter the trajectory partway to **skip early steps** — for conditioning/style
transfer and for shaping plain generation.

**Setup.** Profiling the cached E8 trajectory showed per-band **power locks in *late***
(~25–27 of 28 steps), so a warm-start must commit low-band **phase**, not power (opposite of
what SBN clamps — a genuinely new lever). Re-entry via rectified flow `x_t=(1-σ)x0+σε`;
`gen_sd3_warmstart` noises the warm-start latent and feeds it via `latents=` (bypassing
`prepare_latents`). Four parts in `e20_warmstart.py`: **A. profile** (per-band lock-in via
cross-seed phase coherence → skippable prefix); **B. oracle** (commit a finished run's true
low bands ≤ c, re-enter at strength, measure recovery — the method ceiling); **C. condition**
(commit a reference image's low bands = band-controlled SDEdit, vs full SDEdit); **D.
noiseshape** (color step-0 noise toward a natural-latent spectrum).

**Status.** Built + offline-validated; preflight green (band-split endpoints, scale_noise,
`color_noise` shape_err 0.000). The four generation parts need SD3.5 (cluster). Numbers TBD.

**Artifacts.** `experiments/e20_warmstart.py`; helpers in `e17_sd35.py`
(`gen_sd3_warmstart`, `RecordTraj`, `warmstart_sigma`), `style_ops.color_noise`;
`results/e20/`. See EXPERIMENT_20.md.

---

## E21 — RF-inversion frequency-band editing on SD3.5 (reconstruction gate fails)

**Motivation.** Invert a real image to noise (reverse flow ODE), then regenerate with a NEW
prompt while **locking** chosen frequency content to the source — phase (esp. low-band)
carries layout (E12–E14), so locking source low-band phase should preserve composition while
the new prompt edits appearance.

**Setup.** SD3.5 (rectified flow): inversion = integrate the velocity field from σ=0 (clean)
to σ=1 (noise), reversing the Euler generation step over the same σ grid. Parts:
preflight (reverse-Euler exactness on a toy field), **invert** (the gate — reconstruct with
the same prompt; if reconstruction is poor, editing is moot), edit, analyze. Band-lock
variants via `band_phase_swap` / `restyle_latent`.

**Key result.** **Reconstruction gate FAILS** — naive RF inversion and fixed-point inversion
both drift on SD3.5, so the source can't be recovered faithfully and editing is moot.
Motivated the E22 pivot to an eps-prediction model where DDIM inversion is reliable.

**Artifacts.** `experiments/e21_spectral_edit.py`; `results/e21/`.

---

## E22 — DDIM-inversion frequency-band editing (SDXL pivot)

**Motivation.** E21's RF inversion failed to reconstruct on SD3.5. SDXL is an
eps-prediction model where DDIM inversion is reliable, so pivot here: invert a real image
with `DDIMInverseScheduler`, regenerate with a new prompt while locking chosen source
frequencies. Same 128×128 grid as SD3.5, so `spectral_ops`/`style_ops` apply unchanged.

**Setup.** SDXL 1024px (cached, runs locally). Parts: preflight / **invert** (the gate) /
edit / analyze. Edits: photo → {oil painting, pencil sketch, watercolor}. Variants:
`invert_only`, `lockphase_c{0.1,0.25}_u{0.6,1}` (lock source low-band phase), `lockpower`.
Metrics: struct_clip (CLIP-I to source) vs edit_clip_t (CLIP-T to target prompt).

**Key results.** **DDIM inversion reconstructs (gate passes):** recon CLIP-I **0.91–0.97**
across the three photos. Band-locking is a clean **structure↔edit trade-off**: `invert_only`
edits hardest (edit_clip_t 0.228) but barely preserves structure (struct_clip 0.626);
locking the source's low-band phase pushes structure to **0.89–0.91** while softening the
edit (edit_clip_t ~0.18–0.20). So low-band phase lock = a composition-preservation knob, as
predicted by E12–E14.

**Artifacts.** `experiments/e22_ddim_edit.py`; `results/e22/{invert.json, edit.json,
invert/grid.png}`.

---

## E23 — Real-image spectral target ("real-SBN")

**Motivation.** SBN re-leveled a generated latent's per-band power toward a **cfg=1**
reference — but cfg=1 is just a softer model output, not reality. E23 **measures** the
generated-vs-**real** spectral gap (vs 500 COCO photos) and corrects each generated image's
spectrum toward the real-photo spectrum ("real-SBN").

**Setup.** Flux cfg=3.5. Real target = mean per-(channel, band) power of 500 VAE-encoded COCO
photos (kept per channel). Operator = `psd_match` (mag × √(target/cur), phase kept), with a
strength exponent. Three application modes (offline final-latent / during-gen last-step /
init-noise shaping); we do **not** correct every step (mid-denoising latents are mostly noise).
Driver `e23_real_sbn.py`.

**Key results.**
- **The gap is bimodal:** low-band *excess* (CFG low-freq inflation) + a broad mid/high-band
  **deficit** (ratio ≈ 1.2–1.3) — generated images are systematically under-textured vs real.
- **Correcting toward real helps and beats the old cfg-1 SBN.** real-SBN (offline or
  last-step) gives the **biggest aesthetic gain** of any condition (6.61→6.88/6.89) at ~zero
  adherence (CLIP-T) cost, while the old cfg-1 SBN moves *away* from real (spec-dist 0.582).
  Gain concentrates on photographic portraits (Δaesthetic +0.55…+0.64). **0.5+ over-amplifies
  the empty high bands into grain ⇒ ≈0.25 is the recommended strength** (the LAION predictor
  over-rewards sharpening past human preference).
- **init-noise shaping fails** (the flow prior wants white noise — documented negative).
- spec-dist→real is partly circular at full strength; the independent wins are aesthetic + the
  near-zero adherence cost. Isotropic only.

**Artifacts.** `experiments/e23_real_sbn.py`, `real_spectral.py`, `e23_site.py`;
`results/e23/{report.json, scores.json, examples.json, adherence/}`. See EXPERIMENT_23.md.

---

## E24 — Token-axis FFT on the TEXT conditioning (FNet-motivated)

**Motivation.** A prompt becomes a token-embedding sequence `E ∈ (1, L, 4096)` (Flux T5-XXL)
fed to cross-attention. Can we treat that as a signal, FFT it **along the token axis** (1-D,
per embedding channel, on the real-token span — not 2-D, not the pooled vector), and use the
bands to merge or edit images? Plausible via **FNet** (Lee-Thorp 2021): a parameter-free
token-axis DFT replaced self-attention at ~92–97% of BERT.

**Setup.** Flux true-CFG=1, guidance 3.5, 28 steps, 4 seeds. Ops in `text_spectral_ops.py`
act on `E[:, :L]`. Three parts: **probe** (keep one band — dc/low/high/full — regenerate),
**merge** (A+B: `lowA_highB`, `phaseA_magB`, soft `blend`, vs the `lerp@0.5` bar-to-beat +
`pooled_swap` + `fnet_swap`), **edit** (inject style's high band into base over a cut grid).
Metrics: CLIP-T to each source prompt (attribution), aesthetic, ImageReward.

**Key results.**
- **Probe:** every band-filtered variant is coherent + on-prompt; the **high band alone
  reconstructs ≈ as well as `full`** — the conditioning is robust to band filtering (FNet
  intuition holds at the conditioning level).
- **Merge — NEGATIVE.** No condition blends A and B; results **snap to whichever prompt owns
  the low band + phase**, and the spectral merges **do not beat `lerp`** (which itself
  collapses to A). The token spectrum is a near-**binary identity selector**, not a blender.
- **Edit — PARTIAL POSITIVE.** High-band style injection is a usable **style-strength knob**
  via `cut`; `pooled_inject` does about the same more cheaply.
- **Mechanism:** identity is carried by the token-axis **phase** + low band
  (`phaseA_magB→A`, `magA_phaseB→B`) — mirrors the image-domain "phase = structure".

**Artifacts.** `experiments/e24_text_spectral.py`, `text_spectral_ops.py`;
`results/e24/{probe,merge,edit}/`, `index.html`. See EXPERIMENT_24.md.

---

## E25 — Seed alignment pilot: bias the initial noise toward the prompt (SD1.5)

**Motivation.** The seed `z~N(0,I)` leaves traces in the output (near-linear PF-ODE). Can a
*tiny, cheap* optimization nudge the seed **toward the prompt** before generation, and have
the bias survive? A deliberately gentle, do-no-harm lever — not a big adherence jump.

**Setup.** SD1.5 512px. Objective is **purely latent-space, no UNet**:
`loss = −cos(CLIP_img(VAE.decode(z)), CLIP_text(prompt))`, gradients through frozen VAE+CLIP
only. **Constraint:** re-standardize `z←(z−μ)/σ` each step → zero mean/unit var pins
`‖z‖=√d=128` exactly (stay on the Gaussian sphere). Compare against an `x̂₀` mode that runs
one UNet step. 4 prompts × 2 seeds. `e25_seedalign.py`.

**Key results.** **The seed's trace is a palette/global-appearance trace, not composition.**
- Moments/norm held exactly every run. The **`x̂₀` mode over-optimizes and goes
  CLIP-adversarial** (leopard-print for "cat") → net −0.022…−0.025 (slightly *hurts*).
- The **latent mode is gentlest** (mean Δ CLIP-T **−0.010**, 4↑/4↓), a controlled palette
  nudge that occasionally helps. The latent-space objective is the right do-no-harm form.

**Artifacts.** `experiments/e25_seedalign.py`; `results/e25/`. See EXPERIMENT_26.md.

---

## E26 — Seed alignment on SDXL + DPG-Bench + #-steps sweep

**Motivation.** Extend E25's latent-space seed nudge to a stronger model, long dense prompts,
and characterize how many inner gradient steps are worth taking.

**Setup.** SDXL 1024px (fp16-fix VAE; `√d=256`). Long prompts from DPG-Bench (`dpg_bench.py`,
~55–109 words). **Caveat:** SDXL's CLIP text encoders *and* the CLIP scorer truncate at 77
tokens, so a **long-aware CLIP-T** (per-clause encode → mean-pool + renorm) is the target and
metric. Sweep N = inner latent-mode steps {1, 1-strong, 2, 3, 5}, snapshotting `z` at each N.
`e26_seedalign_sdxl.py`.

**Key results.** **Break-even, and the cheapest setting wins.** Mean Δ long-CLIP-T: N=1
**+0.0015** (only clearly non-negative point); 2/3/5 drift slightly negative; strengthened
single step ~0. Constraint `‖z‖=256=√d` held at every snapshot. Visually the aligned columns
stay close to baseline (gentle palette/saturation shifts, **no structural damage**). A single
cheap gradient step captures whatever benefit there is; the long-prompt 77-token bottleneck
caps how much it can move adherence.

**Artifacts.** `experiments/e26_seedalign_sdxl.py`, `dpg_bench.py`;
`results/e26/{grid.png, deltaclip_vs_N.png, report.json}`. See EXPERIMENT_26.md.

---

## E27 — A single "concept direction" in the seed (CLIP→latent pullback, SDXL)

**Motivation.** Instead of optimizing every seed (E25/E26), compute **one direction per
prompt** in seed space ("more of this prompt") and just **add** it to any seed — a linear
steering vector.

**Setup.** SDXL 1024px. Two-stage construction that reduces to **one chain-rule backward
pass**: Stage 1 CLIP-space cosine gradient `g`; Stage 2 decoder pullback `v=normalize(Jᵀg)`;
composed = `normalize(∇_z cos(CLIP_img(decode(z)), text))`. The only real choice is the
anchor (base image `e₀`) — swept {chain, noise, mean, fit, nofit}. Two uses:
**Arm A** additive `z'=renorm(z₀+s·√d·v)` (s-sweep), **Arm B** heavy per-seed iteration
(N-sweep), each re-standardized. `e27_seeddir.py`.

**Key results.**
- **Arm A additive direction is too blunt:** s≤0.25 break-even (no CLIP-T change), s=0.5
  washes out (−0.083), s=1 collapses to noise (−0.144). No free-lunch additive s.
- **Arm B on-manifold iteration is the well-behaved version:** progressively intensifies
  palette/saturation toward the concept while preserving composition even at N=60; CLIP-T
  stays ~flat (best +0.003 at N=5) — it enhances *appearance*, not object presence.
  Per-step re-standardization is what keeps it safe.
- **The anchor barely matters:** all directions 0.89–0.97 correlated; "the prompt direction
  in seed space" is essentially anchor-independent (Arm A is the chain rule).

**Artifacts.** `experiments/e27_seeddir.py`, `e27_site.py`;
`results/e27/{grid_direction,grid_heavy,grid_anchors,deltaclip}.png, report.json, index.html`.
See EXPERIMENT_27.md.

---

## E28 — Does seed-biasing RESCUE dropped elements on hard compositional prompts?

**Motivation.** E25–E27 found seed-biasing is a do-no-harm palette lever with flat CLIP-T —
but on easy prompts with a metric blind to dropped elements. Is there a regime where the seed
*matters*? Candidate: hard compositional prompts where the baseline **drops** an element,
scored with **B-VQA** (T2I-CompBench attribute binding — product of P(yes), so one
dropped/mis-bound element tanks it).

**Setup.** SDXL 1024px. Scan 30 CompBench prompts (color/shape/texture) × 4 seeds; FAIL =
B-VQA < 0.5; per-prompt seed-dependence = fraction of seeds that pass. Intervene on the worst
37 failures: arm A (bias toward full prompt), arm B (toward the single dropped phrase), via
iterative latent-mode opt (‖z‖=√d). Controls: **re-roll** (fresh random seed) and
**do-no-harm** (apply to passers). `e28_seedrescue.py`.

**Key results.** **A clean negative — biasing the seed does NOT beat re-rolling it.** On the
37 failures, recovery rate: arm A 0.189, arm B 0.243, **re-roll 0.324**; on the
seed-dependent stratum (n=21) **0.571 (re-roll) vs 0.429 / 0.286**. **do-no-harm FAILED:**
applying the bias to *passing* pairs dropped B-VQA by **−0.176** (it breaks working
compositions). Always-fail prompts recover with nothing. The gradient moves palette within
the *same compositional basin*; changing *which* mode renders needs a genuinely different seed.
**Closes the seed-as-adherence line (E25→E28); best-of-N + a B-VQA picker wins.** The
remaining positive use of seed-biasing is appearance/palette steering (E27 Arm B).

**Artifacts.** `experiments/e28_seedrescue.py`, `compbench.py`;
`results/e28/{grid_recovered.png, summary.png, report.json}`. See EXPERIMENT_28.md.

---

## E29 — Phase inheritance: does the seed's FFT phase determine the output's? (SD1.5/DDIM)

**Motivation.** Phase carries structure (Oppenheim–Lim). So when DDIM turns seed `z_T` into
output `z_0`, **how much of the output's phase is inherited from the seed's phase?** If
structure lives in phase, the seed may pre-commit layout before any denoising step. Measured
per band, then confirmed causally. (E28 number was taken, so this is E29.)

**Setup.** SD1.5, deterministic DDIM, 64 seeds/condition, 24 radial bins. Metrics
(`e29_phase_ops.py`): per-bin **circular correlation** (seed↔output phase) with a permutation
null; log-magnitude Pearson; phase-difference resultant; spatial Pearson. Unconditional map +
CFG sweep {1, 3, 7.5}. **Causal transplant:** swap a donor's phase into the base seed's lowest-c
band (magnitude held → variance preserved, std=1.000), regenerate, measure follow-score.

**Key results.**
- **Strong, BROAD-spectrum inheritance — not phase-specific.** Unconditional phase corr
  ≈ 0.40 (low) / 0.53 (high) vs null ≈ 0; **magnitude (log-power) is inherited at least as
  strongly** (≈0.48/0.58) and raw pixel-space corr z_T↔z_0 ≈ **0.76**. So at low guidance the
  seed broadly fixes the *whole* latent — "phase = structure" is what the latent encodes, not a
  privileged inheritance channel (a correction to our prior).
- **CFG erodes inheritance, preferentially in low-freq (composition) bands:** low-band phase
  corr 0.40 (uncond) → 0.35 → 0.23 → 0.15 across CFG.
- **Causal + propagating:** the low-band transplant gives follow-score ≈ **0.66** in the
  swapped band and 0.60–0.63 in *higher* bands too (0.5 = no effect) — the seed's coarse phase
  conditions the entire downstream structure.

**Artifacts.** `experiments/e29_phase_inherit.py`, `e29_phase_ops.py`, `e29_site.py`;
`results/e29/{report.json, transplant.json, index.html}`. See EXPERIMENT_29.md.

---

## E30 — Continuous text-frequency control & extraction (follow-up to E24)

**Motivation.** E24 found token-axis bands of Flux's T5 sequence embedding are meaningful and
on-manifold, that merging snaps to the low-band/phase owner (no win over `lerp`), and that
high-band injection is a style-strength knob. E30 (a) characterizes the bands more finely,
(b) turns the manipulation into a **continuous knob** with image-strip visualizations, and
(c) asks what frequency filtering does to **long** and **compositional** prompts.

**Setup.** Flux. New ops `band_gain_1d` / `band_notch_1d` in `text_spectral_ops.py` (same
1-D token-axis transform as E24). Parts: **probe_deep** (per-band knockout, phase-only vs
mag-only recon), **continuous** (image strips as one knob — low-pass cutoff / high-band gain /
A↔B swap-cut morph), **concat** (spectral merge vs writing "A and B"), **longprompt**
(DPG-Bench: does dropping high freq drop the tail objects?), **compositional** (CompBench
B-VQA). Metrics: CLIP-T, image stats, aesthetic, B-VQA, VQAScore.

**Status.** Code complete and offline-verified (ops + builders); cluster run pending.
**Results: TBD.**

**Artifacts.** `experiments/e30_text_freq_control.py`; `results/e30/`. See EXPERIMENT_30.md.

---

## E31 — Real-image editing via FlowEdit + frequency-surgery conditioning

**Motivation.** FlowEdit (Kulikov 2024) edits a flow model without inversion by integrating
the difference between target- and source-conditioned velocities and adding the delta to the
source latent. E31's twist: the **target conditioning is a token-frequency surgery** of the
source conditioning (E24/E30 ops) — e.g. low band from the source prompt + high band from the
edit prompt — instead of a plain different prompt.

**Setup.** Flux. Manual `flux_velocity` accessor + `flux_sigmas`; FlowEdit ODE
`δ += (σ_next−σ)·(v(x_tar,C_tar) − v(x_src,C_src))`, edited = x0 + δ; `--skip` = edit
strength. `C_tar = band_swap_1d(low: C_src, high: C_style)` at a couple of cuts, plus `full`
(plain prompt-swap). Source `x0` from the source prompt (clean eval) or a real image via
`--real_dir`. **Identity gate:** if `C_tar==C_src` the velocity difference is exactly 0, so
`recon` reproduces the source by construction — validates the VAE/packing path before any GPU.
Metrics: CLIP-to-style (edit) vs CLIP-to-source + pixel-distance (preservation), aesthetic.

**Status.** Code complete; model-free preflight + wiring verified offline (FlowEdit identity
holds by construction); cluster run pending. **Results: TBD.**

**Artifacts.** `experiments/e31_flowedit_freq.py`; `results/e31/`. See EXPERIMENT_31.md.
