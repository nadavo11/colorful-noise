# Fourier Phase in Diffusion Latents — What We Know and What to Probe Next

A discussion/roadmap doc for the *phase* thread of the spectral-latent project. The
band-norm line (E8/E9, `BANDNORM_MATH.md`) deliberately moves **power** and freezes **phase**.
This doc is about the other half: what the Fourier *phase* of a latent actually carries, and
how to test it. It is separate from the chronological `EXPERIMENTS.md` log — that records runs;
this records the *question*.

---

## 1. The motivating question — and the reframing

Observation that started this: in the phase plots the phase "looks really noisy and
non-informative." That is **correct as a statement about the marginal** and **wrong as a
conclusion about information**.

- The FFT phase of a real white-Gaussian field is *exactly* uniform on $[-\pi,\pi]$ with
  $\phi(-f)=-\phi(f)$. A near-uniform phase histogram is therefore the **null**, not a finding.
- Image structure does not live in the phase *marginal*; it lives in **phase relationships
  across frequencies**. Aligned phases across many frequencies = an edge. This is the classic
  **Oppenheim–Lim (1981)** result: swap two images' magnitude and phase, and the *phase*
  image is the one you recognize. Magnitude sets the spectral envelope (palette/contrast);
  phase sets *where things are*.

So "can we classify phase changes to images?" splits into two very different questions:

1. **Does the marginal distribution of phase ever deviate from uniform?** (Mostly no — a
   baseline/control. E13 measures this properly, per band/channel/class.)
2. **Does the joint phase *structure* carry identity, and which part of it?** (Yes — E7 already
   saw identity follow the low-band phase donor. E14–E16 probe this.)

Keeping these separate is the whole point: a flat histogram is *expected* and is not evidence
against phase mattering.

---

## 2. What we've done (phase-touching experiments + results)

### E6 — FFT-phase surgery on SDXL *input* noise (`e6_phase.py`)
Minimal off-manifold probe: rewrite phase, keep Rayleigh magnitudes (PSD stays in-distribution).
- **P0 rerand control:** rerandomizing all phases ≈ fresh noise — confirms Rayleigh×uniform
  factorization.
- **P2 image-phase transplant:** image phase at white amplitude transfers layout at every
  mixing $p$; at $p{=}1$ the output collapses to a near seed-invariant flat poster.
- **P4 phase quantization:** **$k{=}4$ levels already photoreal, $k{\ge}8$ indistinguishable
  from continuous** — 2–3 bits of phase suffice.
- **P4b level omission:** zeroing a phase level pair costs nothing once power is renormalized —
  **the phase hole is free; the power loss does the damage.**
- **Verdict:** *SDXL reads power, not fine phase* (on the input side).

### E7 — FLUX.1-dev *output*-latent phase & spectrum (`e7_flux_phase.py`)
Flip side (output latents, 16ch 128×128), cfg 1.0 vs 3.5.
- **Marginals uniform:** flatness ≈ 0.003, phase ⟂ magnitude (corr ≈ 0.005), both cfg groups.
- **Cross-seed coherence** sits at the $N$-uniform null *except the lowest radial bin* — seeds
  share only gross composition.
- **cfg=1 vs 3.5 is a power story, not a phase story:** low-freq power ~3× lower at cfg=1,
  slope −1.5 vs −2.0; phase statistics unchanged.
- **Band-split phase interpolation** (`band_phase_swap`): **identity flips to the low-band
  phase donor at $c\approx0.1\text{–}0.2$**; the magnitude source mostly sets palette/contrast.
- **Verdict:** *latent identity lives in low-band phase* — the complement to E6.

### Reusable utilities (`spectral_ops.py`)
| Function | Purpose |
|----------|---------|
| `random_hermitian_phase` | uniform iid phase with exact Hermitian symmetry (the null) |
| `quantize_phase` | quantize phase to $k$ levels, optional level omission + renorm |
| `band_phase_swap` | hybrid: phase from A in lowest-$c$ band, B outside; magnitude from one source |
| `condition_latent` | low-band decoupling of phase / magnitude / DC |
| `phase_coherence` | cross-sample phase resultant length, radially averaged, vs null |
| `flatness` | std/mean of the $[-\pi,\pi]$ phase histogram (marginal uniformity) |
| `phase_histogram` | **new (E13):** per-(channel, band) phase histogram + flatness + resultant length |

---

## 3. Open gaps

- No per-band / per-channel / per-class phase **distribution** — only one aggregate histogram
  (E7) and per-band *coherence*. → **E13 (this round)**.
- No **full-spectrum** (not low-band-only) phase/magnitude swap on Flux. → **E14**.
- No systematic **"functions on phase"** sweep (scaling, offset vs ramp, per-band rotation,
  graded phase noise). → **E15**.
- No **clustering** of outputs by phase manipulation (does a phase edit map to a consistent
  output class?). → **E16**.

---

## 4. Roadmap

All follow the house conventions: `results/eN/{images,latents,plots}` + `report.json`,
a `preflight()` of numeric asserts before any GPU work, file-cached generation for free resume,
and an `EXPERIMENTS.md` entry after the run. Target model: **FLUX.1-dev** output latents
(16ch, 128×128), canonical 6 classes in `e9_bandnorm_classes.CLASSES`.

(Numbering note: E10–E12 are unrelated experiments — `e10_cfg_spectral`,
`e11_color_correct`, `e12_sd35_portrait` — so this phase line starts at E13.)

### E13 — Phase distributions *(implemented: `e13_phase_dist.py`)*
- **Question:** Is the phase marginal ever non-uniform, and where (band / channel / class)?
- **Method:** generate `seeds` latents per class; compute `phase_histogram` per (channel,
  band) → per-band heatmaps, flatness-vs-band and resultant-length-vs-band curves, per-class
  global $[-\pi,\pi]$ histogram, and cross-seed `phase_coherence` for the joint-structure
  contrast.
- **Reuses:** `phase_histogram`, `phase_coherence`, `band_index_map`, `random_hermitian_phase`
  (preflight null), `load_flux`/`flux_generate`, `CLASSES`.
- **Expected:** near-flat marginals everywhere except the DC/lowest band, ~identical across
  classes; coherence is what separates classes (elevated only in the low band). Establishes the
  baseline that motivates E14–E16.

### E14 — Full-spectrum phase ↔ magnitude swap (Oppenheim–Lim in latent space)
- **Question:** Across the *whole* spectrum (not just the low band), does perceived identity
  follow phase or magnitude? And what does phase-only / magnitude-only decode to?
- **Method:** for latent pairs A,B per class: (i) A-phase + B-magnitude and the reverse,
  decode; (ii) **phase-only** (A phase + flat/constant magnitude) vs **magnitude-only**
  (A magnitude + `random_hermitian_phase`), decode. Quantify with `image_metrics` + CLIP
  similarity to each source.
- **Reuses:** extend `band_phase_swap` to $c{=}1$ full-spectrum plus a magnitude-flatten path;
  `flux_vae_decode`, `random_hermitian_phase`, `image_metrics`.
- **Expected:** identity tracks phase; magnitude-only ≈ textured palette swatch; phase-only
  ≈ recognizable but flat/desaturated — confirming Oppenheim–Lim holds in the Flux latent.

### E15 — Functions on phase
- **Question:** how does the output deform under parametric phase edits, and which *bands*
  carry the identity?
- **Method (sweeps):**
  - **Scaling** $\phi \to \alpha\phi$ for $\alpha \in \{0, 0.5, 1, 2\}$ (Hermitian-preserving).
  - **Global constant offset $\phi\to\phi+\Delta$ vs a linear frequency *ramp*.** Important
    clarification to bake into the writeup: a *linear ramp* in frequency is exactly a **spatial
    shift** (Fourier shift theorem) and a wrap-around translation in latent space; a *constant*
    offset added to every coefficient is **not** a shift and breaks Hermitian symmetry (must be
    applied antisymmetrically to stay real). This sub-experiment is partly to *demonstrate* the
    shift theorem and dispel the "add a constant to phase" intuition.
  - **Per-band phase rotation** (rotate phase only within selected `band_index_map` annuli).
  - **Graded phase noise** $\phi\to\phi+\varepsilon\eta$ (Hermitian $\eta$), sweep $\varepsilon$
    per band → localize where identity breaks first.
- **Reuses:** `band_index_map`, `random_hermitian_phase`, the per-step `ClampPSD`-style
  callback (intervene during denoising) **and** a one-shot variant on the final latent.
- **Expected:** low-band phase noise destroys identity at small $\varepsilon$; high-band phase
  noise is near-free (mirrors E6 quantization); the ramp produces a clean spatial shift.

### E16 — Classify / cluster outputs by phase manipulation
- **Question:** do phase edits map to *consistent* output classes — i.e. is the
  manipulation→output relation structured enough to "classify"?
- **Method:** run a battery of E14/E15 manipulations across seeds/classes; embed outputs (CLIP)
  + `image_metrics`; cluster / low-dim project to see whether manipulations form coherent
  groups independent of seed.
- **Reuses:** `image_metrics`, a CLIP embedder, outputs cached from E14/E15.
- **Expected:** manipulations that touch the low band cluster by *effect*; high-band-only edits
  collapse near the unmodified cluster.

---

## 5. One-line takeaway

Phase marginals are uniform by construction (E13 confirms, per band/class) — so the signal is
in **phase structure**, concentrated in the **low band** (E7). E14–E16 turn that observation
into method: swap it (E14), perturb it parametrically (E15), and test whether the
manipulation→image map is classifiable (E16).
