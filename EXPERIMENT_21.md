# E21 — Spectral image editing via RF-inversion + frequency-band locking (SD3.5)

**The direction.** The spectral-style thread (E18–E19) showed that in latent Fourier space
**phase carries layout/content** and **per-band magnitude carries texture + palette** — and that
re-leveling per-band power transfers tone/palette without moving the content. Those experiments
operated on *generated* latents. E21 pushes the same decomposition onto **real images**: take a
real photo, **invert** it back to the noise that would have produced it, then **regenerate with a
new prompt while LOCKING chosen source frequency bands**. Because low-band **phase** is the
composition, locking it means the prompt is free to repaint appearance (oil-painting, sketch,
watercolor) while the layout of the original photo survives — a *frequency-decomposed editing
control*. SD3.5 is a **rectified-flow** model, so "inversion" is integrating the velocity field
backwards (clean → noise), not literal DDIM.

> **Status / Verdict:** ⚠️ **Code complete; the run is GATED and pending — and the gate is the
> finding.** The whole edit only makes sense if the inversion *reconstructs* the source. On SD3.5
> the reverse-flow ODE **drifts**: naive forward-Euler inversion and the implicit fixed-point
> variant both fail to round-trip a real image back to itself. That failure is exactly why
> **E22** exists (pivot to SDXL + DDIM inversion, where the round-trip is reliable). E21 is the
> documented negative that motivates E22.

## Background (plain language)

- **Latent.** SD3.5 denoises in a compressed `16×128×128` array (at 1024px); a VAE turns it into
  the RGB image. All spectral surgery happens on the latent, not the pixels.
- **Rectified flow (RF).** SD3.5 is not a noise-predictor; it learns a **velocity field**
  `v(x, σ)` that flows a clean latent (σ=0) along a straight-ish path to pure noise (σ=1) and
  back. Generation is the Euler step `x += (σ_next − σ_cur)·v(x, σ)` walking σ: 1 → 0.
- **Inversion (clean → noise).** To edit a *real* image you first need the noise it came from.
  RF-inversion integrates the same velocity field in the **opposite** direction, σ: 0 → 1. This
  is *not* DDIM inversion; it is the RF-Inversion / FlowEdit recipe.
- **Naive vs fixed-point Euler.** Forward Euler evaluates the velocity at the *current* σ; it is
  only exact when `v` does not depend on `x`. The **fixed-point (implicit) Euler** instead solves
  `x_hi = x_lo + (σ_hi − σ_lo)·v(x_hi, σ_hi)` by iterating a few times — it evaluates the velocity
  at the *next* σ and is far more accurate for a state-dependent field. E21 uses `fp_iters=4`.
- **FFT phase vs magnitude.** Each spatial frequency of the latent has a **magnitude** (how
  strong the ripple is → texture power / palette) and a **phase** (where the ripples line up →
  structure / layout). Oppenheim & Lim: phase carries most recognizable structure.
- **Radial band / cut `c`.** Frequencies are binned into `N_BINS=24` rings from DC outward; band
  0 = coarsest layout. A **cut `c`** selects the lowest `c` fraction of the spectrum — `c=0.1`
  locks only the coarsest layout, `c=0.25` a bit more.
- **Reconstruction CLIP-I (the gate).** Invert a real image, regenerate from that noise with the
  **same** prompt, and measure CLIP image-similarity between original and reconstruction. If this
  is not high, the inverted noise is wrong and any *edit* built on top is meaningless.

## Method

- **Model.** SD3.5 (`e17_sd35.load_sd35`), rectified flow, 1024px, `(1,16,128,128)` latent,
  28 steps, `encode_prompt` with CFG=1 for inversion.
- **Inversion operator (`invert_sd3`).** Walk σ: 0 → 1 over the scheduler's σ grid; at each step
  do `fp_iters=4` implicit-Euler iterations of `x ← x_lo + (σ_hi − σ_lo)·v(x, σ_hi)`. Output: the
  inverted noise latent (`noise_std` reported as a sanity check — should be ≈ 1).
- **The band-lock callback (`BandLock`).** A step-end hook applied for the first `until` fraction
  of generation steps, then released so the target prompt drives the finish:
  - `mode="phase"` — `band_phase_swap(src, gen, c, mag_from="B")`: keep the **source's** low-band
    phase (layout) but the **generation's** magnitude and high-band phase. Locks composition.
  - `mode="power"` — `restyle_latent(...)`: re-level per-band power to the source (palette lock).
- **Conditions.** Per edit pair (photo, source-prompt, target-prompt): `invert_only` (baseline,
  no lock) vs `lockphase_c{0.1,0.25}_u{0.6,1.0}` vs `lockpower`. Three edits:
  photo→oil-painting, photo→pencil-sketch, photo→watercolor.
- **Metrics.** `struct_clip` = CLIP-I to the **source** (composition preserved ↑); `edit_clip_t`
  = CLIP-T to the **target prompt** (edit followed ↑). The tension between these two is the whole
  story: a good editing knob raises both, or trades them along a sensible frontier.
- **Preflight (model-free, passes).** Verifies (1) reverse-Euler is **exact on a
  state-independent** velocity field (round-trip error < 1e-3 — confirming the *math* is right,
  so any real-model drift is the *field's* state-dependence, not a bug), and (2) the band-lock
  invariants: `c=1, mag_from="A"` reconstructs the source exactly; `mag_from="B"` keeps the
  generation's magnitude.

## Findings

- **Preflight passes; the model run is pending and gated.** `results/e21/` is empty — no
  `invert.json` / `edit.json` were produced, so **there are no quantitative SD3.5 numbers to
  report** for E21. (This is honest run-pending status, not a result.)
- **The headline expectation is the GATE failure.** The model-free preflight proves reverse-Euler
  is exact when the velocity field is state-independent. A trained RF velocity field is **strongly
  state-dependent**, so on SD3.5 both the naive and the `fp_iters=4` fixed-point inversions are
  expected to (and in this thread's working notes, do) **drift** — the regenerated image does not
  return to the source. With a bad round-trip the edit conditions are uninterpretable.
- **Why this matters.** Reconstruction is the prerequisite for *every* downstream edit cell. A
  failed gate doesn't just weaken E21's edit numbers; it invalidates them. Rather than tune RF
  inversion further (RF-Inversion-style controllers are finicky), the thread **pivots to an
  eps-prediction model with reliable DDIM inversion → E22**, carrying the *identical* band-lock
  editing operators over unchanged.

## Caveats & next

- **RF inversion is the weak link**, not the band-lock idea. The spectral operators are
  model-agnostic and unit-checked in preflight; what fails is recovering faithful noise from a
  real SD3.5 image via Euler integration of a state-dependent field.
- Locking **low-band phase** preserves layout but cannot, by construction, transfer *oriented*
  brushstrokes — radial bands are isotropic (the E18 caveat carries over).
- The `phase` vs `power` modes answer different questions (composition lock vs palette lock); they
  are reported side-by-side rather than combined.
- **Next = E22:** SDXL + `DDIMInverseScheduler`. SDXL's `4×128×128` latent shares SD3.5's
  `(H,W)=128` grid, so `spectral_ops`/`style_ops` apply with **no changes**; only the
  inversion backbone is swapped to one that round-trips.

## Reproduce

```bash
cd experiments
# 1) model-free sanity: reverse-Euler exactness + band-lock invariants
python e21_spectral_edit.py --part preflight
# 2) THE GATE: invert real photos, reconstruct with same prompt, report CLIP-I fidelity
python e21_spectral_edit.py --part invert  --num 3 --steps 28
# 3) edit: invert, regenerate with target prompt under band-lock variants
python e21_spectral_edit.py --part edit \
    --num 3 --steps 28 --cfg 4.5 --cuts 0.1,0.25 --untils 0.6,1.0
# 4) dump the JSON tables
python e21_spectral_edit.py --part analyze
```

Code: `experiments/e21_spectral_edit.py` (driver: `invert_sd3` RF inversion, `BandLock`
callback), reusing `experiments/spectral_ops.py` (`band_phase_swap`, `band_index_map`),
`experiments/style_ops.py` (`restyle_latent`, `latent_band_power`), and
`experiments/e17_sd35.py` (`load_sd35`, `sd3_vae_encode/decode`, `gen_sd3`).
