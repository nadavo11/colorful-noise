# E22 — Spectral image editing via DDIM inversion + frequency-band locking (SDXL)

**The direction.** E21 tried to turn a real photo into an editable latent on **SD3.5** by
inverting it back to noise with the rectified-flow ODE, then regenerating under a new prompt while
**locking** the source's low-band phase (= composition). The idea is sound but the **inversion
drifted**: SD3.5's reverse-flow integration (naive and fixed-point) would not reconstruct the
source, and reconstruction is the gate for everything downstream. E22 keeps the **exact same
band-lock editing control** and swaps the backbone to one where inversion is reliable: **SDXL**,
an **eps-prediction** model with a standard **DDIM inversion**. Same phase-carries-layout premise,
same operators, a round-trip that actually closes.

> **Status / Verdict:** ✅ **Ran on SDXL. The gate passes** — DDIM reconstruction CLIP-I ≈ **0.94**
> (vs E21's drift). Phase-band locking is a **strong, tunable composition-preservation knob**:
> `lockphase` lifts structure-to-source CLIP-I from **0.60 → 0.90** while the prompt still moves
> appearance — but it **trades away** prompt-adherence (edit CLIP-T **0.24 → ≈ 0.16**). It is a
> genuine structure⇄edit frontier, not a free lunch.

## Background (plain language)

- **Latent.** SDXL denoises a `4×128×128` latent at 1024px; a VAE decodes it to RGB. That
  `(H,W)=128` grid is **identical** to SD3.5's, so the radial-band spectral operators
  (`spectral_ops`, `style_ops`) apply **unchanged** — they are channel-agnostic.
- **eps-prediction vs rectified flow.** SDXL predicts the **noise** `ε` added to a latent
  (classic DDPM/DDIM formulation), unlike SD3.5's velocity field. Eps-prediction has a clean,
  well-behaved **DDIM inversion**: a deterministic map clean-latent → noise that closes back to
  the same image. This is the property E21's RF model lacked.
- **DDIM inversion.** Run the diffusion pipeline with the **`DDIMInverseScheduler`** (timesteps
  0 → T) on a real image's latent to recover the noise that produces it. Reverse with the normal
  `DDIMScheduler` and you should get the image back.
- **FFT phase vs magnitude.** Each spatial frequency carries a **magnitude** (ripple strength →
  texture/palette) and a **phase** (alignment → structure/layout). Locking phase locks layout.
- **Radial band / cut `c`.** Frequencies binned into `N_BINS=24` rings from DC; band 0 = coarsest
  layout. **`c`** = the lowest fraction of the spectrum that gets locked (`c=0.1` coarsest only,
  `c=0.25` a bit more).
- **`until` (`u`).** The band-lock callback is active only for the first `u` fraction of the
  denoising steps, then released so the new prompt drives the finish. `u=0.6` lets go early;
  `u=1.0` holds the lock all the way.
- **The two scores.** `struct_clip` = CLIP-I to the **source** image (composition preserved, ↑).
  `edit_clip_t` = CLIP to the **target prompt** (edit followed, ↑). A good knob negotiates these.

## Method

- **Model.** SDXL (`StableDiffusionXLPipeline`, fp16, `SDXL_ID`), `DDIMScheduler` for generation +
  `DDIMInverseScheduler` for inversion, 1024px, `(1,4,128,128)` latent, 30 steps, inversion
  guidance = 1.0, edit CFG = 5.0. Runs locally (SDXL is cached).
- **Invert (`ddim_invert`).** Encode the photo with the VAE (in fp32 for stability), temporarily
  swap in the inverse scheduler, run the pipe to get the noise latent, restore the normal
  scheduler. `noise_std` reported as a sanity check.
- **Band-lock callback (`BandLock`).** Step-end hook for the first `u` fraction of steps:
  - `mode="phase"` — `band_phase_swap(src, gen, c, mag_from="B")`: keep the **source** low-band
    phase (layout), the **generation's** magnitude + high-band phase. Composition lock.
  - `mode="power"` — `restyle_latent(...)`: re-level per-band power to the source. Palette lock.
- **Conditions** per edit pair: `invert_only` (no lock) · `lockphase_c{0.1,0.25}_u{0.6,1.0}`
  (four phase-lock settings) · `lockpower`. Three edits, all from real photos:
  photo→oil-painting, photo→pencil-sketch, photo→watercolor.
- **The gate (`run_invert`).** Before any edit: invert each photo, regenerate with the **same**
  prompt, measure reconstruction CLIP-I to the original. High ⇒ the inverted noise is faithful ⇒
  edits are meaningful.
- **Preflight (model-free, passes).** `band_phase_swap(c=1, mag_from="A")` reconstructs the source
  (<1e-2); `mag_from="B"` returns a valid same-shape float latent.

## Findings

(SDXL, 3 real photos, 30 steps, edit CFG 5.0. Numbers from
`results/e22/invert.json` and `results/e22/edit.json`.)

**1. The gate passes — DDIM inversion round-trips where RF did not.** Reconstruction CLIP-I to the
source: **0.974 / 0.906 / 0.926** (mean **0.935**), with `noise_std ≈ 0.87–0.90`. This is the
core win of the E21→E22 pivot: on an eps-prediction model the invert→regenerate loop closes, so
the edit cells below are interpretable.

**2. Phase-band locking is a strong composition-preservation knob** (means over the 3 edits):

| condition | struct→source (CLIP-I) ↑ | edit→prompt (CLIP-T) ↑ |
|---|---|---|
| `invert_only` (baseline) | 0.602 | **0.237** |
| `lockphase_c0.1_u0.6` | 0.902 | 0.146 |
| `lockphase_c0.1_u1.0` | 0.900 | 0.169 |
| `lockphase_c0.25_u0.6` | 0.884 | 0.153 |
| `lockphase_c0.25_u1.0` | **0.892** | 0.173 |
| `lockpower` | 0.637 | 0.164 |

Locking the source's low-band phase lifts structure-preservation from **0.60 → ~0.90** — a large,
consistent gain confirming **low-band phase = layout** end-to-end on a real-image edit, not just on
generated latents (E18/E19).

**3. But it is a frontier, not a free lunch — phase-lock costs prompt adherence.** Every
`lockphase` cell **drops** edit CLIP-T from the `invert_only` baseline of **0.237** down to
**≈ 0.15–0.17**. Holding the source layout fixed constrains how far the prompt can repaint the
image, so the oil/sketch/watercolor signal weakens. Within the phase variants, **`u=1.0`
(hold the lock to the end) gives slightly better edit-CLIP than `u=0.6`** at essentially equal
structure — counter-intuitive but consistent across all three edits, suggesting the early release
lets the layout partly wash out without buying back much edit strength. `c=0.1` and `c=0.25` are
close; the smaller cut preserves marginally more structure.

**4. `lockpower` (palette lock) barely preserves structure.** Re-leveling per-band *power* to the
source leaves `struct_clip ≈ 0.64` — only just above the `invert_only` 0.60 and far below the
phase-lock ~0.90. As expected: **magnitude carries palette/texture, not layout**, so a power lock
does not hold composition. It confirms the phase/magnitude split rather than competing with it.

## Caveats & next

- **3 photos, 3 prompts** — directionally clear (the struct gap is huge and unanimous) but a small
  sample; widen the photo/prompt set before any strong quantitative claim on the edit frontier.
- **CLIP-T is a weak edit metric** for style words ("oil painting") — the absolute edit numbers
  are low even at baseline; the *relative* drop under lock is the trustworthy signal, and the
  qualitative grids (`results/e22/edit/grid_*.png`) are the real evidence of the look.
- **Isotropic bands** transfer/lock texture-energy + layout phase, **not oriented strokes** (the
  E18 caveat persists).
- The structure⇄edit trade is a **frontier to tune**, not a single setting: `c` (how much layout
  to lock) and `u` (how long) are the two dials. `c=0.1, u=1.0` is the current sweet spot
  (max structure, best edit-CLIP among the strong-lock cells).
- **Next:** a soft/decaying lock (ramp the lock strength down over steps instead of a hard
  release) to recover edit-adherence without losing the layout; and per-edit picking along the
  `(c,u)` frontier rather than a global setting.

## Reproduce

```bash
cd experiments
# 0) model-free sanity: band-lock reconstruction + shape invariants
python e22_ddim_edit.py --part preflight
# 1) THE GATE: DDIM-invert real photos, reconstruct (same prompt), report CLIP-I  -> invert.json
python e22_ddim_edit.py --part invert --num 3 --steps 30 --inv_guidance 1.0
# 2) edit: invert, regenerate with target prompt under band-lock variants          -> edit.json
python e22_ddim_edit.py --part edit \
    --num 3 --steps 30 --inv_guidance 1.0 --cfg 5.0 --cuts 0.1,0.25 --untils 0.6,1.0
# 3) dump the JSON tables
python e22_ddim_edit.py --part analyze
```

Code: `experiments/e22_ddim_edit.py` (driver: `load_sdxl`, `ddim_invert` DDIM inversion,
`BandLock` callback), reusing `experiments/spectral_ops.py` (`band_phase_swap`, `band_index_map`),
`experiments/style_ops.py` (`restyle_latent`, `latent_band_power`), and `experiments/clip_sim.py`
(`load_clip`, `clip_image_features`, `clip_text_features`, `cosine`). Results in
`experiments/results/e22/` (`invert.json`, `edit.json`, `invert/grid.png`, `edit/grid_*.png`).
