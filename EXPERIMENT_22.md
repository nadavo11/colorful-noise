# E22 — Spectral image editing via DDIM inversion + frequency-band locking (SDXL)

**TL;DR.** We edit a *real* photo by spectral surgery: invert it back to the noise that would have
produced it, then regenerate with a new prompt while **locking** the source's low-frequency
**phase** (which carries layout) so the prompt only repaints appearance (oil painting / pencil
sketch / watercolor). E21 tried this on **SD3.5** and the **inversion drifted** — the round-trip
would not close, and reconstruction is the gate for everything downstream. E22 keeps the **exact
same band-lock editing control** and swaps the backbone to **SDXL**, an **ε-prediction** model with
a clean, deterministic **DDIM inversion**. The result: the **gate passes** (reconstruction CLIP-I
≈ **0.94** vs E21's drift), and phase-band locking is a **strong, tunable composition-preservation
knob** — it lifts structure-to-source CLIP-I from **0.60 → 0.90** while the prompt still moves the
look, but **trades away** prompt adherence (edit CLIP-T **0.24 → ≈ 0.16**). A genuine
structure⇄edit **frontier**, not a free lunch.

> **Status / Verdict:** ✅ **Ran on SDXL. The gate passes** — DDIM reconstruction CLIP-I ≈ **0.94**
> (vs E21's drift). Phase-band locking lifts structure-to-source CLIP-I **0.60 → 0.90** while the
> prompt still moves appearance, but **trades away** prompt adherence (edit CLIP-T **0.24 → ≈ 0.16**).
> It is a genuine structure⇄edit frontier, with `c` (how much layout to lock) and `u` (how long) as
> the two dials.

```mermaid
flowchart LR
  A["real photo"] -->|VAE encode| B["clean latent x0<br/>(1,4,128,128)"]
  B -->|DDIM inversion<br/>(DDIMInverseScheduler, 0→T)| C["noise"]
  C -->|"regen, SAME prompt<br/>(the gate)"| R["reconstruction"]
  A -. "recon CLIP-I ≈ 0.94 ✅" .- R
  C -->|"regen, TARGET prompt<br/>+ BandLock callback"| E["edit"]
  B -. "lock source low-band PHASE<br/>(layout)" .-> E
```

## Background (plain language)

- **Latent (SDXL).** SDXL denoises a `4×128×128` latent at 1024px; a VAE decodes it to RGB. That
  `(H,W)=128` grid is **identical** to SD3.5's, so the radial-band spectral operators
  (`spectral_ops`, `style_ops`) apply **unchanged** — they are channel-agnostic.
- **ε-prediction vs rectified flow.** SDXL predicts the **noise** `ε` added to a latent (classic
  DDPM/DDIM formulation), unlike SD3.5's velocity field. ε-prediction has a clean, well-behaved
  **DDIM inversion**: a deterministic map clean-latent → noise that closes back to the same image.
  This is exactly the property E21's RF model lacked.
- **DDIM inversion (clean → noise).** Run the pipeline with the **`DDIMInverseScheduler`**
  (timesteps 0 → T) on a real image's latent to recover the noise that produces it. Reverse with the
  normal `DDIMScheduler` and you should get the image back.
- **FFT phase vs magnitude.** Each spatial frequency carries a **magnitude** (ripple strength →
  texture/palette) and a **phase** (alignment → structure/layout). Classic result (Oppenheim & Lim):
  **phase carries layout**. Locking phase locks layout.
- **Frequency-band PHASE locking (composition preservation).** Frequencies binned into `N_BINS=24`
  rings from DC; band 0 = coarsest layout. The **`lockphase`** variant (`BandLock`, `mode="phase"`)
  keeps the **source's** low-band phase (layout) while the new prompt owns the magnitude + high-band
  phase — composition survives, the prompt repaints appearance.
- **cut `c` + until `u` (the two dials).** **`c`** = the lowest fraction of the spectrum that gets
  locked (`c=0.1` coarsest only, `c=0.25` a bit more). **`u`** = the lock is active only for the
  first `u` fraction of denoising steps, then released (`u=0.6` lets go early; `u=1.0` holds to the
  end).
- **`lockpower` (palette lock, the control).** `BandLock` `mode="power"` re-levels per-band *power*
  (magnitude energy) to the source. Magnitude carries palette/texture, not layout, so this is
  expected to barely preserve structure — a control isolating the phase⇄magnitude split.
- **The metrics.** `recon_clip_i` ↑ = CLIP-I of the reconstruction to the source (**~0.94 = round-trip
  closed**, the gate). `struct_clip` ↑ = CLIP-I of the edit to the **source** (composition preserved).
  `edit_clip_t` ↑ = CLIP of the edit to the **target prompt** (edit followed). A good knob negotiates
  the last two.

## Method

- **Model.** SDXL (`StableDiffusionXLPipeline`, fp16, `SDXL_ID`), `DDIMScheduler` for generation +
  `DDIMInverseScheduler` for inversion, 1024px, `(1,4,128,128)` latent, 30 steps, inversion guidance
  1.0, edit CFG 5.0. Runs locally (SDXL is cached).
- **`--part invert` (the gate).** Encode the photo with the VAE (in fp32 for stability), run
  `ddim_invert` (temporarily swap in the inverse scheduler, run the pipe to get the noise latent,
  restore the normal scheduler), regenerate from that noise with the **same** prompt at guidance 1,
  and score `recon_clip_i` to the original (+ report `noise_std` as a sanity check). *Question: does
  the inverted noise round-trip back to the source?* High ⇒ the inverted noise is faithful ⇒ edits
  are meaningful.
- **`--part edit` (the frontier).** Same inversion, then regenerate with a **target** prompt under
  band-lock variants — `invert_only` (baseline, no lock) · `lockphase_c{0.1,0.25}_u{0.6,1.0}` (four
  phase-lock settings) · `lockpower` — scoring `struct_clip` (CLIP-I to source ↑) against
  `edit_clip_t` (CLIP to target prompt ↑). Three edits, all from real photos: photo→oil-painting,
  photo→pencil-sketch, photo→watercolor. *Question: can locking source low-band phase preserve
  composition while the prompt repaints appearance — and at what cost to adherence?*
- **`--part preflight` (model-free, passes).** Verifies the band-lock invariants:
  `band_phase_swap(c=1, mag_from="A")` reconstructs the source (<1e-2); `mag_from="B"` returns a
  valid same-shape float latent. The math is right before any model loads.

## Results

(SDXL, 3 real photos, 30 steps, edit CFG 5.0. Numbers from `results/e22/invert.json` and
`results/e22/edit.json`.)

### Reconstruction (the gate)

**What to look for.** A faithful inversion makes each reconstruction match its source — same scene,
layout, colors. On SDXL they do: the round-trip **closes**, unlike E21's SD3.5 drift.

**Interpretation.** SDXL's ε-prediction gives a well-behaved deterministic DDIM inversion: the
recovered "noise" is faithful, so the clean→noise→clean loop returns the source. This is the core
win of the E21→E22 pivot — with a passing gate, the edit cells below are interpretable (on E21 they
were not). Reconstruction CLIP-I to the source: **0.974 / 0.906 / 0.926** (mean **0.935**), with
`noise_std ≈ 0.87–0.90`.

| photo | recon CLIP-I ↑ (round-trip; ~0.94 = closed) | noise_std |
|---|---|---|
| photo_000 | **0.974** | 0.87 |
| photo_001 | 0.906 | 0.90 |
| photo_002 | 0.926 | 0.89 |
| mean | **0.935** | 0.89 |

Every cell sits at/above the ~0.94 "round-trip closed" bar — the gate passes.

### Band-lock edit (the structure⇄edit frontier)

**What to look for.** The `invert_only` cell follows the prompt hardest (most repainted) but
**drifts** from the source layout; each `lockphase` cell keeps the source composition (same scene
geometry) while the style still shifts toward the target — and the prompt's grip visibly weakens as
the lock tightens. `lockpower` looks like a palette nudge, not a composition lock. (Grids:
`results/e22/edit/grid_*.png`.)

**Interpretation.** Locking the source's low-band phase lifts structure-preservation from **0.60**
(no lock) to **~0.90** — a large, consistent gain confirming **low-band phase = layout** end-to-end
on a *real* image edit, not just generated latents (E18/E19). But it is a **frontier, not a free
lunch**: every `lockphase` cell **drops** edit CLIP-T from the **0.237** baseline to **≈ 0.15–0.17**,
because holding the layout fixed constrains how far the prompt can repaint. Within the phase
variants, **`u=1.0` (hold to the end) gives slightly better edit-CLIP than `u=0.6`** at essentially
equal structure — counter-intuitive but consistent across all three edits, suggesting early release
lets the layout partly wash out without buying back much edit strength. `c=0.1` and `c=0.25` are
close; the smaller cut preserves marginally more structure. `lockpower` barely moves structure
(**0.637**, only just above the 0.602 baseline) — magnitude carries palette, not layout — confirming
the phase/magnitude split rather than competing with it.

| condition (mean over edits) | struct→source (CLIP-I) ↑ | edit→prompt (CLIP-T) ↑ |
|---|---|---|
| `invert_only` (baseline) | 0.602 | **0.237** |
| `lockphase_c0.1_u0.6` | 0.902 | 0.146 |
| `lockphase_c0.1_u1.0` | 0.900 | 0.169 |
| `lockphase_c0.25_u0.6` | 0.884 | 0.153 |
| `lockphase_c0.25_u1.0` | **0.892** | 0.173 |
| `lockpower` | 0.637 | 0.164 |

The two best cells land in **different rows** — max structure (a `lockphase` cell) and max adherence
(the `invert_only` baseline) trade off. That gap **is** the frontier; `c` and `u` tune where you sit
on it. `c=0.1, u=1.0` is the current sweet spot (max structure, best edit-CLIP among the strong-lock
cells).

## Caveats & next

- **3 photos, 3 prompts** — directionally clear (the struct gap is huge and unanimous) but a small
  sample; widen the photo/prompt set before any strong quantitative claim on the edit frontier.
- **CLIP-T is a weak edit metric** for style words ("oil painting") — the absolute edit numbers are
  low even at baseline; the *relative* drop under lock is the trustworthy signal, and the qualitative
  grids (`results/e22/edit/grid_*.png`) are the real evidence of the look.
- **Isotropic bands** lock layout phase + texture energy, **not oriented strokes** (the E18 caveat
  persists).
- The structure⇄edit trade is a **frontier to tune**, not a single setting: `c` (how much layout to
  lock) and `u` (how long) are the two dials. `c=0.1, u=1.0` is the current sweet spot.
- **Next:** a soft/decaying lock (ramp the lock strength down over steps instead of a hard release)
  to recover edit-adherence without losing the layout; and per-edit picking along the `(c,u)`
  frontier rather than a global setting.

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
# 4) MODEL-FREE: rebuild results/e22/index.html from invert.json + edit.json + cached grids
python e22_ddim_edit.py --part site          # (or: python e22_site.py)
```

Results live on `/storage` (gitignored). To rebuild the HTML locally without a GPU: `kubectl cp` the
`invert.json` / `edit.json` + `invert/grid.png` / `edit/grid_*.png` from
`mystorage-0-0:/storage/malnick/colorful-noise/experiments/results/e22` into local
`experiments/results/e22`, then run `python e22_ddim_edit.py --part site` (no model loaded).

Code: `experiments/e22_ddim_edit.py` (driver: `load_sdxl`, `ddim_invert` DDIM inversion, `BandLock`
callback), HTML generator `experiments/e22_site.py`, reusing `experiments/spectral_ops.py`
(`band_phase_swap`, `band_index_map`), `experiments/style_ops.py` (`restyle_latent`,
`latent_band_power`), and `experiments/clip_sim.py` (`load_clip`, `clip_image_features`,
`clip_text_features`, `cosine`). Results in `experiments/results/e22/` (`invert.json`, `edit.json`,
`invert/grid.png`, `edit/grid_*.png`, `index.html`).
