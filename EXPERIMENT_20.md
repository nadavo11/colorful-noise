# E20 — spectral warm-start: "skip the beginning" of generation

**Idea.** The early denoising steps fix low-frequency **structure**; the late steps
fix power/detail. So if we hand the model the low-frequency content up front — an
intermediate latent whose bands are pre-set — we can re-enter the trajectory partway
and **skip the early steps**. Two uses: conditioning/style transfer (commit a
reference's structure and skip), and shaping plain generation.

**The pivot that shapes the design.** Profiling the cached E8 trajectory showed
per-band **power locks in *late*** (mean lock-in ~25–27 of 28 steps; the latent's
scale even *dips* mid-trajectory then ramps — E8's "guidance pumps power late"). So
power is **not** what the beginning sets. The early steps fix low-frequency
**structure = phase** (E12: low-band phase coherence rises first). **A warm-start
must commit low-band *phase*, not power** — the opposite of what SBN clamps, which is
why this is a genuinely new lever rather than a reskin of E19.

**Re-entry mechanism (confirmed in installed diffusers).** Rectified flow forward is
`x_t = (1-σ)·x0 + σ·ε` (`FlowMatchEulerDiscreteScheduler.scale_noise`);
`StableDiffusion3Img2ImgPipeline` maps `strength → start-step` via `get_timesteps` +
`set_begin_index`. The pipeline runs `image_processor.preprocess` *before*
`prepare_latents`, so a 16-ch latent can't go through `image=`; instead
`gen_sd3_warmstart` noises the warm-start latent itself and feeds it via `latents=`
(which bypasses `prepare_latents`). `strength=1` → x_t≈pure noise (== a full run);
`strength→0` → x_t≈x0.

## Design (`experiments/e20_warmstart.py`, parts)

- **preflight** (model-free, **passes now**): asserts `band_spectrum_split` endpoints,
  the `scale_noise` interpolation, and `color_noise` (spectrum matches target, unit
  variance); prints the cached power-lock-in (low=27, high=25 of 28) as the motivating
  prior. Run: `python e20_warmstart.py --part preflight`.
- **A. profile** — *when does each band lock in?* `S≈5` seeds × prompts with
  `RecordTraj` (captures the per-step latent); per step, cross-seed phase coherence
  `spectral_ops.phase_coherence` → R[band,t]; per-band lock-in = first t with
  R ≥ 0.9·R_final. Overlays the (late) power trajectory. **Output: which bands/steps
  the beginning actually decides** → the skippable prefix.
- **B. oracle ceiling** — commit a finished run's **true** low bands (≤ cutoff `c`,
  `band_spectrum_split(x0*, noise, c)`), re-enter at `strength`, denoise the rest.
  2-D `(c, strength)` sweep; metric = recovery of the full run (CLIP-I + latent L2).
  Yields "given the true bands up to `c`, you can skip `(1-strength)·T` steps and still
  recover the image" — the method ceiling, and a cross-check of part A (max skip should
  match the measured lock-in steps for bands ≤ `c`).
- **C. condition** — commit a **reference image's** low bands = band-controlled SDEdit;
  baseline = full SDEdit (`c=1`). Hypothesis: committing only low bands keeps structure
  while freeing the model to follow the prompt for detail (better CLIP-T / can skip
  more) than full SDEdit. `(c, strength)` sweep, structure (CLIP-I) vs prompt (CLIP-T).
- **D. noiseshape** — color **step-0** noise toward a natural-latent spectrum
  (`color_noise` = `psd_match` on white noise, from SD3.5-encoded real photos);
  colored-init vs white-init at several step counts (does a natural-spectrum start
  reach quality in fewer steps?). FreeInit/colorful-noise lineage; the regular-gen
  lever.

## Status

- **Built + offline-validated:** helpers `load_sd35_img2img`, `gen_sd3_warmstart`,
  `RecordTraj`, `warmstart_sigma`, `gen_sd3(init_latents=…)` in `e17_sd35.py`;
  `style_ops.color_noise`; the `e20_warmstart.py` driver. Preflight green; all
  construction math asserted (band-split endpoints, scale_noise, color shape_err 0.000).
- **Needs SD3.5 (cluster):** the four generation parts. SD3.5 VAE confirmed
  downloadable here; the full transformer is the larger gated pull. Watch the
  `flux-gen-ops` OOM/`/workspace` traps; one prompt per subprocess.

## Run
```bash
cd experiments
python e20_warmstart.py --part preflight                                   # now
python e20_warmstart.py --part profile  --num_prompts 1 --seeds 3 --steps 8 # smoke (cluster)
python e20_warmstart.py --part oracle   --num_prompts 1 --steps 8
python e20_warmstart.py --part condition --refs results/e18/styles --num_refs 1 --steps 8
python e20_warmstart.py --part noiseshape --step_counts 8,16 --num_prompts 1
python e20_warmstart.py --part analyze                                      # plots
```
