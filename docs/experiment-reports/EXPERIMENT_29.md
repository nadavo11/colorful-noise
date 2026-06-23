# E29 — Phase inheritance: does the seed's FFT phase determine the output latent's phase?

**The direction.** Earlier experiments in this repo established that the **phase** of a latent's 2-D Fourier transform carries the image's *structure / layout*, while the **magnitude** carries palette and texture power. That raises a sharp question about the diffusion map itself: when Stable Diffusion turns a random seed `z_T` into the final latent `z_0`, **how much of the output's phase is inherited from the seed's phase?** If structure lives in phase, and the seed already has a (random) phase field, then the seed may pre-commit the layout before a single denoising step runs. We measure this statistically, per spatial-frequency band, and then confirm it causally.

> **Naming.** The user calls the seed `z0` and the output `z1`. In diffusion convention these are `z_T` (the pure-noise seed) and `z_0` (the final denoised latent, before VAE decode). This doc uses **seed (z_T)** and **output (z_0)**.

## Background (plain language)

- **Seed `z_T` / output `z_0`.** SD1.5 denoises in a compressed latent space. The seed is a `4×64×64` Gaussian array. With the **deterministic DDIM** sampler, the map `z_T → z_0` is a fixed function — no extra randomness is injected — so any relationship between their spectra is a property of the map, not noise.
- **FFT phase vs magnitude.** For each latent channel, the 2-D FFT assigns every spatial frequency a **magnitude** (how strong that ripple is) and a **phase** (where the ripples line up). Oppenheim & Lim's classic demonstration: phase carries most of the *recognizable structure*; magnitude carries the *power envelope*.
- **Radial band.** We bin Fourier coefficients by distance from DC into `n_bins=24` rings: band 0 = lowest frequency (coarse layout), high bands = fine detail.
- **Circular correlation.** Phase is an angle (−π ≡ +π), so ordinary correlation is invalid. We use the **Jammalamadaka–SenGupta circular correlation** per Fourier bin across many seeds: +1 = output phase perfectly predicted by seed phase, 0 = unrelated. Averaged within each band → the **phase-inheritance spectrum**.
- **Permutation null.** Chance level: recompute the correlation after shuffling which output pairs with which seed. Independent phases give ≈0; a band whose correlation exceeds the null band is genuinely inheriting.
- **CFG (classifier-free guidance).** How hard the prompt steers generation; CFG=1 ≈ unsteered, higher overrides more of the seed.

## Method

- **Model.** SD1.5 (`sd-legacy/stable-diffusion-v1-5`), `DDIMScheduler` (deterministic), fp16, 512×512 → `(1,4,64,64)` latent, 50 steps.
- **Capture.** Seed `z_T` is generated explicitly (`torch.randn` with a seeded generator) and passed as `latents=`; output `z_0` is captured with `output_type="latent"`. Both are in the same scaled latent space, so their spectra are directly comparable.
- **Conditions.** An **unconditional** map (empty prompt, CFG=1) isolates the pure `z_T → z_0` function; then a **CFG sweep** `{1.0, 3.0, 7.5}` over 3 prompts, `N=64` seeds each, tests whether stronger guidance overwrites the inherited phase.
- **Metrics** (`e29_phase_ops.py`):
  1. **Phase-inheritance spectrum** — per-bin circular correlation (seed phase vs output phase) across the 64 seeds, radially averaged, with a 20-shuffle **permutation null**. The 4 self-conjugate bins (DC + Nyquist axes, whose phase is real/degenerate) are excluded.
  2. **Magnitude control** — Pearson of `log|FFT|` (does the seed predict output *power*?). Expected weak: the model drives the PSD toward natural-image statistics (cf. E23).
  3. **Phase-difference resultant** — `|mean exp(iΔφ)|` per band: is the output phase the seed phase up to a *consistent* offset (e.g. a global shift)? Secondary, since a constant offset inflates it.
  4. **Spatial Pearson** — per-channel pixel-space correlation of `z_T` vs `z_0` (sanity scalar; expected low — the link lives in frequency).
- **Causal transplant.** For seed pairs (A, B), build `A'` = A's magnitude + A's phase everywhere **except** donor B's phase inside the lowest-`c` band (via `spectral_ops.band_phase_swap`, Hermitian-symmetric, magnitude entirely A's → variance preserved, so `A'` is still a valid `N(0,1)` seed). Regenerate from A, B, A' and compute the **follow score** per band: `d_circ(A',A) / (d_circ(A',A)+d_circ(A',B))`, where `d_circ` is the mean circular distance `1−cos Δφ`. 1 = output phase fully followed the donor; 0.5 = no effect. Sweep `c ∈ {0.1, 0.25, 0.5}`.

## Findings

(SD1.5, N=64 seeds/condition, 24 bands; "low" = lowest third of bands, "high" = highest third. Permutation null ≈ **−0.001** everywhere — there is no measurement floor.)

1. **The output latent is strongly inherited from the seed — across the whole spectrum, not just phase.** Unconditionally, seed→output **phase** circular correlation is ≈ **0.40 (low) / 0.53 (high)**, vs a null of ≈ 0. The link is real and large, but it is *broad-spectrum*: it does not peak at low frequency, and it is essentially flat-to-rising with frequency.

![Phase-inheritance spectrum: circular correlation between seed phase and output phase vs radial frequency. The unconditional map (black dashed) sits at ≈0.4 (low) rising to ≈0.55 (high); raising CFG (1→3→7.5) pulls the whole curve down, most steeply at low frequency. The grey ±2σ permutation null hugs zero everywhere, so there is no measurement floor.](figs/E29/inherit_spectrum.jpg)

![Per-bin phase-correlation heatmap (unconditional, fftshifted; centre = DC). Almost every 2-D Fourier bin is warm (≈0.4–0.7), confirming the inheritance is genuinely spread across the whole spectrum rather than concentrated at a few low frequencies; the radial average of this map is the unconditional curve above.](figs/E29/inherit_heatmap.jpg)

2. **Magnitude is inherited at least as strongly as phase — contradicting our prior.** We expected the model to wash out the seed's power spectrum (it drives the PSD toward natural-image statistics, cf. E23), leaving phase as the surviving carrier. Instead the **magnitude** (log-power) Pearson is ≈ **0.48 (low) / 0.58 (high)**, slightly *above* phase, and the **raw pixel-space** correlation z_T↔z_0 is ≈ **0.76** unconditionally. So with little/no guidance the seed broadly fixes the entire output latent; the "phase = structure" framing describes what the latent *encodes*, not a phase-specific inheritance channel.

![Phase vs magnitude inheritance, unconditional map. The magnitude (log-power) Pearson curve (orange) sits at or slightly above the phase circular-correlation curve (blue) at every radial frequency, and both rise toward high frequency. If phase were a privileged carrier, the blue curve would dominate; it does not.](figs/E29/magnitude_control.jpg)

3. **Guidance erodes inheritance, preferentially in the low-frequency (composition) bands.** Low-band phase correlation falls monotonically with CFG: **0.40 (uncond) → 0.35 (CFG 1) → 0.23 (CFG 3) → 0.15 (CFG 7.5)**; spatial correlation falls **0.76 → 0.70 → ~0.61 → ~0.50**. High-frequency bands are far more stable. Reading: stronger prompts overwrite the coarse layout the seed proposes while leaving fine detail seed-determined — consistent with low frequency = composition.

4. **The link is causal, and the effect is band-localised to exactly the swapped band.** Transplanting a donor seed's phase into the base seed's lowest-`c` band (magnitude held → variance preserved, measured std = **1.000**) moves the output's phase toward the donor **inside that band**: follow score ≈ **0.66 at band 0**, holding **≈0.60–0.66 across the whole swapped region** (0.5 = no effect). Crucially the curve **steps down through 0.5 right at the edge of the swapped band** and falls to ≈0.3 in the untouched higher bands — i.e. those bands revert toward base seed A. The step tracks the cut: `c=0.1` crosses at band ≈5, `c=0.25` at band ≈9, `c=0.5` at band ≈13. This is the clean causal signature — output phase in band *k* is set by seed phase in band *k* — and it confirms the statistical inheritance is a real, per-band, controllable handle.

![Causal transplant follow score per radial band, for three swap cuts c∈{0.1,0.25,0.5}. Each curve stays well above the 0.5 "no-effect" line inside its swapped band (the output phase follows donor B) and steps down through 0.5 to ≈0.3 exactly where the swap stops (the output reverts toward base A). The step edge moves right as c grows — the effect is band-localised, not diffuse.](figs/E29/follow.jpg)

![Qualitative transplant grid (mountain-lake prompt, CFG 7.5). Columns: base seed A's image · donor seed B's image · A′ (A with B's phase in the lowest-c band). Rows: c=0.1, 0.25, 0.5. As c grows, A′ visibly inherits more of B's coarse layout (sky/water split, sun position) while keeping A's magnitude statistics — the low-band seed phase edit re-composes the scene.](figs/E29/transplant_grid.jpg)

## Caveats & next

- SD1.5 + DDIM only; other models/VAEs and stochastic samplers may differ.
- Circular correlation is a statistical link across seeds, not a per-image guarantee.
- The phase-difference resultant can be high for a trivial reason (a consistent global shift), so the circular correlation is the headline metric.
- Phase and magnitude are not fully independent through a nonlinear VAE — "phase = structure" is a strong tendency, not a law.
- The inheritance being broad-spectrum (magnitude ≈ phase, spatial r ≈ 0.76) at low guidance is itself the headline correction to our prior — phase is *what structure lives in*, but it is not a privileged *inheritance channel*; the seed fixes the whole latent.
- Next: repeat on Flux/SD3.5 (16-channel, rectified flow) to test architecture-independence; connect the strong seed→output determinism (and its CFG-driven, low-frequency erosion) to the "golden noise" / seed-trace line (E25–E28), where seed edits that change low-band phase should have the most compositional leverage at low CFG.

## Verdict

**MAPPED.** Under deterministic DDIM the seed fixes the *whole* output latent at low guidance (phase circular corr ≈0.40–0.53, magnitude ≈as strong, spatial r ≈0.76, null ≈0) — *not* a phase-specific channel, the correction to our prior. Guidance erodes it most in the low-frequency composition bands (low-band phase corr 0.40→0.15 over CFG 1→7.5). The link is **causal and band-localised**: a low-band seed-phase transplant moves output phase to the donor *inside* the swapped band (follow ≈0.6–0.66) and leaves the rest near the base (≈0.3). Takeaway: **the seed fixes the whole output spectrum at low CFG, per-band and controllably**, which is the lever the seed-edit line (E25–E28) acts on.

## Artifacts

- **Driver:** `experiments/e29_phase_inherit.py` (parts `preflight | gen | analyze | transplant`).
- **Metrics:** `experiments/e29_phase_ops.py` (Jammalamadaka–SenGupta circular correlation, permutation null, log-magnitude Pearson, phase-diff resultant, follow score), reusing `experiments/spectral_ops.py` (`band_index_map`, `band_phase_swap`, `phase_only`, `_SELF_CONJ`) and `experiments/common.py`.
- **Results location:** local checkout `experiments/results/e29/` — `report.json` (per-condition spectra + null + magnitude/spatial controls), `transplant.json` (per-band follow scores), `plots/*.png`, `examples/transplant_grid.png`, `index.html`. Not on `/storage` (no `roadmap_results/E29/` archive yet; `/storage` is read-only for this user under SSHFS, so the full-res archive must be pushed centrally).
- **Figures (this report):** `figs/E29/{inherit_spectrum, inherit_heatmap, magnitude_control, follow, transplant_grid}.jpg`, all derived directly from `results/e29/`.

## Reproduce

```bash
python experiments/e29_phase_inherit.py preflight     # smoke test + metric sanity
python experiments/e29_phase_inherit.py gen           # N=64 seeds × (uncond + 3 prompts × 3 CFG)
python experiments/e29_phase_inherit.py analyze       # spectra, null, controls, plots, report.json
python experiments/e29_phase_inherit.py transplant    # causal follow-score + grid
python experiments/e29_site.py                        # self-contained results/e29/index.html
# knobs: E29_N, E29_SIZE, E29_STEPS, E29_NBINS, E29_NPERM, E29_TPAIRS
```

Code: `experiments/e29_phase_inherit.py` (driver), `experiments/e29_phase_ops.py` (circular-correlation + follow-score metrics), `experiments/e29_site.py` (HTML), reusing `experiments/spectral_ops.py` (`band_index_map`, `band_phase_swap`, `phase_only`, `_SELF_CONJ`) and `experiments/common.py`.
