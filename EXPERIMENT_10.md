# E10 — CFG inflates spectral power (the SBN motivation)

**Status / Verdict:** RAN (Flux, true-CFG sweep w∈{1,1.5,2,3,4,5}, 6 classes × 3 seeds,
20 real photos). **Confirmed:** the latent's spectral power rises monotonically with CFG
(≈3× over w=1→5), and *real photographs sit at standard guidance* — the unguided field is
spectrally *weaker* than real, full guidance overshoots. This is the foundational fact the
whole Spectral Band Normalization (SBN) line clamps back, and the seed for E23's switch to a
**real**-photo target.

**The direction.** Classifier-free guidance (CFG) is the knob that makes a diffusion model
obey its prompt — but it is an *inference-time extrapolation* that is not in the training
loss, so cranking it pushes the trajectory off the data manifold. This experiment asks the
concrete question: *what does CFG do to the latent's frequency content?* The answer — CFG
**inflates** the latent's per-frequency power, especially at low frequency, above the
natural/real level — is the motivation for re-leveling that power back down (SBN, E9) and,
later, for targeting the **real-photo** spectrum directly instead of the cfg=1 proxy (E23,
which summarizes this finding as "Higher cfg also inflates low-frequency power above the
natural level").

## Background (plain language)

- **Flow model / velocity field.** Flux is trained by flow-matching: the transformer
  predicts a velocity `v_θ(x, t)` that transports noise to data. Integrating the trained
  field is the `w = 1` case.
- **CFG (classifier-free guidance), true CFG.** At inference you extrapolate between the
  conditional and unconditional velocities: `ṽ = v_u + w·(v_c − v_u)`. `w = 1` integrates
  the trained field; `w > 1` pushes off the data manifold to obey the prompt harder. We use
  diffusers' **real two-pass CFG** (`true_cfg_scale` + an empty `negative_prompt`) and hold
  Flux's *distilled* `guidance_scale` fixed at a neutral 1.0, so the sweep isolates the
  cfg-equation effect, not Flux's distilled-guidance embedding.
- **Latent.** Flux denoises a compressed 16-channel 128×128 array; the VAE turns it into the
  1024×1024 image. All spectral metrics here are on the (unpacked) latent.
- **Fourier / radial PSD / band.** Any latent is a sum of spatial-frequency waves (low =
  coarse structure, high = fine texture). Binning frequencies into radial rings and taking
  power-per-ring gives the **radial PSD** — the curve of how much coarse vs fine content the
  latent carries. `LOW_CUT = 0.25` of the radial frequency splits "low band" from "high".
- **Spectral metrics (intensity axis).** `power` = mean |X|² (Parseval: equals mean squared
  latent value), the headline; `lat_std` = latent standard deviation; `spec_norm` = the literal
  top singular value σ_max of each channel (matrix 2-norm), averaged; `low_power` = mean power in
  the low radial band (the steepest riser — inflation is low-frequency-heavy); `low_frac` =
  fraction of total power in the low band (CFG tilts the spectrum toward coarse structure). No
  single "good" direction — the point is that *all* rise above the real level once `w` is large.
- **Image metrics (decoded-pixel correlate).** From the saved PNGs: `rms_contrast` = grayscale
  std (↑ = more contrasty); `saturation` = mean per-pixel (max−min)/max over channels (↑ = more
  saturated); `hf_frac` = fraction of image FFT power above the 0.25 spatial-frequency ring (↑ =
  more fine detail). These show the "over-cooked high-CFG look" in pixel space.
- **Real reference.** Natural photographs encoded through the *same* Flux VAE into the
  generation latent space, so generated and real latents live in one comparable space.

## Method

- **Generation (`--part gen`).** For each class × cfg × seed, run a true-CFG generation and
  cache the image + the unpacked fp32 latent. Text encoders are pre-encoded then dropped to
  avoid OOM on a shared box; runs resume for free.
- **Real pool (`--part download`, `--part coco`).** `download` fetches `--n_real` seeded
  `picsum.photos` images (reproducible). `coco` grows the pool with the first `--n_coco`
  MS-COCO val2017 photos (the cached zip is extracted lazily) — this is the same real pool
  E23 later builds its real-PSD target from.
- **Encode real (`--part real`).** Center-crop-square each photo (so a non-square aspect
  doesn't warp the radial PSD), VAE-encode, and invert the decode convention
  (`lat = (z − shift)·sf`) into the generation latent space; stack to `real_latents.pt`.
- **Analyze (`--part analyze`).** Per-latent spectral metrics + per-image metrics from the
  PNGs, aggregated per cfg and for the real set; writes `cfg_spectral.json` and the
  `cfg_power.png` / `cfg_psd.png` plots.
- **Site (`--part site`).** Model-free: re-templates `results/e10/index.html` from
  `cfg_spectral.json` + the cached plots, loading no model and re-scoring nothing. If
  `cfg_spectral.json` is absent it prints `run --part analyze first` and exits — the page is
  rebuildable anywhere the JSON + plots are present.

## Findings

(Flux, true-CFG sweep, 6 classes × 3 seeds = 18 latents/cfg; real = 20 picsum photos. All
numbers from `results/e10/cfg_spectral.json`.)

**1. Spectral power rises monotonically with CFG — about 3× over the sweep.** Mean Fourier
power climbs `0.636 → 0.784 → 0.922 → 1.224 → 1.564 → 1.928` for `w = 1 → 1.5 → 2 → 3 → 4 →
5` (≈ **3.03×**). Every other intensity metric tracks it: latent std `0.775 → 1.318`,
spectral norm `69.0 → 126.7`, and low-band power `21.4 → 77.7` (**≈ 3.6×**, the steepest —
the inflation is concentrated at low frequency). The low-frequency *fraction* of total power
also creeps up (`0.987 → 0.995`), i.e. CFG tilts the spectrum toward coarse structure.

**2. Real photographs sit at standard guidance — the unguided field is *too weak*.** The
20-photo real set has power **1.229**, std **1.074**, spectral norm **99.6**, low-band power
**43.7** — essentially the `w = 3` row (power 1.224, std 1.069, spec-norm 100.6, low-power
46.3). So the trained field at `w = 1` is spectrally *weaker* than real data, and normal
guidance (`w ≈ 3`) is roughly where the generated spectral scale **crosses** the real one;
above that, CFG overshoots real.

| cfg w | power | lat_std | spec_norm | low_power |
|---|---|---|---|---|
| 1 | 0.636 | 0.775 | 69.0 | 21.4 |
| 2 | 0.922 | 0.933 | 86.2 | 33.6 |
| 3 | 1.224 | 1.069 | 100.6 | 46.3 |
| 4 | 1.564 | 1.197 | 114.1 | 61.5 |
| 5 | 1.928 | 1.318 | 126.7 | 77.7 |
| **real (20)** | **1.229** | **1.074** | **99.6** | **43.7** |

**3. The image-space correlate is contrast/saturation.** As CFG rises the decoded images get
more contrasty and saturated (RMS contrast `0.164 → 0.197 → 0.204`, saturation `0.294 →
0.396 → 0.464` for `w = 1, 3, 5`) — the familiar "over-cooked high-CFG look" is the
image-space shadow of the low-band power inflation. (Real-set image metrics are absent: only
latents are stored for the real pool, no decoded PNGs to score.)

**4. Why this motivates SBN.** The elevated spectral scale CFG produces is *roughly* where
natural data sits at `w ≈ 3`, but past that it overshoots, and the inflation is band-shaped
(low-frequency-heavy), not a flat gain. That is exactly the failure a **per-band** power
clamp fixes: SBN re-levels each (channel, band) back toward a calm reference. E10 used the
cfg=1 output as that reference; the bimodal residual it leaves — too much low-frequency,
too little high — is what E23 later corrects by targeting the **real** PSD instead.

## Caveats & next

- **cfg=1 is a *proxy* for "natural", not natural itself.** Real data sits near `w ≈ 3`, not
  `w = 1`, so clamping to cfg=1 actually moves *away* from real at the low bands — the
  measured motivation for E23's real-target SBN.
- **Isotropic, radial-only.** Bands carry texture-energy + palette, not oriented structure;
  the metric cannot see directional content.
- **Small/seeded sample.** 18 latents per cfg and 20 picsum photos — enough for the
  monotone trend and the crossing point, not for tight per-class claims. E23 rebuilds the
  real target from 500 MS-COCO photos.
- **VAE-space comparison.** Generated vs real are only comparable because both are encoded by
  the *same* VAE; the numbers are not transferable across models (cf. E17, which needs an
  SD3.5-VAE real reference, not this Flux one).
- **Next:** package the band clamp as a method (E9/SBN), then replace the cfg=1 proxy with a
  measured real-photo target (E23).

## Reproduce

```bash
# real pool: 20 seeded picsum photos (+ optional MS-COCO val2017 for the E23 target)
python e10_cfg_spectral.py --part download --n_real 20
python e10_cfg_spectral.py --part coco --n_coco 500        # grows the real pool
# true-CFG sweep generation (6 classes × 3 seeds), encode reals, analyze
python e10_cfg_spectral.py --part gen,real,analyze \
    --cfgs 1,1.5,2,3,4,5 --num_classes 6 --seeds 3 --steps 28 --guidance 1.0
# or all parts at once
python e10_cfg_spectral.py --part download,gen,real,analyze
# rebuild the HTML report anywhere (model-free; needs cfg_spectral.json + plots)
python e10_cfg_spectral.py --part site      # or: python e10_site.py
```

Code: `experiments/e10_cfg_spectral.py` (driver: `run_download`/`run_coco`/`run_gen`/
`run_real`/`run_analyze`/`run_site`) + `experiments/e10_site.py` (the HTML builder), reusing
`experiments/spectral_ops.py` (`radial_psd`), `experiments/e7_flux_phase.py` (`load_flux_vae`),
`experiments/e9_bandnorm_classes.py` (`CLASSES`, `image_metrics`, `agg`), and
`experiments/e27_site.py` (`data_uri`). Artifacts: `results/e10/cfg_spectral.json`,
`results/e10/real_latents.pt`, `results/e10/index.html`,
`results/e10/plots/{cfg_power,cfg_psd}.png`.
