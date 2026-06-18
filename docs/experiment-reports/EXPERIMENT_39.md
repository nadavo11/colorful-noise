# E39 — Spectral band-AdaIN: a soft-band frequency knob on the sampler's output

## TL;DR

A single operator, `spectral_adain` (in `experiments/spectral_adain.py`), that sits **in the
sampler — outside the network — on a velocity or a latent**. It splits the 2-D spatial
spectrum into **soft radial bands** (Gaussian rings that form a partition of unity,
`∑ₖ mₖ(ω)=1`), rewrites the **magnitude's mean *and* std per band** toward a chosen source,
and **reuses the content phase** so the output stays real. One primitive covers four uses:
match a reference's frequency statistics, mix two latents, anchor a velocity's low band to the
running clean estimate, and run real-image SDEdit with a structure lock — plus a tiny
**learned** per-band/per-time table that distills a frequency correction into a few hundred
parameters (and doubles as a concept-erasure probe). Demos: `e39_spectral_adain.py`
(`demo0` pixels, `demo3` learned schedule) and a **Spectral AdaIN** tab in `spectral_demo.py`
(latent mixing, in-sampler correction, real-image SDEdit).

## Where it sits — and why it is *not* AdaLN

A FLUX/DiT block has an internal **AdaLN**: a timestep+pooled-text vector `vec` drives
per-block shift/scale/gate in token-channel space. That is a **semantic / timestep** knob.
This operator is the orthogonal **frequency** knob — it never enters the network:

```
 t, g, c_pooled ─► vec ─► [per-block mod MLP] ─► shift/scale/gate     ◄── internal AdaLN
 x_t (tokens) ─► [transformer] ─► v_θ
                                     │
                              FFT ─► band-AdaIN(sources) ─► iFFT       ◄── spectral AdaIN (this)
                                     │  (latent freq space, in the sampler)
                              x_{t-Δ} = x_t + Δ · v_corr
```

They are not interchangeable: AdaLN's `vec` path is the natural lever for *concept* steering;
spectral AdaIN is the right object for *geometry / frequency*-aware manipulation.

## Background (plain language)

- **Radial frequency band** — the 2-D FFT of a `(C,H,W)` field; binned by distance from DC
  (`0` = global tone … `1` = corner / finest detail). Symmetric under `ω → −ω`.
- **Soft bands / partition of unity** — overlapping Gaussian rings `mₖ(ω)=exp(−½((r−cₖ)/wₖ)²)`,
  normalised so `∑ₖ mₖ = 1`. Wider widths = softer overlap = less ringing. Hard bands (the
  existing `band_index_map`) have sharp edges that ring; soft bands blend convexly.
- **AdaIN in Fourier space** — per band, normalise the magnitude `|V|` to zero-mean/unit-std
  using **mask-weighted** moments, then rewrite to a source's mean+std. Phase (which carries
  layout/structure) is taken from the content and never touched.
- **Realness** — every mask is a function of `|ω|`, so it is symmetric under `ω → −ω`; with the
  content phase reused the spectrum is Hermitian and `ifft2(.).real` loses only ~1e-8 (verified
  by `demo0`). The 4 self-conjugate bins (DC + Nyquist axes) are restored from the content via
  `spectral_ops._restore_self_conj`, keeping realness and the DC mean exact.

### How it differs from the repo's existing SBN

The E8/E16/E23 SBN (`spectral_ops.psd_match`, `latent_spectral_ops.sbn_clamp`) uses **hard**
radial bands and matches **mean power only**. Spectral AdaIN uses **soft** bands (so moments
are mask-weighted convex blends) and matches the **full mean+std** of the magnitude. Same FFT
convention (unshifted, DC at `[0,0]`); normalised radii reuse `latent_spectral_ops.radial_norm`.

## The operator (derivation)

For content `V = FFT(v)`, soft masks `M=(K,H,W)`, and per-band sources `Sₖ`:

```
muc_k, sigc_k = mask-weighted (mean, std) of |V| under m_k          # band_moments(|V|, M)
mus_k, sigs_k = mask-weighted (mean, std) of |S_k| under m_k
|Ṽ|_k = sigs_k · (|V| − muc_k)/sigc_k + mus_k        (per band, then clamp ≥ 0)
|Ṽ|   = ∑_k m_k · |Ṽ|_k                               (convex reassembly)
out   = real( iFFT( |Ṽ| · e^{i∠V} ) )                 (content phase; restore self-conj bins)
```

`sources` is a length-K list, one per band, which makes every use the *same* call:

- `sources = [v, v, …]` → identity (sanity check: `‖adain−v‖∞ < 5e-6`).
- `sources = [ref, ref, …]` → moment-matching (the "matching" formulation, no free params).
- `sources = [A_low, v_high, …]` → a mix (e.g. low anchored, high free).

**Two-latent mixing** (`band_mag_mix`): `|Ṽ|_k = αₖ|Aₖ| + (1−αₖ)|Bₖ|`, phase from the content
`A`. Low-band `α→0` imports B's color/global layout; high-band `α→1` keeps A's texture. Phase is
**not** interpolated (it is circular — a linear blend is meaningless), so it is always taken from
the designated content latent.

**Learned schedule** (`BandSchedule`): replace the reference-derived stats with a learnable
per-band/per-time affine `|Ṽ|_k = g_k(t)·(|V|−muc_k)/sigc_k + b_k(t)`. The params are a tiny
table of size `K × T_bins × 2` — a **learned frequency-shaping schedule over the trajectory**.

## The demos

- **Demo 0 — pixel sanity** (`e39_spectral_adain.py demo0 A.png B.png`, no model). `spectral_adain`
  on two RGB images: low band from B, high band + phase from A. Validates FFT / band / realness
  before any FLUX dynamics. Prints the imaginary residue (want `< 1e-3`; measured ~2.7e-8) and the
  partition-of-unity sum (`[1.0000, 1.0000]`). If you see ringing, widen the band widths.
- **Demo 1 — latent band-AdaIN** (Spectral AdaIN tab). During generation, drive A's low-band
  magnitude toward prompt B's latent (B's palette/tone), keep A's high band + phase. *Caveat:*
  latent frequency ≠ pixel frequency (the VAE), so map bands → perceptual attributes empirically
  before trusting pixel intuition.
- **Demo 2 — in-sampler correction** (tab). Manual Euler loop: each step anchor the velocity's low
  band to the running clean estimate **x̂₀ = x_t − σ·v** (high band = identity). *Caveat:* FLUX-dev
  is guidance-**distilled** — there is no two-pass true `v_∅`, so this is the x̂₀-anchor form, not a
  true-CFG match. (On a real-CFG model the same call with `sources=[v_∅,…]` is the true-CFG
  frequency correction — that is the E37 velocity SBN, of which this is the soft-band, mean+std
  generalisation.)
- **Demo 3 — learned schedule** (`e39_spectral_adain.py demo3`). Fit `BandSchedule` to reshape a
  low-guidance velocity into the high-guidance (distilled-CFG) velocity, per band and per timestep,
  by minimising `‖schedule(v) − v_ref‖²` over a few `(x_t, σ)` pairs. The fitted `g`/`b` heatmaps
  are the per-(band,time) correction. **Concept-erasure angle:** fit conditioned on a concept and
  read off which `(k,t)` cells are driven toward attenuation — a candidate frequency-domain erasure
  operator. Whether concepts have clean, separable band signatures is an **open empirical
  question**; this demo is the test, not an assumption.
- **Demo 4 — real-image SDEdit** (tab, image upload). Encode the photo, noise to level `t`, denoise
  while **re-locking the low band to the real latent** each step (`freq_mixed_init`: low from x₀,
  high from the evolving latent) — keep layout, regenerate texture. A one-shot variant uses a
  frequency-mixed init then denoises. *Caveat:* x̂₀ is a weak anchor in the high-noise regime; use
  the timestep window to act later.

## Use cases

1. **Magnitude/style transfer in frequency space** without disturbing layout (phase).
2. **One-pass CFG magnitude correction** (the soft-band mean+std cousin of E37).
3. **Cheap distillation** of a frequency correction into a `K×T×2` table (Demo 3).
4. **Frequency-domain concept erasure** probe (Demo 3, concept-conditioned).
5. **Two-latent / two-prompt mixing** with per-band control (`band_mag_mix`).
6. **Real-image editing** (SDEdit) with an explicit layout-vs-texture frontier (Demo 4).

## Caveats & next

- FLUX-dev has no true `v_∅`; Demo 2/3 use the distilled velocity (x̂₀ anchor / low-vs-high
  guidance). For a true-CFG study, run on SD3.5 (the Velocity tab's model).
- Latent-band ↔ perceptual-attribute mapping is empirical (VAE); start from Demo 1 sweeps.
- Demo 2/4 run at 1024px (reuse `e31_flowedit_freq`'s velocity/sigmas/pack helpers).
- Next: sweep `(t, cut)` to trace the SDEdit layout-vs-texture frontier; fit Demo 3 per-prompt vs
  global; test full-complex (real/imag) moment matching for the erasure probe.

## Reproduce

```bash
# Demo 0 — pixel sanity, no GPU:
python experiments/e39_spectral_adain.py demo0 A.png B.png --out e39_demo0.png

# Demos 1/2/4 — interactive (Flux):
python experiments/spectral_demo.py --model flux-dev        # open the "Spectral AdaIN" tab

# Demo 3 — fit the learned schedule (Flux + GPU):
python experiments/e39_spectral_adain.py demo3 --bands 4 --tbins 4 --steps 200
```

Operator lives in `experiments/spectral_adain.py`; it reuses `radial_norm` / `hybrid_split_2d`
(`latent_spectral_ops.py`) and `_restore_self_conj` (`spectral_ops.py`).
