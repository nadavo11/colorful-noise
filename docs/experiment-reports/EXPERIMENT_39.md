# E39 — Spectral band-AdaIN: a soft-band frequency knob on the sampler's output

## TL;DR

A single operator, `spectral_adain` (in `experiments/spectral_adain.py`), that sits **in the
sampler — outside the network — on a velocity or a latent**. It splits the 2-D spatial
spectrum into **soft radial bands** (Gaussian rings that form a partition of unity,
`∑ₖ mₖ(ω)=1`), rewrites the **magnitude's mean *and* std per band** toward a chosen source,
and **reuses the content phase** so the output stays real. The library `spectral_adain.py` keeps
several primitives (reference matching `spectral_adain`, two-latent `band_mag_mix`, the learned
`BandSchedule` table); the interactive **Spectral AdaIN** tab in `spectral_demo.py` now exposes the
simplest member — a **single-pass self-AdaIN** with absolute, user-picked per-band targets via
`adain_affine` (modes **global** and **3-band**). Demos: `e39_spectral_adain.py` (`demo0` pixels,
`demo3` learned schedule).

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
*(Deferred — the class stays in the module; design TBD, not wired into the tab.)*

**Single-pass self-AdaIN** (`adain_affine`, what the tab uses): the same affine but with
**user-picked, absolute** targets held *constant* over the run, and `muc_k, sigc_k` taken from the
**current** latent each step (no reference, no second generation):
`|Ṽ|_k = g_k·(|V|−muc_k)/sigc_k + b_k`, with `g_k` = target std and `b_k` = target mean magnitude in
**raw latent units**. Identity is `g_k≈σ_k, b_k≈μ_k`; the tab measures and reports the live per-band
μ_k, σ_k so the targets can be calibrated. `global` uses K=1 (whole spectrum), `3-band` uses 3 soft
rings. One scalar `(g_k,b_k)` is shared across the 16 latent channels.

## The demos

- **Demo 0 — pixel sanity** (`e39_spectral_adain.py demo0 A.png B.png`, no model). `spectral_adain`
  on two RGB images: low band from B, high band + phase from A. Validates FFT / band / realness
  before any FLUX dynamics. Prints the imaginary residue (want `< 1e-3`; measured ~2.7e-8) and the
  partition-of-unity sum (`[1.0000, 1.0000]`). If you see ringing, widen the band widths.
- **Spectral AdaIN tab** (`spectral_demo.py`, Flux). Single prompt, **one** generation pass: each
  step apply `adain_affine` with picked per-band targets. **global** = one `(g,b)` over the whole
  spectrum; **3-band** = `(g,b)` per low/mid/high soft ring. The run reports the measured per-band
  μ_k, σ_k for calibration. *Caveats:* latent frequency ≠ pixel frequency (the VAE), so map bands →
  perceptual attributes empirically; and absolute targets reset the band every step while the
  natural scale drifts along the trajectory — start near the reported μ_k, σ_k. *(The earlier
  B-reference latent mixing, in-sampler x̂₀ correction, and real-image SDEdit were removed from the
  tab; the underlying primitives stay in `spectral_adain.py`, and inversion is handled separately.)*
- **Demo 3 — learned schedule** (`e39_spectral_adain.py demo3`). Fit `BandSchedule` to reshape a
  low-guidance velocity into the high-guidance (distilled-CFG) velocity, per band and per timestep,
  by minimising `‖schedule(v) − v_ref‖²` over a few `(x_t, σ)` pairs. The fitted `g`/`b` heatmaps
  are the per-(band,time) correction. **Concept-erasure angle:** fit conditioned on a concept and
  read off which `(k,t)` cells are driven toward attenuation — a candidate frequency-domain erasure
  operator. Whether concepts have clean, separable band signatures is an **open empirical
  question**; this demo is the test, not an assumption.

## Use cases

1. **Magnitude/style transfer in frequency space** without disturbing layout (phase).
2. **Single-pass per-band magnitude shaping** with picked targets (the tab; `adain_affine`).
3. **Cheap distillation** of a frequency correction into a `K×T×2` table (Demo 3).
4. **Frequency-domain concept erasure** probe (Demo 3, concept-conditioned).
5. **Two-latent / two-prompt mixing** with per-band control (`band_mag_mix`, library primitive).
6. **One-pass CFG magnitude correction** (the soft-band mean+std cousin of E37; `spectral_adain`).

## Caveats & next

- The tab's targets are **absolute** (raw latent units) and **constant** over the run, while the
  band's natural scale drifts along the trajectory — read the reported μ_k, σ_k and start near them.
- One `(g_k,b_k)` is shared across the 16 latent channels (different per-channel scales).
- Latent-band ↔ perceptual-attribute mapping is empirical (VAE); sweep to map it.
- FLUX-dev has no true `v_∅`; Demo 3 uses the distilled velocity. For a true-CFG study, run on SD3.5.
- Next: design the learned/erasure mode (Mode 3) and a separate inversion path; per-channel or
  relative-multiplier targets; full-complex (real/imag) moment matching for the erasure probe.

## Reproduce

```bash
# Demo 0 — pixel sanity, no GPU:
python experiments/e39_spectral_adain.py demo0 A.png B.png --out e39_demo0.png

# Spectral AdaIN tab — interactive (Flux): single-pass self-AdaIN, global / 3-band
python experiments/spectral_demo.py --model flux-dev        # open the "Spectral AdaIN" tab

# Demo 3 — fit the learned schedule (Flux + GPU):
python experiments/e39_spectral_adain.py demo3 --bands 4 --tbins 4 --steps 200
```

Operator lives in `experiments/spectral_adain.py`; it reuses `radial_norm` / `hybrid_split_2d`
(`latent_spectral_ops.py`) and `_restore_self_conj` (`spectral_ops.py`).
