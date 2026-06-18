# E40 — RF inversion + trajectory-matched low-band spectral clamp

## TL;DR

Edit a **real image** with FLUX while preserving its **structure**, even under an aggressive
edit prompt. Three steps: (1) **RF-invert** the image to noise — integrate the velocity ODE
*backwards* (σ: 0 → 1) under the **source** caption, **recording the latent at every σ node**
(the inversion *trajectory* `traj[i]`); (2) regenerate *forward* (σ: 1 → 0) under the **edit**
prompt from that inverted noise; (3) at each step, **clamp the latent's low-frequency band
`[0, cut]` back to `traj[i]`** at the matching σ. Low bands carry coarse layout, so pinning
them to the source trajectory keeps structure while the high band follows the edit. Three clamp
**modes** reuse the repo's existing spectral primitives — `sbn` (per-band power), `phase`
(power + low-band phase lock), `adain` (per-band mean+std). Library + experiment:
`experiments/e40_spectral_invert.py`; interactive **RF inversion** tab in `spectral_demo.py`.

## Where it sits — trajectory reference, not a fixed `x0`

The repo already has structure-locking edits: **BandLock** (E21/E22) clamps a generating latent
to a **single fixed source latent `x0`** at every step. E40's difference is the **reference**:
the per-step **inversion trajectory**, aligned by σ. Because inversion and generation walk the
*same* σ grid, step `i` of the edit sits at the same noise level as `traj[i]`, so the clamp pulls
"the source as it looked at this noise level" rather than "the clean source" at every step.

```
 real image ─► VAE ─► x0 (σ=0)
        │
        │  reverse Euler, source prompt, guidance≈1     ◄── RF inversion
        ▼     x_{σ↑} = x_{σ} + Δσ · v(x_σ, σ, C_src)
   traj[i] = latent at σ=sig[i]     (recorded every node)  ──┐
        │                                                    │  reference per σ
   inverted noise (σ=1)                                      │
        │                                                    ▼
        │  forward Euler, EDIT prompt                FFT ─► low-band clamp(traj[i]) ─► iFFT
        ▼     x_{σ↓} = x_σ + Δσ · v(x_σ, σ, C_edit)          │  (sbn / phase / adain, band [0,cut])
   edited image (σ=0)  ◄──────────────────────────────────────┘
```

## Background (plain language)

- **Rectified-flow inversion** — a flow model generates by following a velocity field
  `v(x, σ, C)` from noise (σ=1) to a clean latent (σ=0). **Inversion runs it backwards**: start
  from the clean image latent and integrate the *same* ODE up to σ=1, recovering the noise that
  would regenerate it. Done under the **source** caption so the recovered noise is "about" the
  source. FLUX is guidance-distilled, so inversion is most faithful at **guidance ≈ 1**.
- **The trajectory** — at every σ node we save the *unpacked* `(1,16,128,128)` latent. `traj[i]`
  is the source's latent at noise level `sig[i]`. Storage is `steps × ~1 MB` on CPU — cheap.
- **Low band `[0, cut]`** — bin the 2-D FFT by distance from DC; `cut` is a **normalized radius**
  (`0` = DC/global tone … `1` = corner / finest detail). Low band ≈ coarse layout, global color,
  large shapes; high band ≈ texture/edges. Pinning the low band keeps *where things are*; freeing
  the high band lets the edit prompt repaint *what they are*.
- **σ-alignment** — inversion records at the same `sig` nodes the generation visits, so reference
  lookup is by index (`traj[i]` ↔ generation step `i`). Verified by the pack/unpack round-trip and
  the off-by-one in the reverse loop.

## The clamp (one new operator, three modes)

At step `i`, with the current unpacked latent `g` and the recorded reference `r = traj[i]`, pull
`g`'s low band toward `r`. All three modes are **low-band only** (bands above `cut` pass through)
and **blended by `strength ∈ [0,1]`** (`0` = no clamp = baseline). They reuse existing primitives:

- **`sbn`** — match per-`(channel, radial-band)` **mean power** (`spectral_ops.psd_match`, the
  E8/E16/E23 SBN). Target power in low bands is a geometric blend
  `cur^{1−s} · ref^{s}`; high bands target their own power (gain 1 → untouched). Magnitudes
  rescaled per band, **phase free**. DC lives in band 0, so the per-channel mean (global tone) is
  matched too.
- **`phase`** — `sbn` on magnitude **plus lock the low-band phase** to the source via
  `spectral_ops.band_phase_swap(r, out, q, mag_from="B")`: phase from `r` inside radius `≤ cut`,
  phase from the `sbn` result outside, magnitude from the `sbn` result. `q` is the bin-fraction
  that makes the quantile mask exactly `{radius ≤ cut}`. **Phase carries layout**, so this is the
  strongest structure preservation. (The phase lock is hard; `strength` governs the magnitude part.)
- **`adain`** — match per-band **mean + std** of the magnitude on **soft** bands
  (`spectral_adain.spectral_adain`, the E39 operator, `K=8` rings). Sources are `r` for rings with
  center `< cut` and `g` (identity) above, then blended `g·(1−s) + out·s`. Richer second-moment
  statistic; phase taken from the content `g`.

```
sbn   :  |F·gain[band]|, gain = sqrt(target/cur), low band only      (mean power)
phase :  sbn(mag)  ⊕  ∠F from r inside radius≤cut                     (+ low-band layout)
adain :  per-ring (|F|−μ_g)/σ_g · σ_r + μ_r, low rings only           (mean+std, soft bands)
```

Identity is exact: clamping a latent to itself (`r = g`) is a no-op for all three modes
(`‖clamp−g‖∞ ≈ 1e-6`, verified in `preflight`).

## Knobs and how they trade off

- **`cut`** (normalized radius, 0–1) — how much of the spectrum is pinned. Small `cut` pins only
  the coarsest structure/color and leaves most of the image editable; large `cut` pins more of the
  source → tighter structure, weaker edit. (One `cut` is consistent across modes because the SBN
  hard-band centers and the AdaIN soft-band centers are both normalized by the corner radius —
  `radial_norm = rr / rr.max()`.)
- **`strength`** (0–1) — clamp blend. `0` = baseline (the no-clamp edit), `1` = full clamp. For
  `phase`, governs the magnitude match only (phase lock is hard).
- **clamp `window`** (step fraction) — which steps clamp. Early/high-σ steps set coarse structure;
  clamping only early preserves layout but frees late steps for edit detail; clamping all steps
  preserves the most.
- **`inv_guidance`** — guidance during inversion; ≈ 1.0 is most faithful for distilled FLUX.

## The experiment / demo

- **`e40_spectral_invert.py`** (`--part preflight,gen,analyze`). `gen`: per source — VAE-encode
  the real image (or generate one from the source prompt), RF-invert under the source caption, then
  produce conditions **`recon`** (edit=source, no clamp — the **inversion + plumbing gate**, should
  reproduce the source), **`recon_clamp`** (edit=source, with clamp — should be *tighter* than
  `recon`, a **drift-correction** probe), **`edit_noclamp`** (plain RF-inversion edit baseline), and
  **`edit_{sbn,phase,adain}`**. `analyze` scores CLIP→edit (adherence), CLIP→source (content kept),
  pixel-distance to source, and aesthetic, and writes `results/e40/index.html`. The run also reports
  the **inverted-noise std** (≈ 1.0 = a clean Gaussian inversion; > 1 = drift, as E21 saw on SD3.5).
- **RF inversion tab** (`spectral_demo.py`, Flux). Upload an image + source caption + edit prompt;
  pick mode / `cut` / `strength` / window / guidances. **Left** = edit with no clamp; **right** = edit
  with the low-band clamp — a direct A/B of what the clamp preserves. Reuses the demo's inline FLUX
  helpers (`pack`/`unpack`/`flux_velocity`/`vae_encode`/`decode_latent`) so it stays decoupled from
  e31's heavy import chain; the only new code is the inversion loop and the clamp.

## Use cases

1. **Structure-preserving real-image editing** — keep layout/geometry, change content/style.
2. **Aggressive edits in the "low regime"** — pin only the coarsest band (`small cut`) so a large
   semantic edit still respects the source composition.
3. **Inversion drift correction** — re-imposing the recorded spectrum counteracts RF-inversion
   drift (the `recon` vs `recon_clamp` comparison).
4. **A controllable knob between content and edit** — `cut` / `strength` / window sweep the
   preservation ↔ adherence trade-off continuously.

## Caveats & next

- **Inversion fidelity is the weak link.** FLUX reverse-Euler can drift (cf. E21's failed SD3.5
  reconstruction, noise std ~1.11). The `recon` gate and the reported inverted-noise std measure it
  directly; if poor, the fallbacks are a fixed-point/implicit Euler step (as E21 tried) or an
  RF-Inversion-style controlled ODE. Not built upfront.
- **`phase` strength only affects magnitude** — the low-band phase lock is hard (phase is circular;
  a linear blend is meaningless).
- **Latent frequency ≠ pixel frequency** (the VAE), so map `cut` → perceptual scale empirically.
- **Cost** — the transformer runs `steps×` for inversion plus `steps×` per edit condition; fine on
  the NF4 (`bnb4`) path, slower than inversion-free FlowEdit (E31).

## Reproduce

```bash
# model-free asserts (pack/unpack, clamp identity, low-band restriction):
python experiments/e40_spectral_invert.py --part preflight

# generate + analyze on one real image (Flux + GPU):
python experiments/e40_spectral_invert.py --part gen,analyze --num 1 \
    --real_dir <dir-of-key.png> --mode all --cut 0.25 --strength 1.0

# interactive (Flux): open the "RF inversion" tab
python experiments/spectral_demo.py --model flux-dev
```

Reuses `psd_match` / `band_power` / `band_phase_swap` / `band_index_map` (`spectral_ops.py`) and
`soft_band_masks` / `spectral_adain` (`spectral_adain.py`); FLUX plumbing mirrors E31/E7. Contrast:
**BandLock** (E21/E22) clamps to a fixed `x0`; E40 clamps to the per-σ inversion trajectory.
