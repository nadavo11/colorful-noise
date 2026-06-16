# E27 — A single "concept direction" in the diffusion seed (CLIP→latent pullback)

**Idea.** A diffusion model starts from a random **seed** `z ~ N(0, I)` (for SDXL a
`4×128×128` latent, dimension `d = 65536`). The seed leaves traces in the output, so:
can we compute **one direction per prompt** in seed space that means "more of this prompt"
and simply **add** it to any seed — a linear *concept/steering vector* — instead of
optimizing every seed (E25/E26)? We keep the edited seed Gaussian by re-standardizing to
zero-mean/unit-variance, so it stays exactly on the `‖z‖ = √d = 256` sphere a real seed
lives on.

**The two-stage construction (and why it's one backward pass).**
- **Stage 1 (CLIP space):** a unit direction `g` that raises
  `cosine(image-embedding, CLIP_text(c))`, taken as the one-step cosine gradient at a base
  image's embedding `e₀`:  `g = normalize(e_text − ⟨e_text, e₀⟩·e₀)`.
- **Stage 2 (decoder pullback):** the latent direction whose decoded image moves along `g`:
  `v = normalize(∇_z ⟨CLIP_image(decode(z_base)), g⟩) = normalize(Jᵀg)`,
  `J` = Jacobian of `CLIP_image∘decode`.
- **Composed = chain rule:** `v_chain = normalize(∇_z cosine(CLIP_image(decode(z)), text))`
  — a single backward pass. The intermediate normalization of `g` is irrelevant because we
  normalize `v` at the end. The *only* substantive choice is **where `g` is anchored** (the
  base image `e₀`), which we sweep:
  `chain` (base's own decoded image = pure chain rule), `noise`, `mean` image, `fit` (an
  image of `c`), `nofit` (an unrelated image).

**Two ways to use it.**
- **Arm A — additive direction:** `z' = renorm(z₀ + s·√d·v)`. Here `s` is the ratio of the
  added vector's norm to the seed's own norm, so `s=1` is a ~45° tilt (expected destructive);
  the useful regime is small `s`.
- **Arm B — heavy optimization:** for contrast, iterate the seed itself for `N` steps on the
  same decode→CLIP→cosine objective, re-standardizing each step (E25/E26 latent-mode taken
  hard) — to see what "pushing on-manifold" does.

## Design (`experiments/e27_seeddir.py`, SDXL 1024px)

Reuses `e26_seedalign_sdxl.{load_sdxl, clip_pixel_values, moments, optimize_seed}`,
`clip_sim.{load_clip, clip_image_features, clip_text_features, cosine}`,
`common.{load_pipe, save_grid, generate}`. Prompts: 5 medium scenes from
`e9_bandnorm_classes.CLASSES` × 2 seeds. fp16 models + fp16-fix VAE; `z` fp32. The pullback
is a single VAE-decode+CLIP backward per (base, direction), averaged over `B=2` random base
latents. Metric: CLIP-T (generated image vs prompt text). All heavy GPU ops have OOM-retry
(shared-GPU contention) and the seed is re-standardized after every edit (`‖z‖≡256` verified).

## Results (5 prompts × 2 seeds; `results/e27/`, mean ΔCLIP-T = aligned − baseline)

**Arm A — additive direction `v_chain`, strength sweep**

| s (·√d) | −0.25 | 0 | 0.1 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|
| mean ΔCLIP-T | −0.002 | 0 | +0.000 | +0.000 | **−0.083** | **−0.144** |

A single additive direction is **too blunt**: at gentle strength (`s ≤ 0.25`) it is
**break-even** (no measurable CLIP-T change, image visually unchanged); at `s = 0.5` it
washes the image out; at `s = 1` (added vector ≈ the seed's own size) it collapses to noise.
(`grid_direction.png`.)

**Arm B — heavy per-seed optimization, step sweep**

| N steps | 0 | 1 | 5 | 20 | 60 |
|---|---|---|---|---|---|
| mean ΔCLIP-T | 0 | +0.002 | **+0.003** | +0.001 | +0.001 |

Iterating *on the sphere* is far better-behaved than the single additive jump: it
**progressively intensifies palette / saturation / detail toward the concept while
preserving composition**, even at `N = 60` (not destroyed — `grid_heavy.png`). CLIP-T stays
~flat (best `+0.003` at `N = 5`): it enhances *appearance*, not object presence. The reason
it doesn't blow up like Arm A is the **per-step re-standardization** (each step is a small,
re-projected, on-manifold move).

**Anchor comparison (fixed `s = 0.25`)** — all break-even, `chain` marginally best:

| anchor | chain | noise | mean | fit | nofit |
|---|---|---|---|---|---|
| mean ΔCLIP-T | +0.000 | −0.006 | −0.016 | −0.009 | −0.008 |
| mean cos(v_chain, v_·) | 1.00 | 0.97 | 0.92 | 0.89 | 0.97 |

**The Stage-1 anchor barely matters**: every anchor yields nearly the same latent direction
(`cos = 0.89–0.97`). The `fit` anchor (an image *of* the prompt) deviates most (0.89) and
helps least — subtracting the already-on-prompt component changes `g` the most. So "the
prompt direction in seed space" is essentially anchor-independent.

## Takeaways

1. **The two stages are one chain-rule backward pass**, confirmed; and the anchor is almost
   irrelevant (directions 0.89–0.97 correlated), with the pure chain-rule (`chain`) marginally
   best.
2. **A single additive concept-direction is too blunt for adherence**: gentle = no effect,
   strong = destruction. There is no "free lunch" additive `s` that raises CLIP-T.
3. **On-manifold iterative steering (Arm B) is the well-behaved version**: it intensifies the
   concept's palette/appearance gracefully up to many steps without wrecking structure — but
   it shifts *appearance*, not composition, so CLIP-T stays flat.
4. Consistent with **E25/E26**: the seed's trace is a **palette / global-appearance** signal,
   not a composition one. Re-standardizing each step (staying on the √d sphere) is what makes
   steering safe.

## Status

Done. Constraint (`‖z‖≡√d`, moments held) verified on every edit. Both arms run; anchor
ablation + chain-rule equivalence quantified. The robust, reproducible parts are the
anchor-independence, the s-collapse threshold, and Arm B's do-no-harm intensification; the
small ΔCLIP-T signs are within noise.

## Run

```bash
python experiments/e27_seeddir.py quick   # 1 prompt smoke
python experiments/e27_seeddir.py         # full -> results/e27/{grid_direction,grid_anchors,grid_heavy,deltaclip}.png, report.json
python experiments/e27_site.py            # -> results/e27/index.html  (self-contained explainer, embeds the grids)
```

Artifacts: `results/e27/grid_direction.png` (Arm A s-sweep), `grid_heavy.png` (Arm B),
`grid_anchors.png` (anchor compare), `deltaclip.png` (trends), `report.json`,
`index.html`. Lineage: see `EXPERIMENT_26.md` (E25/E26 seed-alignment) for the thread.
```
