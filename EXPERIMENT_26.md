# E25–E26 — Seed alignment: biasing the initial noise toward the prompt

**The direction.** Diffusion sampling starts from a random latent "seed" `z ~ N(0, I)`.
There is a well-known observation that **the seed leaves traces in the output**: the final
generated latent stays highly correlated (high cosine similarity) with the seed it started
from — a consequence of the near-linearity of the probability-flow ODE. So instead of
sampling `z` purely at random, can we spend a *tiny, cheap* amount of optimization to nudge
the seed **toward the prompt** before generation, and have that bias survive into the image?

This is a deliberately gentle lever. The goal is **not** a large prompt-adherence jump — it
is to characterize a cheap, do-no-harm "better starting point" for the seed.

## The method (what we optimize, and the one hard constraint)

We optimize a copy of the seed `z` with a few gradient steps to maximize a **text-alignment
objective**, subject to keeping `z` looking like standard Gaussian noise.

- **Objective — purely in latent space, no denoising.** We do **not** predict the clean
  latent `x̂₀` and we **never run the UNet** in the optimization. We simply decode the seed
  itself and compare it to the prompt in CLIP space:

  ```
  loss = − cosine( CLIP_image( VAE.decode(z) ), CLIP_text(prompt) )
  ```

  Gradients flow only through the (frozen) VAE decoder and the (frozen) CLIP image encoder
  back to `z`. The UNet is used **only** in the ordinary generation call afterwards. (An
  alternative objective that *does* run one UNet step to form `x̂₀` was tried in E25 and was
  more aggressive / more destructive — see below. The latent-space version here is the
  gentler, better-behaved one.)

- **Constraint — stay on the Gaussian sphere.** After every gradient step we
  **re-standardize** `z ← (z − mean) / std`, forcing **zero mean and unit variance**. Because
  `‖z‖² = d · (var + mean²)`, this *pins the norm to* `‖z‖ = √d` exactly (d = latent
  dimension), i.e. the optimization is a move **on the sphere of radius √d** that a true
  Gaussian sample lives on — not off into low-probability latent regions. Measured `‖z*‖`
  is `√d` to two decimals on every single run.

**Why this should do anything at all:** if the seed correlates with the output, a seed that
already "looks a little more like the prompt" (in CLIP's eyes) should pull the image that way.

## E25 — pilot on SD1.5 (512px, `experiments/e25_seedalign.py`)

`d = 4·64·64 = 16384`, so `√d = 128`. Standalone `openai/clip-vit-large-patch14` for the
objective (SD1.5's own text encoder is not in the joint image/text CLIP space). Knobs are env
vars (`E25_MODE`, `E25_STEPS`, `E25_TARGET`, `E25_LR`, `E25_DTYPE`).

Findings (mean Δ CLIP-T over baseline, 4 prompts × 2 seeds):
- **Moments/​norm:** held exactly (`mean≈0, std=1.000, ‖z‖=128.00`) in every run — the
  constraint and the objective are jointly satisfiable.
- **`x̂₀` mode (runs one UNet step in the objective):** the inner objective over-optimizes
  (CLIP cosine shoots to ~0.50, higher than any *natural* image's ~0.25–0.30) and the seed
  becomes **CLIP-adversarial**: leopard-print for "cat", swirls for "blue sphere", and lost
  composition. Net **−0.022 to −0.025** — it slightly *hurts*. Early-stopping at a natural
  CLIP-T reduces the damage but not the sign.
- **latent mode (decode `z` directly, no UNet — the method above):** the **gentlest** and
  best-behaved. Mean Δ **−0.010** (4↑/4↓), visually stays very close to baseline and behaves
  like a controlled **palette / global-appearance nudge** that occasionally helps (e.g. it
  nudged a red sphere into existence, kept a bench+umbrella scene and just saturated it).

**Takeaway from E25:** the seed's trace is a **palette/global-appearance** trace, not a
composition one. The latent-space objective is the right, do-no-harm formulation.

## E26 — SDXL + long prompts + #-steps sweep (`experiments/e26_seedalign_sdxl.py`)

Three extensions, all per the latent-space method above.

1. **Better model: SDXL** (1024px, fp16). `d = 4·128·128 = 65536`, `√d = 256`. The stock
   SDXL fp16 VAE NaNs on decode, so we swap in `madebyollin/sdxl-vae-fp16-fix`. Dropping the
   UNet from the objective makes this comfortably fit a 24 GB A5000 (only VAE+CLIP backprop).

2. **Long, dense prompts: DPG-Bench** (`experiments/dpg_bench.py`, cached from the public
   ELLA repo; `load_dpg_prompts`). We used 10 prompts of ~55–109 words.
   **Important caveat:** SDXL's two CLIP text encoders — *and* the CLIP scorer — truncate at
   **77 tokens**, while these prompts run ~70–90 words. So at generation time SDXL literally
   cannot read the whole prompt. We therefore use a **long-aware CLIP-T**: split the prompt
   into clauses (each ≤77 tokens), CLIP-encode each, **mean-pool + renormalize**. This is the
   target for the objective *and* the evaluation metric. (It also reframes the idea: the seed
   is a side-channel that could carry prompt information the truncated text conditioning
   drops — a hypothesis for future work.)

3. **Sweep N = number of inner latent-space gradient steps:** the cheap one-step **linear**
   solution `N=1`, a **strengthened** single step (larger lr), then `N=2, 3, 5`. One
   optimization is run to `N=5` and `z` is **snapshotted** at each N (a prefix reuses work),
   so the sweep is nearly free. Re-standardization keeps `‖z‖=√d` at every snapshot.

### Results (10 DPG prompts, 1 seed; `results/e26/`)

Mean **Δ long-CLIP-T** (aligned − baseline) by number of steps:

| N steps | 1 | 1 (strengthened) | 2 | 3 | 5 |
|---|---|---|---|---|---|
| mean Δ | **+0.0015** | +0.0001 | −0.0004 | −0.0017 | −0.0005 |

- **Break-even, and the cheapest setting wins.** `N=1` (the one-step linear solution) is the
  only clearly non-negative point; adding more steps does **not** help and drifts slightly
  negative. The strengthened single step is ~0. Per-prompt deltas are tiny (±0.01).
- **Constraint holds:** every `z*` has `mean≈0, std=1, ‖z‖=256.00 = √d`.
- **Visually** (`results/e26/grid.png`, columns `baseline | N=1 | N=2 | N=3 | N=5 | N=1*strong`):
  the aligned columns stay very close to baseline — gentle palette / saturation / detail
  shifts, **no structural damage** (unlike E25's `x̂₀` mode). This is the do-no-harm behavior
  the latent-space objective was chosen for.

**Interpretation.** On a stronger model with long prompts, the latent-space seed nudge is a
genuinely *gentle, break-even* operation, and a **single cheap gradient step captures whatever
benefit there is**. That is consistent with the original motivation: a one-step linear nudge is
a sensible, low-cost *better starting point* for the seed, not a heavy optimizer. It does not,
by itself, move long-prompt adherence much — which is expected given SDXL's 77-token bottleneck.

## Status

- E25 (SD1.5): **done**, both objective modes characterized.
- E26 (SDXL + DPG-Bench + N-sweep): **done**; `√d`-sphere constraint verified, N=1 best,
  effect is gentle/break-even on long-aware CLIP-T.
- The constraint (`‖z‖=√d`, moments held) and the do-no-harm behavior are the robust,
  reproducible parts; the small ΔCLIP-T sign is within noise.

## Open directions

- **Beat the 77-token bottleneck** so the seed can actually carry the long-prompt tail:
  Long-CLIP / T5-conditioned models (SD3.5/Flux) — but those are NF4-quantized here, so
  gradient backprop to the seed is heavy (left for later).
- **Structure, not just palette:** restrict the nudge to a **low-frequency band of `z`** (ties
  into the project's spectral toolbox) to bias global layout without touching texture.
- **Stronger metric than CLIP-T:** the official DPG score (VQA-based) or `vqascore.py`.

## Run

```bash
# prompts (downloads + caches the DPG-Bench CSV on first call)
python experiments/dpg_bench.py 8

# SD1.5 pilot (modes via env: E25_MODE=latent|x0, E25_STEPS, E25_TARGET, E25_LR)
python experiments/e25_seedalign.py            # full
E25_MODE=latent python experiments/e25_seedalign.py

# SDXL + DPG-Bench + N-sweep (knobs: E26_PROMPTS, E26_LR, E26_STRONG_LR, E26_SIZE=768 if OOM)
python experiments/e26_seedalign_sdxl.py quick # 2 prompts smoke
python experiments/e26_seedalign_sdxl.py       # full -> results/e26/{grid.png, deltaclip_vs_N.png, report.json}
```
