# E19 — Generation-time spectral style transfer (SD3.5-medium)

**Thread:** style · **Model:** SD3.5-medium (true CFG) · **Status:** pending (code-complete; gen/score awaiting cluster)
**Predecessor:** [E18](EXPERIMENT_18.md) (offline two-image spectral recombination — the foundation)

---

## Motivation

E18 proved, *offline*, that a VAE latent's FFT splits into **phase = content/layout** and
**per-band magnitude/power = style** (palette/tone), and that re-leveling per-band power toward a
second image is **AdaIN on the radial power spectrum** (`psd_match`). E19 is the **headline of the
style thread**: do that split *during generation* — synthesize a **content prompt** while clamping
the denoising trajectory's spectrum toward a **style image's** radial power envelope. The content
supplies the phase and the per-step energy trajectory; the style image supplies the texture/palette
envelope.

## Method

`e19_spectral_style.py`, building on E17's SD3.5 SBN path (`ClampPSD3`, unpacked latents):

1. **Build the style reference.** `ref = build_style_reference(content_ref, style_band, strength, gmax)`
   — take the per-step, per-(channel, radial-band) power trajectory of the **content** generation
   (so the energy schedule stays on-manifold) and bend its band envelope toward the **style** image's
   radial power profile by `strength` (0 = unchanged = plain SBN, 1 = full style envelope), with
   `gmax` clamping the per-band gain so a single band can't blow up.
2. **Generate with the clamp.** Denoise the content prompt at cfg≈4.5; at each step the `ClampPSD3`
   callback rescales the latent's per-(channel, band) **power** toward `ref` while **leaving phase
   untouched** (`psd_match`, mode `band`). So layout (phase) is the content's; the radial
   energy/palette is pulled toward the style.

The operators (`style_ops.py`, unit-tested) already fold in the follow-on **modes**:
- **two-prompt SBN** (`blend_references`) — *reference-image-free* style: e.g. "a cat" carrying a
  Van Gogh **spectral signature** derived from a style *prompt*, no style image needed;
- **hybrid synthesis** and **spectral morph** (interpolate two band envelopes).

**Metrics (planned):** `content_clip` (CLIP-I to the unstyled image, = layout/identity kept) vs
`style_clip` / `style_band_dist` (movement toward the style), with aesthetic / ImageReward / CLIP-T
as guards.

![E19 method — generate a content prompt while a per-step ClampPSD3 re-levels each (channel,band) power toward a style envelope (built from a style image or a second prompt), keeping the content's phase. strength sweeps plain-SBN→full-style; modes: reference image / two-prompt reference-free / hybrid / morph. Status: code-complete, gen numbers pending.](figs/E19/method.jpg)

## Status / results

**Pending.** The model-free preflight passes — at `strength=0` the operator is exactly E17's SBN,
and the `gmax` clamp applies as designed — confirming the wiring. The `gen/score/analyze` parts need
the SD3.5 download and a cluster run; **headline numbers are TBD**. The expectation, from E18's
offline result, is **content-safe spectral tone transfer** (high content_clip, with style_band_dist
shrinking as `strength`→1) — real on the SD3.5 latent, tone/palette only (radial bands are
isotropic, so no oriented brushwork).

## Verdict

**Pending — the style thread's headline, code-complete and awaiting the cluster run.** E18 gives it
solid ground (the offline split holds on SD3.5); E19 is whether it survives *generation-time* and
whether the reference-free two-prompt mode is a usable, content-safe style knob.

## Next

Run `gen/score/analyze`; report the strength sweep (content_clip vs style_band_dist frontier);
exercise the hybrid / morph / two-prompt modes (operators exist). Anisotropic (oriented) bands are
the later extension for real stroke style (radial bands cap at tone/palette).

## Artifacts

`experiments/e19_spectral_style.py` (+ `style_ops.py`, `e17_sd35.py`); results dir `results/e19/`
(pending). Foundation: [E18](EXPERIMENT_18.md).
