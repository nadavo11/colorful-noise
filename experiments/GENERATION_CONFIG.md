# Post-processing conventions

Standing decisions for how we post-process generated images in this project.

## Apply a saturation boost to band-norm (SBN) outputs

The default post-process on a band-norm / SBN image is a **saturation boost
×1.4** (`PIL.ImageEnhance.Color`, see E11 / `e11_color_correct.py`). Apply it on
**any model** we use band-norm with (Flux, and any other backend).

It is **cheap, single-pass, reference-free, and runs on the already-rendered
frame** (no GPU). It restores most of the colour SBN tames away while leaving
`hf_frac` (fine detail) and contrast essentially unchanged. ×1.4 was chosen over
×1.8 because its colorfulness lands closest to the `cfg=3.5` target without
over-saturating (×1.8 overshoots past full-guidance saturation on 5/6 content
classes and reads garish).

### Why not `hist_match`?

`hist_match` (per-channel CDF match of the SBN image to its paired `cfg=3.5`
image) reaches the palette target slightly more precisely, **but it requires a
second, full-guidance `cfg=3.5` generation pass per image** to produce that
reference — roughly doubling generation cost and needing the very full-guidance
output band-norm is meant to avoid. It is an **upper-bound oracle** for "how
close can correction get," **not** a deployable correction. Use it only to
measure the ceiling, never in production.

### Other options
- `contrast`=1.2 — recovers both colour and contrast without a reference (use if
  contrast, not just colour, also needs a lift).
- Avoid `lum_eq` (overshoots contrast, lowers colour, perturbs detail);
  `autocontrast` is a near no-op on SBN images (they already span 0–255).
