---
title: colorful-noise spectral demo
emoji: 🌈
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Spectral image editing demo for FLUX and SD3.5.
---

# colorful-noise spectral demo

This Space runs the interactive Gradio demo from `experiments/spectral_demo.py`.

Default model:

- `flux-dev` via `MODEL_NAME=flux-dev`

Main tabs:

- Token modulation
- Latent modulation
- Spectral AdaIN
- RF inversion
- FlowEdit
- FlowAlign

Notes:

- The app requires GPU hardware.
- The Space uses the owner's Hugging Face token to access gated models when needed.
