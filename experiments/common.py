"""Shared helpers: model loading, image encoding, grid plotting."""
import os
import sys

import torch
from PIL import Image

# Repo root is resolved from this file's location (experiments/common.py ->
# repo root), so paths work regardless of where the repo is mounted (the docker
# image historically mounted it at /workspace; the current compose mounts it at
# its host path). Override with env vars if needed.
_REPO = os.environ.get(
    "CN_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(_REPO, "colorful_noise"))  # reuse paper's code
from utils import encode_img_sdxl, fft_radial_frequency_swap  # noqa: F401,E402

SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
INPUTS = os.environ.get("CN_INPUTS", os.path.join(_REPO, "colorful_noise", "inputs"))
RESULTS = os.environ.get("CN_RESULTS", os.path.join(_REPO, "experiments", "results"))


def load_pipe(model_id=SDXL_ID):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(SDXL_ID, subfolder="vae").to("cuda")


def encode_image(pipe, path, size=1024):
    img = Image.open(path).convert("RGB").resize((size, size))
    return encode_img_sdxl(pipe, img)


def generate(pipe, prompt, latents, steps=50, guidance=5.0):
    return pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=latents.shape[-2] * 8,
        width=latents.shape[-1] * 8,
        latents=latents.to("cuda").type(pipe.dtype),
    ).images[0]


def save_grid(rows, row_labels, col_labels, out_path, thumb=320):
    """rows: list of lists of PIL images -> labeled contact sheet."""
    from PIL import ImageDraw
    pad, label_w, label_h = 6, 200, 36
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    W = label_w + n_cols * (thumb + pad)
    H = label_h + n_rows * (thumb + pad)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for j, cl in enumerate(col_labels):
        d.text((label_w + j * (thumb + pad) + 4, 10), cl, fill="black")
    for i, (label, row) in enumerate(zip(row_labels, rows)):
        y = label_h + i * (thumb + pad)
        d.text((4, y + thumb // 2), label, fill="black")
        for j, im in enumerate(row):
            sheet.paste(im.resize((thumb, thumb)), (label_w + j * (thumb + pad), y))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)


def data_uri(path, max_px=1600, quality=85):
    """Read an image and return a base64 data: URI (downscaled JPEG), so a page
    can embed it and stay self-contained. Keep this name/signature stable."""
    import base64
    from io import BytesIO
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_px:
        r = max_px / max(im.size)
        im = im.resize((round(im.width * r), round(im.height * r)))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return out_path
