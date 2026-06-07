"""Shared helpers: model loading, image encoding, grid plotting."""
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, "/workspace/colorful_noise")  # reuse the paper's code
from utils import encode_img_sdxl, fft_radial_frequency_swap  # noqa: F401,E402

SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
INPUTS = "/workspace/colorful_noise/inputs"
RESULTS = "/workspace/experiments/results"


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
    return out_path
