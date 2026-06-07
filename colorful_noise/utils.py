import argparse
import ast
import torch
import numpy as np
from PIL import Image

def rgb_type(value):
    try:
        rgb = ast.literal_eval(value)

        if not isinstance(rgb, tuple) or len(rgb) != 3:
            raise ValueError

        if not all(isinstance(c, int) and 0 <= c <= 255 for c in rgb):
            raise ValueError

        return rgb

    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(
            "RGB must be in format '(R, G, B)' with values 0-255"
        )
    

def encode_img_sdxl(pipe, image):
    pipe.vae.to(dtype=torch.float32)
    # Encode the image to latents using VAE
    image = pipe.image_processor.preprocess(image).to(pipe.device).type(pipe.vae.dtype)
    with torch.no_grad():
        latent_dist = pipe.vae.encode(image).latent_dist
        latents = latent_dist.sample() * pipe.vae.config.scaling_factor
    
    pipe.vae.to(dtype=torch.float16)
    return latents.type(torch.float16)


def fft_radial_frequency_swap(latents_hi, latents_lo, p, temp=1.0):
    """
    Replace frequencies based on radial percentile.

    Parameters
    ----------
    p : float
        If 0 < p <= 1:
            Replace lowest p fraction of frequencies by radius.
            Example: 0.1 → replace lowest 10% (low-frequency region)

    """

    assert latents_hi.shape == latents_lo.shape

    B, C, H, W = latents_hi.shape

    fft_hi = torch.fft.fftshift(torch.fft.fft2(latents_hi, dim=(-2, -1)), dim=(-2, -1))
    fft_lo = torch.fft.fftshift(torch.fft.fft2(latents_lo, dim=(-2, -1)), dim=(-2, -1))

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=latents_hi.device),
        torch.linspace(-1, 1, W, device=latents_hi.device),
        indexing="ij"
    )

    rr = torch.sqrt(xx**2 + yy**2)

    # compute radial threshold by percentile
    r_flat = rr.flatten()
    cutoff = torch.quantile(r_flat, p)

    low_mask = (rr <= cutoff).float()
    high_mask = 1.0 - low_mask

    low_mask = low_mask[None, None]
    high_mask = high_mask[None, None]

    fft_mix = (fft_lo * low_mask) * temp + fft_hi * high_mask

    fft_mix = torch.fft.ifftshift(fft_mix, dim=(-2, -1))
    return torch.fft.ifft2(fft_mix, dim=(-2, -1)).real


def generate_rgb_mask(
    pil_image,
    target_rgb,
    output_shape=None
):
    """
    Args:
        pil_image: PIL.Image
        target_rgb: (R, G, B) tuple (0–255)
        target_shape: (W, H) resize target (optional)

    Returns:
        torch.Tensor: (4, H, W)
            0 where color matches target_rgb
            1 elsewhere
    """

    img = pil_image.convert("RGB")

    if output_shape is not None:
        img = img.resize(output_shape)

    np_image = np.array(img).astype(np.int32)  # (H, W, 3)
    target_rgb = np.array(target_rgb).astype(np.int32)

    # exact color match
    match = np.all(np_image == target_rgb, axis=-1)  # (H, W)

    mask = (match).astype(np.float32)  # 1 = not target, 0 = target

    # to torch + 4 channels
    mask = torch.from_numpy(mask)[None, :, :]  # (1, H, W)
    mask = mask.repeat(4, 1, 1)                # (4, H, W)

    return mask
