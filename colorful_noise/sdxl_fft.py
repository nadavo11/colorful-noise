
import os
from PIL import Image, ImageEnhance
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline
from utils import encode_img_sdxl, fft_radial_frequency_swap, generate_rgb_mask, generate_background_mask

import argparse
from utils import rgb_type

def main(args):
    
    # 1. Load model and scheduler
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    ).to("cuda")


    if args.seed is not None:
        torch.manual_seed(args.seed)

    input_pil = Image.open(args.input_image).convert("RGB").resize((args.height, args.width))
    vae_latents = encode_img_sdxl(pipe, input_pil).type(pipe.dtype)
    for i in range(args.num_samples):
        noise = torch.randn(1, 4, args.height // 8, args.width // 8).cuda().type(pipe.dtype)
        
        fft_latent = fft_radial_frequency_swap(noise, vae_latents, p=args.alpha, temp=args.gamma).to("cuda").type(pipe.dtype)

        if args.ignore_color:
            mask = torch.tensor(generate_rgb_mask(input_pil, target_rgb=args.ignore_color, output_shape=(args.height // 8, args.width // 8))).to("cuda").type(torch.float16)
            fft_latent = mask * noise + (1-mask) * fft_latent

        generated_image = pipe(
            prompt=args.prompt,
            negative_prompt=None,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            latents=fft_latent
        ).images

        # 6. Save result
        generated_image[0].save(os.path.join(args.output_dir, f"guided_{i}.png"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SDXL images with masked FFT latent mixing."
    )
    parser.add_argument("--input_image", type=str, default="inputs/cat_orange.png", help="Path to the input sketch or image")
    # parser.add_argument("--input_image", type=str, default="inputs/savana.png", help="Path to the input sketch or image")
    parser.add_argument("--prompt", type=str, default="A photo of cat in the park", help="Text prompt for image generation")
    # parser.add_argument("--prompt", type=str, default="A photo of giraffe and an elephant in the savanna", help="Text prompt for image generation")
    parser.add_argument("--alpha", type=float, default=0.015, help="Percentage of frequenciues to replace")
    parser.add_argument("--gamma", type=float, default=0.07, help="Temperature scaling factor for FFT latent mixing")
    parser.add_argument("--ignore_color",type=rgb_type,default=None, help="An RGB color value which will not be used for conditioning. format: '(R, G, B)'")
    parser.add_argument("--output_dir", type=str, default="outputs/sdxl_fft", help="Directory to save generated images")

    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt for image generation")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale")
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
    
