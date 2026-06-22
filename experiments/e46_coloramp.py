"""E46 Approach 3 -- colored amplitude (chair, 1 seed).

Full image phase, but amplitude from a colored (1/f) field instead of white, to suppress the
high-frequency coherent energy that caused Cfull's fringing.
seed = ifft( |FFT(colored_beta)| * exp(i angle(x0)) ), variance-normalized.
"""
import os

import torch

import common
from spectral_ops import colored_noise
from struct_metrics import load_dino, structure_distance, clip_directional, load_clip
from e46_probe1 import sdedit, STEPS, GUID, STRENGTH
from diffusers import StableDiffusionXLImg2ImgPipeline

SRC_PROMPT = "a wooden chair in an empty bright room"
TGT_PROMPT = "a shiny metal chair in an empty bright room"
SRC_SEED, SEED = 100, 0
BETAS = [0.0, -1.0, -2.0]  # 0=white, -1=pink, -2=red


def seed_coloramp(x0, beta, gen):
    cn = colored_noise(x0.shape, beta, device="cuda", generator=gen)
    amp = torch.fft.fft2(cn.float()).abs()
    seed = torch.fft.ifft2(amp * torch.exp(1j * torch.fft.fft2(x0.float()).angle())).real
    seed = (seed - seed.mean()) / (seed.std() + 1e-8)
    return seed.to(x0.dtype)


def main():
    out = os.path.join(common.RESULTS, "e46_coloramp")
    os.makedirs(out, exist_ok=True)
    pipe = common.load_pipe()
    img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
    img2img.set_progress_bar_config(disable=True)
    dino, clip = load_dino(), load_clip()

    gz = torch.randn((1, 4, 128, 128), generator=torch.Generator("cuda").manual_seed(SRC_SEED),
                     device="cuda", dtype=pipe.dtype)
    src = common.generate(pipe, SRC_PROMPT, gz, steps=STEPS, guidance=GUID)
    x0 = common.encode_img_sdxl(pipe, src)

    def score(name, im):
        sd = structure_distance(dino, im, src)
        cd = clip_directional(clip, src, im, SRC_PROMPT, TGT_PROMPT)
        print(f"{name:9s} struct={sd:.4f}  clip_dir={cd:+.4f}", flush=True)
        im.save(os.path.join(out, f"{name}.png"))
        return im

    row, cols = [src], ["source"]
    row.append(score("vanilla", sdedit(img2img, src, TGT_PROMPT, STRENGTH, STEPS, GUID,
                                        torch.Generator("cuda").manual_seed(SEED))))
    cols.append("vanilla")
    for b in BETAS:
        g = torch.Generator("cuda").manual_seed(SEED)
        im = common.generate(pipe, TGT_PROMPT, seed_coloramp(x0, b, g), steps=STEPS, guidance=GUID)
        name = f"beta{b}"
        row.append(score(name, im)); cols.append(name)

    src.save(os.path.join(out, "source.png"))
    common.save_grid([row], ["chair"], cols, os.path.join(out, "grid.png"))
    print(f"\nartifacts: {out}/grid.png", flush=True)


if __name__ == "__main__":
    main()
