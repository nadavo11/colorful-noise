"""E46 Approach 1 -- gamma knob: smoothly whiten the seed phase (chair, 1 seed).

seed_g = ifft( |z| * unit( (1-g)*e^{i angle(z)} + g*e^{i angle(x0)} ) )
g=0 -> white seed z ; g=1 -> Cfull (full image phase). g interpolates the phase factor on the
unit circle, so cross-frequency phase coherence (structure) decays smoothly as g->0.
"""
import os

import torch

import common
from struct_metrics import load_dino, structure_distance, clip_directional, load_clip
from e46_probe1 import sdedit, STEPS, GUID, STRENGTH
from diffusers import StableDiffusionXLImg2ImgPipeline

SRC_PROMPT = "a wooden chair in an empty bright room"
TGT_PROMPT = "a shiny metal chair in an empty bright room"
SRC_SEED, SEED = 100, 0
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def seed_gamma(x0, z, g):
    Fa, Fz = torch.fft.fft2(x0.float()), torch.fft.fft2(z.float())
    mix = (1 - g) * torch.exp(1j * Fz.angle()) + g * torch.exp(1j * Fa.angle())
    phase = mix / mix.abs().clamp(min=1e-8)
    return torch.fft.ifft2(Fz.abs() * phase).real.to(x0.dtype)


def main():
    out = os.path.join(common.RESULTS, "e46_gamma")
    os.makedirs(out, exist_ok=True)
    pipe = common.load_pipe()
    img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
    img2img.set_progress_bar_config(disable=True)
    dino, clip = load_dino(), load_clip()

    gz = torch.randn((1, 4, 128, 128), generator=torch.Generator("cuda").manual_seed(SRC_SEED),
                     device="cuda", dtype=pipe.dtype)
    src = common.generate(pipe, SRC_PROMPT, gz, steps=STEPS, guidance=GUID)
    x0 = common.encode_img_sdxl(pipe, src)
    z = torch.randn(x0.shape, generator=torch.Generator("cuda").manual_seed(SEED),
                    device="cuda", dtype=x0.dtype)

    row, cols = [src], ["source"]
    van = sdedit(img2img, src, TGT_PROMPT, STRENGTH, STEPS, GUID,
                 torch.Generator("cuda").manual_seed(SEED))
    row.append(van); cols.append("vanilla")
    sd = structure_distance(dino, van, src)
    cd = clip_directional(clip, src, van, SRC_PROMPT, TGT_PROMPT)
    print(f"vanilla   struct={sd:.4f}  clip_dir={cd:+.4f}", flush=True)

    for g in GAMMAS:
        im = common.generate(pipe, TGT_PROMPT, seed_gamma(x0, z, g), steps=STEPS, guidance=GUID)
        im.save(os.path.join(out, f"g{g}.png"))
        sd = structure_distance(dino, im, src)
        cd = clip_directional(clip, src, im, SRC_PROMPT, TGT_PROMPT)
        print(f"gamma={g:<4} struct={sd:.4f}  clip_dir={cd:+.4f}", flush=True)
        row.append(im); cols.append(f"g={g}")

    src.save(os.path.join(out, "source.png"))
    common.save_grid([row], ["chair"], cols, os.path.join(out, "grid.png"))
    print(f"\nartifacts: {out}/grid.png", flush=True)


if __name__ == "__main__":
    main()
