"""E46 chair-only debug: full-band phase seed (Cfull) vs vanilla SDEdit vs low-band seed (Blow).

One image, one seed. Generates only the chair source, then the three arms, saves full-res
PNGs + a grid. Fast (~1 min).
"""
import os

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

import common
from latent_spectral_ops import phase_swap_2d
from struct_metrics import load_dino, structure_distance, clip_directional, load_clip
from e46_probe1 import sdedit, STEPS, GUID, STRENGTH

SRC_PROMPT = "a wooden chair in an empty bright room"
TGT_PROMPT = "a shiny metal chair in an empty bright room"
SRC_SEED = 100
SEED = 0


def main():
    out = os.path.join(common.RESULTS, "e46_chair")
    os.makedirs(out, exist_ok=True)
    pipe = common.load_pipe()
    img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
    img2img.set_progress_bar_config(disable=True)
    dino, clip = load_dino(), load_clip()

    # source
    gz = torch.randn((1, 4, 128, 128), generator=torch.Generator("cuda").manual_seed(SRC_SEED),
                     device="cuda", dtype=pipe.dtype)
    src = common.generate(pipe, SRC_PROMPT, gz, steps=STEPS, guidance=GUID)
    x0 = common.encode_img_sdxl(pipe, src)

    z = torch.randn(x0.shape, generator=torch.Generator("cuda").manual_seed(SEED),
                    device="cuda", dtype=x0.dtype)
    seed_full = phase_swap_2d(x0, z, 1.0, mag_from="B")
    # Cnorm: per-freq subtract the noise sample's phase. F = |z| * exp(i(phi_src - phi_z)).
    Fa, Fz = torch.fft.fft2(x0.float()), torch.fft.fft2(z.float())
    seed_norm = torch.fft.ifft2(Fz.abs() * torch.exp(1j * (Fa.angle() - Fz.angle()))).real.to(x0.dtype)

    outs = {
        "vanilla": sdedit(img2img, src, TGT_PROMPT, STRENGTH, STEPS, GUID,
                          torch.Generator("cuda").manual_seed(SEED)),
        "Cfull": common.generate(pipe, TGT_PROMPT, seed_full, steps=STEPS, guidance=GUID),
        "Cnorm": common.generate(pipe, TGT_PROMPT, seed_norm, steps=STEPS, guidance=GUID),
    }
    src.save(os.path.join(out, "source.png"))
    for a, im in outs.items():
        im.save(os.path.join(out, f"{a}.png"))
        sd = structure_distance(dino, im, src)
        cd = clip_directional(clip, src, im, SRC_PROMPT, TGT_PROMPT)
        print(f"{a:8s} struct={sd:.4f}  clip_dir={cd:+.4f}", flush=True)

    common.save_grid([[src, outs["vanilla"], outs["Cfull"], outs["Cnorm"]]],
                     ["chair"], ["source", "vanilla", "Cfull", "Cnorm"],
                     os.path.join(out, "grid.png"))
    print(f"\nartifacts: {out}/  (source.png, vanilla.png, Cfull.png, Blow.png, grid.png)")


if __name__ == "__main__":
    main()
