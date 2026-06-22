"""E46 Approach 2 -- timestep phase injection (chair, 1 seed).

White (on-manifold) seed. During the structure-forming window [lo,hi] of the denoise, replace
the latent's low-band [0,cut] phase with the source latent's (magnitude kept) via a step callback.
Seed is never OOD; structure enters when the model resolves low frequencies. Sweep the window.
"""
import os

import torch

import common
from latent_spectral_ops import phase_swap_2d
from struct_metrics import load_dino, structure_distance, clip_directional, load_clip
from e46_probe1 import sdedit, STEPS, GUID, STRENGTH
from diffusers import StableDiffusionXLImg2ImgPipeline

SRC_PROMPT = "a wooden chair in an empty bright room"
TGT_PROMPT = "a shiny metal chair in an empty bright room"
SRC_SEED, SEED = 100, 0
CUT = 0.2
WINDOWS = [(0.0, 0.3), (0.0, 0.5), (0.0, 0.7)]  # fraction of denoise steps to inject over


def make_cb(x0, cut, lo, hi, total):
    def cb(pipe, i, t, kw):
        if lo * total <= i < hi * total:
            lat = kw["latents"]
            kw["latents"] = phase_swap_2d(x0, lat, cut, mag_from="B").to(lat.dtype)
        return kw
    return cb


def main():
    out = os.path.join(common.RESULTS, "e46_inject")
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

    def score(name, im):
        sd = structure_distance(dino, im, src)
        cd = clip_directional(clip, src, im, SRC_PROMPT, TGT_PROMPT)
        print(f"{name:10s} struct={sd:.4f}  clip_dir={cd:+.4f}", flush=True)
        im.save(os.path.join(out, f"{name}.png"))
        return im

    row, cols = [src], ["source"]
    row.append(score("vanilla", sdedit(img2img, src, TGT_PROMPT, STRENGTH, STEPS, GUID,
                                        torch.Generator("cuda").manual_seed(SEED))))
    cols.append("vanilla")

    for lo, hi in WINDOWS:
        im = pipe(prompt=TGT_PROMPT, num_inference_steps=STEPS, guidance_scale=GUID,
                  height=1024, width=1024, latents=z.clone().type(pipe.dtype),
                  callback_on_step_end=make_cb(x0, CUT, lo, hi, STEPS),
                  callback_on_step_end_tensor_inputs=["latents"]).images[0]
        name = f"inj{int(lo*100)}-{int(hi*100)}"
        row.append(score(name, im)); cols.append(name)

    src.save(os.path.join(out, "source.png"))
    common.save_grid([row], ["chair"], cols, os.path.join(out, "grid.png"))
    print(f"\nartifacts: {out}/grid.png", flush=True)


if __name__ == "__main__":
    main()
