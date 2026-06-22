"""E46 synthesis -- SOFT timestep injection (chair, 1 seed).

White seed. During window [lo,hi], blend the latent's low-band [0,cut] phase TOWARD the source
phase by gamma (slerp on the unit circle), magnitude kept. gamma=1 == hard inject (Approach 2);
gamma<1 leaves the prompt room to edit. Goal: clean (on-manifold) + structured + still editable.
"""
import os

import torch

import common
from latent_spectral_ops import _band_sel
from struct_metrics import load_dino, structure_distance, clip_directional, load_clip
from e46_probe1 import sdedit, STEPS, GUID, STRENGTH
from diffusers import StableDiffusionXLImg2ImgPipeline

SRC_PROMPT = "a wooden chair in an empty bright room"
TGT_PROMPT = "a shiny metal chair in an empty bright room"
SRC_SEED, SEED = 100, 0
CUT, WINDOW = 0.2, (0.0, 0.5)
GAMMAS = [0.3, 0.6, 1.0]


def soft_cb(x0, cut, g, lo, hi, total):
    H, W = x0.shape[-2:]
    low = _band_sel(H, W, 0.0, cut, x0.device, keep_dc=True).to(torch.float32)[None, None]
    Fa = torch.fft.fft2(x0.float())
    def cb(pipe, i, t, kw):
        if lo * total <= i < hi * total:
            lat = kw["latents"]
            Fl = torch.fft.fft2(lat.float())
            mix = (1 - g) * torch.exp(1j * Fl.angle()) + g * torch.exp(1j * Fa.angle())
            mixp = mix / mix.abs().clamp(min=1e-8)
            phase = torch.where(low > 0, mixp, torch.exp(1j * Fl.angle()))
            kw["latents"] = torch.fft.ifft2(Fl.abs() * phase).real.to(lat.dtype)
        return kw
    return cb


def main():
    out = os.path.join(common.RESULTS, "e46_softinject")
    os.makedirs(out, exist_ok=True)
    pipe = common.load_pipe()
    img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
    img2img.set_progress_bar_config(disable=True)
    dino, clip = load_dino(), load_clip()

    gz = torch.randn((1, 4, 128, 128), generator=torch.Generator("cuda").manual_seed(SRC_SEED),
                     device="cuda", dtype=pipe.dtype)
    src = common.generate(pipe, SRC_PROMPT, gz, steps=STEPS, guidance=GUID)
    x0 = common.encode_img_sdxl(pipe, src)
    z = torch.randn((1, 4, 128, 128), generator=torch.Generator("cuda").manual_seed(SEED),
                    device="cuda", dtype=pipe.dtype)

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
    lo, hi = WINDOW
    for g in GAMMAS:
        im = pipe(prompt=TGT_PROMPT, num_inference_steps=STEPS, guidance_scale=GUID,
                  height=1024, width=1024, latents=z.clone(),
                  callback_on_step_end=soft_cb(x0, CUT, g, lo, hi, STEPS),
                  callback_on_step_end_tensor_inputs=["latents"]).images[0]
        row.append(score(f"g{g}", im)); cols.append(f"g={g}")

    src.save(os.path.join(out, "source.png"))
    common.save_grid([row], ["chair"], cols, os.path.join(out, "grid.png"))
    print(f"\nartifacts: {out}/grid.png", flush=True)


if __name__ == "__main__":
    main()
