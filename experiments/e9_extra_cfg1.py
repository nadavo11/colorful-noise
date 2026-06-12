"""E9 add-on: generate the missing cfg=1.0 baseline images (seeds 3..24).

E9 only stored 3 cfg=1.0 reference seeds per class, while cfg=3.5 and band-norm
have 25 each. To put CLIP-T on equal footing -- and make the cfg1/cfg3.5/bandnorm
triptych comparable at every seed on the explainer page -- this fills cfg=1.0
seeds 3..24.

These are PLAIN cfg=1.0 generations (no PSD modification), saved exactly like the
existing cfg1.0_s0..2. We do NOT touch ref_psd.pt: the band-norm reference stays
the 3-seed average the existing bandnorm images were clamped to.

Batched for speed: each pipe call renders `--batch` seeds at once with a list of
per-seed generators (so seeds still match single-image generation) and encodes the
prompt through T5 only ONCE per call (reused across the batch). On CUDA OOM the
batch size halves and retries. --batch 1 reproduces the original per-image path.

    python e9_extra_cfg1.py                       # all classes, seeds 3..24, batch 4
    python e9_extra_cfg1.py --batch 1 --classes animal --start 5 --end 6   # validate
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e7_flux_phase import load_flux, SIZE
from e9_bandnorm_classes import CLASSES


class GrabLast:
    """Capture the final-step packed latents (whole batch)."""
    def __init__(self):
        self.last = None

    def __call__(self, p, i, t, kw):
        self.last = kw["latents"]
        return {}


def gen_batch(pipe, prompt, seeds, cfg, steps):
    """Render a list of seeds in one call -> (list[PIL], unpacked fp32 cpu lat).
    T5 runs once; per-seed generators keep seeds matched to single-image gen."""
    from diffusers import FluxPipeline
    gens = [torch.Generator("cuda").manual_seed(s) for s in seeds]
    grab = GrabLast()
    out = pipe(prompt=prompt, height=SIZE, width=SIZE, guidance_scale=cfg,
               num_inference_steps=steps, num_images_per_prompt=len(seeds),
               generator=gens, callback_on_step_end=grab)
    lat = FluxPipeline._unpack_latents(grab.last, SIZE, SIZE,
                                       pipe.vae_scale_factor).float().cpu()
    return out.images, lat


def save_one(cdir, cond, seed, img, lat):
    img.save(f"{cdir}/images/{cond}_s{seed}.png")
    torch.save(lat, f"{cdir}/latents/{cond}_s{seed}.pt")


def main(args):
    out = os.path.join(RESULTS, "e9")
    pick = [c for c in CLASSES if args.classes is None or c[0] in args.classes]
    cond = f"cfg{args.ref_cfg}"
    print(f"[extra] {len(pick)} classes, seeds {args.start}..{args.end - 1}, "
          f"batch={args.batch}", flush=True)

    pipe = load_flux(args.mem)
    made = 0
    for key, prompt in pick:
        cdir = f"{out}/{key}"
        os.makedirs(f"{cdir}/images", exist_ok=True)
        os.makedirs(f"{cdir}/latents", exist_ok=True)
        # only seeds whose image+latent are not already on disk
        todo = [s for s in range(args.start, args.end)
                if not (os.path.exists(f"{cdir}/images/{cond}_s{s}.png")
                        and os.path.exists(f"{cdir}/latents/{cond}_s{s}.pt"))]
        if not todo:
            print(f"[extra] {key} (all cached)", flush=True)
            continue

        i = 0
        bs = args.batch
        while i < len(todo):
            chunk = todo[i:i + bs]
            try:
                imgs, lat = gen_batch(pipe, prompt, chunk, args.ref_cfg, args.steps)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(f"[extra] OOM -> batch={bs}, retrying", flush=True)
                continue
            for j, s in enumerate(chunk):
                save_one(cdir, cond, s, imgs[j], lat[j:j + 1])
                made += 1
            print(f"[extra] {key} seeds {chunk} done ({made} total)", flush=True)
            i += len(chunk)
        print(f"[extra] {key} complete", flush=True)
    print(f"[extra] all done -- {made} new images", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--start", type=int, default=3)
    ap.add_argument("--end", type=int, default=25, help="exclusive")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--ref_cfg", type=float, default=1.0)
    ap.add_argument("--mem", default="bnb4", choices=["bnb4", "seq_offload"])
    main(ap.parse_args())
