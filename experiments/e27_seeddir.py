"""E27 - Concept directions in seed space (CLIP->latent pullback) + heavy opt.

Idea (continues E25/E26). Instead of optimizing every seed per-prompt, compute ONE
text-conditioned direction `v` in latent space per prompt and just ADD it to any seed:
    z' = renorm(z0 + s * sqrt(d) * v),   renorm = (.-mean)/std  ->  ||z'|| = sqrt(d).

Two stages, which compose into ONE chain-rule backward pass:
  Stage 1 (CLIP space): g = unit direction that raises cosine( image-embedding, CLIP_text(c) ),
    obtained as the one-step cosine gradient at a base image's embedding:
        g = normalize( e_text - <e_text, e0> e0 )   (e0 = a base image's CLIP embedding).
  Stage 2 (decoder pullback): v = normalize( J^T g ),  J = Jacobian of CLIP_image . decode,
    i.e. v = normalize( grad_z <CLIP_image(decode(z_base)), g> ).
Composed: v_chain = normalize( grad_z cosine(CLIP_image(decode(z)), text) )  -- a single pass
(g = e_text anchored at the base latent's OWN decoded image). The intermediate normalization
of g is irrelevant since we normalize v at the end.

We sweep the Stage-1 ANCHOR image e0 (does it matter?):
  chain  - g = e_text directly (pure chain rule, anchored at the base's own decode)
  noise  - e0 from random-pixel images
  mean   - e0 from the mean of a pool of images
  fit    - e0 from an image that MATCHES c
  nofit  - e0 from an image that does NOT match c

Arm A: additive direction, sweep strength s (incl. -v).
Arm B: heavy per-seed optimization (reuse e26.optimize_seed), sweep #steps N -> see the
       over-optimization / CLIP-adversarial regime.

Run:  python experiments/e27_seeddir.py [quick]
Out:  experiments/results/e27/{grid_direction.png, grid_anchors.png, grid_heavy.png,
      deltaclip.png, report.json, pairs/}
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import sys
import time

import numpy as np
import torch
from PIL import Image

from clip_sim import load_clip, clip_image_features, clip_text_features, cosine
from common import RESULTS, save_grid, generate
from e26_seedalign_sdxl import load_sdxl, clip_pixel_values, moments, optimize_seed
from e9_bandnorm_classes import CLASSES

OUT = os.path.join(RESULTS, "e27")
DTYPE = torch.float16
SIZE = int(os.environ.get("E27_SIZE", 1024))
GEN_STEPS = int(os.environ.get("E27_GEN_STEPS", 40))
GUIDANCE = float(os.environ.get("E27_GUIDANCE", 7.0))

SEEDS = [int(s) for s in os.environ.get("E27_SEEDS", "0,1").split(",")]
# strength as fraction of sqrt(d): s == ratio of added-vector norm to the seed norm,
# so s=1 is a ~45deg tilt (destroys structure). Useful regime is small s.
S_SWEEP = [0.0, 0.1, 0.25, 0.5, 1.0, -0.25]
ANCHOR_S = float(os.environ.get("E27_ANCHOR_S", 0.25))   # fixed s for the anchor comparison
HEAVY_N = [0, 1, 5, 20, 60]                  # Arm B optimization steps
HEAVY_LR = float(os.environ.get("E27_HEAVY_LR", 0.05))
ANCHORS = ["chain", "noise", "mean", "fit", "nofit"]
NOFIT_PROMPT = "a close-up photograph of green grass texture"
B_BASES = int(os.environ.get("E27_BASES", 2))    # base latents for the pullback average
R_NOISE = 6                                       # random images for the 'noise' anchor


# ---- image helpers -----------------------------------------------------------
def random_pil(gen, size=256):
    arr = (torch.rand(size, size, 3, generator=gen) * 255).to(torch.uint8).numpy()
    return Image.fromarray(arr)


def mean_pil(pils, size=256):
    stack = np.stack([np.asarray(p.convert("RGB").resize((size, size)), np.float32)
                      for p in pils])
    return Image.fromarray(stack.mean(0).clip(0, 255).astype("uint8"))


def img_emb(clip_model, clip_proc, pils):
    """unit (D,) mean CLIP image embedding over a list of PIL images, on cuda."""
    f = clip_image_features(clip_model, clip_proc, pils).mean(0)   # (D,) cpu
    return (f / f.norm()).to("cuda").float()


# ---- stage 1: CLIP-space directions for a prompt -----------------------------
def clip_directions(clip_model, clip_proc, c, anchor_embs):
    """Return {anchor: unit CLIP-space direction} and e_text. 'chain' = e_text itself."""
    et = clip_text_features(clip_model, clip_proc, [c])[0].to("cuda").float()  # (D,) unit
    dirs = {"chain": et}
    for name, e0 in anchor_embs.items():
        g = et - (et @ e0) * e0          # tangential component (one cosine-grad step)
        dirs[name] = g / (g.norm() + 1e-8)
    return dirs, et


# ---- stage 2: pull CLIP directions back to latent space ----------------------
def pullback(pipe, clip_model, bases, dirs):
    """v_k = normalize( avg_base grad_z <CLIP_image(decode(z_base)), dirs[k]> ).

    Separate forward+backward per (base, direction) so the big 1024px decode graph is
    freed each time (retaining it across all directions OOMs a 24GB card)."""
    sf = pipe.vae.config.scaling_factor
    acc = {k: None for k in dirs}
    for zb in bases:
        for k, d in dirs.items():
            z = zb.clone().detach().requires_grad_(True)
            img = pipe.vae.decode((z / sf).to(pipe.vae.dtype)).sample
            feat = clip_model.get_image_features(
                pixel_values=clip_pixel_values(img).to(DTYPE)).float().squeeze(0)
            e = feat / feat.norm()                          # (D,) unit
            g = torch.autograd.grad(e @ d, z)[0].detach()
            acc[k] = g if acc[k] is None else acc[k] + g
            del z, img, feat, e, g
        torch.cuda.empty_cache()
    return {k: (a / (a.norm() + 1e-8)) for k, a in acc.items()}


def renorm(z):
    return (z - z.mean()) / (z.std() + 1e-8)


def oom_retry(fn, *args, tries=6, wait=30, **kw):
    """Retry a heavy GPU op through transient shared-GPU OOM spikes."""
    for i in range(tries):
        try:
            return fn(*args, **kw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if i == tries - 1:
                raise
            time.sleep(wait)


# ---- main --------------------------------------------------------------------
def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    classes = CLASSES[:1] if quick else CLASSES[:5]
    seeds = SEEDS[:1] if quick else SEEDS
    s_sweep = [0.0, 1.0, -1.0] if quick else S_SWEEP
    heavy_n = [0, 1, 5] if quick else HEAVY_N
    os.makedirs(os.path.join(OUT, "pairs"), exist_ok=True)

    pipe = load_sdxl()
    clip_model, clip_proc = load_clip()
    clip_model = clip_model.to(DTYPE)
    clip_model.requires_grad_(False)

    shape = (1, pipe.unet.config.in_channels, SIZE // 8, SIZE // 8)
    d = shape[1] * shape[2] * shape[3]
    sqrt_d = d ** 0.5
    # shared base latents for the pullback (fixed across prompts/anchors -> fair compare)
    bg = torch.Generator("cuda").manual_seed(1234)
    bases = [torch.randn(shape, generator=bg, device="cuda", dtype=torch.float32)
             for _ in range(B_BASES)]
    cg = torch.Generator().manual_seed(777)
    noise_pils = [random_pil(cg) for _ in range(R_NOISE)]
    print(f"SDXL {SIZE}px d={d} sqrt(d)={sqrt_d:.1f} prompts={len(classes)} seeds={seeds} "
          f"s={s_sweep} N={heavy_n} bases={B_BASES}", flush=True)

    @torch.no_grad()
    def gen(prompt, z):
        import time
        for attempt in range(5):                     # survive transient shared-GPU spikes
            try:
                out = generate(pipe, prompt, z, steps=GEN_STEPS, guidance=GUIDANCE)
                torch.cuda.empty_cache()
                return out
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt == 4:
                    raise
                time.sleep(30)

    # anchor images that don't depend on the prompt's *content* directly
    nofit_img = gen(NOFIT_PROMPT, bases[0])     # one shared "doesn't fit" image
    e_noise = img_emb(clip_model, clip_proc, noise_pils)
    e_nofit = img_emb(clip_model, clip_proc, [nofit_img])

    records = {"config": {"size": SIZE, "d": d, "sqrt_d": sqrt_d, "s_sweep": s_sweep,
                          "anchor_s": ANCHOR_S, "heavy_n": heavy_n, "heavy_lr": HEAVY_LR,
                          "anchors": ANCHORS, "bases": B_BASES, "gen_steps": GEN_STEPS,
                          "guidance": GUIDANCE},
               "prompts": {}}
    rowsA, labA, rowsAnc, labAnc, rowsH, labH = [], [], [], [], [], []
    fit_pils = []

    for key, c in classes:
        # fit anchor = a generation of c; mean anchor = mean of fit images so far + this one
        fit_img = gen(c, bases[0])
        fit_pils.append(fit_img)
        e_fit = img_emb(clip_model, clip_proc, [fit_img])
        e_mean = img_emb(clip_model, clip_proc, [mean_pil(fit_pils)])
        anchor_embs = {"noise": e_noise, "mean": e_mean, "fit": e_fit, "nofit": e_nofit}

        dirs, et = clip_directions(clip_model, clip_proc, c, anchor_embs)
        et_cpu = et.detach().cpu()                             # for cosine vs cpu img feats
        vdirs = oom_retry(pullback, pipe, clip_model, bases, dirs)   # {anchor: unit latent v}

        # pairwise cosine between the latent directions
        cosmat = {a: {b: round(cosine(vdirs[a], vdirs[b]), 3) for b in dirs} for a in dirs}
        prec = {"prompt": c, "v_cos_matrix": cosmat, "seeds": {}}

        for seed in seeds:
            sg = torch.Generator("cuda").manual_seed(seed)
            z0 = torch.randn(shape, generator=sg, device="cuda", dtype=torch.float32)
            srec = {}

            # ---- Arm A: additive direction (v_chain), strength sweep ----
            v = vdirs["chain"]
            imgsA, clipA = [], {}
            for s in s_sweep:
                z = renorm(z0 + s * sqrt_d * v) if s != 0 else z0
                img = gen(c, z)
                imgsA.append(img)
                img.save(os.path.join(OUT, "pairs", f"{key}_s{seed}_A_s{s:+.2f}.png"))
            featsA = clip_image_features(clip_model, clip_proc, imgsA)
            clipA = {f"{s:+.2f}": cosine(featsA[i], et_cpu) for i, s in enumerate(s_sweep)}
            rowsA.append(imgsA)
            labA.append(f"{key} s{seed}")

            # ---- anchor comparison at fixed s=1 ----
            imgsAnc = [imgsA[s_sweep.index(0.0)]]                      # baseline reuse
            anc_cols = ["baseline"]
            clipAnc = {"baseline": cosine(featsA[s_sweep.index(0.0)], et_cpu)}
            for a in ANCHORS:
                z = renorm(z0 + ANCHOR_S * sqrt_d * vdirs[a])
                img = gen(c, z)
                imgsAnc.append(img)
                anc_cols.append(a)
                img.save(os.path.join(OUT, "pairs", f"{key}_s{seed}_anchor_{a}.png"))
            featsAnc = clip_image_features(clip_model, clip_proc, imgsAnc[1:])
            for i, a in enumerate(ANCHORS):
                clipAnc[a] = cosine(featsAnc[i], et_cpu)
            rowsAnc.append(imgsAnc)
            labAnc.append(f"{key} s{seed}")

            # ---- Arm B: heavy per-seed optimization ----
            snap_at = set(n for n in heavy_n if n > 0)
            snaps, hist = oom_retry(optimize_seed, pipe, clip_model, z0,
                                    et.detach().cpu(), HEAVY_LR, max(heavy_n), snap_at)
            imgsH, clipH = [], {}
            for n in heavy_n:
                z = z0 if n == 0 else snaps[n]
                img = gen(c, z)
                imgsH.append(img)
                img.save(os.path.join(OUT, "pairs", f"{key}_s{seed}_B_N{n}.png"))
            featsH = clip_image_features(clip_model, clip_proc, imgsH)
            clipH = {str(n): cosine(featsH[i], et_cpu) for i, n in enumerate(heavy_n)}
            rowsH.append(imgsH)
            labH.append(f"{key} s{seed}")

            srec = {"armA_clip_by_s": clipA, "anchor_clip": clipAnc,
                    "armB_clip_by_N": clipH, "obj_hist": hist,
                    "z_norm_chain_s1": moments(renorm(z0 + sqrt_d * v))["norm"]}
            prec["seeds"][str(seed)] = srec
            torch.cuda.empty_cache()
            print(f"[{key} s{seed}] armA Δclip(s=1)="
                  f"{clipA['+1.00'] - clipA['+0.00']:+.3f}  "
                  f"armB Δclip(N={max(heavy_n)})="
                  f"{clipH[str(max(heavy_n))] - clipH['0']:+.3f}  "
                  f"v_cos(chain,fit)={cosmat['chain']['fit']} "
                  f"v_cos(chain,nofit)={cosmat['chain']['nofit']}", flush=True)
        records["prompts"][key] = prec

    save_grid(rowsA, labA, [f"s={s:+.2f}" for s in s_sweep],
              os.path.join(OUT, "grid_direction.png"))
    save_grid(rowsAnc, labAnc, ["baseline"] + ANCHORS,
              os.path.join(OUT, "grid_anchors.png"))
    save_grid(rowsH, labH, [f"N={n}" for n in heavy_n],
              os.path.join(OUT, "grid_heavy.png"))
    plot_deltaclip(records, s_sweep, heavy_n, os.path.join(OUT, "deltaclip.png"))
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nwrote {OUT}/{{grid_direction.png, grid_anchors.png, grid_heavy.png, "
          f"deltaclip.png, report.json}}")


def plot_deltaclip(records, s_sweep, heavy_n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def collect(key, sub, xs):
        # mean over (prompt, seed) of clip[x] - clip[baseline]
        out = {x: [] for x in xs}
        for pr in records["prompts"].values():
            for sr in pr["seeds"].values():
                tab = sr[sub]
                base = tab[key(0.0)] if sub != "armB_clip_by_N" else tab["0"]
                for x in xs:
                    out[x].append(tab[key(x)] - base)
        return [sum(out[x]) / len(out[x]) for x in xs]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    sm = collect(lambda s: f"{s:+.2f}", "armA_clip_by_s", s_sweep)
    ax[0].plot(s_sweep, sm, "o-")
    ax[0].axhline(0, color="k", lw=0.8); ax[0].axvline(0, color="grey", lw=0.6, ls=":")
    ax[0].set_xlabel("s (fraction of sqrt(d))"); ax[0].set_ylabel("mean ΔCLIP-T")
    ax[0].set_title("Arm A: additive v_chain"); ax[0].grid(alpha=0.3)

    nm = collect(lambda n: str(n), "armB_clip_by_N", heavy_n)
    ax[1].plot(heavy_n, nm, "o-", color="C3")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("N optimization steps"); ax[1].set_ylabel("mean ΔCLIP-T")
    ax[1].set_title("Arm B: heavy optimization"); ax[1].grid(alpha=0.3)

    ancs = ["baseline"] + ANCHORS
    vals = {a: [] for a in ancs}
    for pr in records["prompts"].values():
        for sr in pr["seeds"].values():
            for a in ancs:
                vals[a].append(sr["anchor_clip"][a])
    base = sum(vals["baseline"]) / len(vals["baseline"])
    bars = [sum(vals[a]) / len(vals[a]) - base for a in ANCHORS]
    ax[2].bar(ANCHORS, bars, color="C2")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_ylabel("mean ΔCLIP-T vs baseline")
    ax[2].set_title(f"Anchor (s={records['config'].get('anchor_s', 0.25)})")
    ax[2].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
