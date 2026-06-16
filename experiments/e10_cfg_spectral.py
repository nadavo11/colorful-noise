"""E10: classifier-free guidance inflates spectral power (the SBN motivation).

Flux is a flow model: the transformer predicts a velocity v_theta(x,t) trained by
flow-matching to fit the (conditional) data velocity. The CFG scale equation is
NOT in the training loss -- CFG is an inference-time extrapolation

    v~ = v_u + w * (v_c - v_u).

Only w=1 integrates the trained field; w>1 pushes the trajectory off the data
manifold. This experiment shows that the latent's spectral power / PSD / spectral
norm rises monotonically with the true-CFG scale w, while the trained field (w=1)
and *real images* sit at the bottom -- i.e. CFG inflates the spectral scale above
the natural one, which is exactly what band-normalization clamps back.

We use diffusers' real two-pass CFG (`true_cfg_scale` + an empty `negative_prompt`)
and hold the distilled `guidance_scale` fixed at a neutral 1.0, so the sweep
isolates the cfg-equation effect rather than Flux's distilled guidance embedding.

Parts (--part, comma list):
  download -- fetch --n_real natural photos (seeded picsum.photos, reproducible)
              into results/e10/real_photos/. Network needed only here.
  gen      -- for each class x cfg x seed: true-CFG generation, cache image+latent
              as {key}/tcfg{w}_s{seed}.{png,pt}. Killed runs resume for free.
  real     -- encode each downloaded photo through the Flux VAE into the generation
              latent space; stack to results/e10/real_latents.pt.
  analyze  -- per-latent spectral metrics (lat_std, Fourier power, radial PSD,
              low-band power, spectral norm) + image metrics from the saved PNGs;
              aggregate per cfg and for the real set; write cfg_spectral.json and
              the cfg_power.png / cfg_psd.png plots (into results/e9/plots/ so the
              explainer page can reference them).

Memory: same --mem options as E7 (bnb4 default).
"""
import argparse
import json
import os
import sys
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from spectral_ops import radial_psd
from e7_flux_phase import load_flux_vae, SIZE, REPO
from e9_bandnorm_classes import CLASSES, image_metrics, agg

OUT = os.path.join(RESULTS, "e10")
E9_PLOTS = os.path.join(RESULTS, "e9", "plots")  # site reads ../plots/*.png
LOW_CUT = 0.25  # E7 low/high radial-frequency split


def cfg_tag(w):
    """Filename-safe tag for a cfg value (1.0 -> 'tcfg1.0')."""
    return f"tcfg{w:g}"


# ---------------------------------------------------------------------------
# Part: download real photos
# ---------------------------------------------------------------------------

def run_download(args):
    rdir = os.path.join(OUT, "real_photos")
    os.makedirs(rdir, exist_ok=True)
    got = 0
    for i in range(args.n_real):
        path = os.path.join(rdir, f"photo_{i:03d}.jpg")
        if os.path.exists(path):
            continue
        url = f"https://picsum.photos/seed/e10-{i}/{SIZE}/{SIZE}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "e10/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            Image.open(__import__("io").BytesIO(data)).verify()  # validate
            with open(path, "wb") as f:
                f.write(data)
            got += 1
            print(f"[e10] downloaded {path}", flush=True)
        except Exception as e:
            print(f"[e10] FAILED {url}: {e}", flush=True)
    n = len([p for p in os.listdir(rdir) if p.endswith(".jpg")])
    print(f"[e10] real photos: {got} new, {n} total in {rdir}", flush=True)


# ---------------------------------------------------------------------------
# Part: grow the real pool with MS-COCO val2017 (photographic, ~5k images)
# ---------------------------------------------------------------------------

COCO_ZIP_URLS = ("http://images.cocodataset.org/zips/val2017.zip",
                 "https://images.cocodataset.org/zips/val2017.zip")


def _download_to(path, urls, chunk=1 << 20):
    """Chunked download to path (via a .part temp) trying each url in turn."""
    if os.path.exists(path):
        return path
    tmp = path + ".part"
    last = None
    for url in urls:
        try:
            print(f"[e10] downloading {url} -> {path}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "e10/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
            os.replace(tmp, path)
            return path
        except Exception as e:
            last = e
            print(f"[e10] FAILED {url}: {e}", flush=True)
    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(f"could not fetch COCO zip: {last}")


def run_coco(args):
    """Fetch the first --n_coco MS-COCO val2017 photos into real_photos/ as
    coco_*.jpg. The zip is cached once under results/_cache; only the requested
    images are extracted. Re-run `--part real` afterwards to rebuild
    real_latents.pt from the (now larger) pool."""
    import zipfile
    rdir = os.path.join(OUT, "real_photos")
    os.makedirs(rdir, exist_ok=True)
    cache = os.path.join(RESULTS, "_cache")
    os.makedirs(cache, exist_ok=True)
    zpath = _download_to(os.path.join(cache, "coco_val2017.zip"), COCO_ZIP_URLS)

    got = 0
    with zipfile.ZipFile(zpath) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".jpg"))
        for n in names[: args.n_coco]:
            out_p = os.path.join(rdir, f"coco_{os.path.basename(n)}")
            if os.path.exists(out_p):
                continue
            with zf.open(n) as src, open(out_p, "wb") as dst:
                dst.write(src.read())
            got += 1
    total = len([p for p in os.listdir(rdir) if p.endswith(".jpg")])
    print(f"[e10] COCO: {got} new images extracted; {total} total in {rdir}",
          flush=True)


# ---------------------------------------------------------------------------
# Part: true-CFG sweep generation
# ---------------------------------------------------------------------------

def load_flux_preencoded(prompts, neg=""):
    """Load Flux (bnb4 transformer on GPU), pre-encode every prompt + the negative,
    then DROP the T5/CLIP text encoders so the denoising loop only needs the
    transformer + VAE (both GPU-resident, no CPU offload).

    This removes the ~10GB text-encoder RAM footprint that otherwise gets the
    process OOM-killed on a memory-contended shared box, and skips re-encoding the
    prompt on every generation. Returns (pipe, embeds) with
    embeds[text] = (prompt_embeds_cpu, pooled_embeds_cpu).
    """
    import gc
    from diffusers import (FluxPipeline, FluxTransformer2DModel,
                           BitsAndBytesConfig)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16)
    tr = FluxTransformer2DModel.from_pretrained(
        REPO, subfolder="transformer", quantization_config=qc,
        torch_dtype=torch.bfloat16)
    pipe = FluxPipeline.from_pretrained(REPO, transformer=tr,
                                        torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    embeds = {}
    with torch.no_grad():
        for txt in dict.fromkeys(list(prompts) + [neg]):  # dedup, keep order
            pe, ppe, _ = pipe.encode_prompt(
                prompt=txt, prompt_2=txt, device="cuda",
                num_images_per_prompt=1, max_sequence_length=512)
            embeds[txt] = (pe.cpu(), ppe.cpu())
    pipe.text_encoder = pipe.text_encoder_2 = None
    pipe.tokenizer = pipe.tokenizer_2 = None
    gc.collect()
    torch.cuda.empty_cache()
    pipe.vae.to("cuda")
    print(f"[e10] pre-encoded {len(embeds)} prompts; text encoders dropped",
          flush=True)
    return pipe, embeds


def gen_emb(pipe, emb, neg_emb, seed, true_cfg, guidance, steps):
    """One true-CFG generation from cached embeddings -> (PIL, fp32 cpu latent).

    true_cfg>1 supplies negative embeds -> diffusers' real two-pass CFG
    (v_u + w*(v_c-v_u)); true_cfg=1 is a single pass = the trained field.
    """
    from diffusers import FluxPipeline
    captured = {}

    def grab(p, i, t, kw):
        captured["latents"] = kw["latents"]
        return {}

    kw = dict(prompt_embeds=emb[0].cuda(), pooled_prompt_embeds=emb[1].cuda(),
              height=SIZE, width=SIZE, guidance_scale=guidance,
              true_cfg_scale=true_cfg, num_inference_steps=steps,
              generator=torch.Generator("cuda").manual_seed(seed),
              callback_on_step_end=grab)
    if true_cfg > 1:
        kw["negative_prompt_embeds"] = neg_emb[0].cuda()
        kw["negative_pooled_prompt_embeds"] = neg_emb[1].cuda()
    img = pipe(**kw).images[0]
    lat = FluxPipeline._unpack_latents(captured["latents"], SIZE, SIZE,
                                       pipe.vae_scale_factor)
    return img, lat.float().cpu()


def run_gen(args):
    prompts = [p for _, p in CLASSES[: args.num_classes]]
    pipe, embeds = load_flux_preencoded(prompts, neg="")
    neg_emb = embeds[""]
    made = 0
    for key, prompt in CLASSES[: args.num_classes]:
        cdir = os.path.join(OUT, key)
        os.makedirs(os.path.join(cdir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cdir, "latents"), exist_ok=True)
        for w in args.cfgs:
            tag = cfg_tag(w)
            for s in range(args.seeds):
                ip = f"{cdir}/images/{tag}_s{s}.png"
                lp = f"{cdir}/latents/{tag}_s{s}.pt"
                if os.path.exists(ip) and os.path.exists(lp):
                    continue
                img, lat = gen_emb(pipe, embeds[prompt], neg_emb, s, w,
                                   args.guidance, args.steps)
                img.save(ip)
                torch.save(lat, lp)
                made += 1
                print(f"[e10] {key} {tag} s{s} (std={lat.std():.3f})", flush=True)
        print(f"[e10] {key} generation complete", flush=True)
    print(f"[e10] gen done -- {made} new latents", flush=True)


# ---------------------------------------------------------------------------
# Part: encode real photos into the generation latent space
# ---------------------------------------------------------------------------

def _square(img, size):
    """Center-crop to the shorter side, then resize to size x size -- avoids the
    aspect-ratio distortion (which warps the radial PSD) that a plain resize of a
    non-square photo would introduce. Square inputs (e.g. picsum) are unchanged."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size))


def run_real(args):
    rdir = os.path.join(OUT, "real_photos")
    photos = sorted(p for p in os.listdir(rdir) if p.endswith(".jpg"))
    assert photos, f"no photos in {rdir} -- run --part download/coco first"
    vae = load_flux_vae()
    sf, shift = vae.config.scaling_factor, vae.config.shift_factor
    lats = []
    for p in photos:
        img = _square(Image.open(os.path.join(rdir, p)).convert("RGB"), SIZE)
        x = torch.from_numpy(np.asarray(img).copy()).float() / 255.0
        x = (x.permute(2, 0, 1)[None] * 2 - 1).to(vae.dtype).cuda()  # [-1,1]
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mean  # raw VAE latent
        # invert flux_vae_decode (z = lat/sf + shift) -> generation-space latent
        lat = ((z - shift) * sf).float().cpu()
        lats.append(lat)
        print(f"[e10] encoded {p} (std={lat.std():.3f})", flush=True)
    real = torch.cat(lats, 0)  # (N,16,128,128)
    torch.save(real, os.path.join(OUT, "real_latents.pt"))
    print(f"[e10] saved {real.shape[0]} real latents -> real_latents.pt", flush=True)


# ---------------------------------------------------------------------------
# Part: analyze
# ---------------------------------------------------------------------------

def lat_spectral(lat):
    """Spectral metrics of one (1,16,128,128) latent."""
    centers, psd = radial_psd(lat.cuda())          # psd: (C, n_bins)
    cmean = psd.mean(0)                             # (n_bins,) channel-averaged
    low = centers < LOW_CUT
    spec_norm = float(torch.linalg.matrix_norm(lat[0].float(), ord=2).mean())
    return {
        "lat_std": float(lat.std()),
        "power": float((lat.float() ** 2).mean()),  # Parseval: mean spectral power
        "low_power": float(cmean[low].mean()),
        "low_frac": float(psd[:, low].sum() / psd.sum()),
        "spec_norm": spec_norm,
    }, centers.tolist(), cmean.tolist()


def collect(latents, pngs):
    """Aggregate spectral + image metrics over a list of (latent, png?) pairs."""
    spec_keys = ["lat_std", "power", "low_power", "low_frac", "spec_norm"]
    img_keys = ["rms_contrast", "saturation", "hf_frac"]
    acc = {k: [] for k in spec_keys + img_keys}
    psd_curves, centers = [], None
    for lat, png in zip(latents, pngs):
        sm, centers, curve = lat_spectral(lat)
        for k in spec_keys:
            acc[k].append(sm[k])
        psd_curves.append(curve)
        if png is not None and os.path.exists(png):
            im = image_metrics(Image.open(png))
            for k in img_keys:
                acc[k].append(im[k])
    out = {k: agg(v) for k, v in acc.items() if v}
    out["psd_centers"] = centers
    out["psd"] = np.mean(psd_curves, 0).tolist()
    out["n"] = len(latents)
    return out


def run_analyze(args):
    classes = [k for k, _ in CLASSES[: args.num_classes]]
    per_cfg = {}
    for w in args.cfgs:
        tag = cfg_tag(w)
        lats, pngs = [], []
        for key in classes:
            cdir = os.path.join(OUT, key)
            for s in range(args.seeds):
                lp = f"{cdir}/latents/{tag}_s{s}.pt"
                if os.path.exists(lp):
                    lats.append(torch.load(lp, weights_only=True))
                    pngs.append(f"{cdir}/images/{tag}_s{s}.png")
        if lats:
            per_cfg[f"{w:g}"] = collect(lats, pngs)
            c = per_cfg[f"{w:g}"]
            print(f"[e10] cfg={w:g}: std={c['lat_std']['mean']:.3f} "
                  f"power={c['power']['mean']:.3f} "
                  f"specnorm={c['spec_norm']['mean']:.1f} n={c['n']}", flush=True)

    real_path = os.path.join(OUT, "real_latents.pt")
    real = None
    if os.path.exists(real_path):
        rl = torch.load(real_path, weights_only=True)
        real = collect([rl[i:i + 1] for i in range(rl.shape[0])],
                       [None] * rl.shape[0])
        print(f"[e10] real: std={real['lat_std']['mean']:.3f} "
              f"power={real['power']['mean']:.3f} "
              f"specnorm={real['spec_norm']['mean']:.1f} n={real['n']}", flush=True)

    report = {
        "axis": "true_cfg",
        "guidance_scale": args.guidance,
        "steps": args.steps,
        "num_classes": args.num_classes,
        "seeds": args.seeds,
        "cfgs": [f"{w:g}" for w in args.cfgs],
        "per_cfg": per_cfg,
        "real": real,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cfg_spectral.json"), "w") as f:
        json.dump(report, f, indent=2)
    make_plots(report)
    print(f"[e10] wrote cfg_spectral.json + plots", flush=True)


def make_plots(report):
    os.makedirs(E9_PLOTS, exist_ok=True)
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    cfgs = [float(w) for w in report["cfgs"]]
    pc = report["per_cfg"]
    real = report.get("real")

    # --- cfg_power.png : intensity metrics vs cfg, with real-image band ---
    metrics = [("power", "Fourier power  mean|X|²"),
               ("lat_std", "latent std"),
               ("spec_norm", "spectral norm  (mean σₘₐₓ)"),
               ("low_power", "low-band power")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    for ax, (mk, title) in zip(axes, metrics):
        xs = [w for w in cfgs if f"{w:g}" in pc]
        ys = [pc[f"{w:g}"][mk]["mean"] for w in xs]
        es = [pc[f"{w:g}"][mk]["std"] / max(pc[f"{w:g}"][mk]["n"], 1) ** 0.5
              for w in xs]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color="#58a6ff")
        if real and mk in real:
            rm = real[mk]["mean"]
            rs = real[mk]["std"] / max(real[mk]["n"], 1) ** 0.5
            ax.axhspan(rm - rs, rm + rs, color="#3fb950", alpha=0.18)
            ax.axhline(rm, color="#3fb950", ls="--", lw=1.4, label="real images")
            ax.legend(fontsize=8)
        ax.set(xlabel="true-CFG scale w", title=title)
        ax.grid(alpha=0.2)
    fig.suptitle("CFG inflates the latent's spectral power — real photos "
                 "sit near standard guidance, above the unguided field", fontsize=11)
    fig.tight_layout()
    for d in (E9_PLOTS, os.path.join(OUT, "plots")):
        fig.savefig(os.path.join(d, "cfg_power.png"), dpi=120,
                    bbox_inches="tight")
    plt.close(fig)

    # --- cfg_psd.png : radial PSD curves per cfg + real ---
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cmap = plt.cm.viridis
    xs = [w for w in cfgs if f"{w:g}" in pc]
    for j, w in enumerate(xs):
        c = pc[f"{w:g}"]
        ax.loglog(c["psd_centers"], c["psd"], color=cmap(j / max(len(xs) - 1, 1)),
                  label=f"w={w:g}")
    if real:
        ax.loglog(real["psd_centers"], real["psd"], "k--", lw=2,
                  label="real images")
    ax.set(xlabel="radial frequency", ylabel="power",
           title="Latent radial PSD vs true-CFG scale")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    for d in (E9_PLOTS, os.path.join(OUT, "plots")):
        fig.savefig(os.path.join(d, "cfg_psd.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(args):
    os.makedirs(OUT, exist_ok=True)
    runners = {"download": run_download, "coco": run_coco, "gen": run_gen,
               "real": run_real, "analyze": run_analyze}
    for part in filter(None, (p.strip() for p in args.part.split(","))):
        runners[part](args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="download,gen,real,analyze")
    ap.add_argument("--cfgs", type=lambda s: [float(x) for x in s.split(",")],
                    default=[1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="distilled guidance_scale, held fixed (neutral)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--num_classes", type=int, default=len(CLASSES))
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--n_real", type=int, default=20)
    ap.add_argument("--n_coco", type=int, default=500,
                    help="MS-COCO val2017 photos to add to the real pool")
    ap.add_argument("--mem", default="bnb4", choices=["bnb4", "seq_offload"])
    main(ap.parse_args())
