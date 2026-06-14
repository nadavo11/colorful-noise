"""E19: generation-time spectral style transfer on SD3.5 -- "AdaIN-in-Fourier".

The headline of the two-image-spectrum direction. E18 showed offline that a
latent's PHASE fixes content while its per-(channel, radial band) POWER supplies
style. E19 moves that into generation: a CONTENT PROMPT provides layout (phase +
the per-step energy trajectory) while a STYLE IMAGE supplies the radial power
envelope, by clamping the denoising latent each step to a style-bent SBN
reference (style_ops.build_style_reference -> e17_sd35.ClampPSD3).

This is exactly SBN (E8/E17) with the reference's per-band shape bent toward a
style image instead of the prompt's own cfg=1 spectrum. strength interpolates:
  strength=0  -> plain SBN (content's own spectrum)   == the `sbn` condition
  strength=1  -> content energy redistributed into the style's radial envelope

ISOTROPY CAVEAT (probed, not hidden): the clamp is radial, so it transfers
texture-energy + palette/contrast, NOT oriented strokes. Expect palette/grain/
contrast to move toward the style while brush-direction does not.

Conditions (SD3.5-medium, shared seeded init per seed):
  cfg_hi              guidance=W high-CFG baseline (no clamp)
  sbn                 plain band-norm (strength=0 reference)            [ours, E17]
  style_{sid}_s{p}    SBN clamped to content-energy x style_{sid} envelope (ours)

Metrics: content preservation (CLIP-I to the cfg_hi baseline image) + style match
(CLIP-I to the style image; latent band-power distance to the style; colorfulness
toward the style) + fidelity/adherence guards (aesthetic, ImageReward, CLIP-T).

Parts (--part): preflight (model-free reference-math asserts) / gen / score /
analyze. results/e19/<pid>/... layout mirrors E17.
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid
from spectral_ops import band_index_map
from style_ops import build_style_reference, latent_band_power, style_gain
from e17_sd35 import (load_sd35, load_sd35_vae, gen_sd3, record_reference_sd3,
                      ClampPSD3, RecordPSD3, sd3_vae_encode, H, W, N_CH)
from e16_prompt_adherence import DETAILED, agg
from e9_bandnorm_classes import cached_gen, image_metrics, load_set, METRICS
from e9_clipt import load_clip, clip_scores
from clip_sim import clip_image_features, cosine
from fidelity_metrics import (load_aesthetic, aesthetic_scores,
                              load_imagereward, imagereward_scores)

OUT = os.path.join(RESULTS, "e19")


# ---------------------------------------------------------------------------
# Style bank: real images -> (sid, latent band power) descriptors
# ---------------------------------------------------------------------------

def load_style_bank(args, idx_map):
    """[(sid, band_power (C,n_bins), pil)] from --styles dir (falls back to a few
    e10 photos as placeholder styles so the pipeline runs without extra assets)."""
    sdir = args.styles or os.path.join(RESULTS, "e10", "real_photos")
    files = sorted(p for p in os.listdir(sdir)
                   if p.lower().endswith((".jpg", ".jpeg", ".png")))[: args.num_styles]
    assert files, f"no style images in {sdir}"
    vae = load_sd35_vae(args.mem)
    bank = []
    for f in files:
        pil = Image.open(os.path.join(sdir, f)).convert("RGB").resize((1024, 1024))
        lat = sd3_vae_encode(vae, pil).cuda()
        bank.append((os.path.splitext(f)[0], latent_band_power(lat, idx_map, args.n_bins),
                     pil))
        print(f"[e19] style '{f}' encoded", flush=True)
    del vae
    torch.cuda.empty_cache()
    return bank


# ---------------------------------------------------------------------------
# preflight: reference-building math, no transformer
# ---------------------------------------------------------------------------

def preflight(args, report=None):
    print("[e19] pre-flight asserts (model-free) ...", flush=True)
    torch.manual_seed(0)
    nb = args.n_bins
    steps = args.steps
    content_ref = {"band": torch.rand(steps, N_CH, nb) + 0.1,
                   "total": torch.rand(steps) + 1, "std": torch.rand(steps) + 1}
    style_band = torch.rand(N_CH, nb) + 0.1

    r0 = build_style_reference(content_ref, style_band, strength=0.0)
    assert torch.allclose(r0["band"], content_ref["band"], atol=1e-5), \
        "strength=0 must reproduce plain SBN reference"

    r1 = build_style_reference(content_ref, style_band, strength=1.0)
    assert r1["band"].shape == content_ref["band"].shape
    # final step: content per-channel energy redistributed by the style shape
    from style_ops import band_shape
    exp = content_ref["band"][-1].sum(-1, keepdim=True) * band_shape(style_band)
    assert torch.allclose(r1["band"][-1], exp, atol=1e-4), "style envelope mismatch"

    g = style_gain(content_ref["band"][-1], style_band, 1.0, gmax=args.gmax)
    assert float(g.max()) <= args.gmax + 1e-5 and float(g.min()) >= 1.0 / args.gmax - 1e-5, \
        "gmax clamp not applied"
    print(f"[e19] pre-flight OK (strength0==SBN, gmax in [{1/args.gmax:.2f},{args.gmax}])",
          flush=True)


# ---------------------------------------------------------------------------
# gen
# ---------------------------------------------------------------------------

def style_conds(bank, strengths):
    return [f"style_{sid}_s{p:g}" for sid, _, _ in bank for p in strengths]


def run_gen(args, report):
    idx_map = band_index_map(H, W, args.n_bins, "cuda")
    bank = load_style_bank(args, idx_map)
    pipe = load_sd35(args.mem)
    Wc = args.cfg

    for pid, prompt in DETAILED[: args.num_prompts]:
        cdir = f"{OUT}/{pid}"
        os.makedirs(f"{cdir}/images", exist_ok=True)
        os.makedirs(f"{cdir}/latents", exist_ok=True)

        ref_path = f"{cdir}/ref_psd.pt"
        if os.path.exists(ref_path):
            ref = torch.load(ref_path, weights_only=True)
            print(f"[e19] {pid} reference (cached)", flush=True)
        else:
            ref, _ = record_reference_sd3(pipe, prompt, args.ref_seeds,
                                          args.steps, args.n_bins, 1.0)
            torch.save(ref, ref_path)
            print(f"[e19] {pid} reference recorded", flush=True)

        styled = {sid: {p: build_style_reference(ref, sb, p, gmax=args.gmax)
                        for p in args.strengths} for sid, sb, _ in bank}

        for s in range(args.seeds):
            cached_gen(cdir, f"cfg_hi_s{s}", lambda: gen_sd3(
                pipe, prompt, s, Wc, args.steps))
            cached_gen(cdir, f"sbn_s{s}", lambda: gen_sd3(
                pipe, prompt, s, Wc, args.steps,
                cb_obj=ClampPSD3(ref, idx_map, args.n_bins)))
            for sid, _, _ in bank:
                for p in args.strengths:
                    sref = styled[sid][p]
                    cached_gen(cdir, f"style_{sid}_s{p:g}_s{s}", lambda sref=sref:
                               gen_sd3(pipe, prompt, s, Wc, args.steps,
                                       cb_obj=ClampPSD3(sref, idx_map, args.n_bins)))
        print(f"[e19] {pid} generation complete", flush=True)

    report["styles"] = [sid for sid, _, _ in bank]
    report["conds"] = ["cfg_hi", "sbn"] + style_conds(bank, args.strengths)
    import gc
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# score: content preservation vs style match (+ fidelity/adherence guards)
# ---------------------------------------------------------------------------

def run_score(args, report):
    idx_map = band_index_map(H, W, args.n_bins, "cuda")
    bank = load_style_bank(args, idx_map)
    style_pils = {sid: pil for sid, _, pil in bank}
    style_band = {sid: sb for sid, sb, _ in bank}
    conds = report.get("conds") or (["cfg_hi", "sbn"] + style_conds(bank, args.strengths))

    clip_model, proc = load_clip(args.clip_model)
    mlp, ir = load_aesthetic(), load_imagereward()
    style_feats = {sid: clip_image_features(clip_model, proc, [pil])[0]
                   for sid, pil in style_pils.items()}

    scores = {}
    for pid, prompt in DETAILED[: args.num_prompts]:
        cdir = f"{OUT}/{pid}"
        base_imgs, base_tags = load_set(cdir, "cfg_hi", args.seeds)
        base_feat = (clip_image_features(clip_model, proc, base_imgs)
                     if base_imgs else None)
        per_cond = {}
        for cond in conds:
            imgs, tags = load_set(cdir, cond, args.seeds)
            if not imgs:
                continue
            paths = [f"{cdir}/images/{t}.png" for t in tags]
            lats = [torch.load(f"{cdir}/latents/{t}.pt", weights_only=True).cuda()
                    for t in tags]
            feats = clip_image_features(clip_model, proc, imgs)
            sid = _cond_style(cond)
            entry = {
                "aesthetic": agg(aesthetic_scores(mlp, clip_model, proc, imgs)),
                "imagereward": agg(imagereward_scores(ir, prompt, paths)),
                "clip_t": agg(clip_scores(clip_model, proc, prompt, imgs)),
                # content preservation: similarity to the unstyled cfg_hi image (paired)
                "content_clip": agg([cosine(f, bf) for f, bf in zip(feats, base_feat)])
                                if base_feat is not None else None,
                "colorfulness": agg([image_metrics(i)["colorfulness"] for i in imgs]),
            }
            if sid and sid in style_band:
                entry["style_clip"] = agg([cosine(f, style_feats[sid]) for f in feats])
                entry["style_band_dist"] = agg(
                    [_band_logdist(latent_band_power(l, idx_map, args.n_bins),
                                   style_band[sid]) for l in lats])
            per_cond[cond] = entry
            print(f"[e19] {pid}/{cond}: content={_m(entry,'content_clip')} "
                  f"style={_m(entry,'style_clip')} aes={_m(entry,'aesthetic')}", flush=True)
        scores[pid] = {"prompt": prompt, "conds": per_cond}

    report["scores"] = scores
    with open(f"{OUT}/scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(f"[e19] wrote {OUT}/scores.json", flush=True)


def _cond_style(cond):
    """style_{sid}_s{p}_s{seed} -> sid (None for cfg_hi/sbn)."""
    if not cond.startswith("style_"):
        return None
    return cond[len("style_"):].rsplit("_s", 2)[0]


def _band_logdist(p, q):
    return float((torch.log(p.clamp(min=1e-8)) -
                  torch.log(q.to(p.device).clamp(min=1e-8))).pow(2).mean().sqrt())


def _m(entry, k):
    return "%.3f" % entry[k]["mean"] if entry.get(k) else "NA"


# ---------------------------------------------------------------------------
# analyze: grids + content/style trade-off table
# ---------------------------------------------------------------------------

def run_analyze(args, report):
    if not os.path.exists(f"{OUT}/scores.json"):
        print("[e19] no scores.json; run --part score first", flush=True)
        return
    scores = json.load(open(f"{OUT}/scores.json"))
    conds = report.get("conds") or list(next(iter(scores.values()))["conds"])
    show = ["content_clip", "style_clip", "style_band_dist", "colorfulness",
            "aesthetic", "imagereward", "clip_t"]

    for pid in scores:
        cdir = f"{OUT}/{pid}"
        rows, labels = [], []
        for cond in conds:
            imgs, _ = load_set(cdir, cond, args.grid_n)
            if imgs:
                rows.append(imgs[: args.grid_n])
                labels.append(cond)
        if rows:
            save_grid(rows, labels, [f"s{s}" for s in range(args.grid_n)],
                      f"{OUT}/grid_{pid}.png")

    table = {c: {m: [] for m in show} for c in conds}
    for pdata in scores.values():
        for cond, e in pdata["conds"].items():
            for m in show:
                if e.get(m):
                    table[cond][m].append(e[m]["mean"])
    cell = lambda xs: "%.3f" % (sum(xs) / len(xs)) if xs else "  -  "
    lines = ["# E19 SD3.5 spectral style transfer (mean over prompts)", "",
             f"cfg={args.cfg}, strengths={args.strengths}. content_clip↑ = layout kept; "
             "style_clip↑ / style_band_dist↓ = style matched.", "",
             "| cond | " + " | ".join(show) + " |", "|" + "---|" * (len(show) + 1)]
    for cond in conds:
        lines.append("| " + cond + " | "
                     + " | ".join(cell(table[cond][m]) for m in show) + " |")
    md = "\n".join(lines) + "\n"
    with open(f"{OUT}/summary.md", "w") as f:
        f.write(md)
    print(md, flush=True)
    print(f"[e19] wrote {OUT}/summary.md + grids", flush=True)


def main(args):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/report.json"
    report = json.load(open(path)) if os.path.exists(path) else {}
    report["params"] = vars(args)
    runners = {"preflight": preflight, "gen": run_gen,
               "score": run_score, "analyze": run_analyze}
    for part in filter(None, (p.strip() for p in args.part.split(","))):
        runners[part](args, report)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e19] report -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="preflight,gen,score,analyze")
    ap.add_argument("--styles", default="", help="dir of style images (default: e10 photos)")
    ap.add_argument("--num_styles", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--ref_seeds", type=int, default=3)
    ap.add_argument("--num_prompts", type=int, default=6)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--gmax", type=float, default=4.0, help="per-band gain clamp [1/g,g]")
    ap.add_argument("--strengths", default="0.5,1.0",
                    type=lambda s: [float(x) for x in s.split(",")])
    ap.add_argument("--grid_n", type=int, default=4)
    ap.add_argument("--mem", default="gpu_resident", choices=["gpu_resident", "offload"])
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    main(ap.parse_args())
