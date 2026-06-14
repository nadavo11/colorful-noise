"""E16: fidelity at high CFG -- SBN vs training-free guidance baselines on Flux.

cfg=1.0 Flux already adheres AND looks realistic on simple prompts, but practice
uses high CFG (~3.5) for DETAILED prompts, where guidance buys composition at the
cost of realism (E10: CFG inflates spectral power; over-saturation/contrast). E16
asks: in that regime, does SBN (band-norm) + a cheap saturation postprocess (E11)
yield HIGHER-FIDELITY images than recent training-free guidance methods, WITHOUT
losing prompt adherence? Fidelity is the contest; adherence is a guardrail.

Conditions (all Flux-dev, shared seeded initial latent per seed):
  cfg1.0      realism anchor (distilled guidance 1.0)
  cfg3.5      degraded high-CFG baseline (distilled guidance 3.5) -- what SBN fixes
  bandnorm    SBN: clamp cfg=3.5 latent PSD to the cfg=1.0 per-step reference
  bandnorm_pp SBN + saturation x{sat} postprocess (E11) -- OUR FULL METHOD
  cfgzero     CFG-Zero* (true-CFG, optimal scale + zero-init)
  negprompt   true-CFG + fidelity negative prompt (NAG proxy at 28 steps)
  seg         Smoothed Energy Guidance (blurred-query self-attention), if available

Metrics -- fidelity (primary): LAION aesthetic, ImageReward, spectral-distance-to
-real (E10). guardrail (adherence): CLIP-T, VQAScore. Plus E9 image_metrics.

Parts (--part):
  gen     per prompt: cfg=1.0 reference (cached, doubles as anchor) + every
          condition per seed, image+latent cached so killed runs resume free.
  score   load CLIP / aesthetic / ImageReward / VQAScore (after diffusion freed)
          + spectral-dist; write results/e16/scores.json.
  analyze paired deltas vs cfg=3.5, per-prompt grids, results/e16/summary.md.

Memory: --mem bnb4 (default), same as E7-E10.
"""
import argparse
import json
import os
import shutil
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid
from spectral_ops import band_index_map
from e7_flux_phase import load_flux
from e8_psd_clamp import RecordPSD, gen_with_cb, preflight as e8_preflight
from bandnorm import record_reference, generate_bandnorm
from e9_bandnorm_classes import (cached_gen, image_metrics, lat_metrics,
                                 load_set, METRICS)
from e9_clipt import load_clip, clip_scores
from e11_color_correct import correct_sat
from e16_baselines import gen_cfgzero, gen_negprompt, gen_seg, seg_available
from fidelity_metrics import (load_aesthetic, aesthetic_scores, load_imagereward,
                              imagereward_scores, load_real_psd, spectral_dist_to_real)
from vqascore import load_vqascore, vqa_scores_paths

OUT = os.path.join(RESULTS, "e16")

DETAILED = [
    ("fisherman", "A weathered elderly fisherman in a bright yellow raincoat mending a fishing net on a wet wooden dock at golden hour, dense fog rolling over the harbor, a distant red lighthouse, seagulls overhead, photorealistic"),
    ("apothecary", "A cluttered Victorian apothecary interior with hundreds of glass bottles on dark wooden shelves, polished brass scales on the counter, dried herbs hanging from ceiling beams, warm flickering lamplight, dust motes in the air"),
    ("market", "A bustling outdoor Moroccan spice market at midday, pyramids of red, yellow and orange spices in woven baskets, a vendor in a blue robe weighing saffron, intricate tile work, a crowd in the background, vivid colors"),
    ("cyberpunk", "A lone detective in a long trench coat standing under a flickering neon sign on a rain-soaked cyberpunk street at night, reflections in the puddles, holographic billboards, steam rising from a manhole, cinematic"),
    ("library", "A grand circular library with three floors of mahogany bookshelves, a spiral wrought-iron staircase, tall arched stained-glass windows casting colored light, a wooden reading table with an open book and a green banker's lamp"),
    ("astronaut", "An astronaut in a detailed white spacesuit planting a small fabric flag on a red rocky Martian ridge, a rover in the background, Earth as a pale blue dot in the dark sky, fine dust kicked up, hyperreal"),
    ("banquet", "An opulent medieval banquet table seen from above, roast game, silver goblets of wine, bunches of grapes, fresh bread, lit candelabras, an embroidered tablecloth, a small dog under the table, Flemish still-life style"),
    ("workshop", "A craftsman's woodworking workshop with sawdust floating in a shaft of afternoon sunlight, hand tools hanging on a pegboard, a half-finished violin clamped on the bench, wood shavings curling on the floor, warm and detailed"),
]

# fidelity (higher better, except spectral_dist which is lower better), then guardrail
FID_METRICS = ["aesthetic", "imagereward", "spectral_dist"]
ADH_METRICS = ["clip_t", "vqascore"]
LOWER_BETTER = {"spectral_dist"}
ALL_CONDS = ["cfg1.0", "cfg3.5", "bandnorm", "bandnorm_pp", "cfgzero",
             "negprompt", "seg"]


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return {"mean": m, "std": sd, "n": len(vals)}


# ---------------------------------------------------------------------------
# Part: gen
# ---------------------------------------------------------------------------

def ensure_pp(cdir, s, sat):
    """Derive bandnorm_pp (saturation postprocess of the bandnorm PNG; latent
    unchanged by an image-level op, so copy it) once the bandnorm pair exists."""
    bi, bl = f"{cdir}/images/bandnorm_s{s}.png", f"{cdir}/latents/bandnorm_s{s}.pt"
    pi, pl = f"{cdir}/images/bandnorm_pp_s{s}.png", f"{cdir}/latents/bandnorm_pp_s{s}.pt"
    if os.path.exists(pi) and os.path.exists(pl):
        return
    if not (os.path.exists(bi) and os.path.exists(bl)):
        return
    correct_sat(Image.open(bi).convert("RGB"), None, sat).save(pi)
    shutil.copy(bl, pl)


def run_gen(args, report):
    pipe = load_flux(args.mem)
    seg_ok = (not args.no_seg) and seg_available(pipe)
    report["seg_available"] = seg_ok
    print(f"[e16] SEG {'enabled' if seg_ok else 'SKIPPED'}", flush=True)
    idx_map = band_index_map(128, 128, args.n_bins, "cuda")

    for pid, prompt in DETAILED[: args.num_prompts]:
        cdir = f"{OUT}/{pid}"
        os.makedirs(f"{cdir}/images", exist_ok=True)
        os.makedirs(f"{cdir}/latents", exist_ok=True)

        # cfg=1.0 reference (first ref_seeds) -- cached as ref_psd.pt + images/latents
        ref_path = f"{cdir}/ref_psd.pt"
        ref_tags = [f"cfg1.0_s{s}" for s in range(args.ref_seeds)]
        if os.path.exists(ref_path) and all(
                os.path.exists(f"{cdir}/images/{t}.png") for t in ref_tags):
            ref = torch.load(ref_path, weights_only=True)
            print(f"[e16] {pid} reference (cached)", flush=True)
        else:
            ref, outs = record_reference(pipe, prompt, args.ref_seeds, 1.0,
                                         args.steps, args.n_bins)
            for t, (img, lat) in zip(ref_tags, outs):
                img.save(f"{cdir}/images/{t}.png")
                torch.save(lat, f"{cdir}/latents/{t}.pt")
            torch.save(ref, ref_path)
            print(f"[e16] {pid} reference recorded", flush=True)

        for s in range(args.seeds):
            # cfg=1.0 anchor beyond the reference seeds
            if s >= args.ref_seeds:
                cached_gen(cdir, f"cfg1.0_s{s}", lambda: gen_with_cb(
                    pipe, prompt, s, 1.0, args.steps,
                    RecordPSD(idx_map, args.n_bins, args.steps)))
            # cfg=3.5 baseline
            cached_gen(cdir, f"cfg3.5_s{s}", lambda: gen_with_cb(
                pipe, prompt, s, args.cfg, args.steps,
                RecordPSD(idx_map, args.n_bins, args.steps)))
            # SBN + postprocess
            cached_gen(cdir, f"bandnorm_s{s}", lambda: generate_bandnorm(
                pipe, prompt, s, ref, args.cfg, args.steps, args.n_bins)[:2])
            ensure_pp(cdir, s, args.sat)
            # baselines
            cached_gen(cdir, f"cfgzero_s{s}", lambda: gen_cfgzero(
                pipe, prompt, s, true_cfg=args.cfg, guidance=1.0, steps=args.steps))
            cached_gen(cdir, f"negprompt_s{s}", lambda: gen_negprompt(
                pipe, prompt, s, true_cfg=args.cfg, guidance=1.0, steps=args.steps))
            if seg_ok:
                cached_gen(cdir, f"seg_s{s}", lambda: gen_seg(
                    pipe, prompt, s, seg_scale=args.seg_scale, sigma=args.seg_sigma,
                    guidance=args.cfg, steps=args.steps))
        print(f"[e16] {pid} generation complete", flush=True)

    # free the diffusion model so the score phase (CLIP/aesthetic/ImageReward/
    # VQAScore, ~11GB for clip-flant5-xxl) has VRAM headroom on the 24GB A5000.
    import gc
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Part: score
# ---------------------------------------------------------------------------

def run_score(args, report):
    clip_model, proc = load_clip(args.clip_model)
    mlp = load_aesthetic()
    ir = load_imagereward()
    real_ref = load_real_psd(args.n_bins)
    scorer = None if args.no_vqa else load_vqascore(args.vqa_model)
    print(f"[e16] scorers: aesthetic={mlp is not None} imagereward={ir is not None} "
          f"vqa={scorer is not None} spectral={real_ref is not None}", flush=True)

    scores = {}
    for pid, prompt in DETAILED[: args.num_prompts]:
        cdir = f"{OUT}/{pid}"
        per_cond = {}
        for cond in ALL_CONDS:
            imgs, tags = load_set(cdir, cond, args.seeds)
            if not imgs:
                continue
            paths = [f"{cdir}/images/{t}.png" for t in tags]
            lats = [torch.load(f"{cdir}/latents/{t}.pt", weights_only=True)
                    for t in tags]
            vals = {
                "aesthetic": aesthetic_scores(mlp, clip_model, proc, imgs),
                "imagereward": imagereward_scores(ir, prompt, paths),
                "spectral_dist": [spectral_dist_to_real(l, real_ref, args.n_bins)
                                  for l in lats],
                "clip_t": clip_scores(clip_model, proc, prompt, imgs),
                "vqascore": vqa_scores_paths(scorer, prompt, paths),
            }
            im = [image_metrics(i) for i in imgs]
            entry = {k: agg(v) for k, v in vals.items()}
            entry["per_seed"] = {k: v for k, v in vals.items()}
            entry["seeds"] = [int(t.rsplit("_s", 1)[1]) for t in tags]
            for m in METRICS:
                entry[m] = agg([v[m] for v in im])
            per_cond[cond] = entry
            print(f"[e16] {pid}/{cond}: aes={_m(entry,'aesthetic')} "
                  f"ir={_m(entry,'imagereward')} sdist={_m(entry,'spectral_dist')} "
                  f"clipt={_m(entry,'clip_t')} vqa={_m(entry,'vqascore')}", flush=True)
        scores[pid] = {"prompt": prompt, "conds": per_cond}

    report["scores"] = scores
    with open(f"{OUT}/scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(f"[e16] wrote {OUT}/scores.json", flush=True)


def _m(entry, k):
    return "%.3f" % entry[k]["mean"] if entry.get(k) else "NA"


# ---------------------------------------------------------------------------
# Part: analyze
# ---------------------------------------------------------------------------

def _paired_delta(per_cond, cond, base, metric):
    """Mean paired (cond - base) at matched seeds for one metric, or None."""
    if cond not in per_cond or base not in per_cond:
        return None
    a, b = per_cond[cond], per_cond[base]
    av = dict(zip(a["seeds"], a["per_seed"][metric]))
    bv = dict(zip(b["seeds"], b["per_seed"][metric]))
    d = [av[s] - bv[s] for s in av
         if s in bv and av[s] is not None and bv[s] is not None]
    return agg(d)


def run_analyze(args, report):
    if not os.path.exists(f"{OUT}/scores.json"):
        print("[e16] no scores.json; run --part score first", flush=True)
        return
    scores = json.load(open(f"{OUT}/scores.json"))
    show_metrics = FID_METRICS + ADH_METRICS + ["rms_contrast", "colorfulness",
                                                 "sharpness"]
    base = "cfg3.5"

    # per-prompt grids (rows = conditions, first grid_n seeds)
    for pid in scores:
        cdir = f"{OUT}/{pid}"
        rows, labels = [], []
        for cond in ALL_CONDS:
            imgs, _ = load_set(cdir, cond, args.grid_n)
            if imgs:
                rows.append(imgs[: args.grid_n])
                labels.append(cond)
        if rows:
            save_grid(rows, labels, [f"s{s}" for s in range(args.grid_n)],
                      f"{OUT}/grid_{pid}.png")

    # mean-over-prompts table of absolute values + delta-vs-cfg3.5
    table = {c: {m: [] for m in show_metrics} for c in ALL_CONDS}
    deltas = {c: {m: [] for m in FID_METRICS + ADH_METRICS} for c in ALL_CONDS}
    for pid, pdata in scores.items():
        pc = pdata["conds"]
        for cond in ALL_CONDS:
            if cond not in pc:
                continue
            for m in show_metrics:
                if pc[cond].get(m):
                    table[cond][m].append(pc[cond][m]["mean"])
            for m in FID_METRICS + ADH_METRICS:
                d = _paired_delta(pc, cond, base, m)
                if d:
                    deltas[cond][m].append(d["mean"])

    def cell(xs):
        return "%.3f" % (sum(xs) / len(xs)) if xs else "  -  "

    lines = ["# E16 fidelity comparison (mean over prompts)", "",
             f"Conditions on Flux-dev, cfg={args.cfg}; fidelity is the contest, "
             "adherence (clip_t/vqascore) is the guardrail.",
             "Arrows: higher better except spectral_dist (lower = closer to real).",
             "", "## Absolute means",
             "| cond | " + " | ".join(show_metrics) + " |",
             "|" + "---|" * (len(show_metrics) + 1)]
    for cond in ALL_CONDS:
        lines.append("| " + cond + " | "
                     + " | ".join(cell(table[cond][m]) for m in show_metrics) + " |")
    lines += ["", f"## Paired Δ vs {base} (fidelity + guardrail)",
              "| cond | " + " | ".join(FID_METRICS + ADH_METRICS) + " |",
              "|" + "---|" * (len(FID_METRICS + ADH_METRICS) + 1)]
    for cond in ALL_CONDS:
        if cond == base:
            continue
        lines.append("| " + cond + " | "
                     + " | ".join(cell(deltas[cond][m])
                                  for m in FID_METRICS + ADH_METRICS) + " |")
    md = "\n".join(lines) + "\n"
    with open(f"{OUT}/summary.md", "w") as f:
        f.write(md)
    report["table_abs"] = {c: {m: (sum(v) / len(v) if v else None)
                               for m, v in table[c].items()} for c in ALL_CONDS}
    report["delta_vs_cfg35"] = {c: {m: (sum(v) / len(v) if v else None)
                                    for m, v in deltas[c].items()} for c in ALL_CONDS}
    print(md, flush=True)
    print(f"[e16] wrote {OUT}/summary.md + grids", flush=True)


# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(OUT, exist_ok=True)
    if "gen" in args.part:
        e8_preflight(args)
    report = {"params": vars(args)}
    runners = {"gen": run_gen, "score": run_score, "analyze": run_analyze}
    for part in filter(None, (p.strip() for p in args.part.split(","))):
        runners[part](args, report)

    path = f"{OUT}/report.json"
    if os.path.exists(path):
        merged = json.load(open(path))
        merged.update(report)
        report = merged
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e16] report -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="gen,score,analyze")
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--ref_seeds", type=int, default=3)
    ap.add_argument("--num_prompts", type=int, default=len(DETAILED))
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--sat", type=float, default=1.4, help="E11 saturation factor")
    ap.add_argument("--seg_scale", type=float, default=3.0)
    ap.add_argument("--seg_sigma", type=float, default=3.0)
    ap.add_argument("--no_seg", action="store_true")
    ap.add_argument("--no_vqa", action="store_true")
    ap.add_argument("--grid_n", type=int, default=6)
    ap.add_argument("--mem", default="bnb4",
                    choices=["bnb4", "gpu_resident", "seq_offload"])
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--vqa_model", default="clip-flant5-xxl")
    main(ap.parse_args())
