"""E17-CB: SBN vs CFG-Zero* vs CFG++ on T2I-CompBench (SD3.5), matching cfg*'s
evaluation. CFG-Zero* reported on T2I-CompBench; here the CONTEST is compositional
attribute binding (B-VQA on color/shape/texture), with aesthetic/ImageReward/CLIP-T
as fidelity context. Tests whether SBN's spectral clamp preserves or harms binding,
alone and combined with CFG-Zero*/CFG++.

Conditions = the 8 from E17 (cfg1, cfg_hi, bandnorm, bandnorm_pp, cfgzero,
cfgzero_sbn, cfgpp, cfgpp_sbn). Prompts = balanced subset of T2I-CompBench val
(color/shape/texture). Layout results/e17cb/<pid>/...; scored per category.

Parts: gen / score / analyze (same caching as E17).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid
from spectral_ops import band_index_map
from e17_sd35 import (load_sd35, gen_sd3, gen_rcfgpp_sd3, record_reference_sd3,
                      ClampPSD3, RecordPSD3, make_cfgzero_step, H, W)
from e16_prompt_adherence import agg, _paired_delta
from e9_bandnorm_classes import cached_gen, image_metrics, load_set, METRICS
from e9_clipt import load_clip, clip_scores
from e11_color_correct import correct_sat
from fidelity_metrics import (load_aesthetic, aesthetic_scores,
                              load_imagereward, imagereward_scores,
                              load_real_psd, spectral_dist_to_real,
                              SD35_REAL_LATENTS)
from compbench import load_compbench_prompts, load_bvqa, bvqa_scores

OUT = os.path.join(RESULTS, "e17cb")
ALL_CONDS = ["cfg1", "cfg_hi", "bandnorm", "bandnorm_pp", "cfgzero", "cfgzero_sbn",
             "cfgpp", "cfgpp_sbn"]
BASE = "cfg_hi"
PRIMARY = ["bvqa"]                              # T2I-CompBench binding (the contest)
SECONDARY = ["aesthetic", "imagereward", "spectral_dist", "clip_t"]
LOWER_BETTER = {"spectral_dist"}               # closer-to-real = lower


def _pp(cdir, s, sat):
    bi, bl = f"{cdir}/images/bandnorm_s{s}.png", f"{cdir}/latents/bandnorm_s{s}.pt"
    pi, pl = f"{cdir}/images/bandnorm_pp_s{s}.png", f"{cdir}/latents/bandnorm_pp_s{s}.pt"
    if os.path.exists(pi) and os.path.exists(pl):
        return
    if os.path.exists(bi) and os.path.exists(bl):
        from PIL import Image
        import shutil
        correct_sat(Image.open(bi).convert("RGB"), None, sat).save(pi)
        shutil.copy(bl, pl)


def run_gen(args, report):
    prompts = load_compbench_prompts(args.categories, args.per_cat)
    report["n_prompts"] = len(prompts)
    pipe = load_sd35(args.mem)
    idx_map = band_index_map(H, W, args.n_bins, "cuda")
    Wc = args.cfg
    for pid, prompt, cat in prompts:
        cdir = f"{OUT}/{pid}"
        os.makedirs(f"{cdir}/images", exist_ok=True)
        os.makedirs(f"{cdir}/latents", exist_ok=True)
        ref_path = f"{cdir}/ref_psd.pt"
        ref_tags = [f"cfg1_s{s}" for s in range(args.ref_seeds)]
        if os.path.exists(ref_path) and all(
                os.path.exists(f"{cdir}/images/{t}.png") for t in ref_tags):
            ref = torch.load(ref_path, weights_only=True)
        else:
            ref, outs = record_reference_sd3(pipe, prompt, args.ref_seeds,
                                             args.steps, args.n_bins, 1.0)
            for t, (img, lat) in zip(ref_tags, outs):
                img.save(f"{cdir}/images/{t}.png")
                torch.save(lat, f"{cdir}/latents/{t}.pt")
            torch.save(ref, ref_path)
        for s in range(args.seeds):
            if s >= args.ref_seeds:
                cached_gen(cdir, f"cfg1_s{s}", lambda: gen_sd3(
                    pipe, prompt, s, 1.0, args.steps,
                    cb_obj=RecordPSD3(idx_map, args.n_bins, args.steps)))
            cached_gen(cdir, f"cfg_hi_s{s}", lambda: gen_sd3(pipe, prompt, s, Wc, args.steps))
            cached_gen(cdir, f"bandnorm_s{s}", lambda: gen_sd3(
                pipe, prompt, s, Wc, args.steps, cb_obj=ClampPSD3(ref, idx_map, args.n_bins)))
            _pp(cdir, s, args.sat)
            cached_gen(cdir, f"cfgzero_s{s}", lambda: gen_sd3(
                pipe, prompt, s, Wc, args.steps, step_override=make_cfgzero_step(Wc)))
            cached_gen(cdir, f"cfgzero_sbn_s{s}", lambda: gen_sd3(
                pipe, prompt, s, Wc, args.steps, step_override=make_cfgzero_step(Wc),
                cb_obj=ClampPSD3(ref, idx_map, args.n_bins)))
            cached_gen(cdir, f"cfgpp_s{s}", lambda: gen_rcfgpp_sd3(
                pipe, prompt, s, Wc, args.steps, args.sigma_noise))
            cached_gen(cdir, f"cfgpp_sbn_s{s}", lambda: gen_rcfgpp_sd3(
                pipe, prompt, s, Wc, args.steps, args.sigma_noise,
                cb_obj=ClampPSD3(ref, idx_map, args.n_bins)))
        print(f"[e17cb] {pid} ({cat}) done", flush=True)
    import gc
    del pipe
    gc.collect(); torch.cuda.empty_cache()


def run_score(args, report):
    prompts = load_compbench_prompts(args.categories, args.per_cat)
    clip_model, proc = load_clip(args.clip_model)
    mlp = load_aesthetic()
    ir = load_imagereward()
    bvqa = load_bvqa()
    real_ref = load_real_psd(args.n_bins, path=SD35_REAL_LATENTS)  # SD3.5 VAE space
    print(f"[e17cb] scorers: bvqa={bvqa is not None} aesthetic={mlp is not None} "
          f"imagereward={ir is not None} spectral={real_ref is not None}", flush=True)
    scores = {}
    for pid, prompt, cat in prompts:
        cdir = f"{OUT}/{pid}"
        per_cond = {}
        for cond in ALL_CONDS:
            imgs, tags = load_set(cdir, cond, args.seeds)
            if not imgs:
                continue
            paths = [f"{cdir}/images/{t}.png" for t in tags]
            lats = [torch.load(f"{cdir}/latents/{t}.pt", weights_only=True) for t in tags]
            vals = {
                "bvqa": bvqa_scores(bvqa, prompt, imgs),
                "aesthetic": aesthetic_scores(mlp, clip_model, proc, imgs),
                "imagereward": imagereward_scores(ir, prompt, paths),
                "spectral_dist": [spectral_dist_to_real(l, real_ref, args.n_bins)
                                  for l in lats],
                "clip_t": clip_scores(clip_model, proc, prompt, imgs),
            }
            entry = {k: agg(v) for k, v in vals.items()}
            entry["per_seed"] = vals
            entry["seeds"] = [int(t.rsplit("_s", 1)[1]) for t in tags]
            for m in METRICS:
                entry[m] = agg([image_metrics(i)[m] for i in imgs])
            per_cond[cond] = entry
        scores[pid] = {"prompt": prompt, "category": cat, "conds": per_cond}
        print(f"[e17cb] {pid} bvqa: " + " ".join(
            f"{c}={_m(per_cond.get(c,{}),'bvqa')}" for c in ALL_CONDS), flush=True)
    report["scores"] = scores
    with open(f"{OUT}/scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(f"[e17cb] wrote {OUT}/scores.json", flush=True)


def _m(entry, k):
    return "%.3f" % entry[k]["mean"] if entry.get(k) else "NA"


def run_analyze(args, report):
    if not os.path.exists(f"{OUT}/scores.json"):
        print("[e17cb] no scores.json", flush=True)
        return
    scores = json.load(open(f"{OUT}/scores.json"))
    cats = sorted({p["category"] for p in scores.values()})
    metrics = PRIMARY + SECONDARY

    # per-category + overall B-VQA means, and paired deltas vs cfg_hi
    absm = {c: {m: [] for m in metrics} for c in ALL_CONDS}
    bycat = {cat: {c: [] for c in ALL_CONDS} for cat in cats}   # bvqa only
    deltas = {c: {m: [] for m in metrics} for c in ALL_CONDS}
    for pid, pd in scores.items():
        pc = pd["conds"]
        for cond in ALL_CONDS:
            if cond not in pc:
                continue
            for m in metrics:
                if pc[cond].get(m):
                    absm[cond][m].append(pc[cond][m]["mean"])
            if pc[cond].get("bvqa"):
                bycat[pd["category"]][cond].append(pc[cond]["bvqa"]["mean"])
            for m in metrics:
                d = _paired_delta(pc, cond, BASE, m)
                if d:
                    deltas[cond][m].append(d["mean"])

    def cell(xs):
        return "%.3f" % (sum(xs) / len(xs)) if xs else "  -  "

    L = [f"# E17-CB: T2I-CompBench (SD3.5-medium, cfg={args.cfg})", "",
         "Contest = B-VQA attribute binding; aesthetic/imagereward/clip_t = fidelity context.",
         "", "## Means (all metrics)",
         "| cond | " + " | ".join(metrics) + " |", "|" + "---|" * (len(metrics) + 1)]
    for c in ALL_CONDS:
        L.append("| " + c + " | " + " | ".join(cell(absm[c][m]) for m in metrics) + " |")
    L += ["", "## B-VQA by category",
          "| cond | " + " | ".join(cats) + " | overall |", "|" + "---|" * (len(cats) + 2)]
    for c in ALL_CONDS:
        allv = [v for cat in cats for v in bycat[cat][c]]
        L.append("| " + c + " | " + " | ".join(cell(bycat[cat][c]) for cat in cats)
                 + " | " + cell(allv) + " |")
    L += ["", f"## Paired Δ vs {BASE}",
          "| cond | " + " | ".join(metrics) + " |", "|" + "---|" * (len(metrics) + 1)]
    for c in ALL_CONDS:
        if c == BASE:
            continue
        L.append("| " + c + " | " + " | ".join(cell(deltas[c][m]) for m in metrics) + " |")
    md = "\n".join(L) + "\n"
    with open(f"{OUT}/summary.md", "w") as f:
        f.write(md)
    # small per-category grids (first prompt of each category, all conds)
    for cat in cats:
        pid = next((p for p, d in scores.items() if d["category"] == cat), None)
        if not pid:
            continue
        rows, labels = [], []
        for cond in ALL_CONDS:
            im, _ = load_set(f"{OUT}/{pid}", cond, args.grid_n)
            if im:
                rows.append(im[: args.grid_n]); labels.append(cond)
        if rows:
            save_grid(rows, labels, [f"s{s}" for s in range(args.grid_n)],
                      f"{OUT}/grid_{cat}.png")
    print(md, flush=True)
    print(f"[e17cb] wrote {OUT}/summary.md", flush=True)


def main(args):
    os.makedirs(OUT, exist_ok=True)
    report = {"params": vars(args)}
    for part in filter(None, (p.strip() for p in args.part.split(","))):
        {"gen": run_gen, "score": run_score, "analyze": run_analyze}[part](args, report)
    path = f"{OUT}/report.json"
    if os.path.exists(path):
        merged = json.load(open(path)); merged.update(report); report = merged
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e17cb] report -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="gen,score,analyze")
    ap.add_argument("--categories", nargs="+", default=["color", "shape", "texture"])
    ap.add_argument("--per_cat", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--ref_seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--sat", type=float, default=1.4)
    ap.add_argument("--sigma_noise", type=float, default=0.005)
    ap.add_argument("--grid_n", type=int, default=6)
    ap.add_argument("--mem", default="gpu_resident", choices=["gpu_resident", "offload"])
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    main(ap.parse_args())
