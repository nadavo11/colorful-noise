"""E17: SBN vs CFG-Zero* on Stable Diffusion 3.5 (true CFG, no distillation).

E16 found that on Flux the guidance-distilled cfg makes the high-CFG regime odd;
SD3.5 uses real classifier-free guidance, so it's the cleaner testbed. Question:
in the high-CFG regime, is SBN (band-norm, our method) better than CFG-Zero*, and
do they COMPLEMENT (CFG-Zero* + SBN)? SBN still pays the per-prompt cfg=1 reference
overhead for now (a universal reference is future work).

Conditions (SD3.5-medium, shared seeded init latent per seed):
  cfg1        guidance=1 -> pure conditional field (SBN reference + realism anchor)
  cfg_hi      guidance=W -> high-CFG baseline (default 4.5)
  bandnorm    SBN: clamp cfg=W latent PSD to the cfg=1 per-step reference (ours)
  bandnorm_pp SBN + saturation x{sat} postprocess (E11)
  cfgzero     CFG-Zero* (optimal scale + zero-init), true CFG
  cfgzero_sbn CFG-Zero* AND the SBN PSD clamp (they compose: guidance modifies the
              velocity, SBN clamps the resulting latent at step end)

Metrics (reuse E16): aesthetic + ImageReward (fidelity), CLIP-T (adherence guard).
Spectral-distance-to-real is SKIPPED here: the E10 real reference is in FLUX VAE
latent space, not SD3.5's -- needs an SD3.5-VAE-encoded real ref to be meaningful.

Parts (--part): gen / score / analyze, same layout as E16 (results/e17/<pid>/...).
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
from e17_sd35 import (load_sd35, gen_sd3, gen_rcfgpp_sd3, record_reference_sd3,
                      ClampPSD3, RecordPSD3, make_cfgzero_step, H, W)
from e16_prompt_adherence import (DETAILED, agg, FID_METRICS, ADH_METRICS,
                                  LOWER_BETTER, _paired_delta, ensure_pp)
from e9_bandnorm_classes import cached_gen, image_metrics, load_set, METRICS
from e9_clipt import load_clip, clip_scores
from fidelity_metrics import (load_aesthetic, aesthetic_scores,
                              load_imagereward, imagereward_scores,
                              spectral_dist_to_real, load_real_psd,
                              SD35_REAL_LATENTS)
from vqascore import load_vqascore, vqa_scores_paths

OUT = os.path.join(RESULTS, "e17")
ALL_CONDS = ["cfg1", "cfg_hi", "bandnorm", "bandnorm_pp", "cfgzero", "cfgzero_sbn",
             "cfgpp", "cfgpp_sbn"]
BASE = "cfg_hi"


# ---------------------------------------------------------------------------
# Part: gen
# ---------------------------------------------------------------------------

def run_gen(args, report):
    pipe = load_sd35(args.mem)
    idx_map = band_index_map(H, W, args.n_bins, "cuda")
    Wc = args.cfg
    for pid, prompt in DETAILED[: args.num_prompts]:
        cdir = f"{OUT}/{pid}"
        os.makedirs(f"{cdir}/images", exist_ok=True)
        os.makedirs(f"{cdir}/latents", exist_ok=True)

        # cfg=1 reference (first ref_seeds), cached
        ref_path = f"{cdir}/ref_psd.pt"
        ref_tags = [f"cfg1_s{s}" for s in range(args.ref_seeds)]
        if os.path.exists(ref_path) and all(
                os.path.exists(f"{cdir}/images/{t}.png") for t in ref_tags):
            ref = torch.load(ref_path, weights_only=True)
            print(f"[e17] {pid} reference (cached)", flush=True)
        else:
            ref, outs = record_reference_sd3(pipe, prompt, args.ref_seeds,
                                             args.steps, args.n_bins, 1.0)
            for t, (img, lat) in zip(ref_tags, outs):
                img.save(f"{cdir}/images/{t}.png")
                torch.save(lat, f"{cdir}/latents/{t}.pt")
            torch.save(ref, ref_path)
            print(f"[e17] {pid} reference recorded", flush=True)

        for s in range(args.seeds):
            if s >= args.ref_seeds:
                cached_gen(cdir, f"cfg1_s{s}", lambda: gen_sd3(
                    pipe, prompt, s, 1.0, args.steps,
                    cb_obj=RecordPSD3(idx_map, args.n_bins, args.steps)))
            cached_gen(cdir, f"cfg_hi_s{s}", lambda: gen_sd3(
                pipe, prompt, s, Wc, args.steps))
            cached_gen(cdir, f"bandnorm_s{s}", lambda: gen_sd3(
                pipe, prompt, s, Wc, args.steps,
                cb_obj=ClampPSD3(ref, idx_map, args.n_bins)))
            ensure_pp(cdir, s, args.sat)
            cached_gen(cdir, f"cfgzero_s{s}", lambda: gen_sd3(
                pipe, prompt, s, Wc, args.steps,
                step_override=make_cfgzero_step(Wc)))
            cached_gen(cdir, f"cfgzero_sbn_s{s}", lambda: gen_sd3(
                pipe, prompt, s, Wc, args.steps,
                step_override=make_cfgzero_step(Wc),
                cb_obj=ClampPSD3(ref, idx_map, args.n_bins)))
            if not args.no_cfgpp:
                cached_gen(cdir, f"cfgpp_s{s}", lambda: gen_rcfgpp_sd3(
                    pipe, prompt, s, Wc, args.steps, args.sigma_noise))
                cached_gen(cdir, f"cfgpp_sbn_s{s}", lambda: gen_rcfgpp_sd3(
                    pipe, prompt, s, Wc, args.steps, args.sigma_noise,
                    cb_obj=ClampPSD3(ref, idx_map, args.n_bins)))
        print(f"[e17] {pid} generation complete", flush=True)

    import gc
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Part: score  (spectral_dist skipped: SD3.5 latent space != E10 Flux ref)
# ---------------------------------------------------------------------------

def run_score(args, report):
    clip_model, proc = load_clip(args.clip_model)
    mlp = load_aesthetic()
    ir = load_imagereward()
    scorer = None if args.no_vqa else load_vqascore(args.vqa_model)
    real_ref = load_real_psd(args.n_bins, path=SD35_REAL_LATENTS)  # SD3.5 VAE space
    print(f"[e17] scorers: aesthetic={mlp is not None} imagereward={ir is not None} "
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
            entry = {k: agg(v) for k, v in vals.items()}
            entry["per_seed"] = {k: v for k, v in vals.items()}
            entry["seeds"] = [int(t.rsplit("_s", 1)[1]) for t in tags]
            for m in METRICS:
                entry[m] = agg([image_metrics(i)[m] for i in imgs])
            per_cond[cond] = entry
            print(f"[e17] {pid}/{cond}: aes={_m(entry,'aesthetic')} "
                  f"ir={_m(entry,'imagereward')} clipt={_m(entry,'clip_t')}", flush=True)
        scores[pid] = {"prompt": prompt, "conds": per_cond}

    report["scores"] = scores
    with open(f"{OUT}/scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(f"[e17] wrote {OUT}/scores.json", flush=True)


def _m(entry, k):
    return "%.3f" % entry[k]["mean"] if entry.get(k) else "NA"


# ---------------------------------------------------------------------------
# Part: analyze
# ---------------------------------------------------------------------------

def run_analyze(args, report):
    if not os.path.exists(f"{OUT}/scores.json"):
        print("[e17] no scores.json; run --part score first", flush=True)
        return
    scores = json.load(open(f"{OUT}/scores.json"))
    show = FID_METRICS + ADH_METRICS + ["rms_contrast", "colorfulness", "sharpness"]

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

    table = {c: {m: [] for m in show} for c in ALL_CONDS}
    deltas = {c: {m: [] for m in FID_METRICS + ADH_METRICS} for c in ALL_CONDS}
    for pid, pdata in scores.items():
        pc = pdata["conds"]
        for cond in ALL_CONDS:
            if cond not in pc:
                continue
            for m in show:
                if pc[cond].get(m):
                    table[cond][m].append(pc[cond][m]["mean"])
            for m in FID_METRICS + ADH_METRICS:
                d = _paired_delta(pc, cond, BASE, m)
                if d:
                    deltas[cond][m].append(d["mean"])

    def cell(xs):
        return "%.3f" % (sum(xs) / len(xs)) if xs else "  -  "

    lines = ["# E17 SD3.5: SBN vs CFG-Zero* (mean over prompts)", "",
             f"SD3.5-medium, cfg={args.cfg}. Fidelity = aesthetic + ImageReward; "
             "adherence guardrail = clip_t. (spectral_dist disabled: SD3.5 VAE space.)",
             "", "## Absolute means",
             "| cond | " + " | ".join(show) + " |",
             "|" + "---|" * (len(show) + 1)]
    for cond in ALL_CONDS:
        lines.append("| " + cond + " | "
                     + " | ".join(cell(table[cond][m]) for m in show) + " |")
    lines += ["", f"## Paired Δ vs {BASE}",
              "| cond | " + " | ".join(FID_METRICS + ADH_METRICS) + " |",
              "|" + "---|" * (len(FID_METRICS + ADH_METRICS) + 1)]
    for cond in ALL_CONDS:
        if cond == BASE:
            continue
        lines.append("| " + cond + " | "
                     + " | ".join(cell(deltas[cond][m])
                                  for m in FID_METRICS + ADH_METRICS) + " |")
    md = "\n".join(lines) + "\n"
    with open(f"{OUT}/summary.md", "w") as f:
        f.write(md)
    print(md, flush=True)
    print(f"[e17] wrote {OUT}/summary.md + grids", flush=True)


def run_site(args, report):
    """Retired: per-experiment HTML is superseded by the roadmap site
    (docs/roadmap/, generated from roadmap_registry.py)."""
    print("[e17] --part site retired; see docs/roadmap/ "
          "(regen: python experiments/make_roadmap.py)", flush=True)


def main(args):
    os.makedirs(OUT, exist_ok=True)
    report = {"params": vars(args)}
    runners = {"gen": run_gen, "score": run_score, "analyze": run_analyze,
               "site": run_site}
    parts = [p.strip() for p in args.part.split(",") if p.strip()]
    for part in parts:
        runners[part](args, report)
    # A site-only rebuild must NOT touch report.json (it would overwrite the real
    # run's params with default args); it only re-templates index.html.
    if parts == ["site"]:
        return
    path = f"{OUT}/report.json"
    if os.path.exists(path):
        merged = json.load(open(path))
        merged.update(report)
        report = merged
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e17] report -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="gen,score,analyze")
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--ref_seeds", type=int, default=3)
    ap.add_argument("--num_prompts", type=int, default=len(DETAILED))
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.5, help="SD3.5 high-CFG scale")
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--sat", type=float, default=1.4)
    ap.add_argument("--sigma_noise", type=float, default=0.005,
                    help="Rectified-CFG++ corrector jitter (paper default)")
    ap.add_argument("--no_cfgpp", action="store_true")
    ap.add_argument("--no_vqa", action="store_true")
    ap.add_argument("--grid_n", type=int, default=6)
    ap.add_argument("--mem", default="gpu_resident",
                    choices=["gpu_resident", "offload"])
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--vqa_model", default="clip-flant5-xxl")
    main(ap.parse_args())
