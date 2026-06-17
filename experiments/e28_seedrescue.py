"""E28 - Can biasing the seed RESCUE dropped elements on hard compositional prompts?

E25-E27 showed seed-biasing is a do-no-harm palette/appearance lever with flat CLIP-T,
tested on easy prompts (no headroom) with a metric (CLIP-T) blind to dropped elements.
Here we go to the regime where the seed might matter: HARD compositional prompts
(T2I-CompBench: "a green bench and a blue bowl") where SDXL often DROPS or mis-binds an
element. The prompt is achievable for some seeds but not others, so a seed bias toward the
prompt might tip the sampler into the mode that renders the missing element.

Metric = B-VQA (BLIP-VQA attribute binding: product of P(yes) over the prompt's noun
phrases) — it SEES a dropped element, unlike CLIP-T.

Design:
  Stage 1 scan: baseline gen over prompts × seeds; B-VQA; locate FAILS (B-VQA < tau) and
    each prompt's seed-dependence (how many seeds pass).
  Stage 2 intervene on FAILS: bias the failing seed via iterative latent-mode optimization
    (reuse e26.optimize_seed, ||z||=√d kept), regenerate, re-score. Two targets:
      A. full prompt    B. the single lowest-P(yes) noun phrase (the dropped element).
  Controls: re-roll (fresh random seed, no opt) — does biasing beat luck?; do-no-harm on
    a sample of PASSING pairs.

Run:  python experiments/e28_seedrescue.py [quick]      # generate (loads SDXL + CLIP + B-VQA)
      python experiments/e28_seedrescue.py --part site  # model-free: rebuild index.html only
Out:  experiments/results/e28/{grid_recovered.png, grid_nochange.png, summary.png,
      report.json, pairs/}
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import sys
import time

import torch

import compbench
from clip_sim import load_clip, clip_text_features, clip_image_features, cosine
from common import RESULTS, generate, save_grid
from e26_seedalign_sdxl import load_sdxl, moments, optimize_seed

OUT = os.path.join(RESULTS, "e28")
DTYPE = torch.float16
SIZE = int(os.environ.get("E28_SIZE", 1024))
GEN_STEPS = int(os.environ.get("E28_GEN_STEPS", 40))
GUIDANCE = float(os.environ.get("E28_GUIDANCE", 7.0))

CATS = ("color", "shape", "texture")
PER_CAT = int(os.environ.get("E28_PER_CAT", 10))
SCAN_SEEDS = [int(s) for s in os.environ.get("E28_SEEDS", "0,1,2,3").split(",")]
TAU = float(os.environ.get("E28_TAU", 0.5))          # B-VQA fail threshold
K = int(os.environ.get("E28_K", 8))                  # seed-optimization steps
LR = float(os.environ.get("E28_LR", 0.05))
MAX_INTERVENE = int(os.environ.get("E28_MAX", 40))   # cap worst failures we intervene on
REROLL_BASE = 90000                                  # disjoint from scan seeds
# blip-vqa-base ships safetensors (capfilt-large is .bin-only, blocked on torch<2.6);
# base is a fine attribute-binding scorer for relative recovery measurement.
BVQA_MODEL = os.environ.get("E28_BVQA", "Salesforce/blip-vqa-base")


def oom_retry(fn, *a, tries=6, wait=30, **k):
    for i in range(tries):
        try:
            return fn(*a, **k)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if i == tries - 1:
                raise
            time.sleep(wait)


@torch.no_grad()
def gen(pipe, prompt, z):
    out = oom_retry(generate, pipe, prompt, z, steps=GEN_STEPS, guidance=GUIDANCE)
    torch.cuda.empty_cache()
    return out


def seed_latent(pipe, seed):
    shape = (1, pipe.unet.config.in_channels, SIZE // 8, SIZE // 8)
    g = torch.Generator("cuda").manual_seed(seed)
    return torch.randn(shape, generator=g, device="cuda", dtype=torch.float32)


def bvqa_full(bvqa, prompt, img):
    """(B-VQA score, [(phrase, P(yes)), ...]). Score = product of per-phrase P(yes)."""
    model, proc, nlp = bvqa
    phrases = compbench.noun_phrases(nlp, prompt)
    pys = [(q, compbench._p_yes(model, proc, img.convert("RGB"), f"{q}?", "cuda"))
           for q in phrases]
    score = 1.0
    for _, p in pys:
        score *= p
    return score, pys


def run_site():
    """Rebuild results/e28/index.html from report.json + cached figures. Loads NO model and
    re-scores nothing (all numbers already live in report.json), so it runs anywhere."""
    from e28_site import build
    if build() is None:
        print("[e28] --part site: no results/e28/report.json yet (regeneration BLOCKED until a "
              "run exists). index.html left as-is; generate first with: "
              "python experiments/e28_seedrescue.py", flush=True)


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    per_cat = 2 if quick else PER_CAT
    seeds = SCAN_SEEDS[:2] if quick else SCAN_SEEDS
    kk = 3 if quick else K
    cap = 4 if quick else MAX_INTERVENE
    os.makedirs(os.path.join(OUT, "pairs"), exist_ok=True)

    pipe = load_sdxl()
    clip_model, clip_proc = load_clip()
    clip_model = clip_model.to(DTYPE)
    clip_model.requires_grad_(False)
    bvqa = compbench.load_bvqa(model_id=BVQA_MODEL)
    if bvqa is None:
        print("[e28] B-VQA unavailable; aborting"); return

    prompts = compbench.load_compbench_prompts(CATS, per_cat)
    shape = (1, pipe.unet.config.in_channels, SIZE // 8, SIZE // 8)
    sqrt_d = (shape[1] * shape[2] * shape[3]) ** 0.5
    print(f"SDXL {SIZE}px  prompts={len(prompts)} seeds={seeds} tau={TAU} K={kk} "
          f"sqrt(d)={sqrt_d:.1f}", flush=True)

    # ---- Stage 1: scan baseline, score B-VQA, locate fails ----
    scan = []   # dicts per (prompt,seed)
    for pid, prompt, cat in prompts:
        tf_full = clip_text_features(clip_model, clip_proc, [prompt])[0]
        for seed in seeds:
            z0 = seed_latent(pipe, seed)
            img = gen(pipe, prompt, z0)
            score, pys = bvqa_full(bvqa, prompt, img)
            ct = cosine(clip_image_features(clip_model, clip_proc, [img])[0], tf_full)
            scan.append({"pid": pid, "prompt": prompt, "cat": cat, "seed": seed,
                         "bvqa": score, "clip_t": ct, "pys": pys})
            print(f"  scan [{pid} s{seed}] bvqa={score:.3f} clipT={ct:.3f} "
                  f"phrases={[(q, round(p,2)) for q,p in pys]}", flush=True)

    # per-prompt seed-dependence: how many seeds pass
    by_prompt = {}
    for r in scan:
        by_prompt.setdefault(r["pid"], []).append(r)
    passrate = {pid: sum(rr["bvqa"] >= TAU for rr in rs) / len(rs)
                for pid, rs in by_prompt.items()}

    fails = sorted([r for r in scan if r["bvqa"] < TAU], key=lambda r: r["bvqa"])
    passes = [r for r in scan if r["bvqa"] >= TAU]
    print(f"\n[e28] {len(fails)}/{len(scan)} fails (bvqa<{TAU}); intervening on worst "
          f"{min(cap, len(fails))}", flush=True)

    # ---- Stage 2: intervene on the worst failures ----
    rows, labels, records = [], [], []
    for r in fails[:cap]:
        prompt, seed = r["prompt"], r["seed"]
        z0 = seed_latent(pipe, seed)
        base_img = gen(pipe, prompt, z0)   # re-gen baseline (deterministic) for the grid

        # arm A: optimize toward the full prompt
        tf_full = clip_text_features(clip_model, clip_proc, [prompt])[0]
        snapsA, _ = oom_retry(optimize_seed, pipe, clip_model, z0, tf_full, LR, kk, {kk})
        imgA = gen(pipe, prompt, snapsA[kk])
        scoreA, pysA = bvqa_full(bvqa, prompt, imgA)

        # arm B: optimize toward the single lowest-P(yes) noun phrase (the dropped element)
        drop_phrase = min(r["pys"], key=lambda t: t[1])[0]
        tf_drop = clip_text_features(clip_model, clip_proc, [drop_phrase])[0]
        snapsB, _ = oom_retry(optimize_seed, pipe, clip_model, z0, tf_drop, LR, kk, {kk})
        imgB = gen(pipe, prompt, snapsB[kk])
        scoreB, pysB = bvqa_full(bvqa, prompt, imgB)

        # control: re-roll a fresh random seed (no optimization)
        rr_seed = REROLL_BASE + seed
        imgR = gen(pipe, prompt, seed_latent(pipe, rr_seed))
        scoreR, _ = bvqa_full(bvqa, prompt, imgR)

        tag = f"{r['pid']}_s{seed}"
        for nm, im in [("base", base_img), ("A", imgA), ("B", imgB), ("reroll", imgR)]:
            im.save(os.path.join(OUT, "pairs", f"{tag}_{nm}.png"))
        rows.append([base_img, imgA, imgB, imgR])
        labels.append(f"{r['pid']} s{seed} | drop:{drop_phrase[:14]}")
        rec = {"pid": r["pid"], "prompt": prompt, "cat": r["cat"], "seed": seed,
               "drop_phrase": drop_phrase, "passrate": passrate[r["pid"]],
               "bvqa_base": r["bvqa"], "bvqa_A": scoreA, "bvqa_B": scoreB,
               "bvqa_reroll": scoreR,
               "z_norm_A": moments(snapsA[kk])["norm"]}
        records.append(rec)
        torch.cuda.empty_cache()
        print(f"  fix [{tag}] base={r['bvqa']:.3f} A={scoreA:.3f} B={scoreB:.3f} "
              f"reroll={scoreR:.3f} passrate={passrate[r['pid']]:.2f} "
              f"||z*||={rec['z_norm_A']:.1f}", flush=True)

    # ---- do-no-harm on a few passers (arm A) ----
    harm = []
    for r in passes[:min(8, len(passes))]:
        z0 = seed_latent(pipe, r["seed"])
        tf = clip_text_features(clip_model, clip_proc, [r["prompt"]])[0]
        snaps, _ = oom_retry(optimize_seed, pipe, clip_model, z0, tf, LR, kk, {kk})
        s2, _ = bvqa_full(bvqa, r["prompt"], gen(pipe, r["prompt"], snaps[kk]))
        harm.append(s2 - r["bvqa"])
        torch.cuda.empty_cache()

    # ---- aggregate ----
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def rec_rate(key):  # fraction of intervened fails that cross tau
        return mean([1.0 if rr[key] >= TAU else 0.0 for rr in records])

    sd = [rr for rr in records if rr["passrate"] > 0]      # seed-dependent stratum
    af = [rr for rr in records if rr["passrate"] == 0]     # always-fail stratum
    summary = {
        "config": {"size": SIZE, "cats": CATS, "per_cat": per_cat, "seeds": seeds,
                   "tau": TAU, "K": kk, "lr": LR, "n_scan": len(scan),
                   "n_fail": len(fails), "n_intervened": len(records)},
        "fail_rate": len(fails) / len(scan) if scan else None,
        "mean_dbvqa_fails": {k: mean([rr[f"bvqa_{k}"] - rr["bvqa_base"] for rr in records])
                             for k in ["A", "B", "reroll"]},
        "recovery_rate": {k: rec_rate(f"bvqa_{k}") for k in ["A", "B", "reroll"]},
        "seed_dependent": {"n": len(sd),
                           "dbvqa": {k: mean([rr[f"bvqa_{k}"] - rr["bvqa_base"] for rr in sd])
                                     for k in ["A", "B", "reroll"]},
                           "recovery": {k: mean([1.0 if rr[f"bvqa_{k}"] >= TAU else 0.0
                                                 for rr in sd]) for k in ["A", "B", "reroll"]}},
        "always_fail": {"n": len(af),
                        "recovery": {k: mean([1.0 if rr[f"bvqa_{k}"] >= TAU else 0.0
                                              for rr in af]) for k in ["A", "B", "reroll"]}},
        "do_no_harm_dbvqa_passers": mean(harm),
        "records": records, "passrate": passrate,
    }
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if rows:
        save_grid(rows[:16], labels[:16], ["baseline (fail)", "opt-A full", "opt-B phrase",
                  "re-roll"], os.path.join(OUT, "grid_recovered.png"))
    plot_summary(summary, os.path.join(OUT, "summary.png"))

    print("\n=== E28 summary ===")
    print("fail rate:", round(summary["fail_rate"] or 0, 3))
    print("mean ΔB-VQA on fails:", {k: round(v, 3) for k, v in
                                    summary["mean_dbvqa_fails"].items() if v is not None})
    print("recovery rate:", {k: round(v, 3) for k, v in
                             summary["recovery_rate"].items() if v is not None})
    print("seed-dependent recovery:", {k: (round(v, 3) if v is not None else None)
          for k, v in summary["seed_dependent"]["recovery"].items()},
          f"(n={summary['seed_dependent']['n']})")
    print("do-no-harm Δ on passers:", summary["do_no_harm_dbvqa_passers"])
    print(f"wrote {OUT}/{{grid_recovered.png, summary.png, report.json}}")


def plot_summary(s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = ["A", "B", "reroll"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(arms, [s["mean_dbvqa_fails"][a] or 0 for a in arms],
              color=["C0", "C2", "C7"])
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_title("mean ΔB-VQA on failures")
    ax[0].set_ylabel("ΔB-VQA")
    x = range(len(arms))
    ax[1].bar([i - 0.2 for i in x], [s["recovery_rate"][a] or 0 for a in arms], 0.4,
              label="all fails", color="C0")
    ax[1].bar([i + 0.2 for i in x], [s["seed_dependent"]["recovery"][a] or 0 for a in arms],
              0.4, label="seed-dependent", color="C2")
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(arms)
    ax[1].set_title(f"recovery rate (cross τ={s['config']['tau']})")
    ax[1].set_ylabel("fraction recovered"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


if __name__ == "__main__":
    # Model-free path mirrors E30's `--part site`: rebuild the explainer with no model load.
    if "--part" in sys.argv and "site" in sys.argv[sys.argv.index("--part") + 1:]:
        run_site()
    else:
        main()
