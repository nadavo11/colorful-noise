"""E41: per-image calibration of our spectral-clamp edit + fair RF-inversion (eta) comparison.

For each PIE-Bench image we:
  1. RF-invert under the source prompt (shared invert_core path -> noise + trajectory).
  2. Build two RF-inversion references with the eta controller: VANILLA (eta=0, the plain
     inversion edit) and DEFAULT-ETA (out-of-the-box faithfulness). For showcase images we
     also sweep eta to trace the faithfulness<->editability curve.
  3. Run an Optuna TPE active loop over our knobs (mode/cut/strength/interval/phase band),
     scoring each trial by DINO structure distance (preserve) + CLIP-dir (editability).
     Objective is CONSTRAINED: minimize structure distance s.t. CLIP-dir >= the vanilla
     baseline's (matched editability). Warm-started from the hand-tuned dancers prior + a
     prompt-distance heuristic.
  4. Score ours / vanilla / default-eta with the full suite (DINO, LPIPS, SSIM, background
     PSNR/LPIPS, CLIP-dir, CLIP-T) and write one resumable JSON per image.

Parts: calibrate (per-image loop, GPU) ; analyze (aggregate table + Pareto plot, no GPU).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import invert_core as ic
import struct_metrics as sm
from clip_sim import clip_image_features, clip_text_features
from spectral_ops import band_index_map
from spectral_adain import soft_band_masks

OUT = os.path.join(RESULTS, "e41")
ITEMS = os.path.join(OUT, "items")

ADAIN_K = 8
STEPS = 16
GUIDANCE = 3.5
INV_GUIDANCE = 1.0
SEED = 0
N_TRIALS = 20
PENALTY = 100.0                    # infeasibility weight in the scalarized objective

# RF-inversion baseline knob. eta in [0,1] trades editability<->faithfulness; applied over
# the early step window [0, ETA_STOP*steps]. DEFAULT_ETA is the single out-of-the-box point
# (confirm against the reference Flux RF-inversion repo); ETA_SWEEP traces the full curve.
DEFAULT_ETA = 0.9
ETA_STOP = 0.6
ETA_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SHOWCASE = ("dancers",)            # key substrings that also get the eta sweep
METHODS = ("ours", "vanilla", "etadefault")


# ---------------------------------------------------------------------------
# generation helpers
# ---------------------------------------------------------------------------
def _build(pipe):
    sig = ic.flux_sigmas(pipe, STEPS)
    g_inv = ic.gids(pipe, INV_GUIDANCE)
    g_gen = ic.gids(pipe, GUIDANCE)
    idx = band_index_map(ic.FH, ic.FW, ic.N_BINS, "cuda")
    cen = torch.linspace(0, 1, ADAIN_K)
    M = soft_band_masks(ic.FH, ic.FW, cen.tolist(), [1.0 / ADAIN_K] * ADAIN_K, "cuda")
    return dict(sig=sig, g_inv=g_inv, g_gen=g_gen, idx=idx, M=M, cen_k=cen.tolist())


def _edit(pipe, b, peE, ppeE, x_noise, **kw):
    """Run an edit pass and decode to PIL."""
    from e7_flux_phase import flux_vae_decode
    lat = ic.forward_edit(pipe, peE, ppeE, x_noise, b["sig"], b["g_gen"], **kw)
    return flux_vae_decode(pipe.vae, lat)


def _clip_t(clip, img, prompt):
    model, proc = clip
    fi = clip_image_features(model, proc, [img])
    ft = clip_text_features(model, proc, [prompt])
    return float((fi[0] * ft[0]).sum())


def _full_metrics(met, src_img, edit_img, mask, src_prompt, edit_prompt):
    m = {"struct": sm.structure_distance(met["dino"], src_img, edit_img),
         "clipdir": sm.clip_directional(met["clip"], src_img, edit_img, src_prompt, edit_prompt),
         "clipt": _clip_t(met["clip"], edit_img, edit_prompt)}
    m.update(sm.image_metrics(edit_img, src_img, met["lpips"], met["ssim"]))
    if mask is not None:
        m.update(sm.background_metrics(edit_img, src_img, mask, met["lpips"]))
    return m


# ---------------------------------------------------------------------------
# per-image calibration
# ---------------------------------------------------------------------------
def calibrate_image(pipe, met, item, b, args):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    key, src_img, mask = item["key"], item["src_img"], item.get("mask")
    src_p, edit_p = item["src_prompt"], item["edit_prompt"]
    stem = os.path.join(ITEMS, key.replace("/", "_"))
    if os.path.exists(stem + ".json") and not args.force:
        print(f"[e41] {key}: cached, skip", flush=True)
        return
    os.makedirs(ITEMS, exist_ok=True)

    peS, ppeS = ic.encode_prompt(pipe, src_p)
    peE, ppeE = ic.encode_prompt(pipe, edit_p)
    x0 = ic.pack(pipe, ic.vae_encode(pipe.vae, src_img))
    x_noise, traj = ic.rf_invert(pipe, peS, ppeS, x0, b["sig"], b["g_inv"])

    # RF-inversion references
    eta_win = (0, round(ETA_STOP * (STEPS - 1)))
    vanilla = _edit(pipe, b, peE, ppeE, x_noise)
    etadef = _edit(pipe, b, peE, ppeE, x_noise, x0_packed=x0, eta=DEFAULT_ETA, eta_window=eta_win)
    base_cd = sm.clip_directional(met["clip"], src_img, vanilla, src_p, edit_p)

    # Optuna active loop over our knobs
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({"mode": "phase", "cut": 0.25, "strength": 0.0,
                         "interval_end": 0.1, "phase_hi": 0.15})        # dancers prior
    pdist = sm.prompt_distance(met["clip"], src_p, edit_p)
    lock = float(np.clip(0.3 - pdist, 0.05, 0.4))                       # small text move => lock harder
    study.enqueue_trial({"mode": "phase", "cut": 0.25, "strength": 0.0,
                         "interval_end": min(0.6, lock + 0.05), "phase_hi": lock})
    trial_imgs = {}

    def objective(trial):
        mode = trial.suggest_categorical("mode", ["phase", "sbn", "adain"])
        cut = trial.suggest_float("cut", 0.1, 0.5)
        strength = trial.suggest_float("strength", 0.0, 1.0)
        iend = trial.suggest_float("interval_end", 0.05, 0.6)
        phase_hi = trial.suggest_float("phase_hi", 0.05, 0.4)
        window = (0, max(0, round(iend * (STEPS - 1))))
        img = _edit(pipe, b, peE, ppeE, x_noise, traj=traj, mode=mode, cut=cut,
                    strength=strength, window=window, idx=b["idx"], M=b["M"],
                    cen_k=b["cen_k"], phase_band=(0.0, phase_hi))
        sd = sm.structure_distance(met["dino"], src_img, img)
        cd = sm.clip_directional(met["clip"], src_img, img, src_p, edit_p)
        trial.set_user_attr("struct", sd)
        trial.set_user_attr("clipdir", cd)
        trial_imgs[trial.number] = img
        return sd + PENALTY * max(0.0, base_cd - cd)

    study.optimize(objective, n_trials=args.trials)
    feas = [t for t in study.trials if t.user_attrs.get("clipdir", -9) >= base_cd]
    pick = (min(feas, key=lambda t: t.user_attrs["struct"]) if feas else
            max(study.trials, key=lambda t: t.user_attrs.get("clipdir", -9)))
    ours = trial_imgs[pick.number]

    # save images + full-suite scores
    src_img.save(stem + "_source.png")
    imgs = {"ours": ours, "vanilla": vanilla, "etadefault": etadef}
    for name, im in imgs.items():
        im.save(f"{stem}_{name}.png")
    if mask is not None:
        mask.save(stem + "_mask.png")
    rec = {"key": key, "edit_type": item["edit_type"], "src_prompt": src_p,
           "edit_prompt": edit_p, "base_clipdir": base_cd, "feasible": bool(feas),
           "best_params": pick.params, "prompt_distance": pdist,
           "metrics": {n: _full_metrics(met, src_img, im, mask, src_p, edit_p)
                       for n, im in imgs.items()},
           "trials": [{"struct": t.user_attrs.get("struct"),
                       "clipdir": t.user_attrs.get("clipdir"), "params": t.params}
                      for t in study.trials]}
    if any(s in key for s in SHOWCASE) or args.eta_sweep_all:
        sweep = []
        for e in ETA_SWEEP:
            im = vanilla if e == 0 else _edit(pipe, b, peE, ppeE, x_noise, x0_packed=x0,
                                              eta=e, eta_window=eta_win)
            im.save(f"{stem}_eta{e:.1f}.png")
            sweep.append({"eta": e,
                          "struct": sm.structure_distance(met["dino"], src_img, im),
                          "clipdir": sm.clip_directional(met["clip"], src_img, im, src_p, edit_p)})
        rec["eta_sweep"] = sweep
    tmp = stem + ".json.tmp"
    json.dump(rec, open(tmp, "w"), indent=2)
    os.replace(tmp, stem + ".json")                                    # atomic for sharded jobs
    print(f"[e41] {key}: ours struct={rec['metrics']['ours']['struct']:.4f} "
          f"clipdir={rec['metrics']['ours']['clipdir']:.4f} (base {base_cd:.4f}) "
          f"feasible={bool(feas)} params={pick.params}", flush=True)


def run_calibrate(args):
    from e7_flux_phase import load_flux
    from piebench import load_piebench
    items = load_piebench(args.n_per_type)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        items = items[i::n]
        print(f"[e41] shard {i}/{n}: {len(items)} items", flush=True)
    if args.num:
        items = items[: args.num]
    pipe = load_flux(args.mem)
    met = sm.load_metrics("cuda")
    b = _build(pipe)
    for it in items:
        try:
            calibrate_image(pipe, met, it, b, args)
        except Exception as e:
            print(f"[e41] {it['key']}: ERROR {e}", flush=True)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def _mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def run_analyze(args):
    recs = [json.load(open(os.path.join(ITEMS, f)))
            for f in sorted(os.listdir(ITEMS)) if f.endswith(".json")]
    if not recs:
        print("[e41] no item records; run calibrate first", flush=True)
        return
    keys = ["struct", "lpips", "dssim", "bg_psnr", "bg_lpips", "clipdir", "clipt"]
    lines = [f"# E41 — {len(recs)} images\n",
             "## Means by method (matched-editability calibration)\n",
             "| method | " + " | ".join(keys) + " |",
             "|" + "---|" * (len(keys) + 1)]
    for name in METHODS:
        row = [_mean([r["metrics"][name].get(k) for r in recs]) for k in keys]
        lines.append(f"| {name} | " +
                     " | ".join("—" if x is None else f"{x:.4f}" for x in row) + " |")
    # per-edit-type structure distance (ours vs etadefault)
    lines += ["\n## Structure distance by edit type (ours / etadefault)\n",
              "| edit type | n | ours | etadefault | vanilla |",
              "|---|---|---|---|---|"]
    types = sorted({r["edit_type"] for r in recs})
    for t in types:
        rs = [r for r in recs if r["edit_type"] == t]
        cells = [_mean([r["metrics"][m]["struct"] for r in rs]) for m in METHODS]
        lines.append(f"| {t} | {len(rs)} | " +
                     " | ".join(f"{c:.4f}" for c in cells) + " |")
    feas = sum(r["feasible"] for r in recs)
    lines.append(f"\nfeasible (matched editability reached): {feas}/{len(recs)}")
    md = os.path.join(OUT, "report.md")
    open(md, "w").write("\n".join(lines))
    print("\n".join(lines), flush=True)

    # Pareto plot for showcase images with an eta sweep
    sweepers = [r for r in recs if "eta_sweep" in r]
    if sweepers:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for r in sweepers:
            fig, ax = plt.subplots(figsize=(5, 4))
            es = r["eta_sweep"]
            ax.plot([e["clipdir"] for e in es], [e["struct"] for e in es],
                    "-o", color="#888", label="RF-inv eta sweep")
            for e in es:
                ax.annotate(f"{e['eta']:.1f}", (e["clipdir"], e["struct"]), fontsize=7)
            o = r["metrics"]["ours"]
            ax.scatter([o["clipdir"]], [o["struct"]], color="crimson", zorder=5,
                       label="ours (calibrated)")
            ax.set_xlabel("CLIP-dir (editability) ->")
            ax.set_ylabel("DINO structure distance (lower = better)")
            ax.set_title(r["key"])
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = os.path.join(OUT, f"pareto_{r['key'].replace('/', '_')}.png")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            print(f"[e41] wrote {p}", flush=True)


# ---------------------------------------------------------------------------
# verify: reproduce the hand-tuned dancers run + check the eta controller
# ---------------------------------------------------------------------------
def _mse(a, b):
    a = a.convert("RGB")
    b = b.convert("RGB").resize(a.size)
    aa = np.asarray(a, np.float32) / 255.0
    bb = np.asarray(b, np.float32) / 255.0
    return float(((aa - bb) ** 2).mean())


def run_verify(args):
    from e7_flux_phase import load_flux
    root = os.environ.get("CN_DANCERS")
    if not root or not os.path.exists(os.path.join(root, "config.json")):
        print("[e41] verify: set CN_DANCERS to the saved dancers run dir; skipping", flush=True)
        return
    cfg = json.load(open(os.path.join(root, "config.json")))["config"]
    src_img = Image.open(os.path.join(root, "input_real_img.png"))
    saved_edit = Image.open(os.path.join(root, "edited.png"))
    saved_base = Image.open(os.path.join(root, "baseline.png"))
    pipe = load_flux(args.mem)
    b = _build(pipe)                                                    # STEPS/GUIDANCE/INV match cfg
    peS, ppeS = ic.encode_prompt(pipe, cfg["src_prompt"])
    peE, ppeE = ic.encode_prompt(pipe, cfg["edit_prompt"])
    x0 = ic.pack(pipe, ic.vae_encode(pipe.vae, src_img))
    x_noise, traj = ic.rf_invert(pipe, peS, ppeS, x0, b["sig"], b["g_inv"])
    s = cfg["steps"]
    window = (round(cfg["interval"][0] * (s - 1)), round(cfg["interval"][1] * (s - 1)))
    ours = _edit(pipe, b, peE, ppeE, x_noise, traj=traj, mode=cfg["mode"], cut=cfg["cut"],
                 strength=cfg["strength"], window=window, idx=b["idx"], M=b["M"],
                 cen_k=b["cen_k"], phase_band=tuple(cfg["phase_band"]))
    vanilla = _edit(pipe, b, peE, ppeE, x_noise)                       # eta=0
    recon = _edit(pipe, b, peE, ppeE, x_noise, x0_packed=x0, eta=1.0, eta_window=(0, s - 1))
    print(f"[e41][verify] MSE(ours, saved_edited)   = {_mse(ours, saved_edit):.5f}", flush=True)
    print(f"[e41][verify] MSE(eta0, saved_baseline) = {_mse(vanilla, saved_base):.5f}", flush=True)
    print(f"[e41][verify] MSE(eta1 recon, source)   = {_mse(recon, src_img):.5f} "
          f"(should be small; eta=1 ~ reconstruction)", flush=True)
    print(f"[e41][verify] sanity MSE(source, saved_edited) = {_mse(src_img, saved_edit):.5f} "
          f"(reference scale)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="calibrate", help="verify | calibrate | analyze")
    ap.add_argument("--n_per_type", type=int, default=14)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--num", type=int, default=0, help="cap #images (after sharding)")
    ap.add_argument("--shard", default="", help="i/N stride shard for parallel jobs")
    ap.add_argument("--mem", default="bnb4")
    ap.add_argument("--eta_sweep_all", action="store_true",
                    help="run the eta sweep on every image (default: showcase only)")
    ap.add_argument("--force", action="store_true", help="recompute cached items")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for part in args.part.split(","):
        if part == "verify":
            run_verify(args)
        elif part == "calibrate":
            run_calibrate(args)
        elif part == "analyze":
            run_analyze(args)


if __name__ == "__main__":
    main()
