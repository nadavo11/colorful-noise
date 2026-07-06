"""HorizonCache runner — generation path (FLUX) + capability-gated editing stub.

Smoke:   python -m horizon_cache.run --mode generation --smoke
Full:    python -m horizon_cache.run --mode generation --n 20 --steps 50 --width 1024 --height 1024

Writes per-image + aggregate metrics, per-step action traces, jump logs, compute
accounting, config, git hash. Baselines run on the SAME image/prompt set at a matched
threshold grid; per-image deltas vs SeaCache are computed (deck fair-comparison rules).
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # experiments/ on path

from horizon_cache import capability, metrics as M
from horizon_cache.baselines import FullPolicy, UniformEveryK, RandomK, SeaCachePolicy
from horizon_cache.policy import HorizonCacheV0, HorizonV0Config, HorizonCacheV1, FEATURE_NAMES
from horizon_cache.flux_gen import sample_flux
from horizon_cache.scheduler import ACTIONS


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(HERE)).decode().strip()
    except Exception:
        return "unknown"


def canonical_prompts(n: int) -> list[dict[str, Any]]:
    from fixtures import canonical_prompts as cp
    return cp()[:n]


def build_methods(args, num_steps: int, seed: int) -> dict[str, Any]:
    """Return {method_name: policy}. SeaCache + HorizonCache-v0 swept over a shared grid."""
    methods: dict[str, Any] = {}
    methods["full"] = FullPolicy()
    # uniform / random matched to a caching budget (~ every-2 for a ~1.7x floor)
    methods["uniform_k2"] = UniformEveryK(2)
    methods["uniform_k3"] = UniformEveryK(3)
    methods["random_k"] = RandomK(k_fresh=max(2, num_steps // 2), num_steps=num_steps, seed=seed)
    # shared threshold grid for the causal gates
    for tau in args.tau_grid:
        methods[f"seacache_t{tau:g}"] = SeaCachePolicy(tau)
        cfg = HorizonV0Config(tau_cache=tau, enable_jump=True, jump_mode=args.jump_mode,
                              jf_max_frontier=1.5)
        methods[f"horizon_v0_t{tau:g}"] = HorizonCacheV0(cfg)
    # jump_2.0 ablation at the middle tau (deck: bigger stride overshoots)
    mid = args.tau_grid[len(args.tau_grid) // 2]
    ab = HorizonV0Config(tau_cache=mid, enable_jump=True, jump_mode=args.jump_mode,
                         allow_jump2=True, jf_max_frontier=2.0)
    methods[f"horizon_v0_jump2_t{mid:g}"] = HorizonCacheV0(ab)
    # optional learned policy
    if args.v1_bundle and Path(args.v1_bundle).exists():
        import joblib
        bundle = joblib.load(args.v1_bundle)
        safety = HorizonV0Config(tau_cache=args.tau_grid[len(args.tau_grid) // 2], jump_mode=args.jump_mode)
        methods["horizon_v1"] = HorizonCacheV1(bundle, safety)
    return methods


def run_generation(args) -> dict[str, Any]:
    from flux_seacache_dp_shortcuts import load_flux_pipeline, decode_flux_latents
    caps = capability.detect()
    if not caps["flux_generation"]:
        return {"status": "FAILED", "reason": "no FLUX generation backend", "caps": caps}

    run_dir = Path(args.out) / f"gen_{time.strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)

    prompts = canonical_prompts(args.n)
    seeds = list(range(args.seed_base, args.seed_base + args.seeds_per_prompt))
    print(f"[gen] {len(prompts)} prompts x {len(seeds)} seeds, steps={args.steps} "
          f"{args.width}x{args.height}", flush=True)

    pipe = load_flux_pipeline(caps["flux_model_id"], args.dtype, args.device,
                              offload=False, transformer_only_4bit=args.bnb4)
    L = len(pipe.transformer.transformer_blocks) + len(pipe.transformer.single_transformer_blocks)
    print(f"[gen] FLUX block count L={L}", flush=True)

    methods = build_methods(args, args.steps, seeds[0])
    rows: list[dict[str, Any]] = []
    baseline_wall: dict[str, float] = {}

    for p in prompts:
        for seed in seeds:
            key = f"{p['id']}_s{seed}"
            # reference = full trajectory (also our wall-time baseline)
            ref = sample_flux(pipe, p["prompt"], seed, args.steps, args.height, args.width,
                              args.guidance, args.device, methods["full"], args.max_seq_len,
                              action_source="full", L=L)
            ref_img = decode_flux_latents(pipe, ref["final_latent"], args.height, args.width)
            baseline_wall[key] = ref["ledger"].measured_wall_time
            ref_img.save(run_dir / "samples" / f"{key}__full.png")
            ref_latent = ref["final_latent"].detach().cpu()

            for name, policy in methods.items():
                if name == "full":
                    res = ref
                    img = ref_img
                else:
                    res = sample_flux(pipe, p["prompt"], seed, args.steps, args.height,
                                      args.width, args.guidance, args.device, policy,
                                      args.max_seq_len, action_source=name.split("_")[0], L=L)
                    img = decode_flux_latents(pipe, res["final_latent"], args.height, args.width)
                    if args.save_all_images:
                        img.save(run_dir / "samples" / f"{key}__{name}.png")
                led = res["ledger"]
                led.baseline_wall_time = baseline_wall[key]
                m = M.image_metrics(ref_img, img)
                m["latent_l2"] = M.latent_l2(ref_latent, res["final_latent"].detach().cpu())
                row = {"key": key, "prompt_id": p["id"], "tag": p.get("tag"), "seed": seed,
                       "method": name, **m, **led.as_dict(),
                       "n_jumps": len(res["jumps"]),
                       "actions": {a: sum(1 for t in res["traces"] if t["action"] == a) for a in ACTIONS}}
                rows.append(row)
                # persist trace for timeline/scatter figures
                (run_dir / "traces" / f"{key}__{name}.json").write_text(json.dumps(
                    {"traces": res["traces"], "jumps": res["jumps"],
                     "ledger": led.as_dict()}, indent=2))
            print(f"[gen] done {key}", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    (run_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    _write_csv(run_dir / "metrics.csv", rows)
    summary = summarize_generation(rows, args.tau_grid)
    cfg = {**vars(args), "git": git_hash(), "flux_L": L, "block_stack_cost_note":
           "cached/jump node ~ 1/L of a fresh forward", "fixture": "canonical_prompts v1",
           "caps": caps}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    del pipe; gc.collect(); torch.cuda.empty_cache()
    return {"status": "DONE", "run_dir": str(run_dir), "summary": summary,
            "n_rows": len(rows), "caps": caps, "L": L}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    if not rows:
        path.write_text("")
        return
    base_keys = [k for k in rows[0] if k != "actions"]
    action_keys = [f"act_{a}" for a in ACTIONS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(base_keys + action_keys)
        for r in rows:
            w.writerow([r.get(k) for k in base_keys] + [r["actions"].get(a, 0) for a in ACTIONS])


def summarize_generation(rows: list[dict[str, Any]], tau_grid) -> dict[str, Any]:
    """Per-method aggregates + matched-budget per-image deltas vs SeaCache."""
    import collections
    by_method = collections.defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    def agg(rs):
        return {
            "n": len(rs),
            "psnr": float(np.mean([r["psnr"] for r in rs])),
            "lpips": float(np.mean([r["lpips"] for r in rs if "lpips" in r])) if any("lpips" in r for r in rs) else None,
            "ssim": float(np.mean([r["ssim"] for r in rs if "ssim" in r])) if any("ssim" in r for r in rs) else None,
            "compute_speedup": float(np.mean([r["compute_speedup"] for r in rs])),
            "wall_speedup": float(np.mean([r["wall_speedup"] for r in rs if r.get("wall_speedup")])) if any(r.get("wall_speedup") for r in rs) else None,
            "n_jumps": float(np.mean([r["n_jumps"] for r in rs])),
        }

    method_agg = {m: agg(rs) for m, rs in by_method.items()}

    # matched-budget per-image delta: HorizonCache-v0 vs SeaCache at the SAME tau
    deltas = {}
    for tau in tau_grid:
        sea = {r["key"]: r for r in by_method.get(f"seacache_t{tau:g}", [])}
        hor = {r["key"]: r for r in by_method.get(f"horizon_v0_t{tau:g}", [])}
        keys = set(sea) & set(hor)
        if not keys:
            continue
        dpsnr = [hor[k]["psnr"] - sea[k]["psnr"] for k in keys]
        dspeed = [hor[k]["compute_speedup"] - sea[k]["compute_speedup"] for k in keys]
        wins = sum(1 for d in dpsnr if d > 0)
        deltas[f"t{tau:g}"] = {
            "n": len(keys),
            "mean_delta_psnr": float(np.mean(dpsnr)),
            "mean_delta_compute_speedup": float(np.mean(dspeed)),
            "win_rate_psnr": wins / len(keys),
            "seacache_psnr": float(np.mean([sea[k]["psnr"] for k in keys])),
            "horizon_psnr": float(np.mean([hor[k]["psnr"] for k in keys])),
            "seacache_speedup": float(np.mean([sea[k]["compute_speedup"] for k in keys])),
            "horizon_speedup": float(np.mean([hor[k]["compute_speedup"] for k in keys])),
        }
    # matched-ACHIEVED-speedup frontier delta (the deck's fair rule): interpolate the
    # SeaCache PSNR(speedup) curve at each HorizonCache operating point's achieved speedup.
    sea_pts = sorted([(method_agg[f"seacache_t{t:g}"]["compute_speedup"],
                       method_agg[f"seacache_t{t:g}"]["psnr"]) for t in tau_grid
                      if f"seacache_t{t:g}" in method_agg])
    matched = {}
    if len(sea_pts) >= 2:
        sx = [p[0] for p in sea_pts]; sy = [p[1] for p in sea_pts]
        for t in tau_grid:
            hk = f"horizon_v0_t{t:g}"
            if hk not in method_agg:
                continue
            hs = method_agg[hk]["compute_speedup"]; hp = method_agg[hk]["psnr"]
            sea_at = float(np.interp(hs, sx, sy))
            matched[f"t{t:g}"] = {
                "horizon_speedup": round(hs, 4), "horizon_psnr": round(hp, 4),
                "seacache_psnr_at_matched_speedup": round(sea_at, 4),
                "matched_delta_psnr": round(hp - sea_at, 4),
                "extrapolated": bool(hs > max(sx) or hs < min(sx)),
            }
    return {"method_agg": method_agg, "matched_deltas_v0_vs_seacache": deltas,
            "matched_speedup_deltas": matched, "seacache_frontier": sea_pts}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="HorizonCache runner")
    ap.add_argument("--mode", choices=["generation", "editing"], default="generation")
    ap.add_argument("--smoke", action="store_true", help="tiny fast config")
    ap.add_argument("--n", type=int, default=20, help="#prompts")
    ap.add_argument("--seeds-per-prompt", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--bnb4", action="store_true", default=True,
                    help="4-bit transformer (required to fit FLUX on a 24GB card)")
    ap.add_argument("--no-bnb4", dest="bnb4", action="store_false")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--tau-grid", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    ap.add_argument("--jump-mode", choices=["regrid", "drop"], default="regrid")
    ap.add_argument("--allow-jump2", action="store_true")
    ap.add_argument("--v1-bundle", default="")
    ap.add_argument("--save-all-images", action="store_true")
    ap.add_argument("--out", default="results/horizon_cache")
    return ap


def main():
    args = build_argparser().parse_args()
    if args.smoke:
        args.n = min(args.n, 3)
        args.steps = 28
        args.width = args.height = 512
        args.tau_grid = [0.3]
        args.save_all_images = True
    if args.mode == "editing":
        print(json.dumps({"status": "PARTIAL", "reason":
              "editing path capability-detected but not run in this smoke; see report",
              "caps": capability.detect()}, indent=2))
        return
    res = run_generation(args)
    print(json.dumps({k: v for k, v in res.items() if k != "summary"}, indent=2, default=str))
    if "summary" in res:
        print("---SUMMARY---")
        print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
