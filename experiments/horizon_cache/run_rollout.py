"""Driver: generate the rollout safe-horizon dataset + train HorizonCache-v1.

    python -m horizon_cache.run_rollout --n 2 --steps 28 --width 512 --height 512 \
        --H 5 --stride 2 --out ../metrics/horizon_cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from horizon_cache import capability
from horizon_cache.rollout import build_dataset
from horizon_cache.train_v1 import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seeds-per-prompt", type=int, default=1)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--tau-cache", type=float, default=0.3)
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--tol-l2", type=float, default=0.02)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--out", default="metrics/horizon_cache")
    ap.add_argument("--bundle", default="results/horizon_cache/v1_bundle.joblib")
    ap.add_argument("--mode", choices=["classify", "ordinal"], default="classify")
    args = ap.parse_args()

    caps = capability.detect()
    if not caps["flux_generation"]:
        print(json.dumps({"status": "FAILED", "reason": "no FLUX"})); return

    from flux_seacache_dp_shortcuts import load_flux_pipeline
    from fixtures import canonical_prompts
    prompts = canonical_prompts()[: args.n]
    seeds = list(range(args.seeds_per_prompt))

    pipe = load_flux_pipeline(caps["flux_model_id"], "bf16", "cuda",
                              offload=False, transformer_only_4bit=True)
    L = len(pipe.transformer.transformer_blocks) + len(pipe.transformer.single_transformer_blocks)

    out = Path(args.out)
    csv_p = out / "action_dataset.csv"
    pq_p = out / "action_dataset.parquet"
    ds = build_dataset(pipe, prompts, seeds, args.steps, args.height, args.width,
                       args.guidance, "cuda", csv_p, pq_p, H=args.H, tol_l2=args.tol_l2,
                       max_seq_len=args.max_seq_len, tau_cache=args.tau_cache, L=L,
                       stride=args.stride)
    print("[rollout]", json.dumps(ds, indent=2))
    diag = train(csv_p, Path(args.bundle), mode=args.mode)
    print("[train]", json.dumps(diag, indent=2))
    (out / "v1_diag.json").write_text(json.dumps(diag, indent=2))
    (out / "rollout_meta.json").write_text(json.dumps(ds, indent=2))


if __name__ == "__main__":
    main()
