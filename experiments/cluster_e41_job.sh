#!/bin/bash
# Run:AI entrypoint for E41 (per-image calibration + fair RF-inversion eta comparison).
# Self-gating: dancers verify (Stage-1 regression) -> throwaway 2-image smoke -> full shard.
# Submit one shard with:
#   runai submit --name e41-s0-8 -g 1 -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#       --pvc=storage:/storage --large-shm --command -- \
#       bash /storage/malnick/colorful-noise/experiments/cluster_e41_job.sh 0/8 <extra e41 args>
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export TORCH_HOME=/storage/malnick/torch_hub_cache       # persist the DINO hub weights
export CN_DANCERS=/storage/malnick/colorful-noise/experiments/saved_runs/dancers
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

echo "[e41] installing deps ..."
# No openai/CLIP (needs git, absent in the image) -- metrics use transformers CLIP via clip_sim.
pip install --quiet --no-input \
    diffusers==0.38.0 transformers==4.57.6 accelerate bitsandbytes \
    sentencepiece protobuf matplotlib optuna lpips scikit-image datasets 2>&1 | tail -3 || \
    { echo "[e41] FATAL: dep install failed"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(); print('[e41] gpu', torch.cuda.get_device_name(0))"

SHARD="${1:-0/1}"; shift || true

echo "[e41] verify (dancers regression + eta controller) ..."
python e41_calibrate.py --part verify || echo "[e41] WARN: verify failed (non-fatal)"

echo "[e41] smoke gate: 2 images, 3 trials (throwaway results dir) ..."
CN_RESULTS=/tmp/e41_smoke python e41_calibrate.py --part calibrate \
    --shard "$SHARD" --num 2 --trials 3 "$@" || { echo "[e41] SMOKE FAILED"; exit 1; }

echo "[e41] full shard $SHARD ..."
python e41_calibrate.py --part calibrate --shard "$SHARD" "$@"
echo "[e41] done shard $SHARD"
