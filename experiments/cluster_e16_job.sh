#!/bin/bash
# Run:AI job entrypoint for the E16 sweep. Submitted as e.g.
#   runai submit --name e16-sweep -g 1 -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#       --pvc=storage:/storage --large-shm --command -- \
#       bash /storage/malnick/colorful-noise/experiments/cluster_e16_job.sh \
#       --part gen,score,analyze --num_prompts 8 --seeds 25 --steps 28 --no_vqa --mem gpu_resident
#
# Use a CUDA-12.x torch image (the cluster nodes have driver 12.2; a cu13 image
# reports cuda=False). The public pytorch/pytorch:2.10.0-cuda12.8 image works on
# these nodes (matched torch+torchvision) and other project jobs already use it.
# Reuses FLUX.1-dev + CLIP on the storage PVC (no re-download). All args after the
# script name are forwarded to e16_prompt_adherence.py.
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export PIP_BREAK_SYSTEM_PACKAGES=1     # image pip is PEP-668 externally-managed
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

echo "[job] installing core deps ..."
pip install --quiet --no-input \
    diffusers==0.38.0 transformers==4.57.6 accelerate bitsandbytes \
    sentencepiece protobuf matplotlib

# CUDA must actually be usable before we commit to a long run.
python -c "import torch; print('[job] torch',torch.__version__,'cuda',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE on this node/image'; print('[job] gpu',torch.cuda.get_device_name(0))"
python -c "import diffusers,transformers; print('[job] diffusers',diffusers.__version__,'transformers',transformers.__version__)"
python -c "from diffusers import FluxPipeline; print('[job] FluxPipeline import OK')"

echo "[job] installing optional scoring deps (graceful) ..."
( apt-get update -qq && apt-get install -y -qq git ) >/dev/null 2>&1 || echo "[job] WARN: git install failed"
pip install --quiet --no-input image-reward ftfy regex \
    "git+https://github.com/openai/CLIP.git" 2>&1 | tail -2 || \
    echo "[job] WARN: optional scoring deps failed; ImageReward will degrade gracefully"

echo "[job] launching e16 with args: $*"
python e16_prompt_adherence.py "$@"
echo "[job] done"
