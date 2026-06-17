#!/bin/bash
# Run:AI job entrypoint for E38 (CFG frequency direction). Submit as e.g.
#   runai submit --name e38 -g 1 -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#       --pvc=storage:/storage --large-shm --command -- \
#       bash /storage/malnick/colorful-noise/experiments/cluster_e38_job.sh \
#       --part gen,analyze --n_prompts 10 --cfgs 1.0,3.5,7.0 --steps 28
#
# Use a CUDA-12.x torch image (cluster driver 12.2; a cu13 image reports
# cuda=False). Reuses FLUX.1-dev on the storage PVC (no re-download). All args
# after the script name are forwarded to e38_cfg_direction.py.
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

echo "[job] installing deps ..."
pip install --quiet --no-input \
    diffusers==0.38.0 transformers==4.57.6 accelerate bitsandbytes \
    sentencepiece protobuf matplotlib

python -c "import torch; print('[job] torch',torch.__version__,'cuda',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE on this node/image'; print('[job] gpu',torch.cuda.get_device_name(0))"
python -c "from diffusers import FluxPipeline; print('[job] FluxPipeline import OK')"

echo "[job] launching e38 with args: $*"
python e38_cfg_direction.py "$@"
echo "[job] done"
