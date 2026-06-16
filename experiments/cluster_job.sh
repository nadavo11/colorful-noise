#!/bin/bash
# Generic Run:AI job entrypoint: install the pinned stack, assert CUDA, then run
# the given python script with the remaining args. Submit with:
#   runai submit --name <n> -g 1 -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#       --pvc=storage:/storage --large-shm --command -- \
#       bash /storage/malnick/colorful-noise/experiments/cluster_job.sh \
#       <script.py> <args...>
# (Use the cu12.x pytorch image — cluster nodes have driver 12.2; a cu13 image
# reports cuda=False.) Reuses cached models in HF_HOME on the storage PVC.
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

echo "[job] installing core deps ..."
pip install --quiet --no-input \
    diffusers==0.38.0 transformers==4.57.6 accelerate bitsandbytes \
    sentencepiece protobuf matplotlib

python -c "import torch; print('[job] torch',torch.__version__,'cuda',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; print('[job] gpu',torch.cuda.get_device_name(0))"
python -c "import diffusers,transformers; print('[job] diffusers',diffusers.__version__,'transformers',transformers.__version__)"

echo "[job] installing optional scoring deps (graceful) ..."
( apt-get update -qq && apt-get install -y -qq git ffmpeg ) >/dev/null 2>&1 || echo "[job] WARN: apt (git/ffmpeg) install failed"
pip install --quiet --no-input image-reward ftfy regex \
    "git+https://github.com/openai/CLIP.git" 2>&1 | tail -2 || \
    echo "[job] WARN: optional scoring deps failed; ImageReward degrades gracefully"
# spaCy (+model) for T2I-CompBench B-VQA noun-phrase extraction (e17_compbench)
pip install --quiet --no-input spacy 2>&1 | tail -1 && \
    python -m spacy download en_core_web_sm 2>&1 | tail -1 || \
    echo "[job] WARN: spaCy install failed; B-VQA will degrade gracefully"
# (E23 adherence uses B-VQA = BLIP-VQA via transformers, deps already above. We do
# NOT install t2v-metrics: it pins torch<2.6, which then breaks BLIP's torch.load.)

SCRIPT="$1"; shift
echo "[job] launching $SCRIPT with args: $*"
python "$SCRIPT" "$@"
echo "[job] done"
