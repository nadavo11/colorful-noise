#!/bin/bash
# Run:AI entrypoint for the interactive token-frequency demo (Gradio).
# Submit (interactive, holds 1 GPU until you delete it):
#   runai submit --name e32-demo -g 1 --interactive \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --pvc=storage:/storage --large-shm --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_demo_job.sh
# Then from your machine:  kubectl port-forward e32-demo-0-0 7860:7860
#   and open http://localhost:7860
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

echo "[demo] installing deps ..."
# Pin gradio 5.9 + huggingface-hub<1.0 so diffusers 0.38 keeps working (gradio 6 pulls hf-hub>=1).
pip install --quiet --no-input \
    diffusers==0.38.0 transformers==4.57.6 accelerate bitsandbytes \
    sentencepiece protobuf "huggingface-hub==0.35.3" "gradio==5.9.1"
python -c "import torch; assert torch.cuda.is_available(); print('[demo] gpu', torch.cuda.get_device_name(0))"

echo "[demo] launching Gradio on 0.0.0.0:7860 ..."
exec python spectral_demo.py
