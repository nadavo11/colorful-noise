#!/bin/bash
# Run:AI entrypoint: E48 Probe 0 -- temporal-axis Fourier phasor sanity on LTX latents.
# Pure-math diagnostic (operator correctness + VAE shift-equivariance + fractional coherence)
# on a real clip at LTX-native non-square dims. No cluster sweep; one short run.
#
# Submit:
#   ~/.runai/bin/runai training standard submit e48-phasor -p avidan -g 1 --large-shm \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --existing-pvc claimname=storage,path=/storage --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e48_phasor.sh
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

echo "[job] installing deps ..."
pip install --quiet --no-input \
    diffusers==0.36.0 transformers==4.57.6 accelerate \
    sentencepiece protobuf imageio imageio-ffmpeg
python -c "import torch; assert torch.cuda.is_available(); print('[job] gpu',torch.cuda.get_device_name(0))"

echo "[job] ===== E48 Probe 0: temporal phasor @704x480 landscape, 49f (F_lat=7) ====="
python e48_temporal_phasor.py --width 704 --height 480 --frames 49
echo "[job] E48 Probe 0 done -- results in experiments/results/e48/"
