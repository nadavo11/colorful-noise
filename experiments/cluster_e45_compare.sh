#!/bin/bash
# Run:AI entrypoint for E45 DIAGNOSTIC: canonical FlowEdit (reference baseline) vs my FlowAlign
# variants at LTX-native resolution (704x480 landscape, 49f). Isolates the cause of the distortion:
#   identity                -> sanity (FlowEdit C_tar=C_src must reproduce source)
#   flowedit                -> canonical FlowEdit (src_gs 1.5 / tar_gs 3.5, n_max window, fresh noise)
#   flowalign_hi_allsteps   -> my CURRENT FlowAlign (w=10, edits ALL steps) -- the distorted one
#   flowalign_hi_window     -> + n_max window (skip high-noise early steps)
#   flowalign_lo_window     -> + n_max window AND low guidance (w=3)
# Eyeball the clips in results/e45_compare/ to see which fixes the distortion.
#
# Submit:
#   ~/.runai/bin/runai training standard submit e45-ltx-compare -p avidan -g 1 --large-shm \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --existing-pvc claimname=storage,path=/storage --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e45_compare.sh
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

echo "[job] ===== COMPARE: FlowEdit baseline vs FlowAlign @704x480 ====="
python e45_ltx_flowalign.py --part compare --width 704 --height 480 --frames 49 --steps 30 \
    --w 10 --w_lo 3 --src_gs 1.5 --tar_gs 3.5 --skip_frac 0.15 --n_avg 1 --zeta 0.01 \
    --out_tag compare
echo "[job] COMPARE done -- clips in experiments/results/e45_compare/"
