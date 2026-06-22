#!/bin/bash
# Run:AI entrypoint: RE-RUN the E45 headline comparisons at LTX-NATIVE resolution (non-square),
# since the S2-S7 numbers were measured on resolution-distorted 256/512 square clips.
#   native_toy      : generated toy car->tank @704x480 landscape, 49f
#   native_cockatoo : real cockatoo->parrot @448x768 PORTRAIT (no aspect squashing), 49f
# Each = gen+analyze with the paper's frame-by-frame baseline (fbf) + 2D/3D phase (cut 0.2).
#
# Submit:
#   ~/.runai/bin/runai training standard submit e45-ltx-native -p avidan -g 1 --large-shm \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --existing-pvc claimname=storage,path=/storage --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e45_native.sh
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
python -c "import torchvision; from torchvision.models.optical_flow import raft_small; print('[job] RAFT OK')"

# delete the resolution-distorted (256/512 square) result dirs that these native runs supersede.
echo "[job] cleaning up superseded distorted results ..."
rm -rf results/e45 results/e45_w7.5 results/e45_w10 results/e45_w13.5 results/e45_w18 \
       results/e45_real512
echo "[job] kept results/e45_compare (native diagnostic); removed 256/512 square runs"

echo "[job] ===== NATIVE: toy @704x480 landscape ====="
python e45_ltx_flowalign.py --part gen,analyze --width 704 --height 480 --frames 49 --steps 30 \
    --w 10 --zeta 0.01 --cuts 0.2 --fbf --out_tag native_toy

echo "[job] ===== NATIVE: real cockatoo @448x768 portrait ====="
python e45_ltx_flowalign.py --part gen,analyze --width 448 --height 768 --frames 49 --steps 30 \
    --w 10 --zeta 0.01 --cuts 0.2 --fbf --out_tag native_cockatoo \
    --real_video "imageio:cockatoo.mp4" \
    --src_caption "a white cockatoo bird perched on a branch, moving its head" \
    --edit_prompt "a colorful rainbow parrot perched on a branch, moving its head"
echo "[job] NATIVE done -- results in experiments/results/e45_native_toy/ and e45_native_cockatoo/"
