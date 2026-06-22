#!/bin/bash
# Run:AI entrypoint: E47 CONFIRMATION of the sub20 winners at a larger partial PIE-Bench set
# (n_per_type=10, ~100 imgs). The sub20 sweeps found A_t0.25 and sdg_src_t0.25 beat the vanilla
# SDEdit frontier; this checks whether the t=0.25 operating point holds at scale (pick-on-20,
# validate-on-100). Arg = method: A (geodesic noise in SDEdit) or sdg (spectral-geodesic SDEdit).
#
# Submit (fresh name each):
#   ~/.runai/bin/runai training standard submit e47-amethod-sub100 -p avidan -g 1 --large-shm \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --existing-pvc claimname=storage,path=/storage --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e47_confirm.sh A
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export CN_PIEBENCH_CACHE=/storage/malnick/datasets/pie_bench_hf
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
EXP=/storage/malnick/colorful-noise/experiments
METHOD="${1:-A}"

python -c "import torch; assert torch.cuda.is_available(); print('[job] gpu',torch.cuda.get_device_name(0))"
echo "[job] installing deps ..."
pip install --quiet --no-input \
    diffusers==0.33.1 transformers==4.47.1 accelerate==1.2.1 \
    tokenizers==0.21.0 sentencepiece==0.2.0 protobuf==3.20.3 "numpy<2" \
    datasets safetensors ftfy regex
python -c "from diffusers import StableDiffusionXLPipeline; print('[job] SDXL import OK')"

cd "$EXP"
echo "[job] ===== E47 confirm: method=$METHOD n_per_type=10 (~100 imgs) steps=17 ====="
python e47_geodesic.py --n_per_type 10 --steps 17 --method "$METHOD" --tag "${METHOD}sub100"
echo "[job] E47 confirm method=$METHOD done -- results/e47_${METHOD}sub100/verdict.md"
