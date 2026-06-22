#!/bin/bash
# Run:AI entrypoint for E47 (geodesic phasor-slerp phase whitening vs vanilla SDEdit, SDXL).
# Files staged on the PVC via rsync (the /storage checkout is not git; image has no git).
#
# Stages (1st arg):
#   --sub20    kill test: PIE-Bench subset n_per_type=2 (~20 imgs), draw vanilla frontier +
#              geodesic-global + geodesic-band, report which geodesic points beat the frontier.
#   --sub100   confirmation: n_per_type=10 (~100 imgs), same arms.
#
# Submit (from a shell where `runai` is logged in); SDXL fp16 fits a 24G card:
#   runai submit --name e47-sub20 --gpu-memory 24G \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --pvc=storage:/storage --large-shm --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e47_job.sh --sub20
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export CN_PIEBENCH_CACHE=/storage/malnick/datasets/pie_bench_hf
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
EXP=/storage/malnick/colorful-noise/experiments
STAGE="${1:---sub20}"

echo "[job] torch / cuda check"
python -c "import torch; print('[job] torch',torch.__version__,'cuda',torch.cuda.is_available()); assert torch.cuda.is_available(); print('[job] gpu',torch.cuda.get_device_name(0))"

echo "[job] installing diffusers/transformers/datasets + metric deps ..."
pip install --quiet --no-input \
    diffusers==0.33.1 transformers==4.47.1 accelerate==1.2.1 \
    tokenizers==0.21.0 sentencepiece==0.2.0 protobuf==3.20.3 "numpy<2" \
    datasets safetensors ftfy regex
python -c "from diffusers import StableDiffusionXLPipeline; print('[job] SDXL import OK')"

cd "$EXP"
# NFE=17: FlowAlign's plain-sampling budget on 24GB GPUs (inversion methods 17+17,
# FlowEdit/AlignFlow 33; our SDEdit + seed-phase and the vanilla baseline are sampling).
case "$STAGE" in
  --sub20)    python e47_geodesic.py --n_per_type 2  --steps 17 --method B   --tag sub20 ;;
  --sub100)   python e47_geodesic.py --n_per_type 10 --steps 17 --method B   --tag sub100 ;;
  --sweepA)   python e47_geodesic.py --n_per_type 2  --steps 17 --method A   --tag sweepA ;;
  --sweepSDG) python e47_geodesic.py --n_per_type 2  --steps 17 --method sdg --tag sweepSDG ;;
  --confA)    python e47_geodesic.py --n_per_type 10 --steps 17 --method A   --taus 0.125 0.25 0.375 --tag confA ;;
  --confSDG)  python e47_geodesic.py --n_per_type 10 --steps 17 --method sdg --taus 0.125 0.25 0.375 --taus_white 0.5 --tag confSDG ;;
  *) echo "[job] unknown stage '$STAGE'"; exit 1 ;;
esac
echo "[job] stage $STAGE done"
