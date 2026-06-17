#!/bin/bash
# Run:AI entrypoint for the E35 VS-BASELINE-GENERATION views ONLY (no re-gen).
# Assumes the full E35 sweep already ran (results/e35/report.json + per-(prompt,cond,seed) PNGs
# exist on /storage). Re-references every edit to its same-seed baseline image and writes
# vs_baseline.html (directional ImageReward/aesthetic + distance LPIPS/SSIM/PSD/color).
#
# Submit (after `git pull` lands this file on /storage):
#   runai submit --name e35-vsbase -g 1 \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --pvc=storage:/storage --large-shm --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e35_vsbase_job.sh
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

echo "[vsbase-job] installing deps ..."
pip install --quiet --no-input \
    transformers==4.57.6 accelerate ftfy regex tqdm matplotlib protobuf \
    lpips scikit-image image-reward
# image-reward imports the OpenAI `clip` module but does NOT declare it as a dep;
# clip-anytorch is the pip-installable build (the runtime image has no `git` for git+url)
pip install --quiet --no-input clip-anytorch
python -c "import torch; print('[vsbase-job] gpu', torch.cuda.is_available() and torch.cuda.get_device_name(0))"

# sanity: the sweep must have produced a report to re-reference
if [ ! -f results/e35/report.json ]; then
    echo "[vsbase-job] FAIL: results/e35/report.json missing -- run the full sweep first"; exit 2
fi

echo "[vsbase-job] ===== vs-prompt delta (kept for contrast) ====="
python e35_delta.py || echo "[vsbase-job] e35_delta.py failed (non-fatal)"

echo "[vsbase-job] ===== vs-baseline-generation (directional + distance) ====="
python e35_vs_baseline.py --with_images

echo "[vsbase-job] done -- results/e35/vs_baseline.html (+ vs_baseline.json)"
