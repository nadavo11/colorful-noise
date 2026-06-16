#!/bin/bash
# Run:AI entrypoint for E31 (FlowEdit + frequency-surgery conditioning).
# Files staged on the PVC via kubectl cp (checkout is not git; image has no git).
# SELF-GATING: cheap smoke (1 scene, 8 steps) -> reconstruction gate (recon must
# reproduce the source: px-dist small) -> full run. Resumable (edits cached).
#
# Submit:
#   runai submit --name e31-flowedit -g 1 \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --pvc=storage:/storage --large-shm --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e31_job.sh
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
python -c "from diffusers import FluxPipeline; print('[job] FluxPipeline import OK')"
python e31_flowedit_freq.py --part preflight   # model-free math check
echo "[job] installing optional scoring deps (graceful) ..."
pip install --quiet --no-input image-reward ftfy regex \
    "git+https://github.com/openai/CLIP.git" 2>&1 | tail -1 || echo "[job] WARN: aesthetic/clip deps degrade"

# --- 1) SMOKE: 1 scene, 8 steps ---
echo "[job] ===== SMOKE: 1 scene, 8 steps ====="
python e31_flowedit_freq.py --part gen,analyze --num 1 --steps 8 --out_tag smoke

# --- 2) GATE: recon must reproduce the source (px-dist small) ---
echo "[job] ===== GATE: reconstruction fidelity ====="
python - <<'PY'
import json, os, sys
from common import RESULTS
rp = os.path.join(RESULTS, "e31_smoke", "report.json")
if not os.path.exists(rp):
    print("[gate] FAIL: no smoke report"); sys.exit(1)
rep = json.load(open(rp)); ok = False
for k, e in rep.get("sources", {}).items():
    px = e["conds"].get("recon", {}).get("px_dist_to_src")
    print(f"[gate] {k}: recon px-dist = {px}")
    if px is not None and px < 0.05:
        ok = True
if not ok:
    print("[gate] FAIL: recon did not reproduce source (VAE/packing path broken)")
    sys.exit(1)
print("[gate] PASS: FlowEdit pipeline reconstructs the source")
PY

# --- 3) FULL ---
echo "[job] ===== FULL: 3 scenes, 28 steps ====="
python e31_flowedit_freq.py --part gen,analyze --num 3 --steps 28
echo "[job] done -- results in experiments/results/e31/ (index.html)"
