#!/bin/bash
# Run:AI entrypoint for E45 (FlowAlign on LTX-Video + spectral phase op).
# Files staged on the PVC via rsync (the /storage checkout is not git; image has no git).
# STAGE 2/3 (this script): gen (identity recon + plain-FlowAlign baseline + 2D/3D phase sweep)
# -> IDENTITY GATE -> analyze (metric bundle: DINO struct-dist + CLIP-directional + RAFT
# warp-error global & edited-region-masked) -> GOAL gate (a phase variant beats baseline on
# struct + masked-warp while holding CLIP-directional). 49 frames -> 7 latent temporal bins.
#
# Submit (new CLI):
#   ~/.runai/bin/runai training standard submit e45-ltx-s2 -p avidan -g 1 --large-shm \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --existing-pvc claimname=storage,path=/storage --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e45_job.sh
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

echo "[job] installing deps ..."
pip install --quiet --no-input \
    diffusers==0.36.0 transformers==4.57.6 accelerate \
    sentencepiece protobuf imageio imageio-ffmpeg
python -c "import torch; print('[job] torch',torch.__version__,'cuda',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; print('[job] gpu',torch.cuda.get_device_name(0))"
python -c "from diffusers import LTXPipeline; print('[job] LTXPipeline import OK')"
python -c "import torchvision; from torchvision.models.optical_flow import raft_small; print('[job] torchvision',torchvision.__version__,'RAFT OK')"

# --- S2/S3: gen (identity recon + baseline + 2D/3D phase sweep) ---
echo "[job] ===== S2/S3: FlowAlign-on-LTX gen (sweep) ====="
python e45_ltx_flowalign.py --part gen --steps 24 --frames 49 --size 512 --w 10 --zeta 0.01 --cuts 0.2,0.35

# --- IDENTITY GATE: C_tar==C_src must reproduce the source clip ---
echo "[job] ===== GATE: identity reconstruction ====="
python - <<'PY'
import json, os, sys
OUT = os.path.join(os.path.dirname(os.path.abspath("e45_ltx_flowalign.py")), "results", "e45")
rp = os.path.join(OUT, "gen_report.json")
if not os.path.exists(rp):
    print("[gate] FAIL: no gen_report.json"); sys.exit(1)
l1 = json.load(open(rp)).get("recon_l1")
print(f"[gate] identity recon L1 = {l1}")
if l1 is None or l1 > 0.10:
    print("[gate] FAIL: FlowAlign-on-LTX did not reproduce the source (plumbing broken)")
    sys.exit(1)
print("[gate] PASS: FlowAlign-on-LTX reconstructs the source")
PY

# --- ANALYZE + GOAL: metric bundle, phase vs baseline ---
echo "[job] ===== ANALYZE: metric bundle + GOAL ====="
python e45_ltx_flowalign.py --part analyze
echo "[job] S2/S3 done -- mp4s + gen_report.json in experiments/results/e45/"
