#!/bin/bash
# Run:AI entrypoint for E45 (FlowAlign on LTX-Video + spectral phase op).
# Files staged on the PVC via rsync (the /storage checkout is not git; image has no git).
# STAGE 0 (this script): SMOKE -> VAE round-trip GATE. Confirms LTX loads, generates a tiny
# clip, and the VAE encode->decode reproduces frames (precondition for any latent-space edit).
# Later stages (gen/analyze) are added once the smoke shapes are known.
#
# Submit (from the docker sandbox where runai lives):
#   runai training submit e45-ltx-smoke -g 1 \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --existing-pvc claimname=storage,path=/storage --large-shm --command -- \
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

# --- SMOKE: tiny clip + VAE round-trip, dumps every shape/const ---
echo "[job] ===== SMOKE: LTX gen + VAE round-trip ====="
python e45_ltx_flowalign.py --part smoke --steps 12 --frames 25 --size 256

# --- GATE: VAE round-trip must reproduce the source frames ---
echo "[job] ===== GATE: VAE round-trip fidelity ====="
python - <<'PY'
import os, sys
import numpy as np
import imageio.v3 as iio
OUT = os.path.join(os.path.dirname(os.path.abspath("e45_ltx_flowalign.py")), "results", "e45")
g = os.path.join(OUT, "smoke_gen.mp4"); r = os.path.join(OUT, "smoke_recon.mp4")
if not (os.path.exists(g) and os.path.exists(r)):
    print("[gate] FAIL: smoke videos missing"); sys.exit(1)
a = iio.imread(g); b = iio.imread(r)
n = min(len(a), len(b))
err = np.abs(a[:n].astype(float) - b[:n].astype(float)).mean() / 255.0
print(f"[gate] VAE round-trip L1 = {err:.4f} over {n} frames")
if err > 0.08:
    print("[gate] FAIL: VAE round-trip too lossy for a meaningful identity gate (KILL signal)")
    sys.exit(1)
print("[gate] PASS: LTX VAE reproduces the source clip; safe to build FlowAlign on it")
PY
echo "[job] STAGE0 done -- smoke videos + shape dump in experiments/results/e45/"
