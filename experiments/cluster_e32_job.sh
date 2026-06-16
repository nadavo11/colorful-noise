#!/bin/bash
# Run:AI entrypoint for E32 (per-object token-frequency control on two-object prompts).
# Files are staged on the storage PVC via kubectl cp (the checkout is NOT a git repo and
# the image has no git). SELF-GATING: preflight (spans/bins, no GPU) -> cheap smoke
# (1 prompt, 1 seed, 8 steps) + analyze -> CLIP sanity gate -> full sweep. Resumable
# (every generation is cached by file path).
#
# Submit:
#   runai submit --name e32-obj-freq -g 1 \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --pvc=storage:/storage --large-shm --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e32_job.sh
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

echo "[job] installing optional scoring deps (graceful) ..."
# libgl1/libglib: BLIP / transformers image processors can pull OpenCV (needs libGL.so.1)
( apt-get update -qq && apt-get install -y -qq git libgl1 libglib2.0-0 ) >/dev/null 2>&1 || echo "[job] WARN: apt deps failed"
pip install --quiet --no-input ftfy regex 2>&1 | tail -1 || echo "[job] WARN: clip text deps failed (degrade)"
# B-VQA: spaCy noun-phrase extraction + BLIP-VQA (transformers auto-downloads model)
pip install --quiet --no-input spacy 2>&1 | tail -1 && \
    python -m spacy download en_core_web_sm 2>&1 | tail -1 || echo "[job] WARN: spaCy failed; B-VQA degrades"
# NOTE: VQAScore (t2v-metrics) is deliberately NOT installed (pins transformers==4.49 +
# OpenCV, conflicting with the diffusers-0.38/transformers-4.57 Flux stack, as in E30).
# Per-object presence is covered by B-VQA; VQAScore can be added later from an isolated env.

# --- 0) PREFLIGHT: spans + bins-per-object, no GPU. Fail fast if a phrase won't map. ---
echo "[job] ===== PREFLIGHT (spans / bins) ====="
python e32_object_freq.py --part preflight

# --- 1) SMOKE: 1 prompt, 1 seed, 8 steps + analyze ---
echo "[job] ===== SMOKE (1 prompt, 1 seed, 8 steps) ====="
python e32_object_freq.py --part gen,analyze \
    --num_prompts 1 --seeds 1 --steps 8 --no_vqa --out_tag smoke

# --- 2) GATE: did the filtered embeds still yield an on-prompt image? ---
echo "[job] ===== GATE ====="
python - <<'PY'
import json, os, sys
from common import RESULTS
rp = os.path.join(RESULTS, "e32_smoke", "report.json")
if not os.path.exists(rp):
    print("[gate] FAIL: no smoke report"); sys.exit(1)
rep = json.load(open(rp)); raw = rep.get("raw", {})
ok = False
for k, e in raw.items():
    base = e["conditions"].get("baseline", {})
    vals = [s.get("clipA") for s in base.values()] + [s.get("clipB") for s in base.values()]
    vals = [v for v in vals if v is not None]
    m = max(vals) if vals else None
    print(f"[gate] {k}: baseline max per-object CLIP = {m}")
    if m is not None and m >= 0.18:
        ok = True
sys.exit(0 if ok else 1)
PY
echo "[gate] PASS"

# --- 3) FULL sweep: 10 prompts x 13 conditions x 3 seeds ---
echo "[job] ===== FULL: 10 prompts, 3 seeds ====="
python e32_object_freq.py --part gen,analyze --no_vqa
echo "[job] done -- results in experiments/results/e32/ (index.html)"
