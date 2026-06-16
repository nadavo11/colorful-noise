#!/bin/bash
# Run:AI entrypoint for E30 (continuous text-frequency control & extraction).
# Files are staged on the storage PVC via kubectl cp (the checkout is NOT a git repo
# and the image has no git). SELF-GATING: cheap smoke (probe_deep + analyze) -> CLIP
# sanity gate -> full sweep. Resumable (every generation is cached).
#
# Submit:
#   runai submit --name e30-text-freq -g 1 \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --pvc=storage:/storage --large-shm --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e30_job.sh
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
( apt-get update -qq && apt-get install -y -qq git ) >/dev/null 2>&1 || echo "[job] WARN: git install failed"
pip install --quiet --no-input image-reward ftfy regex \
    "git+https://github.com/openai/CLIP.git" 2>&1 | tail -1 || echo "[job] WARN: imagereward/clip deps failed (degrade)"
# B-VQA: spaCy noun-phrase extraction + BLIP-VQA (transformers auto-downloads model)
pip install --quiet --no-input spacy 2>&1 | tail -1 && \
    python -m spacy download en_core_web_sm 2>&1 | tail -1 || echo "[job] WARN: spaCy failed; B-VQA degrades"
# VQAScore (heavy; optional). If install fails, run with --no_vqa effect via graceful load.
pip install --quiet --no-input t2v-metrics 2>&1 | tail -1 || echo "[job] WARN: t2v-metrics failed; VQAScore degrades"

# --- 1) SMOKE: probe_deep + analyze, cheap ---
echo "[job] ===== SMOKE: probe_deep (1 prompt, 8 steps) ====="
python e30_text_freq_control.py --part probe_deep,analyze \
    --num_prompts 1 --steps 8 --no_vqa --out_tag smoke

# --- 2) GATE: coherent image from filtered embeds? ---
echo "[job] ===== GATE ====="
python - <<'PY'
import json, os, sys
from common import RESULTS
rp = os.path.join(RESULTS, "e30_smoke", "report.json")
if not os.path.exists(rp):
    print("[gate] FAIL: no smoke report"); sys.exit(1)
rep = json.load(open(rp)); pd = rep.get("probe_deep", {})
ok = False
for k, e in pd.items():
    full = e["variants"].get("full", {}).get("clip")
    m = full["mean"] if full else None
    print(f"[gate] {k}: full CLIP = {m}")
    if m is not None and m >= 0.18:
        ok = True
sys.exit(0 if ok else 1)
PY
echo "[gate] PASS"

# --- 3) FULL sweep ---
echo "[job] ===== FULL: probe_deep,continuous,concat,longprompt,compositional,analyze ====="
python e30_text_freq_control.py \
    --part probe_deep,continuous,concat,longprompt,compositional,analyze
echo "[job] done -- results in experiments/results/e30/ (index.html)"
