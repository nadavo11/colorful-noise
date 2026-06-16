#!/bin/bash
# Run:AI entrypoint for E35 (token-frequency operator sweep on SD1.5).
# Ship via kubectl cp (storage PVC is not git). SELF-GATING: preflight (no GPU) -> cheap
# smoke (2 prompts, 2 seeds, 8 steps) -> CLIP sanity gate -> full thorough sweep.
# Resumable: every generation is cached by file path.
#
# Submit:
#   runai submit --name e35-sweep -g 1 \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --pvc=storage:/storage --large-shm --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e35_job.sh
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

echo "[job] installing deps ..."
# SD1.5 is not quantized -> no bitsandbytes. ftfy/regex for the CLIP tokenizer; matplotlib for plots.
pip install --quiet --no-input \
    diffusers==0.38.0 transformers==4.57.6 accelerate \
    ftfy regex matplotlib protobuf
python -c "import torch; assert torch.cuda.is_available(); print('[job] gpu', torch.cuda.get_device_name(0))"
python -c "from diffusers import StableDiffusionPipeline; print('[job] SD pipeline import OK')"

# --- 0) PREFLIGHT (no GPU): prompts/spans/counts/ETA ---
echo "[job] ===== PREFLIGHT ====="
python e35_op_sweep.py --part preflight

# --- 1) SMOKE: 2 prompts, 2 seeds, 8 steps, quick grid ---
echo "[job] ===== SMOKE ====="
python e35_op_sweep.py --part gen,analyze --coverage quick \
    --num_prompts 2 --seeds 2 --steps 8 --out_tag smoke

# --- 2) GATE: did SD1.5 gen-from-embeds produce an on-prompt baseline? ---
echo "[job] ===== GATE ====="
python - <<'PY'
import json, os, sys
from common import RESULTS
rp = os.path.join(RESULTS, "e35_smoke", "report.json")
if not os.path.exists(rp):
    print("[gate] FAIL: no smoke report"); sys.exit(1)
rep = json.load(open(rp)); raw = rep.get("raw", {})
ok = False
for pid, e in raw.items():
    base = e.get("conds", {}).get("baseline", {}).get("seeds", {})
    vals = [s.get("clip") for s in base.values() if s.get("clip") is not None]
    m = max(vals) if vals else None
    print(f"[gate] {pid}: baseline max CLIP = {m}")
    if m is not None and m >= 0.18:
        ok = True
sys.exit(0 if ok else 1)
PY
echo "[gate] PASS"

# --- 3) FULL thorough sweep ---
echo "[job] ===== FULL (thorough) ====="
python e35_op_sweep.py --part gen,analyze --coverage thorough
echo "[job] done -- results in experiments/results/e35/ (index.html)"
