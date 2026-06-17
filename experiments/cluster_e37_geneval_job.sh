#!/bin/bash
# Run:AI entrypoint for E37 GenEval eval (velocity spectral normalization on SD3.5 medium).
# Ship via kubectl cp (storage PVC is not git). SELF-GATING: preflight (no GPU) -> tag-spanning
# smoke (6 prompts x 7 conds, gen+score+summary) -> gate -> full gen (553x7, n=1, 512px, w=4.5)
# -> score -> summary. Resumable: every image cached by file path; scoring re-reads the PNGs.
#
# Submit:
#   runai submit --name e37-geneval -g 1 \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --pvc=storage:/storage --large-shm --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e37_geneval_job.sh
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export TORCH_HOME=/storage/malnick/torch_cache       # torchvision detector weights cache
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

echo "[job] installing deps ..."
# SD3.5 (T5 -> sentencepiece/protobuf). Detector = torchvision (in the pytorch image),
# colour classifier = transformers CLIP. No mmdet / open_clip / clip_benchmark.
pip install --quiet --no-input \
    diffusers==0.38.0 transformers==4.57.6 accelerate sentencepiece protobuf
python -c "import torch,torchvision; assert torch.cuda.is_available(); print('[job] gpu',torch.cuda.get_device_name(0),'| torchvision',torchvision.__version__)"
python -c "from diffusers import StableDiffusion3Pipeline; print('[job] SD3 pipeline import OK')"

# --- 0) PREFLIGHT (no GPU): conditions / prompt counts / image total / ETA ---
echo "[job] ===== PREFLIGHT ====="
python e37_geneval.py --part preflight

# --- 1) SMOKE: 6 prompts spread across all 6 tags x 7 conds, gen+score+summary ---
echo "[job] ===== SMOKE ====="
CN_RESULTS=results/e37_geneval_smoke python e37_geneval.py \
    --part gen,score,summary --num_prompts 6 --spread --steps 20

# --- 2) GATE: smoke produced a report with a scored baseline? ---
echo "[job] ===== GATE ====="
python - <<'PY'
import json, os, sys
rp = "results/e37_geneval_smoke/e37_geneval/report.json"
if not os.path.exists(rp):
    print("[gate] FAIL: no smoke report"); sys.exit(1)
r = json.load(open(rp)); b = r["conditions"].get("baseline")
print("[gate] baseline:", b)
sys.exit(0 if (b and b.get("n", 0) > 0) else 1)
PY
echo "[gate] PASS"

# --- 3) FULL GEN (all 553 x 7 conditions, n=1, 512px, w=4.5) ---
echo "[job] ===== FULL GEN ====="
python e37_geneval.py --part gen --guidance 4.5 --steps 28 --size 512

# --- 4) SCORE + SUMMARY (set +e so a scoring hiccup never discards the cached gen) ---
echo "[job] ===== SCORE + SUMMARY ====="
set +e
python e37_geneval.py --part score,summary --guidance 4.5 --steps 28 --size 512
echo "[job] done -- results/e37_geneval/report.json (+ scores/*.jsonl, per-condition images)"
