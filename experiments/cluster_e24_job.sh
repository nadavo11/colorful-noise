#!/bin/bash
# Run:AI entrypoint for E24 (token-axis spectral surgery on the text conditioning).
# Assumes the E24 files are already present on the storage PVC (staged via kubectl
# cp -- the cluster checkout is a plain directory, NOT a git repo, so there is no
# in-pod git pull). SELF-GATING: runs a cheap smoke (probe + analyze) first, checks
# that band-filtered embeddings still produce coherent images (CLIP sanity), and only
# THEN launches the full probe+merge+edit+analyze sweep. A failed gate exits non-zero
# WITHOUT burning GPU on the full run. Everything is resumable (e24 caches every
# generation), so a re-submit picks up where it left off.
#
# Submit with:
#   runai submit --name e24-text-spectral -g 1 \
#       -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#       --pvc=storage:/storage --large-shm --command -- \
#       bash /storage/malnick/colorful-noise/experiments/cluster_e24_job.sh
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

# --- deps (same pinned stack as the other cluster jobs) -------------------------
echo "[job] installing core deps ..."
pip install --quiet --no-input \
    diffusers==0.38.0 transformers==4.57.6 accelerate bitsandbytes \
    sentencepiece protobuf matplotlib
python -c "import torch; print('[job] torch',torch.__version__,'cuda',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE on this node/image'; print('[job] gpu',torch.cuda.get_device_name(0))"
python -c "from diffusers import FluxPipeline; print('[job] FluxPipeline import OK')"
echo "[job] installing optional scoring deps (graceful) ..."
( apt-get update -qq && apt-get install -y -qq git ) >/dev/null 2>&1 || echo "[job] WARN: git install failed"
pip install --quiet --no-input image-reward ftfy regex \
    "git+https://github.com/openai/CLIP.git" 2>&1 | tail -2 || \
    echo "[job] WARN: optional scoring deps failed; ImageReward degrades gracefully"

# --- 1) SMOKE: probe + analyze, cheap (2 prompts, 1 seed, 8 steps) --------------
echo "[job] ===== SMOKE: probe (2 prompts, 1 seed, 8 steps) ====="
python e24_text_spectral.py --part probe,analyze \
    --num_prompts 2 --seeds 1 --steps 8 --out_tag smoke

# --- 2) GATE: did band-filtered embeds yield a coherent image? ------------------
echo "[job] ===== GATE: checking smoke sanity ====="
python - <<'PY'
import json, os, sys
from common import RESULTS
rp = os.path.join(RESULTS, "e24_smoke", "report.json")
if not os.path.exists(rp):
    print("[gate] FAIL: no smoke report.json"); sys.exit(1)
rep = json.load(open(rp))
probe = rep.get("probe", {})
if not probe:
    print("[gate] FAIL: smoke produced no probe entries"); sys.exit(1)
THRESH = 0.18  # CLIP ViT-L cosine for an on-prompt 1024px image sits ~0.25-0.35
ok = False
for key, e in probe.items():
    full = e.get("conds", {}).get("full", {}).get("clip")
    m = full["mean"] if full else None
    print(f"[gate] {key}: full-variant CLIP = {m}")
    for cond, sc in e["conds"].items():
        c = sc.get("clip")
        print(f"[gate]    {cond:9s} CLIP={c['mean']:.3f}" if c else f"[gate]    {cond:9s} CLIP=NA")
    if m is not None and m >= THRESH:
        ok = True
if not ok:
    print(f"[gate] FAIL: no probe prompt reached full-variant CLIP>={THRESH} "
          "-> generation/conditioning path looks broken; NOT launching full sweep")
    sys.exit(1)
print(f"[gate] PASS: full-variant CLIP>={THRESH}; proceeding to full sweep")
PY

# --- 3) FULL sweep (only reached if the gate passed) ----------------------------
echo "[job] ===== FULL: probe,merge,edit,analyze (defaults) ====="
python e24_text_spectral.py --part probe,merge,edit,analyze
echo "[job] done -- results in experiments/results/e24/ (index.html)"
