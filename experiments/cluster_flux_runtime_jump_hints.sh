#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-smoke} # smoke | main
IMG=${IMG:-pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime}
PROJECT=${PROJECT:-avidan}
SHA=${SHA:-$(git rev-parse --short HEAD)}
JOB=${JOB:-flux-runtime-hints-${MODE}-${SHA}}
GPU_MEMORY=${GPU_MEMORY:-48G}
NODE_TYPE=${NODE_TYPE:-}
RUNAI_DIR=${RUNAI_DIR:-runs/runai/$(date +%Y%m%d_%H%M%S)__flux_runtime_jump_hints_${MODE}__${SHA}}
mkdir -p "$RUNAI_DIR"

if [[ "$MODE" == "smoke" ]]; then
  E54_ARGS="--smoke --num-samples 2"
else
  E54_ARGS="--num-samples ${NUM_SAMPLES:-4}"
fi
GIT_REF=${GIT_REF:-origin/e54-runtime-jump-hints}

cat >"$RUNAI_DIR/resolved_command.sh" <<SCRIPT
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/nada/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "\$HF_HOME/hub"
ln -sfn /storage/malnick/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev \
  "\$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev" || true

echo "== env =="
date -Is
python --version
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git ca-certificates >/dev/null)

echo "== repo =="
cd /storage/nada
if [ -d colorful-noise-e54/.git ]; then
  git -C colorful-noise-e54 fetch -q origin || true
else
  git clone -q https://github.com/nadavo11/colorful-noise.git colorful-noise-e54
fi
cd colorful-noise-e54
git checkout -q -B e54-runtime-jump-hints "${GIT_REF}" || git checkout -q "${SHA}" || git checkout -q -B flux-seacache-dp-shortcuts origin/flux-seacache-dp-shortcuts || true
git rev-parse --short HEAD || true
python -m py_compile experiments/flux_runtime_jump_hints.py

echo "== install =="
python -m pip install --quiet --upgrade \
  'diffusers[torch]==0.38.0' \
  'transformers==4.57.6' \
  accelerate protobuf tokenizers sentencepiece safetensors \
  huggingface-hub hf-transfer bitsandbytes \
  pillow numpy matplotlib scipy scikit-image lpips

echo "== run E54 ${MODE} =="
START=\$(date -Is)
python experiments/flux_runtime_jump_hints.py \
  --device cuda \
  --dtype bf16 \
  --height 1024 \
  --width 1024 \
  --run-root runs/h100 \
  --trajectory-root outputs/flux_runtime_jump_hints/trajectories \
  ${E54_ARGS}
END=\$(date -Is)
echo "== complete =="
echo "start=\$START"
echo "end=\$END"
find runs/h100 -maxdepth 2 -type f -name report.html -path '*flux_runtime_jump_hints*' | sort | tail -5
SCRIPT

chmod +x "$RUNAI_DIR/resolved_command.sh"
ENC=$(base64 -w0 "$RUNAI_DIR/resolved_command.sh")

cat >"$RUNAI_DIR/submit_command.sh" <<SCRIPT
runai submit "$JOB" -p "$PROJECT" --gpu-memory "$GPU_MEMORY" ${NODE_TYPE:+--node-type "$NODE_TYPE"} --large-shm -i "$IMG" --existing-pvc claimname=storage,path=/storage --command -- bash -lc 'echo $ENC | base64 -d | bash'
SCRIPT

cat >"$RUNAI_DIR/run_manifest.yaml" <<YAML
job: "$JOB"
mode: "$MODE"
image: "$IMG"
project: "$PROJECT"
gpu_memory: "$GPU_MEMORY"
node_type: "$NODE_TYPE"
git_sha: "$SHA"
git_ref: "$GIT_REF"
runai_dir: "$RUNAI_DIR"
script: experiments/flux_runtime_jump_hints.py
e54_args: "$E54_ARGS"
expected_outputs:
  - runs/h100/<timestamp>__flux_runtime_jump_hints*/report.html
  - runs/h100/<timestamp>__flux_runtime_jump_hints*/reports/summary.md
  - runs/h100/<timestamp>__flux_runtime_jump_hints*/reports/summary.json
  - runs/h100/<timestamp>__flux_runtime_jump_hints*/metrics/per_method_budget_metrics.csv
  - runs/h100/<timestamp>__flux_runtime_jump_hints*/metrics/per_sample_metrics.csv
  - runs/h100/<timestamp>__flux_runtime_jump_hints*/metrics/local_error_correlations.csv
  - runs/h100/<timestamp>__flux_runtime_jump_hints*/metrics/call_counter_audit.csv
  - runs/h100/<timestamp>__flux_runtime_jump_hints*/metrics/leakage_audit.csv
YAML

echo "Submitting $JOB using $RUNAI_DIR"
runai submit "$JOB" -p "$PROJECT" --gpu-memory "$GPU_MEMORY" ${NODE_TYPE:+--node-type "$NODE_TYPE"} --large-shm -i "$IMG" --existing-pvc claimname=storage,path=/storage --command -- bash -lc "echo $ENC | base64 -d | bash"
echo "Logs: runai logs $JOB"
