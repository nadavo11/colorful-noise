#!/usr/bin/env bash
set -euo pipefail

IMG=${IMG:-pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime}
PROJECT=${PROJECT:-avidan}
SHA=${SHA:-$(git rev-parse --short HEAD)}
JOB=${JOB:-flux-seacache-dp-${SHA}}
RUN_DIR=${RUN_DIR:-runs/runai/$(date +%Y%m%d_%H%M%S)__flux_seacache_dp__${SHA}}
mkdir -p "$RUN_DIR"

cat >"$RUN_DIR/resolved_command.sh" <<'SCRIPT'
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
export HF_HOME=/storage/nada/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME/hub"
ln -sfn /storage/malnick/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev \
  "$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev" || true

echo "== env =="
python --version
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git ca-certificates >/dev/null)

echo "== repo =="
cd /storage/nada
if [ -d colorful-noise/.git ]; then
  git -C colorful-noise fetch -q origin
else
  git clone -q https://github.com/nadavo11/colorful-noise.git
fi
cd colorful-noise
git checkout -q -B flux-seacache-dp-shortcuts origin/flux-seacache-dp-shortcuts || git checkout -q "${SHA:-HEAD}"
git rev-parse --short HEAD

echo "== seacache clone =="
mkdir -p third_party
if [ -d third_party/SeaCache/.git ]; then
  git -C third_party/SeaCache fetch -q origin
else
  git clone -q https://github.com/jiwoogit/SeaCache third_party/SeaCache
fi
git -C third_party/SeaCache checkout -q 8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2

echo "== install =="
python -m pip install --quiet --upgrade \
  'diffusers[torch]==0.38.0' \
  'transformers==4.57.6' \
  accelerate protobuf tokenizers sentencepiece safetensors \
  huggingface-hub hf-transfer bitsandbytes \
  pillow numpy matplotlib scipy scikit-image lpips

echo "== run =="
python experiments/flux_seacache_dp_shortcuts.py run-all \
  --device cuda \
  --dtype bf16 \
  --bnb4 \
  --offload \
  --steps 100 \
  --seacache-steps 50 \
  --height 1024 \
  --width 1024 \
  --seacache-dir third_party/SeaCache \
  --output-root outputs/seacache_replication \
  --trajectory-root outputs/flux100_trajectory \
  --dp-root outputs/flux_dp_shortcuts \
  --metrics-csv metrics/all_metrics.csv \
  --html reports/flux_seacache_dp_shortcuts.html \
  --summary-md reports/summary.md \
  --summary-json reports/summary.json

echo "== artifacts =="
find outputs/seacache_replication outputs/flux100_trajectory outputs/flux_dp_shortcuts reports metrics -maxdepth 3 -type f | sort | sed -n '1,240p'
SCRIPT

chmod +x "$RUN_DIR/resolved_command.sh"
ENC=$(base64 -w0 "$RUN_DIR/resolved_command.sh")
cat >"$RUN_DIR/submit_command.sh" <<SCRIPT
JOB=$JOB IMG=$IMG SHA=$SHA
runai submit "$JOB" -p "$PROJECT" -g 1 --large-shm -i "$IMG" --existing-pvc claimname=storage,path=/storage --command -- bash -lc 'echo $ENC | base64 -d | bash'
SCRIPT
cat >"$RUN_DIR/run_manifest.yaml" <<YAML
job: "$JOB"
image: "$IMG"
project: "$PROJECT"
git_sha: "$SHA"
run_dir: "$RUN_DIR"
outputs:
  - outputs/seacache_replication
  - outputs/flux100_trajectory
  - outputs/flux_dp_shortcuts
  - reports/flux_seacache_dp_shortcuts.html
  - reports/summary.md
  - reports/summary.json
  - metrics/all_metrics.csv
YAML

echo "Submitting $JOB using $RUN_DIR"
runai submit "$JOB" -p "$PROJECT" -g 1 --large-shm -i "$IMG" --existing-pvc claimname=storage,path=/storage --command -- bash -lc "echo $ENC | base64 -d | bash"
echo "Logs: runai logs $JOB"
