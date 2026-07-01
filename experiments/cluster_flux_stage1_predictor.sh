#!/usr/bin/env bash
set -euo pipefail

IMG=${IMG:-pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime}
PROJECT=${PROJECT:-avidan}
SHA=${SHA:-$(git rev-parse --short HEAD)}
MODE=${MODE:-smoke}
PRECISION_MODE=${PRECISION_MODE:-bf16}
GPU_MEMORY=${GPU_MEMORY:-48G}
SEACACHE_COMMIT=${SEACACHE_COMMIT:-8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2}
JOB=${JOB:-flux-stage1-${MODE}-${PRECISION_MODE}-${SHA}}
RUN_DIR=${RUN_DIR:-runs/runai/$(date +%Y%m%d_%H%M%S)__flux_stage1_predictor__${MODE}__${PRECISION_MODE}__${SHA}}
WORKTREE=/storage/nada/job_worktrees/${JOB}
mkdir -p "$RUN_DIR"

case "$MODE" in
  smoke)
    SAMPLE_ARGS="--sample-indices 0 --thresholds 0.3,0.6 --force"
    OUTPUT_ROOT="outputs/stage1_smoke_seacache"
    STEP_CSV="metrics/seacache_step_traces_smoke.csv"
    RUN_CSV="metrics/seacache_reuse_runs_smoke.csv"
    BIN_CSV="metrics/seacache_predictor_binary_metrics_smoke.csv"
    RUNLEN_CSV="metrics/seacache_predictor_runlength_metrics_smoke.csv"
    PRED_SUMMARY="reports/seacache_predictor_summary_smoke.json"
    PRED_ASSETS="reports/assets_seacache_predictor_smoke"
    HTML_OUT="reports/stage1_seacache_predictor_smoke.html"
    ;;
  full)
    SAMPLE_ARGS="--thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.8 --force"
    OUTPUT_ROOT="outputs/stage1_matched_seacache"
    STEP_CSV="metrics/seacache_step_traces.csv"
    RUN_CSV="metrics/seacache_reuse_runs.csv"
    BIN_CSV="metrics/seacache_predictor_binary_metrics.csv"
    RUNLEN_CSV="metrics/seacache_predictor_runlength_metrics.csv"
    PRED_SUMMARY="reports/seacache_predictor_summary.json"
    PRED_ASSETS="reports/assets_seacache_predictor"
    HTML_OUT="reports/stage1_seacache_predictor_dp_comparison.html"
    ;;
  *)
    echo "unknown MODE=$MODE" >&2
    exit 2
    ;;
esac

case "$PRECISION_MODE" in
  bf16)
    LOAD_ARGS="--dtype bf16"
    ;;
  q4offload)
    LOAD_ARGS="--dtype fp16 --offload --bnb4"
    ;;
  *)
    echo "unknown PRECISION_MODE=$PRECISION_MODE" >&2
    exit 2
    ;;
esac

cat >"$RUN_DIR/resolved_command.sh" <<SCRIPT
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/nada/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_CACHE_DIR=/storage/nada/pip_cache
mkdir -p "\$HF_HOME/hub" "\$PIP_CACHE_DIR" /storage/nada/job_worktrees
ln -sfn /storage/malnick/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev \
  "\$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev" || true

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
rm -rf "$WORKTREE"
git clone -q https://github.com/nadavo11/colorful-noise.git "$WORKTREE"
cd "$WORKTREE"
git checkout -q flux-seacache-dp-shortcuts
git rev-parse --short HEAD

echo "== seacache clone =="
mkdir -p third_party
git clone -q https://github.com/jiwoogit/SeaCache third_party/SeaCache
git -C third_party/SeaCache checkout -q $SEACACHE_COMMIT

echo "== install =="
python -m pip install --quiet --upgrade \
  'diffusers[torch]==0.38.0' \
  'transformers==4.57.6' \
  accelerate protobuf tokenizers sentencepiece safetensors \
  huggingface-hub hf-transfer bitsandbytes \
  pillow numpy matplotlib scipy scikit-image scikit-learn \
  lpips open_clip_torch timm

echo "== audit =="
python experiments/flux_seacache_dp_shortcuts.py audit \
  --seacache-dir third_party/SeaCache \
  --audit-md reports/seacache_predictor_audit.md \
  --audit-json reports/seacache_predictor_audit.json

echo "== stage1 =="
python experiments/flux_seacache_dp_shortcuts.py stage1 \
  --device cuda \
  $LOAD_ARGS \
  --height 1024 \
  --width 1024 \
  --model-id black-forest-labs/FLUX.1-dev \
  --trajectory-root /storage/nada/colorful-noise/outputs/flux100_trajectory \
  --output-root $OUTPUT_ROOT \
  --step-traces-csv $STEP_CSV \
  --reuse-runs-csv $RUN_CSV \
  $SAMPLE_ARGS

echo "== analyze =="
python experiments/flux_seacache_dp_shortcuts.py stage1-analyze \
  --step-traces-csv $STEP_CSV \
  --reuse-runs-csv $RUN_CSV \
  --predictor-assets-dir $PRED_ASSETS \
  --predictor-binary-metrics-csv $BIN_CSV \
  --predictor-runlength-metrics-csv $RUNLEN_CSV \
  --predictor-summary-json $PRED_SUMMARY \
  --audit-json reports/seacache_predictor_audit.json

echo "== report =="
python experiments/flux_seacache_dp_shortcuts.py stage1-report \
  --audit-json reports/seacache_predictor_audit.json \
  --predictor-summary-json $PRED_SUMMARY \
  --stage1-html $HTML_OUT

echo "== artifacts =="
find $OUTPUT_ROOT reports metrics -maxdepth 4 -type f | sort | sed -n '1,240p'
SCRIPT

chmod +x "$RUN_DIR/resolved_command.sh"
ENC=$(base64 -w0 "$RUN_DIR/resolved_command.sh")
cat >"$RUN_DIR/submit_command.sh" <<SCRIPT
runai submit "$JOB" -p "$PROJECT" --gpu-memory "$GPU_MEMORY" --large-shm -i "$IMG" \
  --existing-pvc claimname=storage,path=/storage \
  --command -- bash -lc "echo $ENC | base64 -d | bash"
SCRIPT
cat >"$RUN_DIR/run_manifest.yaml" <<YAML
job: "$JOB"
image: "$IMG"
project: "$PROJECT"
git_sha: "$SHA"
mode: "$MODE"
precision_mode: "$PRECISION_MODE"
gpu_memory: "$GPU_MEMORY"
run_dir: "$RUN_DIR"
worktree: "$WORKTREE"
outputs:
  - $OUTPUT_ROOT
  - $STEP_CSV
  - $RUN_CSV
  - $BIN_CSV
  - $RUNLEN_CSV
  - $PRED_SUMMARY
  - $PRED_ASSETS
  - $HTML_OUT
YAML

echo "Submitting $JOB using $RUN_DIR"
runai submit "$JOB" -p "$PROJECT" --gpu-memory "$GPU_MEMORY" --large-shm -i "$IMG" \
  --existing-pvc claimname=storage,path=/storage \
  --command -- bash -lc "echo $ENC | base64 -d | bash"
echo "Logs: runai logs $JOB"
