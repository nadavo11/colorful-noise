#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPACE_ID="${1:-nadavo11/colorful-noise-demo}"
SPACE_FLAVOR="${SPACE_FLAVOR:-cpu-basic}"
MODEL_NAME="${MODEL_NAME:-flux-dev}"
HF_TOKEN_NAME="${HF_TOKEN_NAME:-}"
STAGE_DIR="${ROOT}/.hf-space-stage"

TOKEN_ARGS=()
if [[ -n "${HF_TOKEN_NAME}" ]]; then
  HF_TOKEN_VALUE="$(python - <<'PY' "${HF_TOKEN_NAME}"
import configparser, sys
from pathlib import Path
name = sys.argv[1]
cfg = configparser.ConfigParser()
cfg.read(Path.home()/'.cache/huggingface/stored_tokens')
print(cfg[name]['hf_token'])
PY
)"
  TOKEN_ARGS=(--token "${HF_TOKEN_VALUE}")
fi

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

cp "${ROOT}/deploy/hf_space/app.py" "${STAGE_DIR}/app.py"
cp "${ROOT}/deploy/hf_space/requirements.txt" "${STAGE_DIR}/requirements.txt"
cp "${ROOT}/deploy/hf_space/README.md" "${STAGE_DIR}/README.md"
cp -r "${ROOT}/experiments" "${STAGE_DIR}/experiments"
cp -r "${ROOT}/colorful_noise" "${STAGE_DIR}/colorful_noise"

find "${STAGE_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${STAGE_DIR}" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

hf repo create "${SPACE_ID}" \
  --type space \
  --space-sdk gradio \
  --public \
  --exist-ok \
  --flavor "${SPACE_FLAVOR}" \
  --secrets HF_TOKEN \
  --env "MODEL_NAME=${MODEL_NAME}" \
  "${TOKEN_ARGS[@]}"

hf upload "${SPACE_ID}" "${STAGE_DIR}" . \
  --repo-type space \
  --commit-message "Deploy spectral demo" \
  "${TOKEN_ARGS[@]}"

echo "Published to https://huggingface.co/spaces/${SPACE_ID}"
