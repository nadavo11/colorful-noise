#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPACE_ID="${1:-nadavo11/colorful-noise-demo}"
SPACE_FLAVOR="${SPACE_FLAVOR:-zero-a10g}"
MODEL_NAME="${MODEL_NAME:-flux-dev}"
STAGE_DIR="${ROOT}/.hf-space-stage"

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
  --env "MODEL_NAME=${MODEL_NAME}"

hf upload "${SPACE_ID}" "${STAGE_DIR}" . \
  --repo-type space \
  --commit-message "Deploy spectral demo"

echo "Published to https://huggingface.co/spaces/${SPACE_ID}"
