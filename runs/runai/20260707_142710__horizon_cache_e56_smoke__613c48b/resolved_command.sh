set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/nada/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME/hub"
ln -sfn /storage/malnick/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev   "$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev" || true
# capability.py hardcodes ~/.cache/huggingface/hub -> point the default path at HF_HOME/hub
mkdir -p "$HOME/.cache/huggingface"
ln -sfn "$HF_HOME/hub" "$HOME/.cache/huggingface/hub" || true

echo "== env =="
date -Is
python --version
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git ca-certificates >/dev/null)

echo "== repo =="
REPO_DIR="/storage/nada/colorful-noise-e56"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone -q https://github.com/nadavo11/colorful-noise.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch -q origin || true
git checkout -q e56-horizon-cache || git checkout -q -B e56-horizon-cache origin/e56-horizon-cache || true
git reset -q --hard origin/e56-horizon-cache || true
echo "HEAD=$(git rev-parse --short HEAD)"

echo "== install =="
python -m pip install --quiet --upgrade   'diffusers[torch]==0.38.0'   'transformers==4.57.6'   accelerate protobuf tokenizers sentencepiece safetensors   huggingface-hub hf-transfer bitsandbytes   pillow numpy matplotlib scipy scikit-image scikit-learn lpips pyarrow joblib

echo "== capability =="
cd experiments
python -m horizon_cache.capability || true

echo "== smoke (n<=3, tau[0.3,0.4], regrid_1.25+adaptive_1.25 + baselines) =="
START=$(date -Is)
python -m horizon_cache.run --mode generation --smoke   --out /storage/nada/colorful-noise-e56/results/horizon_cache_smoke
RC=$?
END=$(date -Is)
echo "== complete rc=$RC =="
echo "start=$START"
echo "end=$END"
echo "== outputs =="
find /storage/nada/colorful-noise-e56/results/horizon_cache_smoke -maxdepth 3 -type f 2>/dev/null | sort | tail -30
