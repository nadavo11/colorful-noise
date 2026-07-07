set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/nada/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME/hub"
ln -sfn /storage/malnick/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev   "$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev" || true
mkdir -p "$HOME/.cache/huggingface"
ln -sfn "$HF_HOME/hub" "$HOME/.cache/huggingface/hub" || true

echo "== env =="
date -Is; python --version
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git ca-certificates >/dev/null)

echo "== repo =="
REPO_DIR="/storage/nada/colorful-noise-e57"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone -q https://github.com/nadavo11/colorful-noise.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch -q origin || true
git checkout -q e57-horizon-pc || git checkout -q -B e57-horizon-pc origin/e57-horizon-pc || true
git reset -q --hard origin/e57-horizon-pc || true
echo "HEAD=$(git rev-parse --short HEAD)"

echo "== install =="
python -m pip install --quiet --upgrade   'diffusers[torch]==0.38.0'   'transformers==4.57.6'   accelerate protobuf tokenizers sentencepiece safetensors   huggingface-hub hf-transfer bitsandbytes   pillow numpy matplotlib scipy scikit-image scikit-learn lpips pyarrow joblib

OUT="/storage/nada/colorful-noise-e57/results/horizon_pc_smoke"
echo "== PC SMOKE: N=8 x1 seed, tau 0.4/0.5/0.65 (safe->overshoot) =="
echo "   variants: adaptive_1.25/1.5/2.0 + pc0.5_adaptive_1.5 + pc0.5_adaptive_2.0 (SeaCache auto)"
cd experiments
python -m horizon_cache.capability || true
START=$(date -Is)
python -m horizon_cache.run --mode generation   --n 8 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean   --tau-grid 0.4 0.5 0.65   --variants adaptive_1.25 adaptive_1.5 adaptive_2.0 pc0.5_adaptive_1.5 pc0.5_adaptive_2.0   --jump-mode regrid   --out "$OUT"
RC=$?
END=$(date -Is)
echo "== complete rc=$RC =="
echo "start=$START"; echo "end=$END"
LATEST=$(ls -dt "$OUT"/gen_* 2>/dev/null | head -1)
echo "LATEST=$LATEST"
find "$LATEST" -maxdepth 1 -type f 2>/dev/null | sort
echo "n_traces=$(ls "$LATEST"/traces 2>/dev/null | wc -l)"
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e57/e57_smoke_light.tar.gz metrics.csv metrics.json summary.json config.json traces 2>/dev/null || true
ls -la /storage/nada/colorful-noise-e57/e57_smoke_light.tar.gz 2>/dev/null || true
echo "== DONE_MARKER rc=$RC latest=$LATEST =="
