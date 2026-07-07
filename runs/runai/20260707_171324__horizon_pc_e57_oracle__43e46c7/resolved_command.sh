set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/nada/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME/hub"
ln -sfn /storage/malnick/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev   "$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev" || true
mkdir -p "$HOME/.cache/huggingface"; ln -sfn "$HF_HOME/hub" "$HOME/.cache/huggingface/hub" || true
date -Is; python --version
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git ca-certificates >/dev/null)
REPO_DIR="/storage/nada/colorful-noise-e57"
[ -d "$REPO_DIR/.git" ] || git clone -q https://github.com/nadavo11/colorful-noise.git "$REPO_DIR"
cd "$REPO_DIR"; git fetch -q origin || true
git checkout -q e57-horizon-pc || git checkout -q -B e57-horizon-pc origin/e57-horizon-pc || true
git reset -q --hard origin/e57-horizon-pc || true
echo "HEAD=$(git rev-parse --short HEAD)"
python -m pip install --quiet --upgrade 'diffusers[torch]==0.38.0' 'transformers==4.57.6' accelerate protobuf tokenizers sentencepiece safetensors huggingface-hub hf-transfer bitsandbytes pillow numpy matplotlib scipy scikit-image scikit-learn lpips pyarrow joblib
OUT="/storage/nada/colorful-noise-e57/results/horizon_pc_oracle"
cd experiments
echo "== ORACLE SMOKE: plain vs cached-PC vs FRESH-endpoint oracle, adaptive_2.0, tau 0.5/0.65 =="
python -m horizon_cache.run --mode generation --n 8 --seeds-per-prompt 1 --seed-base 0 \
  --steps 28 --width 512 --height 512 --guidance 3.5 --dtype bf16 --lean \
  --tau-grid 0.5 0.65 \
  --variants adaptive_2.0 pc0.5_adaptive_2.0 pcoracle0.5_adaptive_2.0 pcoracle1.0_adaptive_2.0 \
  --jump-mode regrid --out "$OUT"
RC=$?
LATEST=$(ls -dt "$OUT"/gen_* 2>/dev/null | head -1)
echo "LATEST=$LATEST"
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e57/e57_oracle_light.tar.gz metrics.csv metrics.json summary.json config.json traces 2>/dev/null || true
echo "== DONE_MARKER rc=$RC latest=$LATEST =="
