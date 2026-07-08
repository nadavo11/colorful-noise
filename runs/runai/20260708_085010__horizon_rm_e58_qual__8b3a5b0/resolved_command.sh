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

echo "== env =="; date -Is; python --version
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git ca-certificates >/dev/null)

echo "== repo =="
REPO_DIR="/storage/nada/colorful-noise-e58"
if [ ! -d "$REPO_DIR/.git" ]; then git clone -q https://github.com/nadavo11/colorful-noise.git "$REPO_DIR"; fi
cd "$REPO_DIR"
git fetch -q origin || true
git checkout -q e58-residual-motion || git checkout -q -B e58-residual-motion origin/e58-residual-motion || true
git reset -q --hard origin/e58-residual-motion || true
echo "HEAD=$(git rev-parse --short HEAD)"

echo "== install =="
python -m pip install --quiet --upgrade   'diffusers[torch]==0.38.0'   'transformers==4.57.6'   accelerate protobuf tokenizers sentencepiece safetensors   huggingface-hub hf-transfer bitsandbytes   pillow numpy matplotlib scipy scikit-image scikit-learn lpips pyarrow joblib

cd experiments
# Qualitative: SAVE ALL per-method PNGs for a few prompts at a safe band + the overshoot band.
# full + SeaCache + plain adaptive_1.5 + RM raw0.5 adaptive_1.5, tau 0.4 (~2.5x) / 0.5 (~2.74x) / 0.65 (~3.4x).
OUT="/storage/nada/colorful-noise-e58/results/horizon_rm_qual"
echo "== RM QUALITATIVE: N=6 x1 seed, tau 0.4/0.5/0.65, save all images =="
START=$(date -Is)
python -m horizon_cache.run --mode generation   --n 6 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean --save-all-images   --tau-grid 0.4 0.5 0.65   --variants adaptive_1.5 rmraw0.5_adaptive_1.5   --rm-lambda-mode sigma --rm-lambda-max 1.5   --jump-mode regrid   --out "$OUT"
RC=$?
END=$(date -Is)
echo "== qual complete rc=$RC =="; echo "start=$START"; echo "end=$END"
LATEST=$(ls -dt "$OUT"/gen_* 2>/dev/null | head -1)
echo "LATEST_QUAL=$LATEST"
echo "n_samples=$(ls "$LATEST"/samples 2>/dev/null | wc -l)"
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e58/e58_qual.tar.gz metrics.csv metrics.json summary.json config.json samples 2>/dev/null || true
ls -la /storage/nada/colorful-noise-e58/e58_qual.tar.gz 2>/dev/null || true
echo "== DONE_MARKER rc=$RC qual=$LATEST =="
