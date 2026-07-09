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
REPO_DIR="/storage/nada/colorful-noise-e61"
if [ ! -d "$REPO_DIR/.git" ]; then git clone -q https://github.com/nadavo11/colorful-noise.git "$REPO_DIR"; fi
cd "$REPO_DIR"
git fetch -q origin || true
git checkout -q e61-innovation-gate || git checkout -q -B e61-innovation-gate origin/e61-innovation-gate || true
git reset -q --hard origin/e61-innovation-gate || true
echo "HEAD=$(git rev-parse --short HEAD)"

echo "== install =="
python -m pip install --quiet --upgrade   'diffusers[torch]==0.38.0'   'transformers==4.57.6'   accelerate protobuf tokenizers sentencepiece safetensors   huggingface-hub hf-transfer bitsandbytes   pillow numpy matplotlib scipy scikit-image scikit-learn lpips pyarrow joblib

cd experiments
echo "== STAGE 0: math preflight =="
python horizon_cache/test_e61_math.py

echo "== STAGE 1: E61 SMOKE2 — kappa recalibrated to the measured I_a scale (crossover ~0.75) =="
OUT="/storage/nada/colorful-noise-e61/results/horizon_gate_smoke2"
START=$(date -Is)
python -m horizon_cache.run --mode generation   --n 8 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean --save-all-images   --tau-grid 0.3 0.5 0.575 0.65 0.8 1.0 1.2 1.4   --variants adaptive_1.25 rmraw0.5_adaptive_1.25 rmgh0.65_adaptive_1.25 rmgh0.75_adaptive_1.25 rmgh0.85_adaptive_1.25 rmgs0.7_adaptive_1.25 rmgs0.9_adaptive_1.25 rmgf0.8g0.5_adaptive_1.25 rmgf0.8g0.5a0.5_adaptive_1.25   --rm-lambda-mode sigma --rm-lambda-max 1.5   --jump-mode regrid   --out "$OUT"
RC=$?
END=$(date -Is)
echo "== smoke2 complete rc=$RC =="; echo "start=$START"; echo "end=$END"
LATEST=$(ls -dt "$OUT"/gen_* 2>/dev/null | head -1)
echo "LATEST_SMOKE2=$LATEST"
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e61/e61_smoke2_light.tar.gz metrics.csv metrics.json summary.json config.json traces 2>/dev/null || true
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e61/e61_smoke2_samples.tar.gz samples 2>/dev/null || true
ls -la /storage/nada/colorful-noise-e61/e61_smoke2_light.tar.gz 2>/dev/null || true
echo "== DONE_MARKER rc=$RC smoke2=$LATEST =="
