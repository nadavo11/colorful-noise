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
REPO_DIR="/storage/nada/colorful-noise-e60"
if [ ! -d "$REPO_DIR/.git" ]; then git clone -q https://github.com/nadavo11/colorful-noise.git "$REPO_DIR"; fi
cd "$REPO_DIR"
git fetch -q origin || true
git checkout -q e60-closed-loop || git checkout -q -B e60-closed-loop origin/e60-closed-loop || true
git reset -q --hard origin/e60-closed-loop || true
echo "HEAD=$(git rev-parse --short HEAD)"

echo "== install =="
python -m pip install --quiet --upgrade   'diffusers[torch]==0.38.0'   'transformers==4.57.6'   accelerate protobuf tokenizers sentencepiece safetensors   huggingface-hub hf-transfer bitsandbytes   pillow numpy matplotlib scipy scikit-image scikit-learn lpips pyarrow joblib

cd experiments
echo "== STAGE 0: math preflight (fail-fast) =="
python horizon_cache/test_e60_math.py

echo "== STAGE 1: E60 SMOKE2 — gate/scale recalibration, N=8, tau 0.3..1.4 =="
OUT="/storage/nada/colorful-noise-e60/results/horizon_cl_smoke2"
START=$(date -Is)
python -m horizon_cache.run --mode generation   --n 8 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean --save-all-images   --tau-grid 0.3 0.5 0.575 0.65 0.8 1.0 1.2 1.4   --variants adaptive_1.25 rmraw0.5_adaptive_1.25 rmcl0.5m0.25g0.08_adaptive_1.25 rmcl0.5m0.25g0.12_adaptive_1.25 rmcl0.2m0.25k2.5_adaptive_1.25 rmcl0.2m0.25k2.5mid_adaptive_1.25   --rm-lambda-mode sigma --rm-lambda-max 1.5   --jump-mode regrid   --out "$OUT"
RC=$?
END=$(date -Is)
echo "== smoke2 complete rc=$RC =="; echo "start=$START"; echo "end=$END"
LATEST=$(ls -dt "$OUT"/gen_* 2>/dev/null | head -1)
echo "LATEST_SMOKE2=$LATEST"
echo "n_traces=$(ls "$LATEST"/traces 2>/dev/null | wc -l)"
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e60/e60_smoke2_light.tar.gz metrics.csv metrics.json summary.json config.json traces 2>/dev/null || true
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e60/e60_smoke2_samples.tar.gz samples 2>/dev/null || true
ls -la /storage/nada/colorful-noise-e60/e60_smoke2_light.tar.gz 2>/dev/null || true
echo "== DONE_MARKER rc=$RC smoke2=$LATEST =="
