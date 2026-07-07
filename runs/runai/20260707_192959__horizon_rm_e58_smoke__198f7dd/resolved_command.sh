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
REPO_DIR="/storage/nada/colorful-noise-e58"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone -q https://github.com/nadavo11/colorful-noise.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch -q origin || true
git checkout -q e58-residual-motion || git checkout -q -B e58-residual-motion origin/e58-residual-motion || true
git reset -q --hard origin/e58-residual-motion || true
echo "HEAD=$(git rev-parse --short HEAD)"

echo "== install =="
python -m pip install --quiet --upgrade   'diffusers[torch]==0.38.0'   'transformers==4.57.6'   accelerate protobuf tokenizers sentencepiece safetensors   huggingface-hub hf-transfer bitsandbytes   pillow numpy matplotlib scipy scikit-image scikit-learn lpips pyarrow joblib

cd experiments
python -m horizon_cache.capability || true

# ---------------- 1) RESIDUAL-MOTION SMOKE: N=8 x1 seed, tau 0.4/0.5/0.65 ----------------
OUT="/storage/nada/colorful-noise-e58/results/horizon_rm_smoke"
echo "== RM SMOKE: N=8 x1 seed, tau 0.4/0.5/0.65 (safe->overshoot) =="
echo "   plain adaptive_1.25/1.5/2.0 + raw/lowpass ResidualMotion beta 0.25/0.5 (SeaCache auto)"
START=$(date -Is)
python -m horizon_cache.run --mode generation   --n 8 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean   --tau-grid 0.4 0.5 0.65   --variants adaptive_1.25 adaptive_1.5 adaptive_2.0 rmraw0.25_adaptive_1.5 rmraw0.5_adaptive_1.5 rmlowpass0.25_adaptive_1.5 rmlowpass0.5_adaptive_1.5 rmlowpass0.25_adaptive_2.0 rmlowpass0.5_adaptive_2.0   --rm-lambda-mode sigma --rm-lambda-max 1.5   --jump-mode regrid   --out "$OUT"
RC=$?
END=$(date -Is)
echo "== smoke complete rc=$RC =="
echo "start=$START"; echo "end=$END"
LATEST=$(ls -dt "$OUT"/gen_* 2>/dev/null | head -1)
echo "LATEST_SMOKE=$LATEST"
find "$LATEST" -maxdepth 1 -type f 2>/dev/null | sort
echo "n_traces=$(ls "$LATEST"/traces 2>/dev/null | wc -l)"
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e58/e58_smoke_light.tar.gz metrics.csv metrics.json summary.json config.json traces 2>/dev/null || true
ls -la /storage/nada/colorful-noise-e58/e58_smoke_light.tar.gz 2>/dev/null || true

# ---------------- 2) ORACLE RESIDUAL DIAGNOSTIC: N=4, true-residual scoring ----------------
# The key mechanism probe: does r_pred predict the true block-residual motion better than
# the frozen r_anchor? Extra full forward per cached step -> N=4, tau 0.5 only, 2 RM variants.
OUT2="/storage/nada/colorful-noise-e58/results/horizon_rm_oracle"
echo "== RM ORACLE DIAG: N=4 x1 seed, tau 0.5, rm-oracle on =="
STARTO=$(date -Is)
python -m horizon_cache.run --mode generation   --n 4 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean   --tau-grid 0.5   --variants rmraw0.5_adaptive_1.5 rmlowpass0.5_adaptive_1.5   --rm-lambda-mode sigma --rm-lambda-max 1.5 --rm-oracle   --jump-mode regrid   --out "$OUT2"
RCO=$?
ENDO=$(date -Is)
echo "== oracle complete rc=$RCO =="
echo "starto=$STARTO"; echo "endo=$ENDO"
LATESTO=$(ls -dt "$OUT2"/gen_* 2>/dev/null | head -1)
echo "LATEST_ORACLE=$LATESTO"
tar -C "$LATESTO" -czf /storage/nada/colorful-noise-e58/e58_oracle_light.tar.gz metrics.csv metrics.json summary.json config.json traces 2>/dev/null || true
ls -la /storage/nada/colorful-noise-e58/e58_oracle_light.tar.gz 2>/dev/null || true

echo "== DONE_MARKER rc=$RC rco=$RCO smoke=$LATEST oracle=$LATESTO =="
