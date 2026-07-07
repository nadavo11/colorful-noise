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

OUT="/storage/nada/colorful-noise-e56/results/horizon_cache_n50"
echo "== full FLUX N=50 x 2 seeds scale-up (512px/28/bf16+bnb4) =="
echo "   method set: full + uniform + teacache + seacache + regrid_1.25 + adaptive_1.25/1.5/2.0"
echo "   tau grid: 0.2 0.3 0.4 0.5 0.65 (safe + overshoot bands)"
cd experiments
python -m horizon_cache.capability || true
START=$(date -Is)
python -m horizon_cache.run --mode generation   --n 50 --seeds-per-prompt 2 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16   --tau-grid 0.2 0.3 0.4 0.5 0.65   --variants regrid_1.25 adaptive_1.25 adaptive_1.5 adaptive_2.0   --teacache-taus 0.5 0.8 1.2   --jump-mode regrid   --save-all-images   --out "$OUT"
RC=$?
END=$(date -Is)
echo "== complete rc=$RC =="
echo "start=$START"; echo "end=$END"
echo "== newest run dir =="
LATEST=$(ls -dt "$OUT"/gen_* 2>/dev/null | head -1)
echo "LATEST=$LATEST"
echo "== outputs =="
find "$LATEST" -maxdepth 1 -type f 2>/dev/null | sort
echo "n_traces=$(ls "$LATEST"/traces 2>/dev/null | wc -l)  n_samples=$(ls "$LATEST"/samples 2>/dev/null | wc -l)"
head -1 "$LATEST/metrics.csv" 2>/dev/null; wc -l "$LATEST/metrics.csv" 2>/dev/null
# lightweight tarball of everything except the bulky per-image PNGs, for fast fetch
echo "== packing fetch tarball (metrics + traces + summary, no PNGs) =="
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e56/e56_n50_light.tar.gz   metrics.csv metrics.json summary.json config.json traces 2>/dev/null || true
ls -la /storage/nada/colorful-noise-e56/e56_n50_light.tar.gz 2>/dev/null || true
echo "== DONE_MARKER rc=$RC latest=$LATEST =="
