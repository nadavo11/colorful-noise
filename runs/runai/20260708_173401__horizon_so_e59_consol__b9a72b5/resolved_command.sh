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
REPO_DIR="/storage/nada/colorful-noise-e59"
if [ ! -d "$REPO_DIR/.git" ]; then git clone -q https://github.com/nadavo11/colorful-noise.git "$REPO_DIR"; fi
cd "$REPO_DIR"
git fetch -q origin || true
git checkout -q e59-second-order || git checkout -q -B e59-second-order origin/e59-second-order || true
git reset -q --hard origin/e59-second-order || true
echo "HEAD=$(git rev-parse --short HEAD)"

echo "== install =="
python -m pip install --quiet --upgrade   'diffusers[torch]==0.38.0'   'transformers==4.57.6'   accelerate protobuf tokenizers sentencepiece safetensors   huggingface-hub hf-transfer bitsandbytes   pillow numpy matplotlib scipy scikit-image scikit-learn lpips pyarrow joblib

cd experiments
python horizon_cache/test_e59_math.py

# E59 CONSOLIDATION. N=100 x2 seeds = 200 paired. Priorities (smoke-gated):
#   1. Extreme-speed SeaCache comparison for FIRST-ORDER RM (E58 gains reproduced in smoke).
#   2. Second-order: smoke FAILED both proceed criteria (best SO-FO +0.23 noise, rho2 p50=1.4,
#      cos p50=0.08, quad KILL) -> keep ONE uniform variant (b0.1) to make the verdict
#      definitive at N=200 pairs; quad + gated dropped (gate never opens; quad collapses).
# tau grid fills all bands 1.8-2.2 .. 4.8-5.2 (+1.4 so the SeaCache frontier covers ~5.3x
# method points without extrapolation).
OUT="/storage/nada/colorful-noise-e59/results/horizon_so_consol"
echo "== E59 CONSOLIDATION: N=100 x2 seeds, tau 0.3..1.4 =="
START=$(date -Is)
python -m horizon_cache.run --mode generation   --n 100 --seeds-per-prompt 2 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean   --tau-grid 0.3 0.4 0.5 0.575 0.65 0.8 1.0 1.2 1.4   --variants adaptive_1.25 adaptive_1.5 rmraw0.5_adaptive_1.25 rmraw0.5_adaptive_1.5 rm2raw0.5b0.1_adaptive_1.25   --rm-lambda-mode sigma --rm-lambda-max 1.5   --jump-mode regrid   --out "$OUT"
RC=$?
END=$(date -Is)
echo "== consolidation complete rc=$RC ==" ; echo "start=$START"; echo "end=$END"
LATEST=$(ls -dt "$OUT"/gen_* 2>/dev/null | head -1)
echo "LATEST_CONSOL=$LATEST"
echo "n_traces=$(ls "$LATEST"/traces 2>/dev/null | wc -l)"
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e59/e59_consol_light.tar.gz metrics.csv metrics.json summary.json config.json traces 2>/dev/null || true
ls -la /storage/nada/colorful-noise-e59/e59_consol_light.tar.gz || true

echo "== E59 ORACLE stage: N=4, extreme tau, FO vs SO true-residual scoring =="
OUT2="/storage/nada/colorful-noise-e59/results/horizon_so_oracle"
python -m horizon_cache.run --mode generation   --n 4 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean   --tau-grid 0.65 1.0   --variants rmraw0.5_adaptive_1.25 rm2raw0.5b0.1_adaptive_1.25   --rm-lambda-mode sigma --rm-lambda-max 1.5 --rm-oracle   --jump-mode regrid   --out "$OUT2"
LATESTO=$(ls -dt "$OUT2"/gen_* 2>/dev/null | head -1)
echo "LATEST_ORACLE=$LATESTO"
tar -C "$LATESTO" -czf /storage/nada/colorful-noise-e59/e59_oracle_light.tar.gz metrics.csv metrics.json summary.json config.json traces 2>/dev/null || true
ls -la /storage/nada/colorful-noise-e59/e59_oracle_light.tar.gz || true
echo "== DONE_MARKER rc=$RC consol=$LATEST oracle=$LATESTO =="
