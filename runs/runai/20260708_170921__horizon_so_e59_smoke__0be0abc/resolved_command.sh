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
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available(): print("gpu", torch.cuda.get_device_name(0))
PY
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
python -m horizon_cache.capability || true

echo "== STAGE 0: E59 math preflight (CPU, fail-fast) =="
python horizon_cache/test_e59_math.py

echo "== STAGE 1: micro pipeline check (N=2, extreme tau, new code paths) =="
OUTM="/storage/nada/colorful-noise-e59/results/horizon_so_micro"
python -m horizon_cache.run --mode generation   --n 2 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean   --tau-grid 0.5 1.2   --variants rm2raw0.5b0.1_adaptive_1.25 rmqraw0.5_adaptive_1.25   --rm-lambda-mode sigma --rm-lambda-max 1.5   --jump-mode regrid   --out "$OUTM"
LATESTM=$(ls -dt "$OUTM"/gen_* 2>/dev/null | head -1)
echo "MICRO_DIR=$LATESTM"
python - "$LATESTM" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
rows = json.loads((d / "metrics.json").read_text())
assert len(rows) >= 10, f"micro produced only {len(rows)} rows"
so = [r for r in rows if "rm2" in r["method"]]
assert so and all(r["num_second_order_applications"] > 0 for r in so), "no SO applications recorded"
q = [r for r in rows if "rmq" in r["method"]]
assert q and all(r["num_second_order_applications"] > 0 for r in q), "no quad applications recorded"
# traces must carry the anchor-triple diagnostics
t = next((d / "traces").glob("*rm2*.json"))
tr = json.loads(t.read_text())["traces"]
assert any("so_rho2" in x for x in tr), "so_rho2 missing from traces"
print("MICRO OK:", len(rows), "rows;",
      "SO apps:", [r["num_second_order_applications"] for r in so])
PY
echo "== micro green =="

echo "== STAGE 2: E59 SMOKE — N=8, extreme-speed sweep (tau 0.3..1.2), FO+SO variants =="
OUT="/storage/nada/colorful-noise-e59/results/horizon_so_smoke"
START=$(date -Is)
python -m horizon_cache.run --mode generation   --n 8 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean --save-all-images   --tau-grid 0.3 0.5 0.575 0.65 0.8 1.0 1.2   --variants adaptive_1.25 adaptive_1.5 rmraw0.5_adaptive_1.25 rmraw0.5_adaptive_1.5 rm2raw0.5b0.05_adaptive_1.25 rm2raw0.5b0.1_adaptive_1.25 rm2raw0.5b0.25_adaptive_1.25 rm2raw0.5b0.1g0.25r0.5_adaptive_1.25 rmqraw0.5_adaptive_1.25   --rm-lambda-mode sigma --rm-lambda-max 1.5   --jump-mode regrid   --out "$OUT"
RC=$?
END=$(date -Is)
echo "== smoke complete rc=$RC =="; echo "start=$START"; echo "end=$END"
LATEST=$(ls -dt "$OUT"/gen_* 2>/dev/null | head -1)
echo "LATEST_SMOKE=$LATEST"
find "$LATEST" -maxdepth 1 -type f 2>/dev/null | sort
echo "n_traces=$(ls "$LATEST"/traces 2>/dev/null | wc -l)"
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e59/e59_smoke_light.tar.gz metrics.csv metrics.json summary.json config.json traces 2>/dev/null || true
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e59/e59_smoke_samples.tar.gz samples 2>/dev/null || true
ls -la /storage/nada/colorful-noise-e59/e59_smoke_light.tar.gz /storage/nada/colorful-noise-e59/e59_smoke_samples.tar.gz 2>/dev/null || true
echo "== DONE_MARKER rc=$RC smoke=$LATEST =="
