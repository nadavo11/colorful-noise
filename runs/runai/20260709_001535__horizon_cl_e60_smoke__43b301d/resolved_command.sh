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
python -m horizon_cache.capability || true

echo "== STAGE 0: E60 math preflight (CPU, fail-fast; E58/E59 regressions included) =="
python horizon_cache/test_e60_math.py
python horizon_cache/test_e59_math.py

echo "== STAGE 1: micro pipeline check (N=2, new rmcl/mid code paths) =="
OUTM="/storage/nada/colorful-noise-e60/results/horizon_cl_micro"
python -m horizon_cache.run --mode generation   --n 2 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean   --tau-grid 0.5 1.4   --variants rmcl0.5_adaptive_1.25 rmcl0.5mid_adaptive_1.25 rmraw0.5mid_adaptive_1.25   --rm-lambda-mode sigma --rm-lambda-max 1.5   --jump-mode regrid   --out "$OUTM"
LATESTM=$(ls -dt "$OUTM"/gen_* 2>/dev/null | head -1)
echo "MICRO_DIR=$LATESTM"
python - "$LATESTM" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
rows = json.loads((d / "metrics.json").read_text())
assert len(rows) >= 10, f"micro produced only {len(rows)} rows"
cl = [r for r in rows if "rmcl" in r["method"]]
assert cl, "no rmcl rows"
assert all("cl_beta_hat_final" in r and r.get("cl_n_obs", 0) > 0 for r in cl), \
    f"cl aggregates missing: {[{k: r.get(k) for k in ('method','cl_n_obs')} for r in cl]}"
print("beta_hat_final by run:", [(r["method"], r["achieved_speedup"] if "achieved_speedup" in r else None,
      round(r["cl_beta_hat_final"], 3)) for r in cl])
# traces: beta must actually adapt within a trajectory, innovation must be logged
t = next((d / "traces").glob("*rmcl0.5_*.json"))
tr = json.loads(t.read_text())["traces"]
betas = [x["rm_beta"] for x in tr if x.get("rm_beta_mode") == "cl"]
assert len(set(round(b, 5) for b in betas)) >= 2, f"beta_hat never adapted: {betas}"
assert any("cl_innov_rel" in x for x in tr), "cl_innov_rel missing from fresh traces"
# midpoint: lambda used must differ from the point value on at least one jump step
tm = next((d / "traces").glob("*rmcl0.5mid_*.json"))
trm = json.loads(tm.read_text())["traces"]
mids = [x for x in trm if x.get("rm_lambda_eval") == "mid" and x.get("rm_lambda_point") is not None]
assert mids and any(abs(x["rm_lambda"] - x["rm_lambda_point"]) > 1e-6 for x in mids), \
    "midpoint lambda never differed from point"
print("MICRO OK:", len(rows), "rows;", len(betas), "cl cached steps; midpoint engaged on",
      sum(1 for x in mids if abs(x["rm_lambda"] - x["rm_lambda_point"]) > 1e-6), "steps")
PY
echo "== micro green =="

echo "== STAGE 2: E60 SMOKE — N=8, tau 0.3..1.4, closed-loop + midpoint variants =="
OUT="/storage/nada/colorful-noise-e60/results/horizon_cl_smoke"
START=$(date -Is)
python -m horizon_cache.run --mode generation   --n 8 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean --save-all-images   --tau-grid 0.3 0.5 0.575 0.65 0.8 1.0 1.2 1.4   --variants adaptive_1.25 adaptive_1.5 rmraw0.5_adaptive_1.25 rmraw0.5_adaptive_1.5 rmcl0.5_adaptive_1.25 rmcl0.5_adaptive_1.5 rmcl0.5mid_adaptive_1.25 rmcl0.5mid_adaptive_1.5 rmraw0.5mid_adaptive_1.25 rmcl0.5w1.0_adaptive_1.25   --rm-lambda-mode sigma --rm-lambda-max 1.5   --jump-mode regrid   --out "$OUT"
RC=$?
END=$(date -Is)
echo "== smoke complete rc=$RC =="; echo "start=$START"; echo "end=$END"
LATEST=$(ls -dt "$OUT"/gen_* 2>/dev/null | head -1)
echo "LATEST_SMOKE=$LATEST"
find "$LATEST" -maxdepth 1 -type f 2>/dev/null | sort
echo "n_traces=$(ls "$LATEST"/traces 2>/dev/null | wc -l)"
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e60/e60_smoke_light.tar.gz metrics.csv metrics.json summary.json config.json traces 2>/dev/null || true
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e60/e60_smoke_samples.tar.gz samples 2>/dev/null || true
ls -la /storage/nada/colorful-noise-e60/e60_smoke_light.tar.gz /storage/nada/colorful-noise-e60/e60_smoke_samples.tar.gz 2>/dev/null || true
echo "== DONE_MARKER rc=$RC smoke=$LATEST =="
