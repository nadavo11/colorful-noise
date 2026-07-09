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
echo "== STAGE 0: E61 math preflight (CPU, fail-fast; E59/E60 regressions included) =="
python horizon_cache/test_e61_math.py
python horizon_cache/test_e60_math.py
python horizon_cache/test_e59_math.py

echo "== STAGE 1: micro pipeline check (N=2, new gate code paths) =="
OUTM="/storage/nada/colorful-noise-e61/results/horizon_gate_micro"
python -m horizon_cache.run --mode generation   --n 2 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean   --tau-grid 0.5 1.2   --variants rmgh0.08_adaptive_1.25 rmgs0.15_adaptive_1.25 rmgf0.20g0.5a0.5_adaptive_1.25 rmbg0.25_adaptive_1.25   --rm-lambda-mode sigma --rm-lambda-max 1.5   --jump-mode regrid   --out "$OUTM"
LATESTM=$(ls -dt "$OUTM"/gen_* 2>/dev/null | head -1)
echo "MICRO_DIR=$LATESTM"
python - "$LATESTM" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
rows = json.loads((d / "metrics.json").read_text())
assert len(rows) >= 8, f"micro produced only {len(rows)} rows"
gate_rows = [r for r in rows if "rmg" in r["method"] or "rmbg" in r["method"]]
def _ok(r):
    if "rmbg" in r["method"]:
        return r.get("rm_beta_used_mean") is not None
    return r.get("gate_n_obs", 0) > 0
assert gate_rows and all(_ok(r) for r in gate_rows), \
    f"gate aggregates missing: {[(r['method'], r.get('gate_n_obs'), r.get('rm_beta_used_mean')) for r in gate_rows]}"
t = next((d / "traces").glob("*rmgh0.08*.json"))
tr = json.loads(t.read_text())["traces"]
assert any("gate_g" in x for x in tr), "gate_g missing from fresh-anchor traces"
gs = [x["rm_beta"] for x in tr if x.get("rm_beta_mode") == "gate"]
assert len(gs) > 0, "no gated cached steps recorded"
print("MICRO OK:", len(rows), "rows;", len(gate_rows), "gate rows; example beta_i span",
      (min(gs), max(gs)) if gs else None)
PY
echo "== micro green =="

echo "== STAGE 2: E61 SMOKE — N=8, tau 0.3..1.4, fixed/cl/gate variants =="
OUT="/storage/nada/colorful-noise-e61/results/horizon_gate_smoke"
START=$(date -Is)
python -m horizon_cache.run --mode generation   --n 8 --seeds-per-prompt 1 --seed-base 0   --steps 28 --width 512 --height 512 --guidance 3.5   --dtype bf16 --lean --save-all-images   --tau-grid 0.3 0.5 0.575 0.65 0.8 1.0 1.2 1.4   --variants adaptive_1.25 rmraw0.5_adaptive_1.25 rmcl0.5_adaptive_1.25 rmgh0.08_adaptive_1.25 rmgs0.15_adaptive_1.25 rmgf0.20g0.5_adaptive_1.25 rmgf0.20g0.5a0.5_adaptive_1.25   --rm-lambda-mode sigma --rm-lambda-max 1.5   --jump-mode regrid   --out "$OUT"
RC=$?
END=$(date -Is)
echo "== smoke complete rc=$RC =="; echo "start=$START"; echo "end=$END"
LATEST=$(ls -dt "$OUT"/gen_* 2>/dev/null | head -1)
echo "LATEST_SMOKE=$LATEST"
echo "n_traces=$(ls "$LATEST"/traces 2>/dev/null | wc -l)"
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e61/e61_smoke_light.tar.gz metrics.csv metrics.json summary.json config.json traces 2>/dev/null || true
tar -C "$LATEST" -czf /storage/nada/colorful-noise-e61/e61_smoke_samples.tar.gz samples 2>/dev/null || true
ls -la /storage/nada/colorful-noise-e61/e61_smoke_light.tar.gz /storage/nada/colorful-noise-e61/e61_smoke_samples.tar.gz 2>/dev/null || true
echo "== DONE_MARKER rc=$RC smoke=$LATEST =="
