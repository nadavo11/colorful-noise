#!/bin/bash
# Run:AI entrypoint for E44 (reproduce FlowAlign on PIE-Bench + ours on top).
# Files staged on the PVC via rsync (the /storage checkout is not git; image has no git).
#
# Stages (1st arg):
#   --smoke         foundation: official env + pull SD3.0-medium + run their run_edit.py on 1 sample.
#   --mini          end-to-end harness check: gen+analyze on a tiny stratified subset, cfg 7.5.
#   --cfg <w>       full 700-image reproduction at CFG <w> (gen+analyze). Run per w in {5,7.5,10,13.5}.
#
# Submit (from a shell where `runai` is logged in); force a big card so SD3+T5 fit at 1024px:
#   runai training submit e44-smoke -g 1 \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --existing-pvc claimname=storage,path=/storage --large-shm \
#     --node-pool a6000 --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e44_job.sh --smoke
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
export CN_FLOWALIGN=/storage/malnick/flowalign_official
export CN_PNP=/storage/malnick/pnpinversion
EXP=/storage/malnick/colorful-noise/experiments
SD3=stabilityai/stable-diffusion-3-medium-diffusers
STAGE="${1:---smoke}"

echo "[job] torch / cuda check"
python -c "import torch; print('[job] torch',torch.__version__,'cuda',torch.cuda.is_available()); assert torch.cuda.is_available(); print('[job] gpu',torch.cuda.get_device_name(0))"

echo "[job] installing FlowAlign stack + metric deps ..."
pip install --quiet --no-input \
    diffusers==0.33.1 transformers==4.47.1 accelerate==1.2.1 \
    tokenizers==0.21.0 sentencepiece==0.2.0 protobuf==3.20.3 "numpy<2" \
    datasets torchmetrics lpips scikit-image ftfy
python -c "from diffusers import StableDiffusion3Pipeline; print('[job] SD3 import OK')"

echo "[job] ensure SD3.0-medium present in shared cache (gated; shared token) ..."
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$SD3", allow_patterns=["*.json","*.txt","*.model","*.safetensors"])
print("[job] SD3.0-medium at", p)
PY

cd "$EXP"
case "$STAGE" in
  --smoke)
    echo "[job] ===== SMOKE: official run_edit.py on samples/bicycle.jpg ====="
    cd "$CN_FLOWALIGN"
    python run_edit.py --method flowalign --img_path "samples/bicycle.jpg" \
      --src_prompt "a slanted mountain bicycle on the road in front of a building" \
      --tgt_prompt "a slanted rusty mountain bicycle on the road in front of a building" \
      --cfg_scale 13.5 --NFE 33 --seed 123 --workdir "$EXP/results/e44_smoke"
    ls -la "$EXP/results/e44_smoke/edited" && echo "[job] SMOKE OK" || echo "[job] SMOKE FAIL"
    ;;
  --mini)
    echo "[job] ===== MINI: gen+analyze on subset (2/type=20 imgs), cfg 7.5 ====="
    python e44_flowalign_repro.py --part gen,analyze --cfg 7.5 --n_per_type 2 --tag mini
    ;;
  --cfg)
    W="$2"; TAG="cfg$(echo "$W" | tr -d '.')"
    echo "[job] ===== FULL repro: all 700, cfg $W -> tag $TAG ====="
    python e44_flowalign_repro.py --part gen,analyze --cfg "$W" --n_per_type 0 --tag "$TAG"
    ;;
  *) echo "[job] unknown stage '$STAGE'"; exit 1 ;;
esac
echo "[job] stage $STAGE done"
