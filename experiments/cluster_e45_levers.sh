#!/bin/bash
# Run:AI entrypoint for E45 follow-ups (levers 1 & 2).
# LEVER 1 (CFG-match frontier): sweep w in {7.5,10,13.5,18} for the video edit AND the paper's
#   frame-by-frame baseline; the question is whether video editing stays low-flicker even as w
#   rises to match fbf's edit strength (CLIP). Per-w results in results/e45_w<W>/.
# LEVER 2 (real clip @512): edit a REAL clip (imageio's bundled cockatoo.mp4) at 512px instead of
#   an LTX-generated source, with the same baseline/phase/fbf comparison. results/e45_real512/.
#
# Submit (new CLI):
#   ~/.runai/bin/runai training standard submit e45-ltx-levers -p avidan -g 1 --large-shm \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --existing-pvc claimname=storage,path=/storage --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e45_levers.sh
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

echo "[job] installing deps ..."
pip install --quiet --no-input \
    diffusers==0.36.0 transformers==4.57.6 accelerate \
    sentencepiece protobuf imageio imageio-ffmpeg
python -c "import torch; assert torch.cuda.is_available(); print('[job] gpu',torch.cuda.get_device_name(0))"
python -c "import torchvision; from torchvision.models.optical_flow import raft_small; print('[job] RAFT OK')"

# --- LEVER 1: w frontier (video edit + fbf at each w) ---
for W in 7.5 10 13.5 18; do
  echo "[job] ===== LEVER1: w=$W ====="
  python e45_ltx_flowalign.py --part gen,analyze --steps 24 --frames 25 --size 256 \
      --w $W --zeta 0.01 --cuts 0.2 --fbf --out_tag w$W
done

echo "[job] ===== LEVER1 FRONTIER: editability vs flicker (video baseline vs paper fbf) ====="
python - <<'PY'
import json, os, glob
rows = []
for rp in sorted(glob.glob(os.path.join("results", "e45_w*", "gen_report.json"))):
    rep = json.load(open(rp)); m = rep.get("metrics", {})
    w = rep["params"]["w"]; b = m.get("baseline", {}); f = m.get("fbf", {})
    rows.append((w, b.get("clip_dir"), b.get("warp_masked"), f.get("clip_dir"), f.get("warp_masked")))
rows.sort()
print(f"{'w':>5} | {'video clip':>10} {'video warpM':>11} | {'fbf clip':>9} {'fbf warpM':>10}")
for w, vc, vw, fc, fw in rows:
    print(f"{w:>5} | {vc:>10.4f} {vw:>11.5f} | {fc:>9.4f} {fw:>10.5f}")
print("\n[frontier] If video warpM stays << fbf warpM even where video clip ~ fbf clip,")
print("[frontier] then video editing beats the paper at MATCHED edit strength.")
PY

# --- LEVER 2: real clip @512 (cockatoo -> colorful parrot) ---
echo "[job] ===== LEVER2: real clip @512 ====="
python e45_ltx_flowalign.py --part gen,analyze --steps 24 --frames 25 --size 512 \
    --w 10 --zeta 0.01 --cuts 0.2 --fbf --out_tag real512 \
    --real_video "imageio:cockatoo.mp4" \
    --src_caption "a white cockatoo bird perched on a branch, moving its head" \
    --edit_prompt "a colorful rainbow parrot perched on a branch, moving its head"
echo "[job] LEVERS done -- results in experiments/results/e45_w*/ and results/e45_real512/"
