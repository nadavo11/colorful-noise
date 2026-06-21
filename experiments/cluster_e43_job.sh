#!/bin/bash
# Run:AI entrypoint for E43 (FlowAlign on FLUX + two spectral twists).
# Files staged on the PVC via rsync (the /storage checkout is not git; image has no git).
# SELF-GATING: cheap smoke (1 scene, 8 steps) -> identity gate (recon must reproduce the
# source: DINO struct-dist small) -> full small sweep over w -> GOAL gate. Resumable
# (edits + reports cached on disk; delete results/e43* to force a clean rerun).
#
# Submit:
#   runai training submit e43-flowalign -g 1 \
#     -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
#     --existing-pvc claimname=storage,path=/storage --large-shm --command -- \
#     bash /storage/malnick/colorful-noise/experiments/cluster_e43_job.sh
set -e
export HF_HOME=/storage/malnick/huggingface_cache
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
cd /storage/malnick/colorful-noise/experiments

echo "[job] installing core deps ..."
pip install --quiet --no-input \
    diffusers==0.38.0 transformers==4.57.6 accelerate bitsandbytes \
    sentencepiece protobuf matplotlib
echo "[job] installing scoring deps (DINO/CLIP via transformers; LPIPS+SSIM here) ..."
pip install --quiet --no-input lpips scikit-image || echo "[job] WARN: lpips/skimage degrade"
python -c "import torch; print('[job] torch',torch.__version__,'cuda',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; print('[job] gpu',torch.cuda.get_device_name(0))"
python -c "from diffusers import FluxPipeline; print('[job] FluxPipeline import OK')"
python -c "import invert_core as IC; assert callable(IC.flowalign) and callable(IC.vel_sbn) and callable(IC.gauss_lowpass); print('[job] invert_core.flowalign OK')"

# --- 1) SMOKE: 1 scene, 8 steps ---
echo "[job] ===== SMOKE: 1 scene, 8 steps ====="
python e43_flowalign.py --part gen,analyze --num 1 --steps 8 --out_tag smoke

# --- 2) IDENTITY GATE: recon (C_tar==C_src) must reproduce the source ---
echo "[job] ===== GATE: identity reconstruction ====="
python - <<'PY'
import json, os, sys
from common import RESULTS
rp = os.path.join(RESULTS, "e43_smoke", "report.json")
if not os.path.exists(rp):
    print("[gate] FAIL: no smoke report"); sys.exit(1)
rep = json.load(open(rp)); ok = False
for k, e in rep.get("sources", {}).items():
    sd = e["conds"].get("recon", {}).get("struct_dist")
    print(f"[gate] {k}: recon struct-dist = {sd}")
    if sd is not None and sd < 0.05:
        ok = True
if not ok:
    print("[gate] FAIL: recon did not reproduce source (FLUX/packing path broken)")
    sys.exit(1)
print("[gate] PASS: FlowAlign reconstructs the source")
PY

# --- 3) FULL small sweep over w (cut/zeta at defaults) ---
echo "[job] ===== FULL: 3 scenes, 28 steps, w sweep ====="
python e43_flowalign.py --part gen,analyze --num 3 --steps 28 --w 5  --out_tag w5
python e43_flowalign.py --part gen,analyze --num 3 --steps 28 --w 7  --out_tag w7
python e43_flowalign.py --part gen,analyze --num 3 --steps 28 --w 10 --out_tag w10

# --- 4) GOAL gate: does any variant beat the FlowAlign baseline on struct-dist
#        (lower) while not dropping clip-directional below baseline? ---
echo "[job] ===== GOAL: variant vs FlowAlign baseline ====="
python - <<'PY'
import json, os
from common import RESULTS
VARIANTS = ["sbn_bp", "sbn_phase", "term_anneal", "sbn_bp+term"]
wins = []  # (tag, cond, n_scenes, mean_struct_delta, mean_clip_delta)
for tag in ("w5", "w7", "w10"):
    rp = os.path.join(RESULTS, f"e43_{tag}", "report.json")
    if not os.path.exists(rp):
        print(f"[goal] missing {rp}"); continue
    rep = json.load(open(rp))
    scenes = rep.get("sources", {})
    print(f"\n[goal] === {tag} ({len(scenes)} scenes) ===")
    for cond in VARIANTS:
        sd_deltas, clip_deltas, n = [], [], 0
        held = True
        for k, e in scenes.items():
            c = e["conds"]; base = c.get("flowalign", {}); v = c.get(cond, {})
            bs, vs = base.get("struct_dist"), v.get("struct_dist")
            bc, vc = base.get("clip_dir"), v.get("clip_dir")
            if None in (bs, vs, bc, vc):
                continue
            n += 1
            sd_deltas.append(vs - bs)            # want < 0 (struct closer)
            clip_deltas.append(vc - bc)          # want >= 0 (editability kept)
            if not (vs < bs and vc >= bc):
                held = False
        if n == 0:
            continue
        msd = sum(sd_deltas)/n; mcd = sum(clip_deltas)/n
        flag = "  <-- BEATS baseline on all scenes" if held else ""
        print(f"[goal] {cond:13s}: meanΔstruct={msd:+.4f}  meanΔclip={mcd:+.4f}"
              f"  ({n} scenes){flag}")
        if held:
            wins.append((tag, cond, n, msd, mcd))
print()
if wins:
    print(f"[goal] PASS: {len(wins)} (config,variant) settings beat FlowAlign-baseline on "
          "struct-dist while holding clip-directional, every scene:")
    for tag, cond, n, msd, mcd in wins:
        print(f"[goal]   {tag} / {cond}: meanΔstruct={msd:+.4f} meanΔclip={mcd:+.4f}")
else:
    print("[goal] NO all-scene winner; inspect per-scene tables in results/e43_w*/index.html")
PY
echo "[job] done -- results in experiments/results/e43_w5|w7|w10/ (index.html)"
