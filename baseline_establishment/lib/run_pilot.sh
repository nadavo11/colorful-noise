#!/usr/bin/env bash
# Phase B pilot: StyleID (anaconda env) then the 4 FLUX models (uv env), sequenced.
set -u
cd "$(dirname "$0")"
ANA=/home/nada/anaconda3/bin/python
UVPY=/home/nada/.cache/uv/environments-v2/spectral-demo-ef53f7caffa88925/bin/python3

echo "=== [$(date +%H:%M:%S)] StyleID pilot (anaconda) ==="
$ANA runner.py --phase pilot --models styleid

echo "=== [$(date +%H:%M:%S)] FLUX pilot (uv) ==="
$UVPY runner.py --phase pilot --models flux_img2img,flux_redux,flux_ipadapter,flux_kontext

echo "=== [$(date +%H:%M:%S)] PILOT GENERATION DONE ==="
