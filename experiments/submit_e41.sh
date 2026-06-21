#!/bin/bash
# Submit N sharded E41 jobs to Run:AI (each job verifies + smoke-gates, then does its shard).
# Usage: bash submit_e41.sh [N_SHARDS] [SUFFIX] [extra e41 args...]
#   N_SHARDS  number of parallel jobs (default 8)
#   SUFFIX    appended to job names (bump on resubmit after deletion, e.g. b, c)
N="${1:-8}"; SUF="${2:-a}"; shift 2 2>/dev/null || shift $#
IMG=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
PART="${PART:-calibrate}"                 # forwarded into the job (calibrate | gridsweep | ...)
CPU_MEM="${CPU_MEM:-}"                     # optional system RAM request --memory (e.g. 32G)
GPU_MEM="${GPU_MEM:-}"                     # optional --gpu-memory (e.g. 40G) to force a big card
# FLUX loads ~24GB bf16 (bnb4 quantization is a no-op in this image), so it needs an
# A6000/H100; GPU_MEM=40G schedules onto blaufer/bengal. Default -g 1 may land on a 24GB A5000.
if [ -n "$GPU_MEM" ]; then GPU_ARGS="--gpu-memory $GPU_MEM"; else GPU_ARGS="-g 1"; fi
TAG=""; [ "$PART" = "calibrate" ] || TAG="${PART:0:4}-"   # keep names distinct per part
for i in $(seq 0 $((N - 1))); do
  name="e41-${TAG}s${i}-${N}${SUF}"
  echo "submitting $name (shard $i/$N, part=$PART, gpu=$GPU_ARGS)"
  runai submit --name "$name" $GPU_ARGS -i "$IMG" --pvc=storage:/storage --large-shm \
    ${CPU_MEM:+--memory "$CPU_MEM"} \
    --environment "PART=$PART" --command -- \
    bash /storage/malnick/colorful-noise/experiments/cluster_e41_job.sh "$i/$N" "$@"
done
