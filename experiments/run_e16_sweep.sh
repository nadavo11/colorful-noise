#!/usr/bin/env bash
# Wait for the GPU to free (it is currently held by an external process), then run
# the full E16 sweep. Resumable: e16 caches every generation, so a restart picks up
# where it left off. VQAScore is skipped (--no_vqa) -- t2v-metrics pins
# transformers==4.49 + needs the llava package, which conflicts with our
# diffusers-0.38/transformers-4.57 Flux stack; run VQAScore later from an isolated
# env over the saved results/e16/<id>/images/*.png. CLIP-T is the adherence guardrail.
set -u
cd "$(dirname "$0")"
LOG=results/e16_sweep.log
mkdir -p results
echo "[launcher] $(date) waiting for GPU (<2000 MiB used) ..." | tee -a "$LOG"
free_count=0
for _ in $(seq 1 720); do          # up to ~12h of polling at 60s
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${used:-99999}" -lt 2000 ]; then
    free_count=$((free_count+1))
  else
    free_count=0
  fi
  if [ "$free_count" -ge 2 ]; then
    echo "[launcher] $(date) GPU free (${used} MiB) -- starting sweep" | tee -a "$LOG"
    python e16_prompt_adherence.py --part gen,score,analyze \
        --num_prompts 8 --seeds 25 --steps 28 --no_vqa >>"$LOG" 2>&1
    echo "[launcher] $(date) sweep exited with code $?" | tee -a "$LOG"
    exit 0
  fi
  sleep 60
done
echo "[launcher] $(date) gave up: GPU still busy after ~12h" | tee -a "$LOG"
exit 1
