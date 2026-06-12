#!/bin/bash
# Wait for the other GPU job to finish, then run the expanded freqctrl sweep
# and rebuild the site. Logs heartbeats so progress is tail-able.
set -u
cd "$(dirname "$0")"
LOG=.freqctrl_watcher.log

used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
ts()   { date '+%H:%M:%S'; }
say()  { echo "[$(ts)] $*" | tee -a "$LOG"; }

BUSY_MIB=2000      # > this => a job is on the GPU
IDLE_MIB=1500      # < this => GPU considered idle
IDLE_NEED=15       # consecutive idle polls required (15 x 20s = 5 min sustained)
POLL=20
PHASE_A_MAX=180    # up to 60 min waiting for the job to appear

: > "$LOG"
say "watcher start; current GPU used=$(used) MiB"

# Phase A: wait until the other job actually starts (GPU becomes busy).
saw_busy=0
for ((i=0; i<PHASE_A_MAX; i++)); do
  u=$(used)
  if [ "${u:-0}" -gt "$BUSY_MIB" ]; then saw_busy=1; say "GPU busy (used=$u MiB) — job detected"; break; fi
  sleep "$POLL"
done
[ "$saw_busy" -eq 0 ] && say "no busy GPU seen within window — proceeding (job may have finished or not started)"

# Phase B: wait for sustained idle (bridges model-reload gaps between the 3 experiments).
idle=0
while [ "$idle" -lt "$IDLE_NEED" ]; do
  u=$(used)
  if [ "${u:-99999}" -lt "$IDLE_MIB" ]; then
    idle=$((idle+1))
    say "idle $idle/$IDLE_NEED (used=$u MiB)"
  else
    [ "$idle" -gt 0 ] && say "busy again (used=$u MiB) — idle counter reset"
    idle=0
  fi
  [ "$idle" -lt "$IDLE_NEED" ] && sleep "$POLL"
done
say "GPU idle sustained — starting expanded freqctrl run"

# Run the sweep (caching skips the 36 already-generated cells) + cfg3.5 baseline.
python e9_freqctrl.py --seeds 5 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then say "e9_freqctrl.py FAILED (rc=$rc) — not rebuilding site"; exit "$rc"; fi
say "freqctrl run done — rebuilding site (standalone)"

python make_e9_site.py --standalone 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
say "site rebuild rc=$rc — watcher complete"
exit "$rc"
