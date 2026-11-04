#!/usr/bin/env bash
# docs/rive-design.md section 6.13 end to end (scripts/rive_think_fix.py). Runs on the FP8 model, then the direction
# and the edited copies with the server stopped (the forward passes need the GPU), the edited models one after the
# other, the longer cap and, if an edit works, its confirmation with another seed. Leaves the FP8 server running.
#   bash scripts/rive_think_fix_chain.sh >> results/logs/rive_think_fix.log 2>&1
# Steps already done are skipped (their output files exist).
set -euo pipefail
cd "$(dirname "$0")/.."
FIX=(~/alter/.venv/bin/python scripts/rive_think_fix.py)
BASE=~/alter/models/Qwen3.5-4B
SERVER_LOG=results/logs/vllm_serve_rive_fp8.log

stop() {
  pkill -f "^$HOME/alter/.venv/bin/python $HOME/alter/.venv/bin/vllm serve" || true
  for _ in $(seq 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 1500 ] && return
    sleep 3
  done
  echo "$(date '+%F %T') GPU still in use after stopping the server"; exit 1
}

serve() {  # serve <model directory>, unless the server already has it
  if curl -sf 127.0.0.1:8000/v1/models | grep -q "\"root\":\"$1\""; then return; fi
  stop
  echo "$(date '+%F %T') serving $1"
  setsid nohup bash scripts/serve.sh "$1" >> "$SERVER_LOG" 2>&1 < /dev/null &
  sleep 10  # until serve.sh has become the vllm process
  for _ in $(seq 120); do
    curl -sf 127.0.0.1:8000/v1/models > /dev/null && return
    pgrep -f "vllm serve $1" > /dev/null || { echo "$(date '+%F %T') server for $1 exited"; exit 1; }
    sleep 3
  done
  echo "$(date '+%F %T') server for $1 not up after 6 minutes"; exit 1
}

echo "$(date '+%F %T') chain start"
serve "$BASE"
"${FIX[@]}" run t0 tb tw
if [ ! -f ~/alter/rive/thinkfix/direction.json ]; then stop; "${FIX[@]}" direction; fi
for e in e1 e2; do
  [ -f ~/alter/rive/thinkfix/$e.json ] && continue
  [ -f "$BASE-nowait-$e/config.json" ] || { stop; "${FIX[@]}" edit $e; }
  serve "$BASE-nowait-$e"
  "${FIX[@]}" run $e
done
best=$("${FIX[@]}" best-edit)
if [ -n "$best" ]; then
  echo "$(date '+%F %T') $best works on seed 0: confirming with seed 1"
  serve "$BASE-nowait-$best"
  "${FIX[@]}" run e_seed1
  serve "$BASE"
  "${FIX[@]}" run t0_seed1
fi
serve "$BASE"
"${FIX[@]}" run t16
"${FIX[@]}" report
echo "$(date '+%F %T') chain done"
