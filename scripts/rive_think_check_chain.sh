#!/usr/bin/env bash
# docs/rive-design.md section 6.14 end to end (scripts/rive_think_check.py): the original model's runs, then e1's, then
# the original model served again. Leaves the FP8 server running.
#   bash scripts/rive_think_check_chain.sh >> results/logs/rive_think_check.log 2>&1
# Steps already done are skipped (their output files exist).
set -euo pipefail
cd "$(dirname "$0")/.."
CHECK=(~/alter/.venv/bin/python scripts/rive_think_check.py)
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
"${CHECK[@]}" run s_t0 s_tb s_nt
serve "$BASE-nowait-e1"
"${CHECK[@]}" run s_e1 s_e1b s_e1_16 s_e1_16b s_nt_e1
serve "$BASE"
"${CHECK[@]}" report
echo "$(date '+%F %T') chain done"
