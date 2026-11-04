#!/usr/bin/env bash
# docs/rive-design.md section 6.16 end to end (scripts/rive_claude_edit.py): the stored states with the server
# stopped, the FP8 server with LoRA (adapters loaded at run time), Claude's session, the held-out check, then the
# plain FP8 server again. Steps already done are skipped (their output files exist).
#   bash scripts/rive_claude_edit_chain.sh >> results/logs/rive_claude_edit.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
EDIT=(~/alter/.venv/bin/python scripts/rive_claude_edit.py)
OUT=~/alter/rive/claude_edit
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

start() {  # start <options for scripts/serve.sh after the model and the served name>
  stop
  echo "$(date '+%F %T') serving $BASE $*"
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True setsid nohup bash scripts/serve.sh "$BASE" qwen3.5-4b "$@" \
    >> "$SERVER_LOG" 2>&1 < /dev/null &
  sleep 10
  for _ in $(seq 120); do
    curl -sf 127.0.0.1:8000/v1/models > /dev/null && return
    pgrep -f "vllm serve $BASE" > /dev/null || { echo "$(date '+%F %T') server exited"; exit 1; }
    sleep 3
  done
  echo "$(date '+%F %T') server not up after 6 minutes"; exit 1
}

echo "$(date '+%F %T') chain start"
[ -f $OUT/positions.json ] || { stop; "${EDIT[@]}" positions; }
start --enable-lora --max-lora-rank 8 --max-loras 1
[ -f $OUT/chosen.json ] || "${EDIT[@]}" session
"${EDIT[@]}" verify
start
echo "$(date '+%F %T') chain done"
