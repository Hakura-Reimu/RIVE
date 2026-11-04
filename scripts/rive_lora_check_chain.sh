#!/usr/bin/env bash
# docs/rive-design.md section 6.15 end to end: the FP8 server with the e1 adapter, the checks of
# scripts/rive_lora_check.py, then the plain FP8 server again.
#   bash scripts/rive_lora_check_chain.sh >> results/logs/rive_lora_check.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
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
  setsid nohup bash scripts/serve.sh "$BASE" qwen3.5-4b "$@" >> "$SERVER_LOG" 2>&1 < /dev/null &
  sleep 10
  for _ in $(seq 120); do
    curl -sf 127.0.0.1:8000/v1/models > /dev/null && return
    pgrep -f "vllm serve $BASE" > /dev/null || { echo "$(date '+%F %T') server exited"; exit 1; }
    sleep 3
  done
  echo "$(date '+%F %T') server not up after 6 minutes"; exit 1
}

echo "$(date '+%F %T') chain start"
start --enable-lora --max-lora-rank 1 --max-loras 1 --lora-modules "nowait-e1=$HOME/alter/models/lora-nowait-e1"
grep "GPU KV cache size\|Available KV cache memory" "$SERVER_LOG" | tail -2
~/alter/.venv/bin/python scripts/rive_lora_check.py run lora_think lora_plain lora_mixed
start
echo "$(date '+%F %T') chain done"
