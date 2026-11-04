#!/usr/bin/env bash
# docs/rive-design.md sections 6.23 (e1) and 6.24 (e1h) after the direction and the edit (scripts/rive_9b.py): the
# edited 9B on the development pack with and without thinking (after a check that it answers at all), the
# confirmation with seed 1 if the edit works, then the report.
# Every server has the baseline's options (LoRA support on, no adapter loaded). Leaves the unedited 9B served.
#   bash scripts/rive_9b_chain.sh [e1|e1h] >> results/logs/rive_9b.log 2>&1
# Runs already done are skipped (their output files exist).
set -euo pipefail
cd "$(dirname "$0")/.."
R9=(~/alter/.venv/bin/python scripts/rive_9b.py)
BASE=~/alter/models/Qwen3.5-9B
E=${1:-e1}
case $E in
  e1) COPY=$BASE-nowait-e1l8 ;;
  e1h) COPY=$BASE-nowait-e1l8h ;;
  *) echo "unknown edit $E"; exit 1 ;;
esac
SERVER_LOG=results/logs/vllm_serve_9b.log

stop() {
  pkill -f "^$HOME/alter/.venv/bin/python $HOME/alter/.venv/bin/vllm serve" || true
  for _ in $(seq 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 2000 ] && return  # Windows keeps about 1.2 GiB of the card
    sleep 3
  done
  echo "$(date '+%F %T') GPU still in use after stopping the server"; exit 1
}

serve() {  # serve <model directory>, unless the server already has it
  if curl -sf 127.0.0.1:8000/v1/models | grep -q "\"root\":\"$1\""; then return; fi
  stop
  echo "$(date '+%F %T') serving $1"
  MODEL="$1" VLLM_ALLOW_RUNTIME_LORA_UPDATING=True setsid nohup bash scripts/serve9b.sh \
    --enable-lora --max-lora-rank 16 --max-loras 1 >> "$SERVER_LOG" 2>&1 < /dev/null &
  sleep 10  # until serve9b.sh has become the vllm process
  for _ in $(seq 200); do
    curl -sf 127.0.0.1:8000/v1/models > /dev/null && return
    pgrep -f "vllm serve $1" > /dev/null || { echo "$(date '+%F %T') server for $1 exited"; exit 1; }
    sleep 3
  done
  echo "$(date '+%F %T') server for $1 not up after 10 minutes"; exit 1
}

echo "$(date '+%F %T') chain start ($E)"
serve "$COPY"
"${R9[@]}" talks
"${R9[@]}" run ${E}_think16k_s0 ${E}_plain_s0
if [ "$("${R9[@]}" works $E)" = yes ]; then
  echo "$(date '+%F %T') $E works on seed 0: confirming with seed 1"
  "${R9[@]}" run ${E}_think16k_s1
  serve "$BASE"
  "${R9[@]}" run think16k_s1
fi
serve "$BASE"
"${R9[@]}" report
echo "$(date '+%F %T') chain done"
