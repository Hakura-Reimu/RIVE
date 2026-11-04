#!/usr/bin/env bash
# DeepSeek teaches with its reasoning (scripts/rive_reason.py, docs/rive-design.md section 6.20): run the given steps
# in order. Solving needs only the teacher; training needs the GPU, so the server is stopped before it (unless the
# adapter is already trained); canary and eval need the server with both adapters. A failed step ends the chain with
# its exit code, so that scripts/watchdog.py starts it again; finished work is kept and not repeated.
#   python scripts/watchdog.py --heartbeat ~/alter/rive/reason/heartbeat --stall-minutes 45 -- \
#       bash scripts/rive_reason_chain.sh train canary eval report
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."
log=results/logs/rive_reason.log
up() {  # the server answers and has the adapter named $1
    curl -sf 127.0.0.1:8000/v1/models 2> /dev/null | grep -q "\"$1\""
}
gpu_free() {
    pkill -f "bin/vllm serve" || true  # this script's own command line does not contain the pattern
    for _ in $(seq 60); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ] && return 0
        sleep 3
    done
}
serve() {  # serve WANTED NAME=PATH ...: a server that has the adapter WANTED, started with the adapters given
    local wanted=$1
    shift
    up $wanted && return 0
    gpu_free
    echo "$(date '+%F %T') serving with the adapters $*" >> $log
    setsid nohup bash scripts/serve.sh ~/alter/models/Qwen3.5-4B qwen3.5-4b \
        --enable-lora --max-lora-rank 16 --max-loras 2 --lora-modules "$@" \
        >> results/logs/vllm_serve_rive_fp8.log 2>&1 < /dev/null &
    for _ in $(seq 90); do
        up $wanted && return 0
        sleep 10
    done
    echo "$(date '+%F %T') server did not start" >> $log
    exit 3
}
e1=nowait-e1=$HOME/alter/models/lora-nowait-e1
r1=reason-s1=$HOME/alter/models/lora-reason-s1
for step in "$@"; do
    case $step in
        train) [ -d ~/alter/rive/reason/adapters/seed1-vllm ] || gpu_free ;;
        canary|eval) serve reason-s1 $e1 $r1 ;;
    esac
    echo "$(date '+%F %T') step $step" >> $log
    $py scripts/rive_reason.py $step >> $log 2>&1
    code=$?
    echo "$(date '+%F %T') step $step exit $code" >> $log
    [ $code -eq 0 ] || exit $code
done
pkill -f "bin/vllm serve"
exit 0
