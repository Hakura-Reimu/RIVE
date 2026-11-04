#!/usr/bin/env bash
# Longer answers (scripts/rive_long.py, docs/rive-design.md section 6.22): run the given steps in order.
# The memory probe needs the GPU, so the server is stopped before it (unless the probe is done); the evaluations
# need the server with reason-s1. The probe comes first, so the watchdog never sees the server go away on purpose.
# A failed step ends the chain with its exit code, so that scripts/watchdog.py starts it again.
#   python scripts/watchdog.py --heartbeat ~/alter/rive/long/heartbeat --stall-minutes 45 -- \
#       bash scripts/rive_long_chain.sh probe eval report
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."
log=results/logs/rive_long.log
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
        probe) [ -f ~/alter/rive/long/train_probe-4096.json ] || gpu_free ;;
        eval) serve reason-s1 $e1 $r1 ;;
    esac
    echo "$(date '+%F %T') step $step" >> $log
    $py scripts/rive_long.py $step >> $log 2>&1
    code=$?
    echo "$(date '+%F %T') step $step exit $code" >> $log
    [ $code -eq 0 ] || exit $code
done
pkill -f "bin/vllm serve"
exit 0
