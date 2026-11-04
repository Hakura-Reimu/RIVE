#!/usr/bin/env bash
# Training-recipe check (scripts/rive_recipe.py, docs/rive-design.md section 6.21): run the given steps in order.
# Training needs the GPU, so the server is stopped before it (unless the adapter is already trained); the other steps
# need the server with self-s1 and gentle-s1. Training comes first, so the watchdog never sees the server go away
# on purpose. A failed step ends the chain with its exit code, so that scripts/watchdog.py starts it again.
#   python scripts/watchdog.py --heartbeat ~/alter/rive/recipe/heartbeat --stall-minutes 45 -- \
#       bash scripts/rive_recipe_chain.sh train canary eval fidelity report
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."
log=results/logs/rive_recipe.log
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
        --enable-lora --max-lora-rank 16 --max-loras 2 --max-cpu-loras 3 --lora-modules "$@" \
        >> results/logs/vllm_serve_rive_fp8.log 2>&1 < /dev/null &
    for _ in $(seq 90); do
        up $wanted && return 0
        sleep 10
    done
    echo "$(date '+%F %T') server did not start" >> $log
    exit 3
}
e1=nowait-e1=$HOME/alter/models/lora-nowait-e1
s1=self-s1=$HOME/alter/models/lora-self-s1
g1=gentle-s1=$HOME/alter/models/lora-gentle-s1
for step in "$@"; do
    case $step in
        train) [ -d ~/alter/rive/recipe/adapters/seed1-vllm ] || gpu_free ;;
        canary|eval|fidelity) serve gentle-s1 $e1 $s1 $g1 ;;
    esac
    echo "$(date '+%F %T') step $step" >> $log
    $py scripts/rive_recipe.py $step >> $log 2>&1
    code=$?
    echo "$(date '+%F %T') step $step exit $code" >> $log
    [ $code -eq 0 ] || exit $code
done
pkill -f "bin/vllm serve"
exit 0
