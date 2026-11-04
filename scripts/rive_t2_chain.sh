#!/usr/bin/env bash
# T2's environment check (scripts/rive_t2.py, docs/rive-design.md sections 6.11 and 6.17): serve the model with the
# e1 adapter unless it is up so, run the given steps in order, stop the server. A failed step ends the chain with its
# exit code, so that scripts/watchdog.py starts it again; finished work is kept and not repeated.
#   python scripts/watchdog.py --heartbeat ~/alter/rive/t2b/heartbeat -- \
#       bash scripts/rive_t2_chain.sh packs scan repeat gate m0c check report
# Extra arguments for every step go in T2_ARGS (no ~: it is not expanded there).
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."
log=results/logs/rive_t2b.log
adapter=nowait-e1
up() {  # the server answers and has the adapter that thinking calls use (rive/meta_rsi/synth.py, T2Model)
    curl -sf 127.0.0.1:8000/v1/models 2> /dev/null | grep -q "\"$adapter\""
}
if ! up; then
    # This script's own command line does not contain the pattern.
    pkill -f "bin/vllm serve" || true
    for _ in $(seq 60); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ] && break
        sleep 3
    done
    echo "$(date '+%F %T') serving with the $adapter adapter" >> $log
    setsid nohup bash scripts/serve.sh ~/alter/models/Qwen3.5-4B qwen3.5-4b \
        --enable-lora --max-lora-rank 1 --max-loras 1 --lora-modules $adapter=$HOME/alter/models/lora-$adapter \
        >> results/logs/vllm_serve_rive_fp8.log 2>&1 < /dev/null &
    for _ in $(seq 90); do
        up && break
        sleep 10
    done
    up || { echo "$(date '+%F %T') server did not start" >> $log; exit 3; }
fi
for step in "$@"; do
    echo "$(date '+%F %T') step $step" >> $log
    $py scripts/rive_t2.py $step $T2_ARGS >> $log 2>&1
    code=$?
    echo "$(date '+%F %T') step $step exit $code" >> $log
    [ $code -eq 0 ] || exit $code
done
pkill -f "bin/vllm serve"
exit 0
