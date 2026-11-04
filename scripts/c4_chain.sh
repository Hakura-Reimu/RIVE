#!/usr/bin/env bash
# C4 pilot, queued after the C1 check (scripts/c1_chain.sh): start the vLLM server, run 3 generations of the Phase C
# loop, stop the server. c4_pilot.py itself refuses to start if the C1 check did not reproduce Phase 0.
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."

while pgrep -f "c1_chain.sh|phaseA_final_eval.sh|night_v2_chain.sh" > /dev/null; do sleep 60; done
for _ in $(seq 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && break
    sleep 5
done

bash scripts/serve.sh > results/logs/vllm_serve_c4.log 2>&1 &
for _ in $(seq 120); do
    $py -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2)" \
        2> /dev/null && break
    sleep 10
done

$py scripts/c4_pilot.py --generations 3 > results/logs/c4_pilot.log 2>&1
echo "exit: $?" >> results/logs/c4_pilot.log

# This script's own command line does not contain the pattern.
pkill -f "bin/vllm serve"
