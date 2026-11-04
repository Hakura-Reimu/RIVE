#!/usr/bin/env bash
# C1 check, queued after the Phase A milestone check (scripts/phaseA_final_eval.sh): start the vLLM server, run the
# initial Phase C candidate on the maths, code and RTL development items, compare with Phase 0, stop the server.
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."

while pgrep -f "phaseA_final_eval.sh|night_v2_chain.sh" > /dev/null; do sleep 60; done
for _ in $(seq 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && break
    sleep 5
done

bash scripts/serve.sh > results/logs/vllm_serve_c1.log 2>&1 &
for _ in $(seq 120); do
    $py -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2)" \
        2> /dev/null && break
    sleep 10
done

$py scripts/c1_check_baseline.py > results/logs/c1_baseline_check.log 2>&1
echo "exit: $?" >> results/logs/c1_baseline_check.log

# This script's own command line does not contain the pattern.
pkill -f "bin/vllm serve"
