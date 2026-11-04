#!/usr/bin/env bash
# C5b (docs/phase-c-design.md section 8): C5 again, with the gates and the research step as changed after it. Start
# the vLLM server, run 10 generations on all maths, code and RTL development tasks, stop the server. Started again
# after an interruption, c4_pilot.py continues the run.
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."

for _ in $(seq 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && break
    sleep 5
done

bash scripts/serve.sh > results/logs/vllm_serve_c5b.log 2>&1 &
for _ in $(seq 120); do
    $py -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2)" \
        2> /dev/null && break
    sleep 10
done

$py scripts/c4_pilot.py --generations 10 --sizes math=39,code=66,rtl=80 \
    --run-dir ~/alter/evo/runs/c5b-main >> results/logs/c5b_main.log 2>&1
echo "exit: $?" >> results/logs/c5b_main.log

# This script's own command line does not contain the pattern.
pkill -f "bin/vllm serve"
