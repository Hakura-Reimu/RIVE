#!/usr/bin/env bash
# Before the run after C5 (docs/phase-c-design.md section 8): start the vLLM server, check the changed research step
# live (scripts/c5b_research_check.py), stop the server.
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."

for _ in $(seq 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && break
    sleep 5
done

bash scripts/serve.sh > results/logs/vllm_serve_c5b_check.log 2>&1 &
for _ in $(seq 120); do
    $py -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2)" \
        2> /dev/null && break
    sleep 10
done

$py scripts/c5b_research_check.py > results/logs/c5b_research_check.log 2>&1
echo "exit: $?" >> results/logs/c5b_research_check.log

# This script's own command line does not contain the pattern.
pkill -f "bin/vllm serve"
