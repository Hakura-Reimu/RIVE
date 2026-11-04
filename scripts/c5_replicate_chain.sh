#!/usr/bin/env bash
# After C5 (docs/phase-c-design.md section 8): start the vLLM server, run g09c036 against the seed again on the code
# and RTL development tasks (scripts/c5_replicate.py), stop the server.
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."

for _ in $(seq 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && break
    sleep 5
done

bash scripts/serve.sh > results/logs/vllm_serve_c5_replication.log 2>&1 &
for _ in $(seq 120); do
    $py -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2)" \
        2> /dev/null && break
    sleep 10
done

$py scripts/c5_replicate.py --candidates g09c036 --domains code,rtl --rounds 3 \
    > results/logs/c5_replication.log 2>&1
echo "exit: $?" >> results/logs/c5_replication.log

# This script's own command line does not contain the pattern.
pkill -f "bin/vllm serve"
