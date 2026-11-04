#!/usr/bin/env bash
# Meta-RSI with the component agent (scripts/meta_rsi_components.py, docs/meta-rsi-design.md section 6): serve V0,
# scan the hand-picked configurations, run the configuration-only M0, write the report, stop the server. Started
# again, it goes on where it stopped.
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."
log=results/logs/meta_rsi_components.log
for attempt in 1 2 3; do
    bash scripts/serve.sh >> results/logs/vllm_serve_meta_rsi.log 2>&1 &
    pid=$!
    for _ in $(seq 120); do
        $py -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2)" \
            2> /dev/null && break 2
        kill -0 $pid 2> /dev/null || break
        sleep 10
    done
    echo "server start, try $attempt, failed" >> $log
    sleep 30
done
for step in scan run report; do
    $py scripts/meta_rsi_components.py $step >> $log 2>&1
    echo "$step exit: $?" >> $log
done
# This script's own command line does not contain the pattern.
pkill -f "bin/vllm serve"
