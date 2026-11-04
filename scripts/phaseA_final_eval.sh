#!/usr/bin/env bash
# Phase A milestone check (docs/phase-a-design.md section 15). Waits for scripts/night_v2_chain.sh (the thinking
# baseline) to finish, starts the vLLM server, then compares the original router, the v2 research champion and
# the random-control champion on the selection split (per-category accuracy, router recall) and on the sealed
# split, whose first use this is (section 3 reserves it for milestones). Stops the server at the end.
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."

while pgrep -f night_v2_chain.sh > /dev/null; do sleep 60; done

bash scripts/serve.sh > results/logs/vllm_serve_final.log 2>&1 &
for _ in $(seq 120); do
    $py -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2)" \
        2> /dev/null && break
    sleep 10
done

routers="base=$HOME/alter/evo/runs/a3-main/baseline_candidate
v2_champ=$HOME/alter/evo/runs/a3-v2/cycle_014/candidate
random_champ=$HOME/alter/evo/runs/a5-main/attempt_002"
for split in selection sealed; do
    $py scripts/eval_candidates.py --split $split --out results/phaseA_v2_$split.json $routers \
        > results/logs/phaseA_v2_$split.log 2>&1
    echo "exit: $?" >> results/logs/phaseA_v2_$split.log
done

# This script's own command line does not contain the pattern.
pkill -f "bin/vllm serve"
