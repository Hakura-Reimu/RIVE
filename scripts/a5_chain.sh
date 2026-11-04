#!/usr/bin/env bash
# Phase A main run: 20 research cycles, then the 20-attempt random-change control, on the same
# vLLM server (scripts/serve.sh must be running). Each step logs to results/logs/ and records its exit code.
cd "$(dirname "$0")/.."
logs=results/logs
py=~/alter/.venv/bin/python

$py scripts/a3_run_cycles.py --cycles 20 --k 32 --dev-n 200 --run-dir ~/alter/evo/runs/a3-main > $logs/a3_main.log 2>&1
echo "exit: $?" >> $logs/a3_main.log

$py scripts/a5_random_control.py --attempts 20 --k 32 --dev-n 200 --run-dir ~/alter/evo/runs/a5-main > $logs/a5_main.log 2>&1
echo "exit: $?" >> $logs/a5_main.log
