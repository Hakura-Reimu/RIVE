#!/usr/bin/env bash
# Meta-RSI P0 calibration (scripts/meta_rsi_calibrate.py, docs/meta-rsi-design.md section 5): serve V0, run every
# run of the experiment that is not done yet, write the report, stop the server. Started again, it goes on where
# it stopped.
#   bash scripts/meta_rsi_calibrate_chain.sh [calibration|m1]
experiment=${1:-calibration}
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."
log=results/logs/meta_rsi_$experiment.log
[ -f ~/alter/meta-rsi/calibration/packs.json ] || $py scripts/meta_rsi_calibrate.py packs >> $log 2>&1 || exit 1
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
$py scripts/meta_rsi_calibrate.py run --experiment $experiment >> $log 2>&1
echo "run exit: $?" >> $log
$py scripts/meta_rsi_calibrate.py report --experiment $experiment >> $log 2>&1
echo "report exit: $?" >> $log
# This script's own command line does not contain the pattern.
pkill -f "bin/vllm serve"
