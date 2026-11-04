#!/usr/bin/env bash
# P2 of Meta-RSI (scripts/meta_rsi_p2.py, docs/meta-rsi-design.md section 7): serve V0 unless it is up, run the
# given steps in order, stop the server. A failed step ends the chain with its exit code, so that
# scripts/watchdog.py starts it again; finished work is kept and not repeated.
#   python scripts/watchdog.py --heartbeat ~/alter/meta-rsi/p2/heartbeat -- \
#       bash scripts/meta_rsi_p2_chain.sh packs check scan m0c choose report
# Extra arguments for every step go in P2_ARGS, e.g. P2_ARGS="--out-dir $HOME/alter/meta-rsi/p2b" (no ~: it is
# not expanded there).
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."
log=results/logs/meta_rsi_p2.log
up() {
    $py -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5)" 2> /dev/null
}
if ! up; then
    bash scripts/serve.sh >> results/logs/vllm_serve_meta_rsi.log 2>&1 &
    for _ in $(seq 90); do
        up && break
        sleep 10
    done
    up || { echo "$(date '+%F %T') server did not start" >> $log; exit 3; }
fi
for step in "$@"; do
    echo "$(date '+%F %T') step $step" >> $log
    $py scripts/meta_rsi_p2.py $step $P2_ARGS >> $log 2>&1
    code=$?
    echo "$(date '+%F %T') step $step exit $code" >> $log
    [ $code -eq 0 ] || exit $code
done
# This script's own command line does not contain the pattern.
pkill -f "bin/vllm serve"
exit 0
