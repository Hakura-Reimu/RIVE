#!/usr/bin/env bash
# Section 6.25 end to end (scripts/rive_rl.py): the loop, the chosen checkpoint's evaluations, the report. The loop
# starts and drives the server itself (sleep mode while it trains). A failed step ends the chain with its exit code,
# so that scripts/watchdog.py starts it again; finished iterations and evaluations are kept.
#   python scripts/watchdog.py --heartbeat ~/alter/rive/rl/heartbeat --stall-minutes 45 -- bash scripts/rive_rl_chain.sh
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."
log=results/logs/rive_rl.log
for step in run eval report; do
    echo "$(date '+%F %T') step $step" >> $log
    $py scripts/rive_rl.py $step >> $log 2>&1
    code=$?
    echo "$(date '+%F %T') step $step exit $code" >> $log
    [ $code -eq 0 ] || exit $code
done
exit 0
