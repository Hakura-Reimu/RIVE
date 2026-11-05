#!/usr/bin/env bash
# Section 6.26 end to end (scripts/rive_rl.py --version 2): the loop, the chosen checkpoint's evaluations, the report.
# As scripts/rive_rl_chain.sh, in ~/alter/rive/rl2:
#   python scripts/watchdog.py --heartbeat ~/alter/rive/rl2/heartbeat --stall-minutes 45 -- bash scripts/rive_rl2_chain.sh
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."
log=results/logs/rive_rl2.log
for step in run eval report; do
    echo "$(date '+%F %T') step $step" >> $log
    $py scripts/rive_rl.py $step --version 2 >> $log 2>&1
    code=$?
    echo "$(date '+%F %T') step $step exit $code" >> $log
    [ $code -eq 0 ] || exit $code
done
exit 0
