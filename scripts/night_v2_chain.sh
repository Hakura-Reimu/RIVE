#!/usr/bin/env bash
# After the Phase A main run: research loop v2 (feedback changes, docs/phase-a-design.md section 12) on the
# running vLLM server, then stop the server and run the thinking-mode sampled baseline on the hard sets
# (phase0/REPORT.md 7.3). Each step logs and records its exit code.
py=~/alter/.venv/bin/python

# Let the v1 selection re-evaluation (scripts/eval_candidates.py) finish first so the two do not share the server.
while pgrep -f eval_candidates.py > /dev/null; do sleep 30; done

cd "$(dirname "$0")/.."
$py scripts/a3_run_cycles.py --cycles 20 --k 32 --dev-n 200 --run-dir ~/alter/evo/runs/a3-v2 \
    > results/logs/a3_v2.log 2>&1
echo "exit: $?" >> results/logs/a3_v2.log

# The server was started from another shell; this script's own command line does not contain the pattern.
pkill -f "bin/vllm serve"
for _ in $(seq 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && break
    sleep 5
done

cd "$(dirname "$0")/../../phase0"
smoke=~/alter/runs/phase0/kvfp8-smoke
$py baseline_eval.py --thinking --only aime2025 --limit 2 --kv-fp8 --out $smoke --report-dir $smoke \
    > logs/thinking_kvfp8_smoke.log 2>&1
status=$?
echo "exit: $status" >> logs/thinking_kvfp8_smoke.log
# FP8 KV roughly doubles how many long traces run at once; fall back to BF16 KV if it does not work.
kv=""
[ $status -eq 0 ] && kv="--kv-fp8"
$py baseline_eval.py --thinking --only hard --sample 30 $kv > logs/thinking_hard_sample30.log 2>&1
echo "exit: $?" >> logs/thinking_hard_sample30.log
