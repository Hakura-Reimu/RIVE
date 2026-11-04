#!/usr/bin/env bash
# Phase D, D1 to D4 (docs/phase-d-design.md section 7; the decision rules there were written before any result).
# D1 sample and build -> D2 two seeds (2 epochs when both fit in about 5 hours, else 1) -> D3 canary and the pass
# criterion -> D4 only if D3 passed. Any failure stops the chain. Steps already done are skipped when it is
# started again (samples, trained seeds and evaluation runs are kept).
# Run by path: this file's own command line must not contain the pkill pattern.
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."
run=~/alter/evo/runs/phase-d log=results/logs/phase_d.log
say() { echo "$(date '+%F %T') $*" >> $log; }
stop() { say "STOPPED: $*"; pkill -f "bin/vllm serve"; exit 1; }
start() {  # the vLLM server with these extra options; three tries (the desktop's share of VRAM varies)
    for attempt in 1 2 3; do
        bash scripts/serve.sh "$HOME/alter/models/Qwen3.5-4B" qwen3.5-4b "$@" \
            >> results/logs/vllm_serve_phase_d.log 2>&1 &
        pid=$!
        for _ in $(seq 120); do
            $py -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2)" \
                2> /dev/null && return 0
            kill -0 $pid 2> /dev/null || break
            sleep 10
        done
        say "server start, try $attempt, failed"
        pkill -f "bin/vllm serve"
        sleep 30
    done
    return 1
}
# D0 (2026-10-30): with two adapter slots vLLM had 0.17 GiB of KV cache left and did not start; one slot on the
# GPU (both adapters registered), 2048-token batches and 0.88 of the memory gave 1.7 GiB.
LORA="--enable-lora --max-lora-rank 16 --max-loras 1 --max-cpu-loras 2 --max-num-batched-tokens 2048
      --gpu-memory-utilization 0.88"

if [ ! -f $run/train.jsonl ]; then
    say "D1: sampling"
    start || stop "the server did not start for D1"
    [ -f $run/pool.json ] || $py scripts/phase_d_data.py pool >> $log 2>&1 || stop "pool"
    $py scripts/phase_d_data.py sample >> $log 2>&1 || stop "sample"
    $py scripts/phase_d_data.py build >> $log 2>&1 || stop "build"
    pkill -f "bin/vllm serve"
    sleep 30
fi

epochs=$($py -c "
import json
s = json.load(open('$run/train_summary.json'))
hours = 2 * 2 * s['answer_tokens'] * 1.3 / 770 / 3600  # two seeds, two epochs, prompts about 30% more
print(2 if hours <= 5 else 1)")
say "D2: training two seeds, $epochs epoch(s)"
for seed in 1 2; do
    [ -f $run/train_seed$seed.json ] && continue
    $py scripts/phase_d_train.py --seed $seed --epochs $epochs --no-upcast >> $log 2>&1 || stop "training seed $seed"
done

say "D3: evaluation"
start $LORA --lora-modules seed1=$run/adapters/seed1-vllm seed2=$run/adapters/seed2-vllm \
    || stop "the server did not start for D3"
$py scripts/phase_d_eval.py canary --adapters seed1,seed2 >> $log 2>&1 || stop "canary: an adapter does not change the answers"
$py scripts/phase_d_eval.py evaluate --adapters seed1,seed2 >> $log 2>&1 || stop "evaluate"
passed=$($py -c "import json; print(json.load(open('results/phase_d_eval.json'))['verdict']['passed'])")
say "D3 passed: $passed"

if [ "$passed" = True ]; then
    best=$($py -c "
import json, statistics
m = json.load(open('results/phase_d_eval.json'))['models']
print(max(('seed1', 'seed2'), key=lambda k: statistics.mean(m[k]['selection_macro'])))")
    say "D4: research ability test, researcher $best, solver qwen3.5-4b, with the guards"
    out=~/alter/evo/runs/research-ability-v1 rlog=results/logs/research_ability_v1.log
    $py scripts/research_ability.py propose --researcher $best --solver qwen3.5-4b --guards --out-dir $out \
        >> $rlog 2>&1 || stop "D4 propose"
    $py scripts/research_ability.py evaluate --researchers $best --solver qwen3.5-4b --out-dir $out \
        --out results/research_ability_v1.json >> $rlog 2>&1 || stop "D4 evaluate"
fi
pkill -f "bin/vllm serve"
say "done"
