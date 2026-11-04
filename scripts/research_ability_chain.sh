#!/usr/bin/env bash
# Research ability test (scripts/research_ability.py, docs/phase-c-design.md section 8): each researcher model in
# turn makes its 16 changes, then Qwen3.5-4B runs them all. Started again after an interruption, both steps go on
# where they stopped.
#   bash scripts/research_ability_chain.sh [smoke|guarded]
#   smoke: one change per model, three tasks per domain;
#   guarded: the 4B alone, with the locality check and the pre-screen (--guards), in its own directory.
py=~/alter/.venv/bin/python
cd "$(dirname "$0")/.."

specs=("Qwen3.5-0.8B qwen3.5-0.8b" "Qwen3.5-2B qwen3.5-2b" "Qwen3.5-4B qwen3.5-4b")
case "${1:-}" in
    smoke)
        out=~/alter/evo/runs/research-ability-smoke log=results/logs/research_ability_smoke.log
        propose_args="--limit 1" evaluate_args="--tasks-per-domain 3 --seed-runs 2" ;;
    guarded)
        out=~/alter/evo/runs/research-ability-guarded log=results/logs/research_ability_guarded.log
        specs=("Qwen3.5-4B qwen3.5-4b")
        propose_args="--guards" evaluate_args="--researchers qwen3.5-4b --out results/research_ability_guarded.json" ;;
    *)
        out=~/alter/evo/runs/research-ability log=results/logs/research_ability.log
        propose_args="" evaluate_args="" ;;
esac

serve() {  # model directory under ~/alter/models, served name
    for _ in $(seq 60); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && break
        sleep 5
    done
    bash scripts/serve.sh "$HOME/alter/models/$1" "$2" >> "results/logs/vllm_serve_ra_$2.log" 2>&1 &
    for _ in $(seq 120); do
        $py -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2)" \
            2> /dev/null && return
        sleep 10
    done
}

# This script's own command line does not contain the pattern.
for spec in "${specs[@]}"; do
    set -- $spec
    serve "$1" "$2"
    $py scripts/research_ability.py propose --researcher "$2" --out-dir $out $propose_args >> $log 2>&1
    echo "propose $2 exit: $?" >> $log
    [ "$2" = qwen3.5-4b ] || pkill -f "bin/vllm serve"
done
# The 4B's server stays up: it is also the solver.
$py scripts/research_ability.py evaluate --out-dir $out $evaluate_args >> $log 2>&1
echo "evaluate exit: $?" >> $log
pkill -f "bin/vllm serve"
