"""Phase A control group: same number of attempts as the research loop, but router changes are drawn at
random from fixed templates. Also scores the hand-written rule baseline. Needs scripts/serve.sh running.

  python scripts/a5_random_control.py --attempts 20 --k 16 --dev-n 200
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from transformers import AutoTokenizer  # noqa: E402

from a3_run_cycles import evaluate_on, load_tasks  # noqa: E402
from rive.benchmark.stats import paired_bootstrap  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.research.random_control import RULE_BASELINE, random_router  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/alter/models/Qwen3.5-4B"))
    p.add_argument("--attempts", type=int, default=20)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--dev-n", type=int, default=200)
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", default=os.path.expanduser(f"~/alter/evo/runs/a5-{time.strftime('%Y%m%d-%H%M%S')}"))
    return p.parse_args()


def candidate_with(src, base, dest):
    shutil.copytree(base, dest, dirs_exist_ok=True)
    (dest / "agent" / "tool_router.py").write_text(src)
    return dest


def acc(results):
    return round(sum(r["correct"] for r in results) / len(results), 4)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tok, llm = AutoTokenizer.from_pretrained(args.model), LLM()
    dev = load_tasks(args.k, args.round, "dev")
    dev = sorted(random.Random(0).sample(dev, min(args.dev_n, len(dev))), key=lambda t: t["id"])
    selection = load_tasks(args.k, args.round, "selection")
    template = ROOT / "candidate_template"

    base_dir = candidate_with((template / "agent" / "tool_router.py").read_text(), template, run_dir / "baseline")
    parent_dir, parent_res = base_dir, evaluate_on(dev, base_dir, llm, tok)
    rule_dir = candidate_with(RULE_BASELINE, template, run_dir / "rule_baseline")
    summary = {"args": vars(args), "baseline_dev": acc(parent_res), "rule_dev": acc(evaluate_on(dev, rule_dir, llm, tok)),
               "attempts": []}
    rng = random.Random(args.seed)
    for i in range(1, args.attempts + 1):
        name, params, src = random_router(rng)
        child = candidate_with(src, template, run_dir / f"attempt_{i:03d}")
        child_res = evaluate_on(dev, child, llm, tok)
        diff, p = paired_bootstrap([int(r["correct"]) for r in parent_res], [int(r["correct"]) for r in child_res])
        improved = diff > 0 and p < 0.05
        summary["attempts"].append({"attempt": i, "template": name, "params": params, "parent_dev": acc(parent_res),
                                    "child_dev": acc(child_res), "diff": round(diff, 4), "p": round(p, 4),
                                    "improved": improved})
        print(json.dumps(summary["attempts"][-1]), flush=True)
        if improved:
            parent_dir, parent_res = child, child_res

    base_sel = evaluate_on(selection, base_dir, llm, tok)
    summary["selection"] = {"baseline": acc(base_sel), "rule": acc(evaluate_on(selection, rule_dir, llm, tok))}
    if parent_dir != base_dir:
        champ_sel = evaluate_on(selection, parent_dir, llm, tok)
        diff, p = paired_bootstrap([int(r["correct"]) for r in base_sel], [int(r["correct"]) for r in champ_sel])
        summary["selection"].update(champion=acc(champ_sel), diff=round(diff, 4), p_one_sided=round(p, 4),
                                    champion_dir=str(parent_dir))
    summary["improved_attempts"] = sum(a["improved"] for a in summary["attempts"])
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "attempts"}, indent=2))


if __name__ == "__main__":
    main()
