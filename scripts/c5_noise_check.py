"""Before C5 (docs/phase-c-design.md section 8): how much does the same candidate change between two runs?

Runs the initial candidate twice on every maths, code and RTL development task and compares run 2 with run 1 the
way the promotion gates compare a child with its champion: domain-balanced difference and one-sided p (the dev
gate), and the share of run-1 solved tasks that run 2 gets wrong (the regression gate, limit 5%). Also gives the
same numbers on random 80-task samples like the C4 pilot's. Needs the vLLM server (scripts/serve.sh).

  python scripts/c5_noise_check.py --out results/c5_noise_check.json
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

from rive.benchmark.hard import load_tasks  # noqa: E402
from rive.evolution.controller import compare  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.runtime.workflow import run_workflow_tasks  # noqa: E402

PILOT_SIZES = {"math": 20, "code": 30, "rtl": 30}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=os.path.expanduser("~/alter/evo/runs/c5-noise"))
    p.add_argument("--out", default=str(ROOT / "results" / "c5_noise_check.json"))
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--samples", type=int, default=200, help="random pilot-sized samples to compare on")
    args = p.parse_args()

    cand = Path(args.run_dir) / "candidate"
    shutil.rmtree(cand, ignore_errors=True)
    shutil.copytree(ROOT / "candidate_template", cand)
    tasks = load_tasks(("math", "code", "rtl"), ("dev",))
    llm = LLM()
    runs = []
    for _ in range(2):
        start = time.perf_counter()
        runs.append(run_workflow_tasks(tasks, cand, llm, workers=args.workers))
        print(f"run {len(runs)}: {time.perf_counter() - start:.0f} s", flush=True)
    first, second = runs

    out = {"n": len(tasks), "all_dev": compare(first, second), "by_domain": {}}
    for domain in ("math", "code", "rtl"):
        a = [r for r in first if r["domain"] == domain]
        b = {r["id"]: r for r in second if r["domain"] == domain}
        solved = [r for r in a if r["correct"]]
        out["by_domain"][domain] = {
            "n": len(a), "accuracy": [round(sum(r["correct"] for r in rs) / len(a), 4) for rs in (a, b.values())],
            "changed": sum(r["correct"] != b[r["id"]]["correct"] for r in a),
            "regression": round(sum(not b[r["id"]]["correct"] for r in solved) / len(solved), 4) if solved else 0.0}
    # The pilot's gates on pilot-sized samples: how often would an unchanged candidate pass or fail them?
    rng = random.Random(0)
    by_domain = {d: [i for i, t in enumerate(tasks) if t["domain"] == d] for d in PILOT_SIZES}
    stats = []
    for _ in range(args.samples):
        idx = [i for d, k in PILOT_SIZES.items() for i in rng.sample(by_domain[d], k)]
        stats.append(compare([first[i] for i in idx], [second[i] for i in idx]))
    out["pilot_sized_samples"] = {
        "samples": args.samples,
        "dev_gate_passed": sum(s["p"] < 0.05 for s in stats) / len(stats),
        "regression_gate_failed": sum(s["regression"] > 0.05 for s in stats) / len(stats),
        "median_regression": sorted(s["regression"] for s in stats)[len(stats) // 2]}
    out["items"] = [{"id": a["id"], "domain": a["domain"], "run1": a["correct"], "run2": b["correct"]}
                    for a, b in zip(first, second)]
    out["llm_usage"] = llm.usage
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    main()
