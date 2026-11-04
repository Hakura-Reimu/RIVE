"""Meta-RSI P0 pilot (docs/meta-rsi-design.md section 5, steps 1 and 2): T1 bug-fix tasks from MBPP, then the
initial agent A0 on every difficulty setting, to find settings where A0 solves 30-60% of the tasks.

  python scripts/meta_rsi_pilot.py tasks    # mutants checked in the judge sandbox; no model needed
  python scripts/meta_rsi_pilot.py run      # A0 answers every task (vLLM serving qwen3.5-4b)

MBPP problems are split once (dev 40%, selection 20%, test 40%); the pilot draws only from dev, so the problems
of the later selection and test sets stay unseen.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.benchmark.hard import DATA  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.meta_rsi.bugfix import SETTINGS, candidates, check, failing, judge, make_task  # noqa: E402
from rive.runtime.workflow import run_workflow_tasks  # noqa: E402

SPLIT_SALT = "meta-rsi-v1"


def split(problem_id):
    """dev, selection or test, fixed per MBPP problem: every task made from one problem lands in one split."""
    h = int.from_bytes(hashlib.sha256(f"{SPLIT_SALT}:{problem_id}".encode()).digest()[:8], "big") / 2**64
    return "dev" if h < 0.4 else "selection" if h < 0.6 else "test"


def tasks_step(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in open(DATA / "mbpp.jsonl")]
    dev = [i for i in items if split(i["id"]) == "dev"]
    pilot = sorted(random.Random(0).sample(dev, min(args.problems, len(dev))), key=lambda i: i["id"])
    cands = {(item["id"], bugs): candidates(item, bugs, n=args.candidates) for item in pilot for bugs in (1, 2)}
    jobs = [(item, None) for item in pilot] + [(item, c) for item in pilot for bugs in (1, 2)
                                               for c in cands[(item["id"], bugs)]]
    start = time.perf_counter()
    with ThreadPoolExecutor(args.threads) as pool:
        results = list(pool.map(lambda job: check(job[0], job[0]["reference"] if job[1] is None else job[1]["code"]),
                                jobs))
    reference_ok = {}
    for (item, cand), r in zip(jobs, results):
        if cand is None:
            reference_ok[item["id"]] = failing(r) == []
        else:
            cand["failing"] = failing(r)  # None: did not load or timed out; []: passes every test
            cand["check"] = r
    tasks, stats = [], Counter()
    for item in pilot:
        if not reference_ok[item["id"]]:
            stats["reference_fails"] += 1
            continue
        for bugs in (1, 2):
            stats[f"bugs{bugs}_candidates"] += len(cands[(item["id"], bugs)])
            stats[f"bugs{bugs}_usable"] += sum(bool(c["failing"]) for c in cands[(item["id"], bugs)])
        for setting, (_, _, bugs) in SETTINGS.items():
            usable = [c for c in cands[(item["id"], bugs)] if c["failing"]]
            if usable:
                tasks.append(make_task(item, usable[0], setting))
            else:
                stats[f"{setting}_no_mutant"] += 1
    (out / "pilot_tasks.json").write_text(json.dumps(tasks, ensure_ascii=False))
    summary = {"problems": len(pilot), "tasks": dict(Counter(t["item"]["setting"] for t in tasks)), **stats,
               "check_seconds": round(time.perf_counter() - start)}
    (out / "pilot_tasks_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


def run_step(args):
    out = Path(args.out_dir)
    tasks = json.loads((out / "pilot_tasks.json").read_text())
    agent = out / "agent_a0"
    if not agent.exists():
        shutil.copytree(ROOT / "candidate_template", agent)
    path = out / "pilot_a0.json"
    if path.exists():
        res = json.loads(path.read_text())
    else:
        start = time.perf_counter()
        res = run_workflow_tasks(tasks, agent, LLM(model=args.model), workers=args.workers, judge_fn=judge)
        for r, t in zip(res, tasks):
            r["setting"] = t["item"]["setting"]
        path.write_text(json.dumps(res, ensure_ascii=False))
        print(f"A0 on {len(tasks)} tasks: {time.perf_counter() - start:.0f} s", flush=True)
    summary = {}
    for setting in SETTINGS:
        rs = [r for r in res if r["setting"] == setting]
        if rs:
            summary[setting] = {"tasks": len(rs), "solved": sum(r["correct"] for r in rs),
                                "rate": round(statistics.mean(r["correct"] for r in rs), 3),
                                "failures": dict(Counter(r["failure"] for r in rs if not r["correct"])),
                                "mean_tokens": round(statistics.mean(r["tokens"] for r in rs)),
                                "mean_seconds": round(statistics.mean(r["seconds"] for r in rs), 1)}
    result = {"tasks": json.loads((out / "pilot_tasks_summary.json").read_text()), "a0": summary}
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("tasks", "run"))
    p.add_argument("--out-dir", default=os.path.expanduser("~/alter/meta-rsi/pilot"))
    p.add_argument("--problems", type=int, default=40, help="tasks: MBPP problems from the dev split")
    p.add_argument("--candidates", type=int, default=6, help="tasks: mutants tried per problem and bug count")
    p.add_argument("--threads", type=int, default=8, help="tasks: sandbox checks at a time")
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", default=str(ROOT / "results" / "meta_rsi_t1_pilot.json"))
    args = p.parse_args()
    (tasks_step if args.step == "tasks" else run_step)(args)


if __name__ == "__main__":
    main()
