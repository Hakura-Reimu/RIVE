"""C1 check (docs/phase-c-design.md section 8): the initial candidate, run through the Phase C workflow sandbox,
must reproduce the Phase 0 non-thinking baseline on the maths, code and RTL development items.
Needs the vLLM server (scripts/serve.sh).

  python scripts/c1_check_baseline.py --out results/c1_baseline_check.json
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.benchmark.hard import load_tasks  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.runtime.workflow import run_workflow_tasks  # noqa: E402

PHASE0 = Path(os.path.expanduser("~/alter/runs/phase0/baseline"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=os.path.expanduser("~/alter/evo/runs/c1-check"))
    p.add_argument("--out", default=str(ROOT / "results" / "c1_baseline_check.json"))
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    cand = Path(args.run_dir) / "candidate"
    shutil.rmtree(cand, ignore_errors=True)
    shutil.copytree(ROOT / "candidate_template", cand)
    tasks = load_tasks(("math", "code", "rtl"), ("dev",))
    llm = LLM()
    start = time.perf_counter()
    results = run_workflow_tasks(tasks, cand, llm, workers=args.workers)
    seconds = time.perf_counter() - start

    phase0 = {}
    for source in {t["source"] for t in tasks}:
        for line in open(PHASE0 / f"items_{source}.jsonl"):
            r = json.loads(line)
            phase0[r["id"]] = r["correct"]
    by_domain = {}
    for domain in ("math", "code", "rtl"):
        rs = [r for r in results if r["domain"] == domain]
        by_domain[domain] = {
            "n": len(rs), "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4),
            "phase0_accuracy": round(sum(phase0[r["id"]] for r in rs) / len(rs), 4),
            "same_verdict": round(sum(r["correct"] == phase0[r["id"]] for r in rs) / len(rs), 4),
            "failures": dict(Counter(r["failure"] for r in rs if r["failure"]).most_common()),
            "mean_tokens": round(sum(r["tokens"] for r in rs) / len(rs)),
            "truncated": sum(r["truncated_calls"] > 0 for r in rs)}
    out = {"seconds": round(seconds), "llm_usage": llm.usage, "by_domain": by_domain,
           "items": [{k: r[k] for k in ("id", "domain", "correct", "failure", "tokens", "truncated_calls", "seconds")}
                     for r in results]}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"seconds": out["seconds"], "by_domain": by_domain}, indent=1))


if __name__ == "__main__":
    main()
