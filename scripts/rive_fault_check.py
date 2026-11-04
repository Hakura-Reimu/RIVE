"""R-000 fault check (docs/rive-design.md section 4): 100 tasks in the real sandbox with injected faults, and what the
evaluation does with them.

- Faults: workflows that crash, hang past the deadline or call the model until the budget is gone; a model server
  that fails calls; the same evaluation asked for twice.
- Passes if:
  - the evaluation with server failures raises and keeps nothing;
  - once the server is back, every task gets a verdict of the right kind;
  - asking again is answered from the store;
  - every evaluation that started is logged as done or failed.

  python scripts/rive_fault_check.py      (WSL, docker; no GPU)
"""
import json
import shutil
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path

import httpx
import openai

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.meta_rsi.reset_meta import EvaluationFailed, Evaluator  # noqa: E402
from rive.infra.events import EventLog, read_events  # noqa: E402
from rive.runtime.workflow import run_workflow_tasks  # noqa: E402

KINDS = ["normal"] * 70 + ["crash"] * 10 + ["hang"] * 8 + ["loop"] * 6 + ["server"] * 6
WORKFLOW = '''
import time


def solve(task, ctx):
    kind = task["prompt"]
    if kind == "crash":
        raise ValueError("injected crash")
    if kind == "hang":
        time.sleep(60)
    answers = []
    for _ in range(50 if kind == "loop" else 1):
        try:
            answers.append(ctx.llm([{"role": "user", "content": kind}], max_tokens=1000))
        except RuntimeError:
            break
    return {"answer": answers[-1] if answers else ""}
'''


class FlakyServer:
    """Answers at once; calls for "server" tasks fail while it is down."""

    def __init__(self):
        self.down, self.lock, self.calls = True, threading.Lock(), Counter()

    def generate(self, messages, max_tokens, temperature, thinking, seed=0):
        kind = messages[-1]["content"]
        with self.lock:
            self.calls[kind] += 1
        if kind == "server" and self.down:
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:8000/v1"))
        return {"text": "ok", "finish_reason": "stop", "prompt_tokens": 3, "completion_tokens": min(900, max_tokens)}


def main():
    work = Path(tempfile.mkdtemp(prefix="rive_fault_", dir=Path.home()))
    cand = work / "cand"
    shutil.copytree(ROOT / "candidate_template", cand)
    (cand / "agent" / "workflow.py").write_text(WORKFLOW)
    for p in [cand, *cand.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    tasks = [{"id": f"t{i:03d}-{k}", "domain": "code", "public": {"domain": "code", "prompt": k}, "item": {}}
             for i, k in enumerate(KINDS)]
    server, events = FlakyServer(), EventLog(work / "events.jsonl", "fault-check")

    def run(ts, d):
        events.log("evaluation_start", tasks=len(ts))
        try:
            out = run_workflow_tasks(ts, d, server, workers=8, seconds=8, judge_fn=lambda t, a: (a == "ok", None, {}))
        except Exception as e:
            events.log("evaluation_failed", error=str(e)[:200])
            raise
        failed = sum(r["server_errors"] > 0 for r in out)
        events.log("evaluation_failed" if failed else "evaluation_done", server_errors=failed)
        return out
    evaluate = Evaluator(run, work / "evals")
    checks = {}
    try:
        evaluate(cand, "faults", tasks)
        checks["server failure raises"] = False
    except EvaluationFailed as e:
        checks["server failure raises"] = "6 of 100" in str(e)
    checks["nothing kept after it"] = not any((work / "evals").iterdir())
    server.down = False
    results = evaluate(cand, "faults", tasks)
    by_kind = {k: Counter((r["correct"], r["failure"]) for r in results if r["id"].endswith(k)) for k in set(KINDS)}
    checks["every task has a verdict"] = len(results) == 100 and [r["id"] for r in results] == [t["id"] for t in tasks]
    checks["normal and recovered tasks solved"] = by_kind["normal"] == {(True, None): 70} and \
        by_kind["server"] == {(True, None): 6}
    checks["crashes and hangs are workflow errors"] = by_kind["crash"] == {(False, "workflow_error"): 10} and \
        by_kind["hang"] == {(False, "workflow_error"): 8}
    loops = [r for r in results if r["id"].endswith("loop")]
    checks["loops stop at the token budget"] = all(r["tokens"] <= 8192 and r["llm_calls"] == 10 for r in loops)
    calls_before = sum(server.calls.values())
    again = evaluate(cand, "faults", tasks)
    checks["asking again uses the store"] = again == results and sum(server.calls.values()) == calls_before
    log = read_events(work / "events.jsonl")
    starts = sum(e["event"] == "evaluation_start" for e in log)
    ends = sum(e["event"] in ("evaluation_done", "evaluation_failed") for e in log)
    checks["event log complete"] = starts == ends == 2
    print(json.dumps({"checks": checks, "verdicts": {k: {str(x): n for x, n in v.items()} for k, v in by_kind.items()},
                      "work_dir": str(work)}, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
