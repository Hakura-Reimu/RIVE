"""Phase D evaluation (docs/phase-d-design.md section 5): is V1 better than V0? Both answer through the initial
agent's workflow, interleaved in one vLLM session that serves V0 and hot-loads the adapters (--enable-lora), on the
selection split and the easy sets; the Phase 0 pass criterion (rive.phase_d.criterion) decides.

  python scripts/phase_d_eval.py canary --adapters seed1,seed2     # each adapter changes the model's answers
  python scripts/phase_d_eval.py evaluate --adapters seed1,seed2

An adapter vLLM cannot match to the model's layers is skipped without an error and the answers stay V0's
(phase0/REPORT.md section 4): the canary runs before any evaluation. Each run is kept in --run-dir/evals and not
repeated after an interruption.
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.benchmark.hard import DATA, load_tasks, public_task  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.phase_d import accuracy, criterion, macro  # noqa: E402
from rive.runtime.workflow import run_workflow_tasks  # noqa: E402

DOMAINS = ("math", "code", "rtl")
EASY = (("math", "gsm8k"), ("math", "math500"), ("code", "mbpp"))  # the Phase 0 regression sets this harness judges


def easy_tasks():
    tasks = []
    for domain, source in EASY:
        for line in open(DATA / f"{source}.jsonl"):
            item = json.loads(line)
            tasks.append({"id": f"{source}:{item['id']}", "domain": domain, "source": source,
                          "public": public_task(source, item), "item": item})
    return tasks


def canary(args):
    """Greedy answers to six development prompts, two per domain, sent one at a time: an adapter must change at
    least `--min-changed` of them, more than a second V0 pass changes."""
    dev = sorted(load_tasks(DOMAINS, ("dev",)), key=lambda t: t["id"])
    prompts = [t["public"]["prompt"] for d in DOMAINS for t in [t for t in dev if t["domain"] == d][:2]]

    def answers(model):
        llm = LLM(model=model)
        return [llm.chat([{"role": "user", "content": p}], max_tokens=args.tokens) for p in prompts]
    base, again = answers(args.base), answers(args.base)
    noise = sum(a != b for a, b in zip(base, again))
    out = {"prompts": len(prompts), "v0_against_itself": noise, "adapters": {}}
    for name in args.adapters.split(","):
        changed = sum(a != b for a, b in zip(base, answers(name)))
        out["adapters"][name] = {"changed": changed, "passed": changed >= args.min_changed and changed > noise}
    out["passed"] = all(a["passed"] for a in out["adapters"].values())
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.run_dir) / "canary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)
    sys.exit(0 if out["passed"] else 1)


def describe(results):
    """Accuracy by domain and source, share of tasks with a call cut off by the token limit, mean output tokens."""
    def by(key):
        groups = {}
        for r in results:
            groups.setdefault(r[key], []).append(r)
        return {k: round(accuracy(v), 4) for k, v in sorted(groups.items())}
    return {"by_domain": by("domain"), "by_source": by("source"),
            "cut_off": round(statistics.mean(r["truncated_calls"] > 0 for r in results), 4),
            "tokens": round(statistics.mean(r["tokens"] for r in results))}


def evaluate(args):
    run_dir = Path(args.run_dir)
    evals, agent = run_dir / "evals", run_dir / "agent"
    evals.mkdir(parents=True, exist_ok=True)
    if not agent.exists():
        shutil.copytree(ROOT / "candidate_template", agent)
    selection, easy = load_tasks(DOMAINS, ("selection",)), easy_tasks()
    adapters = args.adapters.split(",")
    models = [args.base] + adapters

    def run(model, name, tasks):
        path = evals / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text())
        start = time.perf_counter()
        res = run_workflow_tasks(tasks, agent, LLM(model=model), workers=args.workers)
        for r, t in zip(res, tasks):
            r["source"] = t["source"]
        path.write_text(json.dumps(res, ensure_ascii=False))
        print(f"{name}: {time.perf_counter() - start:.0f} s, macro {macro(res):.4f}, "
              f"{sum(r['correct'] for r in res)} of {len(res)} correct", flush=True)
        return res

    # V0 opens and closes the session, the adapters run in between: V0, A, B, V0, A, B, V0.
    plan = [(m, k) for k in range(1, args.runs + 1) for m in models] + [(args.base, args.runs + 1)]
    sel = {m: [] for m in models}
    for model, k in plan:
        sel[model].append(run(model, f"{model}_selection_{k}", selection))
    easy_runs = {m: run(m, f"{m}_easy", easy) for m in models}
    verdict = criterion(sel[args.base], {a: sel[a] for a in adapters}, easy_runs[args.base],
                        {a: easy_runs[a] for a in adapters})
    result = {"verdict": verdict, "models": {m: {"selection_macro": [round(macro(r), 4) for r in sel[m]],
                                                 "selection": describe([r for run in sel[m] for r in run]),
                                                 "easy": describe(easy_runs[m])} for m in models},
              "selection_tasks": len(selection), "easy_tasks": len(easy)}
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(verdict, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("canary", "evaluate"))
    p.add_argument("--run-dir", default=os.path.expanduser("~/alter/evo/runs/phase-d"))
    p.add_argument("--base", default="qwen3.5-4b", help="the served name of V0")
    p.add_argument("--adapters", default="seed1,seed2", help="the served names of the adapters (--lora-modules)")
    p.add_argument("--runs", type=int, default=2, help="evaluate: selection runs per adapter; V0 gets one more")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--tokens", type=int, default=200, help="canary: answer length")
    p.add_argument("--min-changed", type=int, default=3, help="canary: answers an adapter must change")
    p.add_argument("--out", default=str(ROOT / "results" / "phase_d_eval.json"))
    args = p.parse_args()
    (canary if args.step == "canary" else evaluate)(args)


if __name__ == "__main__":
    main()
