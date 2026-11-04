"""Meta-RSI with the component agent (docs/meta-rsi-design.md section 6).

  python scripts/meta_rsi_components.py scan     # hand-picked configurations on the dev and selection packs
  python scripts/meta_rsi_components.py run      # the configuration-only M0, RESET-META from A0, seeds 1-6
  python scripts/meta_rsi_components.py report

Same packs as the calibration (~/alter/meta-rsi/calibration/packs.json). The scan stays off the test pack, so no
design decision is made on it. Every evaluation and run is kept and not repeated after an interruption.
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.llm import LLM  # noqa: E402
from rive.meta_rsi.bugfix import judge  # noqa: E402
from rive.meta_rsi.components import DEFAULT_CONFIG, make_agent  # noqa: E402
from rive.meta_rsi.config_improver import ConfigImprover  # noqa: E402
from rive.meta_rsi.reset_meta import Evaluator, reset_meta, solved  # noqa: E402
from rive.runtime.workflow import run_workflow_tasks  # noqa: E402

SEEDS = tuple(range(1, 7))
SCAN = {  # name: changes to the default configuration (A0)
    "a0": {},
    "retry2": {"test_retry": 2},
    "retry2_show_test": {"test_retry": 2, "feedback": "stderr_and_test"},
    "retry3_1024": {"test_retry": 3, "max_tokens": 1024},
    "sample4_pass": {"attempts": 4, "temperature": 0.7, "select": "passes_test", "max_tokens": 1024},
    "sample4_pass_retry2": {"attempts": 4, "temperature": 0.7, "select": "passes_test", "test_retry": 2,
                            "max_tokens": 1024},
    "sample4_majority": {"attempts": 4, "temperature": 0.7, "select": "majority", "max_tokens": 1024},
    "sample3_greedy_pass": {"attempts": 3, "temperature": 0.0, "select": "passes_test", "max_tokens": 1024},
    "review": {"review": True},
    "explain_first": {"instruction": "First explain in one sentence what is wrong, then give the corrected code."},
}


def setup(args):
    out = Path(args.out_dir)
    packs = json.loads(Path(args.packs).read_text())
    solver = LLM(model=args.model)
    evaluate = Evaluator(lambda tasks, d: run_workflow_tasks(tasks, d, solver, workers=args.workers, judge_fn=judge),
                         out / "evals")
    return out, packs, evaluate


def scan_step(args):
    out, packs, evaluate = setup(args)
    for name, change in SCAN.items():
        agent = make_agent(out / "scan" / name, {**DEFAULT_CONFIG, **change})
        dev, sel = evaluate(agent, "dev", packs["dev"]), evaluate(agent, "select", packs["select"])
        print(f"scan {name}: dev {solved(dev)} of {len(dev)}, select {solved(sel)} of {len(sel)}, "
              f"mean tokens {statistics.mean(r['tokens'] for r in dev + sel):.0f}", flush=True)


def run_step(args):
    out, packs, evaluate = setup(args)
    a0 = make_agent(out / "agent_a0", DEFAULT_CONFIG)
    print(f"A0 test: {solved(evaluate(a0, 'test', packs['test']))} of {len(packs['test'])}", flush=True)
    improver = ConfigImprover("m0c", lambda: LLM(model=args.model))
    for seed in SEEDS:
        path = out / "runs" / f"m0c_s{seed}.json"
        if path.exists():
            continue
        result = reset_meta(improver, a0, packs, out / "runs" / f"m0c_s{seed}", evaluate, rounds=args.rounds,
                            margin=args.margin, seed=seed)
        path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"m0c seed {seed}: test {result['test_solved']} of {result['test_tasks']}; rounds "
              f"{[(s['status'], s.get('select_gain'), s.get('kept')) for s in result['rounds']]}", flush=True)


def report_step(args):
    out, packs, evaluate = setup(args)
    scan = {}
    for name, change in SCAN.items():
        agent = out / "scan" / name
        if agent.exists():
            dev, sel = evaluate(agent, "dev", packs["dev"]), evaluate(agent, "select", packs["select"])
            scan[name] = {"config": change, "dev": solved(dev), "select": solved(sel),
                          "mean_tokens": round(statistics.mean(r["tokens"] for r in dev + sel))}
    runs = {p.stem: json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))}
    a0_test = solved(evaluate(out / "agent_a0", "test", packs["test"])) if (out / "agent_a0").exists() else None
    result = {"packs": {k: len(v) for k, v in packs.items()}, "scan": scan, "a0_test": a0_test,
              "runs": {k: {"test": v["test_solved"], "kept": sum(bool(s.get("kept")) for s in v["rounds"]),
                           "rounds": [{k2: s.get(k2) for k2 in ("status", "select_gain", "kept", "hypothesis",
                                                                "config")} for s in v["rounds"]],
                           "improver_tokens": v["improver_usage"].get("completion_tokens")}
                       for k, v in runs.items()}}
    if a0_test is not None and runs:
        gains = [v["test_solved"] - a0_test for v in runs.values()]
        result["gain_over_a0"] = {"each": gains, "mean": statistics.mean(gains),
                                  "runs_with_a_kept_change": sum(v["kept"] > 0 for v in result["runs"].values())}
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("scan", "run", "report"))
    p.add_argument("--out-dir", default=os.path.expanduser("~/alter/meta-rsi/components"))
    p.add_argument("--packs", default=os.path.expanduser("~/alter/meta-rsi/calibration/packs.json"))
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--margin", type=int, default=3)
    p.add_argument("--out", default=str(ROOT / "results" / "meta_rsi_components.json"))
    args = p.parse_args()
    {"scan": scan_step, "run": run_step, "report": report_step}[args.step](args)


if __name__ == "__main__":
    main()
