"""Meta-RSI P0 steps 3-5 (docs/meta-rsi-design.md section 5): calibrate RESET-META before any evolution.

  python scripts/meta_rsi_calibrate.py packs     # dev, selection and test packs, checked in the sandbox
  python scripts/meta_rsi_calibrate.py run       # A0 baseline and the improvers (vLLM serving qwen3.5-4b)
  python scripts/meta_rsi_calibrate.py report

The runs, every one from A0 with 4 proposals:
- oracle: installs the hand-written A* (run the shown failing test, retry). Checks that the ruler sees a known gain.
- m0: the Phase C research step on T1 tasks, 3 seeds. The spread of its results is the noise floor.
- m0_hint: m0 with one strategy hint in its policy layer, 3 seeds. Checks that RESET-META can tell a better
  improver from m0.
A0 is also scored twice on the test pack, for the noise of scoring alone. Every run is kept and not repeated.

  python scripts/meta_rsi_calibrate.py run --experiment m1      # M1 against M0, seeds 1-8
  python scripts/meta_rsi_calibrate.py report --experiment m1

The m1 experiment (section 5, after the calibration): M1 with seeds 1-8, and M0 with seeds 4-8 next to the
calibration's 1-3, on the same packs; the A0 results are reused.
"""
import argparse
import itertools
import json
import math
import os
import random
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.benchmark.hard import DATA  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.meta_rsi.bugfix import judge, make_pack  # noqa: E402
from rive.meta_rsi.improvers import (M1_POLICY, TEST_HINT, EffectCheck, OracleImprover,  # noqa: E402
                                       ResearchImprover, task_example)
from rive.meta_rsi.reset_meta import Evaluator, agent_hash, reset_meta, solved  # noqa: E402
from rive.runtime.workflow import run_workflow_tasks  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from meta_rsi_pilot import split  # noqa: E402

SEEDS = (1, 2, 3)
M1_SEEDS = tuple(range(1, 9))


def packs_step(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in open(DATA / "mbpp.jsonl")]
    by = {s: [i for i in items if split(i["id"]) == s] for s in ("dev", "selection", "test")}
    dev = sorted(random.Random(0).sample(by["dev"], min(40, len(by["dev"]))), key=lambda i: i["id"])  # the pilot's
    packs = {"dev": make_pack(dev, {1: 1, 2: 1}), "select": make_pack(by["selection"], {1: 1, 2: 1}),
             "test": make_pack(by["test"], {1: 2, 2: 1})}
    (out / "packs.json").write_text(json.dumps(packs, ensure_ascii=False))
    print(json.dumps({k: len(v) for k, v in packs.items()}), flush=True)


def run_step(args):
    out = Path(args.out_dir)
    packs = json.loads((out / "packs.json").read_text())
    a0 = out / "agent_a0"
    if not a0.exists():
        shutil.copytree(ROOT / "candidate_template", a0)
    solver = LLM(model=args.model)
    evaluate = Evaluator(lambda tasks, d: run_workflow_tasks(tasks, d, solver, workers=args.workers, judge_fn=judge),
                         out / "evals")
    for pack in ("dev", "select", "test"):
        print(f"A0 {pack}: {solved(evaluate(a0, pack, packs[pack]))} of {len(packs[pack])}", flush=True)
    print(f"A0 test again: {solved(evaluate(a0, 'test_repeat', packs['test']))}", flush=True)
    smoke = packs["dev"][:2]
    improvers = {"oracle": OracleImprover(),
                 "m0": ResearchImprover("m0", lambda: LLM(model=args.model), smoke, solver),
                 "m0_hint": ResearchImprover("m0_hint", lambda: LLM(model=args.model), smoke, solver, TEST_HINT),
                 "m1": ResearchImprover("m1", lambda: LLM(model=args.model), smoke, solver,
                                        M1_POLICY.format(example=task_example(packs["dev"][0])), EffectCheck(solver))}
    if args.experiment == "m1":  # M1 and M0 take turns; M0's seeds 1-3 are the calibration's
        plan = [(name, s) for s in M1_SEEDS for name in ("m1", "m0") if name == "m1" or s not in SEEDS]
    else:
        plan = [("oracle", 1)] + [(name, s) for s in SEEDS for name in ("m0", "m0_hint")]
    for name, seed in plan:
        path = out / "runs" / f"{name}_s{seed}.json"
        if path.exists():
            continue
        result = reset_meta(improvers[name], a0, packs, out / "runs" / f"{name}_s{seed}", evaluate,
                            rounds=args.rounds, margin=args.margin, seed=seed)
        path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"{name} seed {seed}: test {result['test_solved']} of {result['test_tasks']}; rounds "
              f"{[(s['status'], s.get('select_gain'), s.get('kept')) for s in result['rounds']]}", flush=True)


def binomial_p(a, b):
    """Two-sided exact sign test: a tasks solved only by one side, b only by the other."""
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def report_step(args):
    out = Path(args.out_dir)
    packs = json.loads((out / "packs.json").read_text())
    a0 = agent_hash(out / "agent_a0")
    a0_test = json.loads((out / "evals" / f"{a0}_test.json").read_text())
    a0_again = json.loads((out / "evals" / f"{a0}_test_repeat.json").read_text())
    runs = {p.stem: json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))}

    def correct(run):
        return {r["id"]: r["correct"] for r in run}

    def paired(x, y):  # tasks solved only by x, only by y, sign-test p
        cx, cy = correct(x), correct(y)
        a, b = sum(cx[t] and not cy[t] for t in cx), sum(cy[t] and not cx[t] for t in cx)
        return {"only_first": a, "only_second": b, "p": round(binomial_p(a, b), 4)}
    result = {"packs": {k: len(v) for k, v in packs.items()},
              "a0": {"test": solved(a0_test), "test_again": solved(a0_again),
                     "scoring_noise": paired(a0_test, a0_again)},
              "runs": {k: {"test": v["test_solved"], "rate": round(v["test_rate"], 4),
                           "valid": sum(s["status"] == "valid" for s in v["rounds"]),
                           "kept": sum(bool(s.get("kept")) for s in v["rounds"]),
                           "rounds": v["rounds"], "improver_tokens": v["improver_usage"].get("completion_tokens")}
                       for k, v in runs.items()}}
    if "oracle_s1" in runs:
        result["oracle_vs_a0"] = paired(runs["oracle_s1"]["test_results"], a0_test)
    mig = {}
    for s in SEEDS:
        good, base = runs.get(f"m0_hint_s{s}"), runs.get(f"m0_s{s}")
        if good and base:
            mig[s] = {"mig": good["test_solved"] - base["test_solved"],
                      **paired(good["test_results"], base["test_results"])}
    result["mig_hint_vs_m0"] = mig
    m0 = [runs[f"m0_s{s}"]["test_solved"] for s in SEEDS if f"m0_s{s}" in runs]
    if len(m0) > 1:
        result["m0_spread"] = {"test_solved": m0, "sd": round(statistics.stdev(m0), 2)}
    Path(args.out or ROOT / "results" / "meta_rsi_calibration.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=1), flush=True)


def sign_flip(d):
    """Exact two-sided sign-flip test that the mean of the paired differences `d` is 0."""
    obs = abs(sum(d))
    hits = sum(abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-9
               for signs in itertools.product((1, -1), repeat=len(d)))
    return hits / 2 ** len(d)


def report_m1(args):
    out = Path(args.out_dir)
    runs = {p.stem: json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))}
    pairs = {s: (runs[f"m1_s{s}"], runs[f"m0_s{s}"]) for s in M1_SEEDS if f"m1_s{s}" in runs and f"m0_s{s}" in runs}
    mig = {s: m1["test_solved"] - m0["test_solved"] for s, (m1, m0) in pairs.items()}

    def per_side(f):
        return {"m1": [f(m1) for m1, _ in pairs.values()], "m0": [f(m0) for _, m0 in pairs.values()]}
    result = {"pairs": len(pairs), "test_solved": per_side(lambda r: r["test_solved"]), "mig": mig,
              "mean_mig": statistics.mean(mig.values()) if mig else None,
              "sign_flip_p": sign_flip(list(mig.values())) if mig else None,
              "kept_a_change": per_side(lambda r: any(s.get("kept") for s in r["rounds"])),
              "valid_proposals": per_side(lambda r: sum(s["status"] == "valid" for s in r["rounds"])),
              "effect_fixed": [s["checks"].get("effect_fixed") for m1, _ in pairs.values() for s in m1["rounds"]
                               if s.get("checks")],
              "improver_tokens": per_side(lambda r: r["improver_usage"].get("completion_tokens")),
              "rounds": {k: v["rounds"] for k, v in runs.items() if k.startswith(("m1_", "m0_"))}}
    Path(args.out or ROOT / "results" / "meta_rsi_m1.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "rounds"}, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("packs", "run", "report"))
    p.add_argument("--experiment", choices=("calibration", "m1"), default="calibration")
    p.add_argument("--out-dir", default=os.path.expanduser("~/alter/meta-rsi/calibration"))
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--rounds", type=int, default=4, help="proposals per improver run")
    p.add_argument("--margin", type=int, default=3, help="more selection tasks solved for a change to be kept")
    p.add_argument("--out", default=None, help="report: the result file (default results/meta_rsi_<experiment>.json)")
    args = p.parse_args()
    if args.step == "report" and args.experiment == "m1":
        report_m1(args)
    else:
        {"packs": packs_step, "run": run_step, "report": report_step}[args.step](args)


if __name__ == "__main__":
    main()
