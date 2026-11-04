"""Research ability test (docs/phase-c-design.md section 8, after C5): the same research situations given to
different researcher models, and the changes they make measured on one scale. The user asked whether, as the model
improves, its ability to improve itself improves too. This is the instrument for it: first across sizes of one
model family (Qwen3.5 0.8B, 2B and 4B), later for an improved version of the 4B.

- Situations: the initial agent and its last development results from C5 as the feedback, the four problems with the
  most failing tasks there, four seeds each: 16 research steps per researcher. Each has its problem assigned and no
  history or blocked list, so every researcher meets exactly the same situations.
- Measure: every change that passes validation is run by the same solver (Qwen3.5-4B) on the domains it can reach,
  and judged with the development gates (evolution.controller.judge) against the initial agent's runs made in the
  same session, spread over it. A change that passes both gates runs a second time. Duplicates of a researcher's
  own earlier change are not run again.

  python scripts/research_ability.py propose --researcher qwen3.5-0.8b    (vLLM serving that model)
  python scripts/research_ability.py evaluate                              (vLLM serving qwen3.5-4b)

Both steps can be started again after an interruption: finished proposals and runs are kept in --out-dir.

--guards (after the first test, 2026-10-30) adds the research step's two checks: a change must leave the other
domains' tasks as they were (agent_loop.behaviour), and it first answers a few development tasks the initial agent
solves (the pre-screen). The pre-screen needs the solver on the server too: the researcher is the solver itself,
or, in Phase D, an adapter vLLM serves next to it (--researcher seed1 --solver qwen3.5-4b).
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.benchmark.hard import load_tasks  # noqa: E402
from rive.benchmark.leakscan import LeakScanner  # noqa: E402
from rive.evolution.controller import Settings, judge  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.research.agent_loop import UNFIXABLE, AgentResearch, affected_domains, prediction_check  # noqa: E402
from rive.runtime.workflow import run_workflow_tasks  # noqa: E402

DOMAINS = ("math", "code", "rtl")
PROBLEMS, SEEDS = 4, 4


def setup(c5_dir, out_dir):
    """The initial agent and its last C5 development results, copied once into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = out_dir / "seed"
    if not seed.exists():
        shutil.copytree(c5_dir / "candidates" / "seed", seed)
    feedback = out_dir / "feedback.json"
    if not feedback.exists():
        shutil.copy(sorted((c5_dir / "evals" / "seed").glob("dev_*.json"))[-1], feedback)
    return seed, json.loads(feedback.read_text())


def situations(feedback):
    """[(proposal id, research seed, assigned problem)], the same for every researcher."""
    counts = Counter((r["domain"], r["failure"]) for r in feedback if r["failure"] and r["failure"] not in UNFIXABLE)
    problems = [k for k, _ in counts.most_common(PROBLEMS)]
    return [(f"r{n:02d}", n, p) for n, p in enumerate((p for p in problems for _ in range(SEEDS)), 1)]


def propose(args):
    out = Path(args.out_dir)
    seed_dir, feedback = setup(Path(args.c5), out)
    dev = load_tasks(DOMAINS, ("dev",))
    tasks_by_id = {t["id"]: t for t in dev}
    scanner = LeakScanner().add_hard_tasks(load_tasks(DOMAINS, ("selection", "anchor", "sealed")), dev)
    smoke = [min((t for t in dev if t["domain"] == d), key=lambda t: t["id"]) for d in DOMAINS]
    work = out / args.researcher
    work.mkdir(parents=True, exist_ok=True)
    path = work / "proposals.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    lock = threading.Lock()

    def one(job):
        pid, seed, problem = job
        if pid in done:
            return
        llm = LLM(model=args.researcher)  # one client per step, so its token use is its own
        solver = LLM(model=args.solver) if args.guards else None
        start = time.perf_counter()
        try:
            child = AgentResearch(llm, work / "candidates", smoke, leak_scanner=scanner, locality=args.guards,
                                  solver=solver).propose(pid, seed_dir, feedback, tasks_by_id, [], seed=seed,
                                                         problem=problem)
        except Exception as e:  # noqa: BLE001 - a failure outside the research step (sandbox, server) is kept apart
            child = {"status": "error", "module": None, "dir": None, "record": {"stages": {}, "error": repr(e)[:500]}}
        rec, valid, domains = child["record"], child["status"] == "valid", None
        if valid:  # the parent is the initial agent, the champion here: a change kept to its domain is judged there
            domains = [child["local_to"]] if child.get("local_to") else affected_domains(child["dir"], seed_dir)
        row = {"id": pid, "researcher": args.researcher, "seed": seed, "assigned": list(problem),
               "status": child["status"], "module": child["module"], "code_hash": child.get("code_hash"),
               "dir": child["dir"], "kind": rec.get("kind"), "checks": rec.get("checks"),
               "attempts": rec["stages"].get("implement", {}).get("attempts"),
               "problems": rec["stages"].get("implement", {}).get("problems"), "error": rec.get("error"),
               "hypothesis": rec.get("hypothesis"), "local_to": child.get("local_to"),
               "prescreen": rec["stages"].get("prescreen"), "domains": domains,
               "seconds": round(time.perf_counter() - start), "usage": dict(llm.usage),
               "prescreen_usage": dict(solver.usage) if solver else None}
        with lock:
            done[pid] = row
            path.write_text(json.dumps(done, indent=1, ensure_ascii=False))
        print(f"{args.researcher} {pid} {problem} {child['status']} {child['module']} {row['seconds']} s", flush=True)

    jobs = situations(feedback)[:args.limit or None]
    if args.guards:  # the pre-screen judges maths with math-verify, which works only on the main thread
        for job in jobs:
            one(job)
    else:
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(one, jobs))
    first = {}  # the same change made twice by one researcher is run once
    for pid in sorted(done):
        row = done[pid]
        if row["status"] == "valid":
            original = first.setdefault(row["code_hash"], pid)
            row["duplicate_of"] = original if original != pid else None
    path.write_text(json.dumps(done, indent=1, ensure_ascii=False))
    print(json.dumps(Counter(r["status"] for r in done.values())), flush=True)


def summarize(rows, alpha):
    measured = [r for r in rows if "verdict" in r]
    diffs = [r["verdict"]["diff"] for r in measured]
    # A change that is not run changes nothing: the expected effect of one research step counts it as 0.
    per_step = diffs + [0.0] * (len(rows) - len(measured))
    return {"steps": len(rows), "status": dict(Counter(r["status"] for r in rows)),
            "duplicates": sum(bool(r.get("duplicate_of")) for r in rows), "measured": len(measured),
            "evidence_matches_records": sum(bool((r.get("checks") or {}).get("diagnose_evidence_matches_records"))
                                            for r in rows),
            "mean_diff_measured": statistics.mean(diffs) if diffs else None,
            "mean_diff_per_step": statistics.mean(per_step) if per_step else None,
            "se_per_step": statistics.stdev(per_step) / len(per_step) ** 0.5 if len(per_step) > 1 else None,
            "better": sum(r["verdict"]["pass_dev"] and r["verdict"]["pass_regression"] for r in measured),
            "worse": sum(r["verdict"]["p_worse"] < alpha for r in measured),
            "prediction_hits": {str(k): v for k, v in Counter(r["prediction"]["hit"] for r in measured).items()},
            # --guards: changes rejected at the end by each check, and pre-screen runs (one per attempt that got there)
            "rejected_by_locality": sum(any("alters how other domains" in p for p in r.get("problems") or [])
                                        for r in rows),
            "rejected_by_prescreen": sum(any(p.startswith("the pre-screen") for p in r.get("problems") or [])
                                         for r in rows),
            "prescreens": sum(len(r.get("prescreen") or []) for r in rows),
            "median_seconds": statistics.median(r["seconds"] for r in rows) if rows else None,
            "median_completion_tokens":
                statistics.median(r["usage"]["completion_tokens"] for r in rows) if rows else None}


def evaluate(args):
    out, s = Path(args.out_dir), Settings()
    seed_dir, _ = setup(Path(args.c5), out)
    dev = load_tasks(DOMAINS, ("dev",))
    if args.tasks_per_domain:  # a short smoke run
        dev = [t for d in DOMAINS for t in [t for t in dev if t["domain"] == d][:args.tasks_per_domain]]
    props = {r: json.loads((out / r / "proposals.json").read_text()) for r in args.researchers.split(",")
             if (out / r / "proposals.json").exists()}
    researchers = list(props)
    ids = sorted({pid for p in props.values() for pid in p})
    queue = [props[r][pid] for pid in ids for r in researchers  # researchers take turns, so drift hits all alike
             if pid in props[r] and props[r][pid]["status"] == "valid" and not props[r][pid].get("duplicate_of")]
    evals = out / ("evals" if not args.tasks_per_domain else "evals_smoke")
    evals.mkdir(exist_ok=True)
    llm = LLM(model=args.solver)

    def run(name, cand_dir, domains):
        path = evals / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text())
        start = time.perf_counter()
        res = run_workflow_tasks([t for t in dev if t["domain"] in domains], Path(cand_dir), llm, workers=12)
        path.write_text(json.dumps(res, ensure_ascii=False))
        print(f"{name} on {','.join(domains)}: {time.perf_counter() - start:.0f} s, "
              f"{sum(r['correct'] for r in res)} of {len(res)} correct", flush=True)
        return res

    # The initial agent runs args.seed_runs times, spread over the session from its start to its end.
    seed_runs, k = [], 0
    for n in range(len(queue) + 1):
        while k < args.seed_runs and n >= len(queue) * k / max(1, args.seed_runs - 1):
            seed_runs.append(run(f"seed_{k + 1}", seed_dir, DOMAINS))
            k += 1
        if n < len(queue):
            p = queue[n]
            run(f"{p['researcher']}_{p['id']}", p["dir"], p["domains"] or DOMAINS)
    rows = {r: [dict(p) for p in props[r].values()] for r in researchers}
    for row in (row for rs in rows.values() for row in rs):
        if row["status"] != "valid" or row.get("duplicate_of"):
            continue
        domains = row["domains"] or list(DOMAINS)
        runs = [run(f"{row['researcher']}_{row['id']}", row["dir"], domains)]
        verdict = judge(seed_runs, runs, domains, s)
        if verdict["pass_dev"] and verdict["pass_regression"]:
            runs.append(run(f"{row['researcher']}_{row['id']}_2", row["dir"], domains))
            verdict = {**judge(seed_runs, runs, domains, s), "first_run": verdict}
        row["verdict"] = verdict
        row["prediction"] = prediction_check(row["hypothesis"]["prediction"], seed_runs, runs, domains)
    out_path = Path(args.out) if not args.tasks_per_domain else evals / "summary.json"
    result = {"researchers": researchers, "seed_runs": len(seed_runs),
              "situations": situations(json.loads((out / "feedback.json").read_text())),
              "summary": {r: summarize(rows[r], s.alpha) for r in researchers}, "proposals": rows,
              "solver_usage": dict(llm.usage)}
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result["summary"], indent=1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("propose", "evaluate"))
    p.add_argument("--c5", default=os.path.expanduser("~/alter/evo/runs/c5-main"))
    p.add_argument("--out-dir", default=os.path.expanduser("~/alter/evo/runs/research-ability"))
    p.add_argument("--researcher", default="qwen3.5-4b", help="propose: the served name of the researcher model")
    p.add_argument("--researchers", default="qwen3.5-0.8b,qwen3.5-2b,qwen3.5-4b", help="evaluate: whose proposals")
    p.add_argument("--workers", type=int, default=4, help="propose: research steps at a time")
    p.add_argument("--limit", type=int, default=0, help="propose: only the first N situations (a smoke run)")
    p.add_argument("--seed-runs", type=int, default=4, help="evaluate: runs of the initial agent")
    p.add_argument("--tasks-per-domain", type=int, default=0, help="evaluate: a smoke run on this many tasks")
    p.add_argument("--out", default=str(ROOT / "results" / "research_ability.json"))
    p.add_argument("--guards", action="store_true", help="propose: the locality check and the pre-screen")
    p.add_argument("--solver", default="qwen3.5-4b", help="the served name of the model the agent answers with")
    args = p.parse_args()
    if args.step == "propose" and args.guards and not {args.researcher, args.solver} <= served():
        p.error("--guards: the pre-screen needs the solver on the server as well as the researcher (Phase D serves "
                "V0 with the adapters hot-loaded)")
    (propose if args.step == "propose" else evaluate)(args)


def served(url="http://127.0.0.1:8000/v1/models"):
    """The model names the vLLM server answers to (an adapter loaded with --lora-modules has its own)."""
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return {m["id"] for m in json.load(r)["data"]}
    except OSError:
        return set()


if __name__ == "__main__":
    main()
