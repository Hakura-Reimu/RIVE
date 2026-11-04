"""Live check of the research step as changed after C5 (docs/phase-c-design.md section 8), before the next run.
Needs the vLLM server; nothing is evaluated, and the C5 run directory is only read.

1. Preview: one generation's four research steps from the initial candidate on its last C5 feedback, with problems
   assigned as PhaseC.research assigns them and no blocked kinds (a new run has none yet).
2. Tool plans: the three fixed tool plans of scripts/c5_research_check.py, with two seeds each, now also through
   the checks against copies of the example tool and against reading result fields a tool does not return.
3. REFLECT: the rule's conclusion and the model's explanation for one previewed child. Its own results are not
   measured here, so the seed's stand in for both sides and the conclusion is "inconclusive".

  python scripts/c5b_research_check.py --out results/c5b_research_check.json
"""
import argparse
import json
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c5_research_check import PLANS, Forced, summary  # noqa: E402
from rive.archive.db import Archive  # noqa: E402
from rive.benchmark.hard import load_tasks  # noqa: E402
from rive.benchmark.leakscan import LeakScanner  # noqa: E402
from rive.evolution.controller import Settings, judge  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.research.agent_loop import AgentResearch, affected_domains, assign_problem  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c5", default=os.path.expanduser("~/alter/evo/runs/c5-main"))
    p.add_argument("--work", default=os.path.expanduser("~/alter/evo/runs/c5b-research-check"))
    p.add_argument("--out", default=str(ROOT / "results" / "c5b_research_check.json"))
    p.add_argument("--seeds", type=int, default=2)
    args = p.parse_args()
    c5, work = Path(args.c5), Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    shutil.copy(c5 / "archive.sqlite", work / "archive.sqlite")  # the C5 archive itself is not opened
    archive = Archive(work / "archive.sqlite")

    domains = ("math", "code", "rtl")
    dev = load_tasks(domains, ("dev",))
    tasks_by_id = {t["id"]: t for t in dev}
    scanner = LeakScanner().add_hard_tasks(load_tasks(domains, ("selection", "anchor", "sealed")), dev)
    smoke = [min((t for t in dev if t["domain"] == d), key=lambda t: t["id"]) for d in domains]
    seed_dir = Path(archive.candidate("seed")["dir"])
    champion = json.loads(sorted((c5 / "evals" / "seed").glob("dev_*.json"))[-1].read_text())
    llm = LLM()

    taken = []
    for n in range(1, 5):
        taken.append(assign_problem(champion, [], taken, random.Random(n)))

    def preview(job):
        n, problem = job
        step = AgentResearch(llm, work / "preview", smoke, leak_scanner=scanner)
        child = step.propose(f"p{n:03d}", seed_dir, champion, tasks_by_id, [], seed=n, problem=problem)
        valid = child["status"] == "valid"
        return {**summary(child), "assigned": problem,
                "domains": affected_domains(child["dir"], seed_dir) if valid else None}, child, step

    with ThreadPoolExecutor(4) as pool:
        previews = list(pool.map(preview, enumerate(taken, 1)))
    print(json.dumps([r for r, _, _ in previews], indent=1, ensure_ascii=False), flush=True)

    def forced(job):
        plan, seed = job
        step = Forced(llm, work / "tools", smoke, leak_scanner=scanner, plan=plan)
        return {"plan": plan["tool_name"], "seed": seed,
                **summary(step.propose(f"{plan['tool_name']}_{seed}", seed_dir, champion, tasks_by_id, [],
                                       seed=seed))}

    jobs = [(plan, seed) for plan in PLANS for seed in range(1, args.seeds + 1)]
    with ThreadPoolExecutor(len(jobs)) as pool:
        tools = list(pool.map(forced, jobs))
    print(f"tool plans: {sum(r['status'] == 'valid' for r in tools)} of {len(jobs)} valid", flush=True)

    reflected = None
    child, step = next(((c, s) for _, c, s in previews if c["status"] == "valid"), (None, None))
    if child:
        judged = affected_domains(child["dir"], seed_dir) or list(domains)
        verdict = {**judge([champion], [champion], judged, Settings()), "alpha": Settings().alpha}
        note = step.reflect(child, verdict, [champion], [champion])
        reflected = {"id": child["id"], "verdict": child["record"]["verdict"], "note": note,
                     "reflect": child["record"]["stages"]["reflect"]}

    out = {"assigned": taken, "preview": [r for r, _, _ in previews], "tool_plans": tools,
           "valid_tool_plans": sum(r["status"] == "valid" for r in tools), "reflect": reflected,
           "llm_usage": dict(llm.usage)}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"assigned": taken, "valid_tool_plans": out["valid_tool_plans"],
                      "reflect": reflected and reflected["verdict"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
