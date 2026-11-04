"""Checks the research-step changes made after C5 generation 3 (docs/phase-c-design.md section 8) on the live model,
before the run continues. Needs the vLLM server; nothing is evaluated, and the C5 run directory is only read.

1. Preview: four research steps from the C5 champion (seeds 14-17, those of generation 4's children) with the
   blocked kinds computed from the C5 archive. Shows what the blocked list makes the model work on and whether the
   changes pass validation.
2. Tool template: three fixed tool plans (maths, code, RTL), each implemented with three seeds with and without the
   example tool in the request. In the first three C5 generations 0 of 7 tools passed validation.

  python scripts/c5_research_check.py --c5 ~/alter/evo/runs/c5-main --out results/c5_research_check.json

The first run (results/c5_research_check.json) found that 3 of the 4 previewed steps only raised max_tokens above
the per-call cap; after the check for that, the preview was run again on its own:

  python scripts/c5_research_check.py --parts preview --out results/c5_research_check_preview2.json
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.archive.db import Archive  # noqa: E402
from rive.benchmark.hard import load_tasks  # noqa: E402
from rive.benchmark.leakscan import LeakScanner  # noqa: E402
from rive.evolution.phase_c import experiment_history  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.research import agent_loop  # noqa: E402
from rive.research.agent_loop import AgentResearch, blocked_kinds  # noqa: E402

PLANS = [
    {"domain": "math", "failure": "no_final_answer", "tool_name": "python_calc",
     "hypothesis": "A tool that runs a short Python computation lets the workflow check the final number before "
                   "answering.",
     "steps": ["Create tools/python_calc.py: run(args, public) runs args['code'] with public.python_exec and returns "
               "its stdout and whether it crashed.",
               "In agent/workflow.py, for maths: after the first answer, ask the model for a short Python program "
               "that checks its final answer, run it with ctx.tool('python_calc', ...), and if the check fails ask "
               "once more for a corrected \\boxed{} answer."]},
    {"domain": "code", "failure": "runtime_error", "tool_name": "run_samples",
     "hypothesis": "Running the solution on the sample input from the statement catches crashes before submitting.",
     "steps": ["Create tools/run_samples.py: run(args, public) runs args['code'] on args['stdin'] with "
               "public.python_exec and returns whether it crashed and its output.",
               "In agent/workflow.py, for code read from stdin: take the first ```python block, run it on the first "
               "sample input of the statement with ctx.tool('run_samples', ...), and if it crashes ask the model once "
               "to fix it, giving the error."]},
    {"domain": "rtl", "failure": "compile_error", "tool_name": "verilog_compile",
     "hypothesis": "Compiling the generated design with iverilog and feeding the errors back fixes compile errors.",
     "steps": ["Create tools/verilog_compile.py: run(args, public) compiles args['design'] with public.verilog_sim and "
               "the testbench 'module tb; endmodule', and returns whether it compiled and the compiler log.",
               "In agent/workflow.py, for rtl: take the first ```verilog block, compile it with "
               "ctx.tool('verilog_compile', ...), and if it does not compile ask the model once to fix it, giving the "
               "compiler log."]},
]


class Forced(AgentResearch):
    """DIAGNOSE, HYPOTHESIZE and DESIGN return a fixed tool plan; IMPLEMENT and validation run as usual."""

    def __init__(self, *args, plan, **kw):
        super().__init__(*args, **kw)
        self.plan = plan

    def _structured(self, messages, schema, check=None):
        p, fields = self.plan, schema["properties"]
        if "selected_domain" in fields:
            return {"selected_domain": p["domain"], "selected_problem": p["failure"],
                    "evidence": [{"domain": p["domain"], "failure_type": p["failure"], "count": 1, "example_ids": []}],
                    "suspected_cause": "missing_tool", "explanation": "fixed plan",
                    "recommended_next_action": "fixed plan"}, {}
        h = {"hypothesis_id": "H1", "hypothesis": p["hypothesis"], "target": "tool", "tool_name": p["tool_name"],
             "proposed_change": " ".join(p["steps"])[:600],
             "prediction": {"metric": f"{p['domain']}_accuracy", "direction": "increase"},
             "control": "the unchanged agent", "potential_regression": "more tokens per task"}
        if "hypotheses" in fields:
            return {"hypotheses": [h, {**h, "hypothesis_id": "H2"}], "selected": "H1", "why_selected": "fixed"}, {}
        return {"target": "tool", "tool_name": p["tool_name"], "change_steps": p["steps"],
                "primary_metric": f"{p['domain']}_accuracy", "expected_direction": "increase",
                "rollback_condition": "accuracy drops"}, {}


def summary(child):
    rec = child["record"]
    impl = rec["stages"].get("implement", {})
    diag = rec["stages"].get("diagnose", {})
    return {"id": child["id"], "status": child["status"], "module": child["module"], "kind": rec.get("kind"),
            "diagnose_attempts": diag.get("attempts"), "diagnose_errors": diag.get("errors"),
            "hypothesis": rec.get("hypothesis", {}).get("hypothesis"), "implement_attempts": impl.get("attempts"),
            "problems": impl.get("problems"), "error": rec.get("error"), "note": child.get("note")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c5", default=os.path.expanduser("~/alter/evo/runs/c5-main"))
    p.add_argument("--work", default=os.path.expanduser("~/alter/evo/runs/c5-research-check"))
    p.add_argument("--out", default=str(ROOT / "results" / "c5_research_check.json"))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--parts", default="preview,template")
    args = p.parse_args()
    parts = args.parts.split(",")
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
    notes = [c["note"] for c in archive.candidates() if c["note"]][-15:]
    blocked = blocked_kinds(experiment_history(archive))
    hashes = {c["code_hash"] for c in archive.candidates() if c["code_hash"]}
    llm = LLM()

    def preview(seed):
        step = AgentResearch(llm, work / "preview", smoke, leak_scanner=scanner)
        step.seen_hashes |= hashes
        return summary(step.propose(f"p{seed:03d}", seed_dir, champion, tasks_by_id, notes, seed=seed,
                                    blocked=blocked))

    previews = []
    if "preview" in parts:
        with ThreadPoolExecutor(4) as pool:
            previews = list(pool.map(preview, range(14, 18)))
        print(json.dumps(previews, indent=1, ensure_ascii=False), flush=True)

    def forced(job):
        plan, seed, variant = job
        step = Forced(llm, work / variant, smoke, leak_scanner=scanner, plan=plan)
        return {"plan": plan["tool_name"], "seed": seed, "variant": variant,
                **summary(step.propose(f"{plan['tool_name']}_{seed}", seed_dir, champion, tasks_by_id, [],
                                       seed=seed))}

    template = {}
    guide = agent_loop.TOOL_GUIDE
    for variant in ("with_example", "without_example") if "template" in parts else ():
        agent_loop.TOOL_GUIDE = guide if variant == "with_example" else ""
        jobs = [(plan, seed, variant) for plan in PLANS for seed in range(1, args.seeds + 1)]
        with ThreadPoolExecutor(len(jobs)) as pool:
            template[variant] = list(pool.map(forced, jobs))
        valid = sum(r["status"] == "valid" for r in template[variant])
        print(f"{variant}: {valid} of {len(jobs)} tools valid", flush=True)
    agent_loop.TOOL_GUIDE = guide

    out = {"blocked": blocked, "preview": previews, "tool_template": template,
           "valid_tools": {v: sum(r["status"] == "valid" for r in rs) for v, rs in template.items()},
           "llm_usage": dict(llm.usage)}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["valid_tools"]), flush=True)


if __name__ == "__main__":
    main()
