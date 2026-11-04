"""Phase C runs (docs/phase-c-design.md section 8): the C4 pilot, and the C5 main run with other arguments. Needs the
vLLM server.

  python scripts/c4_pilot.py --generations 3 --run-dir ~/alter/evo/runs/c4-pilot
  python scripts/c4_pilot.py --generations 10 --sizes math=39,code=66,rtl=80 --run-dir ~/alter/evo/runs/c5-main

Started again on a run directory that already holds candidates, it continues after the last finished generation
(rive.evolution.phase_c.resume).

Refuses to start unless the C1 checks showed that the initial candidate reproduces Phase 0; otherwise a difference
found here could come from the harness rather than from evolution. scripts/c1_harness_check.py must find the same
prompts and judges, and in scripts/c1_check_baseline.py the items whose verdict changed must look like generation
noise (most verdicts agree and the changes do not lean one way).
"""
import argparse
import json
import os
import shutil
import sys
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.archive.db import Archive  # noqa: E402
from rive.benchmark.hard import load_easy_tasks, load_tasks  # noqa: E402
from rive.benchmark.leakscan import LeakScanner  # noqa: E402
from rive.evolution.controller import Evolution, Settings  # noqa: E402
from rive.evolution.phase_c import PhaseC, resume  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.research.agent_loop import AgentResearch  # noqa: E402


def sign_test(wins, losses):
    """Exact two-sided sign test on the items where exactly one side is right."""
    n = wins + losses
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0


def c1_problems(check_path, harness_path, min_same=0.85):
    for path in (check_path, harness_path):
        if not Path(path).exists():
            return [f"{path} does not exist"]
    check = json.loads(Path(check_path).read_text())
    harness = json.loads(Path(harness_path).read_text())
    problems = []
    if harness["rendered_prompt_tokens"] != harness["server_prompt_tokens"]:
        problems.append("the server counted different prompt tokens than Phase 0's prompts have")
    for d, v in check["by_domain"].items():
        h = harness["by_domain"][d]
        if h["prompt_mismatches"] or h["rejudged_same_verdict"] < 0.97 * h["n"]:
            problems.append(f"{d}: prompts or judges differ from Phase 0")
        changed = round((1 - v["same_verdict"]) * v["n"])
        net = round((v["accuracy"] - v["phase0_accuracy"]) * v["n"])
        gained, lost = (changed + net) // 2, (changed - net) // 2
        if v["same_verdict"] < min_same or sign_test(gained, lost) < 0.05:
            problems.append(f"{d}: {gained} items gained and {lost} lost against Phase 0, of {v['n']}")
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--run-dir", default=os.path.expanduser("~/alter/evo/runs/c4-pilot"))
    p.add_argument("--sizes", default="math=20,code=30,rtl=30", help="development tasks per domain and generation")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--c1-check", default=str(ROOT / "results" / "c1_baseline_check.json"))
    p.add_argument("--c1-harness", default=str(ROOT / "results" / "c1_harness_check.json"))
    args = p.parse_args()
    problems = c1_problems(args.c1_check, args.c1_harness)
    if problems:
        print("C1 check not passed, not starting:", problems, flush=True)
        sys.exit(2)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sizes = {k: int(v) for k, v in (kv.split("=") for kv in args.sizes.split(","))}
    domains = tuple(sizes)
    dev = load_tasks(domains, ("dev",))
    selection = load_tasks(domains, ("selection",))
    scanner = LeakScanner().add_hard_tasks(load_tasks(domains, ("selection", "anchor", "sealed")), dev)
    smoke = [min((t for t in dev if t["domain"] == d), key=lambda t: t["id"]) for d in domains]

    llm = LLM()
    archive = Archive(run_dir / "archive.sqlite")
    if archive.candidate("seed") is None:
        seed_dir = run_dir / "candidates" / "seed"
        shutil.copytree(ROOT / "candidate_template", seed_dir, dirs_exist_ok=True)
        archive.add_candidate("seed", None, 0, seed_dir, "seed", status="evaluated")
    research = AgentResearch(llm, run_dir / "candidates", smoke, leak_scanner=scanner, solver=llm)
    glue = PhaseC(llm, run_dir, archive, research, load_easy_tasks(50), selection, workers=args.workers)
    evo = Evolution(archive, glue.research, glue.evaluate, glue.easy_guard, glue.selection, dev, sizes, Settings(),
                    seed=0, report_dir=run_dir / "reports", reflect=glue.reflect)

    summary = {"args": vars(args), "generations": []}
    first = 1
    if len(archive.candidates()) > 1:
        if (run_dir / "summary.json").exists():
            summary = json.loads((run_dir / "summary.json").read_text())
        first = resume(archive, glue, evo, summary["generations"], dev, seed=0)
        summary.setdefault("resumed_at_generation", []).append(first)
        print(f"resuming at generation {first}", flush=True)
    for g in range(first, args.generations + 1):
        start = time.perf_counter()
        before = archive.champion()
        rep = evo.run_generation(g)
        champ_eval = archive.evaluation(before, rep["task_set"])
        summary["generations"].append({
            "generation": g, "champion_before": before, "champion_after": rep["champion_after"],
            "champion_macro": champ_eval["summary"]["macro"], "champion_by_domain": champ_eval["summary"]["by_domain"],
            "candidates": [{k: c[k] for k in ("id", "parent_source", "module", "status", "macro", "vs_champion")}
                           for c in rep["candidates"]],
            "seconds": round(time.perf_counter() - start), "llm_usage": dict(llm.usage)})
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        print(json.dumps(summary["generations"][-1], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
