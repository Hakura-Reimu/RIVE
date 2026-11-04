"""After C5 (docs/phase-c-design.md section 8): runs C5 candidates again, interleaved with the seed, on chosen domains
of the development set, to see whether a gain seen once comes back. Copies the candidates out of the C5 run directory
and writes only to its own run directory and output. Needs the vLLM server (scripts/serve.sh).

  python scripts/c5_replicate.py --candidates g09c036 --domains code,rtl --rounds 3

- The order alternates every round (the seed first, then the candidates first), so slow drift in the server's state
  does not favour one side.
- Per domain, each candidate is compared with the seed run of the same round: one-sided paired bootstrap over the
  items of all rounds together. Fixed before the runs: a domain's gain counts as replicated at p < 0.05 divided by
  the number of domains tested.
- The seed's runs in C5 (one per generation) are listed per domain for reference.
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

from rive.archive.db import Archive  # noqa: E402
from rive.benchmark.hard import load_tasks  # noqa: E402
from rive.benchmark.stats import paired_bootstrap  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.runtime.workflow import run_workflow_tasks  # noqa: E402


def accuracy(results, domain):
    rs = [r for r in results if r["domain"] == domain]
    return sum(r["correct"] for r in rs) / len(rs)


def c5_seed_runs(c5_dir, domains):
    archive = Archive(c5_dir / "archive.sqlite")
    evals = archive.db.execute("SELECT id FROM evaluations WHERE candidate_id = 'seed' AND split = 'dev' "
                               "AND task_set LIKE 'gen-%' ORDER BY id").fetchall()
    runs = [archive.item_results(e["id"]) for e in evals]
    return {d: [accuracy(r, d) for r in runs] for d in domains}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c5-dir", default=os.path.expanduser("~/alter/evo/runs/c5-main"))
    p.add_argument("--run-dir", default=os.path.expanduser("~/alter/evo/runs/c5-replication"))
    p.add_argument("--candidates", default="g09c036")
    p.add_argument("--domains", default="code,rtl")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", default=str(ROOT / "results" / "c5_replication.json"))
    args = p.parse_args()
    c5_dir, run_dir = Path(args.c5_dir), Path(args.run_dir)
    names = ["seed", *args.candidates.split(",")]
    domains = tuple(args.domains.split(","))
    for name in names:
        dst = run_dir / "candidates" / name
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(c5_dir / "candidates" / name, dst)

    tasks = load_tasks(domains, ("dev",))
    llm = LLM()
    runs = {n: [] for n in names}
    for k in range(args.rounds):
        for name in names if k % 2 == 0 else names[::-1]:
            start = time.perf_counter()
            res = run_workflow_tasks(tasks, run_dir / "candidates" / name, llm, workers=args.workers)
            (run_dir / f"{name}_round{k + 1}.json").write_text(json.dumps(res, ensure_ascii=False))
            runs[name].append(res)
            print(f"round {k + 1} {name}: {time.perf_counter() - start:.0f} s, "
                  + ", ".join(f"{d} {accuracy(res, d):.3f}" for d in domains), flush=True)

    alpha = 0.05 / len(domains)
    seed_c5 = c5_seed_runs(c5_dir, domains)
    out = {"domains": domains, "rounds": args.rounds, "alpha_per_domain": alpha,
           "seed": {d: {"runs": [accuracy(r, d) for r in runs["seed"]],
                        "c5_runs": seed_c5[d], "c5_mean": statistics.mean(seed_c5[d]),
                        "c5_sd": statistics.stdev(seed_c5[d])} for d in domains},
           "candidates": {}, "llm_usage": llm.usage}
    for name in names[1:]:
        out["candidates"][name] = {}
        for d in domains:
            seed_items, child_items = [], []
            for s_run, c_run in zip(runs["seed"], runs[name]):
                by_id = {r["id"]: r for r in c_run}
                for r in s_run:
                    if r["domain"] == d:
                        seed_items.append(int(r["correct"]))
                        child_items.append(int(by_id[r["id"]]["correct"]))
            diff, p_value = paired_bootstrap(seed_items, child_items)
            out["candidates"][name][d] = {"runs": [accuracy(r, d) for r in runs[name]], "diff": diff, "p": p_value,
                                          "replicated": diff > 0 and p_value < alpha}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
