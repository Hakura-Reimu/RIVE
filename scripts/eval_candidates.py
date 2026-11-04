"""Evaluate candidate directories on one split, with per-category accuracy and failure counts.

  python scripts/eval_candidates.py --split selection --out results/a3_selection_breakdown.json \
      base=~/alter/evo/runs/a3-main/baseline_candidate champ=~/alter/evo/runs/a3-main/cycle_002/candidate

The first candidate is the reference for the paired comparison. Listing the same directory twice
measures run-to-run noise of the server (batching makes outputs not fully deterministic).
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from transformers import AutoTokenizer  # noqa: E402

from a3_run_cycles import evaluate_on, load_tasks  # noqa: E402
from rive.benchmark.stats import paired_bootstrap  # noqa: E402
from rive.benchmark.toolpool import category_of  # noqa: E402
from rive.llm import LLM  # noqa: E402


def breakdown(results, tasks):
    cats = sorted({category_of(r["id"]) for r in results})
    acc = {c: round(sum(r["correct"] for r in results if category_of(r["id"]) == c) /
                    sum(category_of(r["id"]) == c for r in results), 4) for c in cats}
    # Router recall: share of tasks that need tools where every expected function was shown, so a lost
    # task can be told apart as "the router hid the tool" or "the model chose wrongly".
    needed = [(r, [next(iter(g)) for g in t["ground_truth"]]) for r, t in zip(results, tasks) if t["ground_truth"]]
    recall = sum(all(e in r["shown"] for e in exp) for r, exp in needed) / len(needed) if needed else None
    return {"accuracy": round(sum(r["correct"] for r in results) / len(results), 4), "by_category": acc,
            "failures": dict(Counter(r["failure"] for r in results if r["failure"]).most_common()),
            "mean_tools_shown": round(sum(r["n_tools_shown"] for r in results) / len(results), 2),
            "router_recall": round(recall, 4) if recall is not None else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidates", nargs="+", help="name=candidate_dir")
    p.add_argument("--split", default="selection")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--model", default=os.path.expanduser("~/alter/models/Qwen3.5-4B"))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM()
    tasks = load_tasks(args.k, args.round, args.split)
    out = {"split": args.split, "n": len(tasks), "k": args.k, "candidates": {}}
    ref = None
    for spec in args.candidates:
        name, path = spec.split("=", 1)
        t0 = time.perf_counter()
        results = evaluate_on(tasks, Path(os.path.expanduser(path)), llm, tok)
        entry = {"dir": path, **breakdown(results, tasks), "seconds": round(time.perf_counter() - t0)}
        vec = [int(r["correct"]) for r in results]
        if ref is None:
            ref = vec
        else:
            diff, pv = paired_bootstrap(ref, vec)
            entry["vs_first"] = {"diff": round(diff, 4), "p_one_sided": round(pv, 4),
                                 "changed_items": sum(a != b for a, b in zip(ref, vec))}
        entry["correct"] = {r["id"]: r["correct"] for r in results}
        out["candidates"][name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != "correct"}), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
