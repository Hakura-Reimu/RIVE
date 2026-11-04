"""Run Phase A research cycles against the vLLM server (scripts/serve.sh must be running).

  python scripts/a3_run_cycles.py --cycles 3 --k 16 --dev-n 200

Champion rule inside a run: a child replaces the current parent only if it beats it on the dev tasks
with one-sided p < 0.05. At the end the final champion and the original router are compared on the
selection split (paired), which is the second Phase A pass criterion.
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from rive.benchmark.leakscan import LeakScanner  # noqa: E402
from rive.benchmark.splits import split_of  # noqa: E402
from rive.benchmark.stats import paired_bootstrap  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.research.loop import ResearchLoop  # noqa: E402
from rive.runtime.toolcall import run_tasks  # noqa: E402
from rive.sandbox.container import CandidateSandbox  # noqa: E402
from rive.benchmark.toolpool import build_tasks  # noqa: E402

DATA = Path(os.path.expanduser("~/alter/data"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/alter/models/Qwen3.5-4B"))
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--dev-n", type=int, default=200, help="dev tasks used inside the loop (fixed subset)")
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--thinking", action="store_true", help="thinking mode for analysis and implementation")
    p.add_argument("--run-dir", default=os.path.expanduser(f"~/alter/evo/runs/a3-{time.strftime('%Y%m%d-%H%M%S')}"))
    return p.parse_args()


CATEGORIES = ("multiple", "parallel", "parallel_multiple", "irrelevance")


def load_items():
    # parallel added after A1: with distractors only, accuracy stayed near 80% (see docs/phase-a-design.md 2.2)
    return [json.loads(l) for c in CATEGORIES for l in open(DATA / f"bfcl_{c}.jsonl")]


def load_tasks(k, round_id, split):
    return build_tasks(load_items(), k, round_id, (split,))


def leak_scanner():
    items = load_items()
    protected = [i for i in items if split_of(i["id"]) != "dev"]
    dev_names = {f["name"] for i in items if split_of(i["id"]) == "dev" for f in i["functions"]}
    return LeakScanner(protected, dev_names)


def evaluate_on(tasks, candidate_dir, llm, tok):
    with CandidateSandbox(candidate_dir) as sb:
        return run_tasks(tasks, sb, llm, tok)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM()

    dev = load_tasks(args.k, args.round, "dev")
    dev = sorted(random.Random(0).sample(dev, min(args.dev_n, len(dev))), key=lambda t: t["id"])
    selection = load_tasks(args.k, args.round, "selection")
    tasks_by_id = {t["id"]: t for t in dev}

    base_dir = run_dir / "baseline_candidate"
    shutil.copytree(ROOT / "candidate_template", base_dir, dirs_exist_ok=True)
    start = time.perf_counter()
    parent_dir, parent_results = base_dir, evaluate_on(dev, base_dir, llm, tok)
    print(f"baseline dev accuracy {sum(r['correct'] for r in parent_results) / len(dev):.3f} "
          f"({time.perf_counter() - start:.0f}s)", flush=True)

    loop = ResearchLoop(llm, lambda d: evaluate_on(dev, d, llm, tok), CandidateSandbox, dev[:5], run_dir,
                        thinking=args.thinking, leak_scanner=leak_scanner())
    notes, records = [], []
    for c in range(1, args.cycles + 1):
        t0 = time.perf_counter()
        rec = loop.run_cycle(c, parent_dir, parent_results, tasks_by_id, notes)
        rec["seconds"] = round(time.perf_counter() - t0, 1)
        records.append(rec)
        notes.append(rec["note"])
        if rec.get("checks", {}).get("improved_on_dev"):
            parent_dir = Path(rec["child_dir"])
            parent_results = json.loads((Path(rec["child_dir"]).parent / "dev_results.json").read_text())
            rec["became_parent"] = True
        print(f"cycle {c}: {rec['status']} checks={rec.get('checks')} metrics="
              f"{ {k: rec.get('metrics', {}).get(k) for k in ('parent_accuracy', 'child_accuracy', 'p_one_sided')} } "
              f"({rec['seconds']}s)", flush=True)

    summary = {"args": vars(args), "llm_usage": llm.usage, "cycles": len(records),
               "status_counts": {s: sum(r["status"] == s for r in records) for s in {r["status"] for r in records}},
               "structured_and_testable": sum(bool(r.get("checks", {}).get("diagnose_evidence_matches_records"))
                                              and bool(r.get("checks", {}).get("hypothesis_testable"))
                                              for r in records),
               "patch_valid": sum(bool(r.get("checks", {}).get("patch_valid")) for r in records),
               "improved_on_dev": sum(bool(r.get("checks", {}).get("improved_on_dev")) for r in records)}
    if parent_dir != base_dir:
        base_sel = evaluate_on(selection, base_dir, llm, tok)
        champ_sel = evaluate_on(selection, parent_dir, llm, tok)
        diff, p = paired_bootstrap([int(r["correct"]) for r in base_sel], [int(r["correct"]) for r in champ_sel])
        summary["selection"] = {"n": len(selection), "baseline_acc": round(sum(r["correct"] for r in base_sel) /
                                                                           len(selection), 4),
                                "champion_acc": round(sum(r["correct"] for r in champ_sel) / len(selection), 4),
                                "diff": round(diff, 4), "p_one_sided": round(p, 4), "champion": str(parent_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
