"""Paired comparison of two candidates from one scripts/eval_candidates.py output, overall and per category.

  python scripts/pairwise_compare.py results/phaseA_v2_sealed.json random_champ v2_champ

Tests whether the second candidate beats the first on the same items (paired bootstrap, one-sided).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.benchmark.stats import paired_bootstrap  # noqa: E402
from rive.benchmark.toolpool import category_of  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("first")
    p.add_argument("second")
    args = p.parse_args()
    candidates = json.loads(Path(args.results).read_text())["candidates"]
    a, b = candidates[args.first]["correct"], candidates[args.second]["correct"]
    if set(a) != set(b):
        raise SystemExit("the two candidates were not run on the same items")
    # Task order as evaluated, so the overall p value matches vs_first in the eval_candidates.py output.
    groups = {"all": list(a)}
    for task_id in groups["all"]:
        groups.setdefault(category_of(task_id), []).append(task_id)
    for name, ids in groups.items():
        first, second = [int(a[i]) for i in ids], [int(b[i]) for i in ids]
        diff, pv = paired_bootstrap(first, second)
        print(f"{name:18} n={len(ids):3}  {args.first}={sum(first):3}  {args.second}={sum(second):3}  "
              f"diff={diff:+.4f}  p_one_sided={pv:.4f}  "
              f"only_first={sum(x > y for x, y in zip(first, second))}  "
              f"only_second={sum(y > x for x, y in zip(first, second))}")


if __name__ == "__main__":
    main()
