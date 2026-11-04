"""The development gates changed after C5 (docs/phase-c-design.md section 8), checked offline on the C5 archive;
writes results/c5_rescore.json. No model or GPU needed.

  python scripts/c5_rescore.py [--run-dir ~/alter/evo/runs/c5-main]

- Calibration: one of the initial candidate's 11 C5 runs plays a child, K of its other runs the champion's runs.
  Nothing changed between them, so every pass of the development gate is a false pass, and every stop at the
  regression gate a false stop. Both should be near alpha (5%) at every K, on all domains and on one domain. The
  noise prior comes from these same runs, so this checks the arithmetic, not how the prior generalizes.
- The same with a second run: the share of unchanged children that pass once and still pass with two runs.
- C5's candidates under the new gates: each child's C5 run against the seed's runs made before it, on the domains
  its change can reach (agent_loop.affected_domains against the seed). C5's own gate decisions are listed beside.
"""
import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.archive.db import Archive  # noqa: E402
from rive.evolution.controller import Settings, judge  # noqa: E402
from rive.research.agent_loop import affected_domains  # noqa: E402

DOMAIN_SETS = (["code", "math", "rtl"], ["math"], ["code"], ["rtl"])


def calibration(runs, s, draws=20, seed=0):
    """False passes and stops with K champion runs, for one child run and for a child run twice."""
    rng = random.Random(seed)
    out = {}
    for k in (1, 2, 3, 5, 9):
        once, twice, stops = [], [], []
        for i, child in enumerate(runs):
            others = [r for j, r in enumerate(runs) if j != i]
            for _ in range(draws):
                second, *rest = rng.sample(others, len(others))
                champion = rng.sample(rest, k)
                for domains in DOMAIN_SETS:
                    first = judge(champion, [child], domains, s)
                    once.append(first["pass_dev"])
                    stops.append(not first["pass_regression"])
                    both = first["pass_dev"] and first["pass_regression"]
                    twice.append(both and judge(champion, [child, second], domains, s)["pass_dev"])
        out[k] = {"trials": len(once), "false_pass_one_run": statistics.mean(once),
                  "false_pass_after_second_run": statistics.mean(twice),
                  "false_stop_regression": statistics.mean(stops)}
    return out


def candidates(archive, s):
    seed_dir = archive.candidate("seed")["dir"]
    seed_evals = archive.db.execute("SELECT id, created FROM evaluations WHERE candidate_id = 'seed' AND split = 'dev' "
                                    "ORDER BY id").fetchall()
    old = {e["detail"]["candidate"]: e["detail"] for e in archive.events("gates")}
    rows = []
    for cid, gate in old.items():
        ev = archive.db.execute("SELECT id, created FROM evaluations WHERE candidate_id = ? AND split = 'dev' "
                                "ORDER BY id", (cid,)).fetchone()
        child = [dict(r, id=r["item_id"]) for r in archive.item_results(ev["id"])]
        champion = [[dict(r, id=r["item_id"]) for r in archive.item_results(e["id"])] for e in seed_evals
                    if e["created"] < ev["created"]]
        domains = affected_domains(archive.candidate(cid)["dir"], seed_dir) or ["code", "math", "rtl"]
        new = judge(champion, [child], domains, s)
        rows.append({"id": cid, "domains": domains, "champion_runs": len(champion),
                     "old": {k: gate[k] for k in ("diff", "p", "regression", "pass_dev", "pass_regression")},
                     "new": {k: new[k] for k in ("diff", "p", "p_worse", "sd", "regression", "expected_regression",
                                                 "p_regression", "pass_dev", "pass_regression")},
                     "second_run_needed": new["pass_dev"] and new["pass_regression"]})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=os.path.expanduser("~/alter/evo/runs/c5-main"))
    p.add_argument("--out", default=str(ROOT / "results" / "c5_rescore.json"))
    args = p.parse_args()
    archive, s = Archive(Path(args.run_dir) / "archive.sqlite"), Settings()
    out = {"calibration": calibration(archive.runs("seed"), s), "candidates": candidates(archive, s)}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["calibration"], indent=1))
    for r in out["candidates"]:
        o, n = r["old"], r["new"]
        print(f"{r['id']} {','.join(r['domains']):14} K={r['champion_runs']:2}  old {o['diff']:+.3f} p={o['p']:.3f} "
              f"dev={o['pass_dev']!s:5} reg={o['pass_regression']!s:5} | new {n['diff']:+.3f} p={n['p']:.3f} "
              f"dev={n['pass_dev']!s:5} reg={n['pass_regression']!s:5} ({n['regression']:.2f} vs "
              f"{n['expected_regression']:.2f})")


if __name__ == "__main__":
    main()
