"""C5 analysis (docs/phase-c-design.md section 8): reads the C5 archive and writes results/c5_main_analysis.json.

  python scripts/c5_analyze.py [--run-dir ~/alter/evo/runs/c5-main]

- Two parts: generations 1-3 ran the original research step, 4-10 the changed one (blocked directions, the tool
  example, the tool-name check).
- Predictions: each hypothesis named a metric and a direction (research/schemas.py); the metric is measured for the
  child and for the champion run it was compared with, on the same items.
- Rule verdict: what REFLECT would say if the comparison decided it. Supported = significantly better and the
  predicted metric moved as predicted; refuted = significantly worse (one-sided p > 1 - alpha); otherwise
  inconclusive.
- The seed against itself: the champion, still the initial candidate, was evaluated on the same 185 items at the
  start of every generation. Its runs compared in pairs show what the gates do when nothing changed, and each child
  is also placed against the mean and spread of these runs.
"""
import argparse
import itertools
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.archive.db import Archive  # noqa: E402
from rive.evolution.controller import compare  # noqa: E402
from rive.research.agent_loop import kind_of  # noqa: E402

CHANGED_FROM = 4  # first generation with the changed research step
C5_GATES = SimpleNamespace(alpha=0.05, max_regression=0.15)  # the gates C5 ran with


def metric(rows, name):
    """A prediction metric (research/schemas.py AGENT_METRICS) over item results."""
    if name == "macro_accuracy":
        domains = sorted({r["domain"] for r in rows})
        return sum(metric(rows, f"{d}_accuracy") for d in domains) / len(domains)
    if name == "completion_tokens":
        return sum(r["tokens"] or 0 for r in rows) / len(rows)
    if name.endswith("_accuracy"):
        rs = [r for r in rows if r["domain"] == name[:-len("_accuracy")]]
        return sum(r["correct"] for r in rs) / len(rs)
    if name.endswith("_count"):
        return sum(r["failure"] == name[:-len("_count")] for r in rows)
    raise ValueError(name)


def item_rows(archive, evaluation_id):
    return [dict(r, id=r["item_id"]) for r in archive.item_results(evaluation_id)]


def reference_run(archive, child_eval):
    """The champion run a child was compared with: the reference's latest run on the task set before the child's.
    Generation 1 has two champion runs (the first start and the restart)."""
    row = archive.db.execute("SELECT id FROM evaluations WHERE candidate_id = ? AND task_set = ? AND id < ? "
                             "ORDER BY id DESC", (child_eval["reference_id"], child_eval["task_set"],
                                                  child_eval["id"])).fetchone()
    return item_rows(archive, row["id"])


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def candidate_rows(archive, s):
    recs = archive.experiment_records()
    gates = {e["detail"]["candidate"]: e["detail"] for e in archive.events("gates")}
    out = []
    for c in archive.candidates():
        if c["origin"] == "seed":
            continue
        r = recs.get(c["id"], {})
        kind = kind_of(r) or (None, None, None)  # generations 1-3 stored no "kind" of their own
        row = {"id": c["id"], "generation": c["generation"], "parent_source": c["parent_source"],
               "module": c["module"], "problem": "/".join(map(str, kind[:2])), "module_kind": kind[2],
               "status": c["status"]}
        if c["status"] == "invalid_patch":
            row["problems"] = r.get("stages", {}).get("implement", {}).get("problems")
        g = gates.get(c["id"])
        if g:
            ev = archive.evaluation(c["id"], f"gen-{c['generation']:03d}")
            child, champ = item_rows(archive, ev["id"]), reference_run(archive, ev)
            pred = r.get("hypothesis", {}).get("prediction", {})
            before, after = metric(champ, pred["metric"]), metric(child, pred["metric"])
            hit = None if after == before else (after > before) == (pred["direction"] == "increase")
            better = g["diff"] > 0 and g["p"] < s.alpha
            row.update({
                "macro": metric(child, "macro_accuracy"),
                "diff": g["diff"], "p": g["p"], "regression": g["regression"], "pass_dev": g["pass_dev"],
                "pass_regression": g["pass_regression"],
                "by_domain": {d: metric(child, f"{d}_accuracy") - metric(champ, f"{d}_accuracy")
                              for d in sorted({x["domain"] for x in champ})},
                "prediction": {"metric": pred["metric"], "direction": pred["direction"], "champion": before,
                               "child": after, "hit": hit},
                "reflect": r.get("stages", {}).get("reflect", {}).get("output", {}).get("conclusion"),
                "rule_verdict": ("supported" if better and hit else "refuted" if g["p"] > 1 - s.alpha
                                 else "inconclusive")})
        out.append(row)
    return out


def part_summary(cands):
    ev = [c for c in cands if "diff" in c]
    tools = [c for c in cands if c["module_kind"] == "tool"]
    diffs = [c["diff"] for c in ev]
    per_gen = {}
    for c in cands:
        per_gen.setdefault(c["generation"], Counter())[c["problem"]] += 1
    return {
        "candidates": len(cands),
        "evaluated": len(ev),
        "invalid": sum(c["status"] == "invalid_patch" for c in cands),
        "tools": {"tried": len(tools), "valid": sum(c["status"] != "invalid_patch" for c in tools)},
        "problems": dict(Counter(c["problem"] for c in cands).most_common()),
        "module_kinds": dict(Counter(c["module_kind"] for c in cands).most_common()),
        "most_slots_on_one_problem": {g: max(n.values()) for g, n in sorted(per_gen.items())},
        "pass_dev": sum(c["pass_dev"] for c in ev),
        "pass_regression": sum(c["pass_regression"] for c in ev),
        "diff": {"mean": statistics.mean(diffs), "median": statistics.median(diffs), "max": max(diffs),
                 "positive": sum(d > 0 for d in diffs), "below_minus_5pp": sum(d < -0.05 for d in diffs)},
        "macro": {"median": statistics.median(c["macro"] for c in ev), "max": max(c["macro"] for c in ev)},
        "prediction_hits": {str(k): v for k, v in Counter(c["prediction"]["hit"] for c in ev).items()},
        "reflect": dict(Counter(c["reflect"] for c in ev)),
        "rule_verdict": dict(Counter(c["rule_verdict"] for c in ev)),
    }


def seed_null(archive, s):
    """Every pair of the seed's development runs, both ways round, through the same comparison as the gates."""
    runs = [e for e in archive.db.execute("SELECT * FROM evaluations WHERE candidate_id = 'seed' AND split = 'dev' "
                                          "AND task_set LIKE 'gen-%' ORDER BY id")]
    rows = {r["id"]: item_rows(archive, r["id"]) for r in runs}
    items = {frozenset(x["id"] for x in v) for v in rows.values()}
    assert len(items) == 1, "the seed's runs are not on the same items"
    pairs = []
    for a, b in itertools.permutations(rows, 2):
        cmp = compare(rows[a], rows[b])
        pairs.append({**cmp, "pass_dev": cmp["diff"] > 0 and cmp["p"] < s.alpha,
                      "pass_regression": cmp["regression"] <= s.max_regression})
    diffs = [p["diff"] for p in pairs]
    regs = [p["regression"] for p in pairs]
    macros = [metric(rows[r["id"]], "macro_accuracy") for r in runs]
    return {
        "runs": [{"task_set": r["task_set"], "macro": m,
                  **{d: metric(rows[r["id"]], f"{d}_accuracy") for d in ("math", "code", "rtl")}}
                 for r, m in zip(runs, macros)],
        "macro": {"mean": statistics.mean(macros), "sd": statistics.stdev(macros), "min": min(macros),
                  "max": max(macros)},
        "pairs": len(pairs),
        "pass_dev_share": sum(p["pass_dev"] for p in pairs) / len(pairs),
        "fail_regression_share": sum(not p["pass_regression"] for p in pairs) / len(pairs),
        "diff": {"sd": statistics.pstdev(diffs), "q95": quantile(diffs, 0.95), "max": max(diffs)},
        "regression": {"median": statistics.median(regs), "q95": quantile(regs, 0.95), "max": max(regs)},
        "diffs": sorted(diffs),
    }


def usage(summary):
    """LLM use and time per generation; the counters restart with every process."""
    out, prev = [], None
    for g in summary["generations"]:
        u = g["llm_usage"]
        delta = u if prev is None or u["calls"] < prev["calls"] else {k: u[k] - prev[k] for k in u}
        out.append({"generation": g["generation"], "seconds": g["seconds"], **delta})
        prev = u
    total = {k: sum(x[k] for x in out) for k in ("seconds", "calls", "prompt_tokens", "completion_tokens")}
    return {"per_generation": out, "total": total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=os.path.expanduser("~/alter/evo/runs/c5-main"))
    p.add_argument("--out", default=str(ROOT / "results" / "c5_main_analysis.json"))
    args = p.parse_args()
    run_dir, s = Path(args.run_dir), C5_GATES
    archive = Archive(run_dir / "archive.sqlite")

    cands = candidate_rows(archive, s)
    null = seed_null(archive, s)
    m = null["macro"]
    for c in cands:
        if "diff" in c:
            # Against one champion run, that run's own noise is in every child's difference; against the mean of
            # all the seed's runs, mostly the child's own noise is left.
            c["share_of_seed_pairs_at_least_as_good"] = sum(d >= c["diff"] for d in null["diffs"]) / null["pairs"]
            c["vs_seed_runs"] = {"diff_from_mean": c["macro"] - m["mean"],
                                 "z": (c["macro"] - m["mean"]) / (m["sd"] * (1 + 1 / len(null["runs"])) ** 0.5)}
    out = {
        "promotions": [e["detail"] for e in archive.events("promotion")],
        "selection_retests": [e["detail"] for e in archive.events("selection_retest")],
        "parts": {"generations 1-3": part_summary([c for c in cands if c["generation"] < CHANGED_FROM]),
                  "generations 4-10": part_summary([c for c in cands if c["generation"] >= CHANGED_FROM])},
        "seed_against_itself": null,
        "usage": usage(json.loads((run_dir / "summary.json").read_text())),
        "candidates": cands,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    brief = {k: out[k] for k in ("promotions", "selection_retests", "parts")}
    brief["seed_against_itself"] = {k: v for k, v in null.items() if k != "diffs"}
    brief["usage"] = out["usage"]["total"]
    print(json.dumps(brief, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
