"""Tables for the research ability test (scripts/research_ability.py, docs/phase-c-design.md section 8): the initial
agent's runs, one row per researcher, the same 16 situations compared pairwise, and every measured change.

  python scripts/research_ability_report.py [results/research_ability.json] [evals dir] [--against other.json]

--against compares each researcher with the same researcher in another result file, situation by situation (the
guarded rerun against the first test).
"""
import argparse
import itertools
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("math", "code", "rtl")


def pp(x):
    return "—" if x is None else f"{100 * x:+.1f}"


def effect(row):
    """The difference on the domains the change was judged on. A change that was not run changes nothing: 0."""
    return row["verdict"]["diff"] if "verdict" in row else 0.0


def macro(row):
    """The same, as a change of the agent's domain-balanced accuracy: a change judged on one domain moves it by a
    third of its difference there."""
    return effect(row) * len(row["verdict"]["domains"]) / len(DOMAINS) if "verdict" in row else 0.0


def worse(row, alpha=0.05):
    return "verdict" in row and row["verdict"]["p_worse"] < alpha


def better(row):
    return "verdict" in row and row["verdict"]["pass_dev"] and row["verdict"]["pass_regression"]


def evidence(row):
    return bool((row.get("checks") or {}).get("diagnose_evidence_matches_records"))


def fisher(a, b, c, d):
    """Two-sided Fisher exact test on [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    total = math.comb(n1 + n2, k)
    prob = {x: math.comb(n1, x) * math.comb(n2, k - x) / total for x in range(max(0, k - n2), min(k, n1) + 1)}
    return sum(p for p in prob.values() if p <= prob[a] * (1 + 1e-9))


def sign_flip(d):
    """Exact two-sided paired sign-flip test of mean(d) = 0."""
    obs = abs(sum(d))
    hits = sum(abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-12
               for signs in itertools.product((1, -1), repeat=len(d)))
    return hits / 2 ** len(d)


def mean_se(values):
    se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else float("nan")
    return f"{pp(statistics.mean(values))} ± {100 * se:.1f}"


def compare(name_a, rows_a, name_b, rows_b):
    """Two sets of research steps over the same situations: counts (Fisher exact tests) and the per-situation
    difference of their effects (exact paired sign-flip tests)."""
    n, other = len(rows_a), {x["id"]: x for x in rows_b}
    parts = []
    for k, f in (("valid", lambda x: x["status"] == "valid"), ("evidence", evidence), ("better", better),
                 ("worse", worse)):
        ca, cb = sum(map(f, rows_a)), sum(map(f, rows_b))
        parts.append(f"{k} {ca} vs {cb} (Fisher p={fisher(ca, n - ca, cb, n - cb):.4f})")
    for k, f in (("effect/step", effect), ("macro effect/step", macro)):
        d = [f(x) - f(other[x["id"]]) for x in rows_a]
        parts.append(f"{k} {pp(statistics.mean(d))} pp (paired sign-flip p={sign_flip(d):.3f})")
    return f"{name_a} vs {name_b}: " + "; ".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="?", default=str(ROOT / "results" / "research_ability.json"))
    p.add_argument("evals", nargs="?", default=os.path.expanduser("~/alter/evo/runs/research-ability/evals"))
    p.add_argument("--against", help="another result file with the same situations")
    args = p.parse_args()
    res, evals = json.loads(Path(args.results).read_text()), Path(args.evals)
    researchers = res["researchers"]
    rows = {r: sorted(res["proposals"][r], key=lambda x: x["id"]) for r in researchers}

    print("## initial agent runs")
    for f in sorted(evals.glob("seed_*.json")):
        ok, n = Counter(), Counter()
        for x in json.loads(f.read_text()):
            ok[x["domain"]] += x["correct"]
            n[x["domain"]] += 1
        print(f.stem, f"{sum(ok.values())}/{sum(n.values())} = {100 * sum(ok.values()) / sum(n.values()):.1f}%",
              " ".join(f"{d} {ok[d]}/{n[d]}" for d in DOMAINS))

    print("\n## per researcher")
    print("| researcher | valid | evidence matches | measured | mean effect/step (pp) ± SE | macro effect/step ± SE "
          "| mean effect measured | better | worse | prediction hit/miss/none | median s | median out tokens |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in researchers:
        s = res["summary"][r]
        hits = Counter(str(x["prediction"]["hit"]) for x in rows[r] if "verdict" in x)
        print(f"| {r} | {s['status'].get('valid', 0)}/{s['steps']} | {s['evidence_matches_records']} "
              f"| {s['measured']} | {mean_se([effect(x) for x in rows[r]])} | {mean_se([macro(x) for x in rows[r]])} "
              f"| {pp(s['mean_diff_measured'])} | {s['better']} | {s['worse']} "
              f"| {hits.get('True', 0)}/{hits.get('False', 0)}/{hits.get('None', 0)} "
              f"| {s['median_seconds']} | {s['median_completion_tokens']} |")

    print("\n## mean effect per step by assigned problem (pp; changes run in brackets)")
    problems = sorted({tuple(x["assigned"]) for r in researchers for x in rows[r]})
    print("| researcher | " + " | ".join("/".join(p) for p in problems) + " |")
    print("|---|" + "---|" * len(problems))
    for r in researchers:
        by = defaultdict(list)
        for x in rows[r]:
            by[tuple(x["assigned"])].append(x)
        print(f"| {r} | " + " | ".join(f"{pp(statistics.mean(effect(x) for x in by[p]))} "
                                        f"({sum('verdict' in x for x in by[p])})" for p in problems) + " |")

    print("\n## pairwise (the same situations)")
    for a, b in itertools.combinations(researchers, 2):
        print(compare(a, rows[a], b, rows[b]))
    if args.against:
        other = json.loads(Path(args.against).read_text())
        print(f"\n## against {args.against}")
        for r in (r for r in researchers if r in other["proposals"]):
            print(compare(f"{r} here", rows[r], f"{r} there", other["proposals"][r]))

    print("\n## measured changes")
    print("| researcher | id | problem | module | judged on | diff (pp) | p | p_worse | lost | gates | 2nd run "
          "| prediction |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in researchers:
        for x in (x for x in rows[r] if "verdict" in x):
            v, p = x["verdict"], x["prediction"]
            gates = f"dev {('fail', 'pass')[v['pass_dev']]}, regression {('fail', 'pass')[v['pass_regression']]}"
            second = "—" if "first_run" not in v else f"first {pp(v['first_run']['diff'])}, p={v['first_run']['p']:.3f}"
            print(f"| {r} | {x['id']} | {'/'.join(x['assigned'])} | {x['module']} | {','.join(v['domains'])} "
                  f"| {pp(v['diff'])} | {v['p']:.3f} | {v['p_worse']:.3f} | {100 * v['regression']:.1f}% | {gates} "
                  f"| {second} | {p['metric']} {p['direction']}: {p['hit']} |")
    print("\nsolver usage:", res.get("solver_usage"))


if __name__ == "__main__":
    main()
