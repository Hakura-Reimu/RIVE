"""Summarize a Phase A research run: why evidence checks failed, what each cycle tried, how failures moved.

  python scripts/a3_analyze.py ~/alter/evo/runs/a3-main
"""
import json
import sys
from pathlib import Path


def evidence_problems(rec):
    """Re-run the DIAGNOSE fact check and say which claim was wrong."""
    obs = rec["stages"]["observe"]
    counts = obs["failure_counts"]
    shown = {}
    for ex in obs["failure_examples"]:
        shown.setdefault(ex["failure"], set()).add(ex["id"])
    diag = rec["stages"]["diagnose"]["output"]
    out = []
    if counts.get(diag["selected_problem"], 0) == 0:
        out.append(f"selected '{diag['selected_problem']}' has no failures")
    for e in diag["evidence"]:
        real = counts.get(e["failure_type"], 0)
        if e["count"] != real:
            out.append(f"{e['failure_type']} count {e['count']} (real {real})")
        bad = set(e["example_ids"]) - shown.get(e["failure_type"], set())
        if bad:
            out.append(f"{e['failure_type']} ids not shown for it: {sorted(bad)}")
    return out


def main():
    run = Path(sys.argv[1]).expanduser()
    kinds = {"count": 0, "ids": 0, "selected": 0}
    for d in sorted(run.glob("cycle_*")):
        rec = json.loads((d / "record.json").read_text())
        m = rec.get("metrics", {})
        hyp = rec["stages"].get("hypothesize", {}).get("output", {})
        chosen = next((h for h in hyp.get("hypotheses", []) if h["hypothesis_id"] == hyp.get("selected")), {})
        print(f"== cycle {rec['cycle']} {rec['status']}  dev {m.get('parent_accuracy')} -> {m.get('child_accuracy')}"
              f"  p={m.get('p_one_sided')}")
        print(f"   problem: {rec['stages'].get('diagnose', {}).get('output', {}).get('selected_problem')}"
              f" | hypothesis: {chosen.get('hypothesis', '')[:150]}")
        if "diagnose" in rec["stages"] and not rec["checks"].get("diagnose_evidence_matches_records"):
            probs = evidence_problems(rec)
            for p in probs:
                kinds["count" if " count " in p else "ids" if " ids " in p else "selected"] += 1
            print(f"   evidence: {'; '.join(probs)}")
        if "parent_failures" in m:
            keys = sorted(set(m["parent_failures"]) | set(m["child_failures"]))
            print("   failures: " + ", ".join(f"{k} {m['parent_failures'].get(k, 0)}->{m['child_failures'].get(k, 0)}"
                                             for k in keys))
        if rec["status"] == "invalid_patch":
            print(f"   patch: {rec['stages'].get('implement', {}).get('problems')}")
    print("evidence problem kinds:", kinds)


if __name__ == "__main__":
    main()
