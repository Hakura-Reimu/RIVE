"""Research loop control flow with a scripted fake model (needs docker for the router smoke test)."""
import json
import re
import shutil
from pathlib import Path

import pytest

from rive.research import schemas
from rive.research.loop import ResearchLoop, check_router_source, observe
from rive.sandbox.container import CandidateSandbox

TEMPLATE = Path(__file__).resolve().parents[1] / "candidate_template"
pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")

GOOD_ROUTER = '''import re

def select(task, tools, history=None):
    words = set(re.findall(r"[a-z]+", task.lower()))
    def score(t):
        text = (t["name"] + " " + t.get("description", "")).lower()
        return len(words & set(re.findall(r"[a-z]+", text)))
    ranked = sorted(tools, key=score, reverse=True)
    return {"tools": [t["name"] for t in ranked[:5]], "hint": ""}
'''


def tasks(n=10):
    out = []
    for i in range(n):
        out.append({"id": f"t{i}", "messages": [{"role": "user", "content": f"add numbers {i}"}],
                    "functions": [{"name": "add", "description": "add numbers", "parameters": {}},
                                  {"name": "weather", "description": "weather forecast", "parameters": {}}],
                    "ground_truth": [{"add": {}}], "irrelevance": False})
    return out


class FakeLLM:
    """Answers by stage. `script` overrides: {"diagnose_counts": int, "bad_json_first": bool, "router": [src...]}."""

    def __init__(self, counts, **script):
        self.counts, self.script, self.routers = counts, script, list(script.get("router", [GOOD_ROUTER]))
        self.bad_json_left = 1 if script.get("bad_json_first") else 0

    def chat(self, messages, schema=None, max_tokens=0, temperature=0.0, thinking=False, seed=0):
        last = " ".join(m["content"] for m in messages if m["role"] == "user")
        if schema is None and "IMPLEMENT" in last and "Do not output JSON" not in messages[-1]["content"]:
            src = self.routers.pop(0) if len(self.routers) > 1 else self.routers[0]
            return f"```python\n{src}```"
        if schema is None:
            return "Some analysis."
        if self.bad_json_left:
            self.bad_json_left -= 1
            return "{not json"
        if schema is schemas.DIAGNOSE:
            n = self.script.get("diagnose_counts", self.counts["wrong_tool"])
            shown = re.findall(r'"id": "(t\d+)", "failure": "wrong_tool"', last)
            return json.dumps({"selected_problem": "wrong_tool",
                               "evidence": [{"failure_type": "wrong_tool", "count": n, "example_ids": shown[:1]}],
                               "suspected_cause": "tool_router", "explanation": "too many tools",
                               "recommended_next_action": "filter tools"})
        if schema is schemas.HYPOTHESIZE:
            h = {"hypothesis": "keyword filter", "target_file": "agent/tool_router.py", "proposed_change": "rank",
                 "prediction": {"metric": "accuracy", "direction": "increase"}, "control": "identity router",
                 "potential_regression": "may drop the right tool"}
            return json.dumps({"hypotheses": [{"hypothesis_id": "H1", **h}, {"hypothesis_id": "H2", **h}],
                               "selected": "H1", "why_selected": "cheap"})
        if schema is schemas.DESIGN:
            return json.dumps({"target_file": "agent/tool_router.py", "change_steps": ["rank by overlap"],
                               "primary_metric": "accuracy", "expected_direction": "increase",
                               "rollback_condition": "accuracy drops"})
        if schema is schemas.REFLECT:
            return json.dumps({"conclusion": "supported", "summary": "better", "limitations": "small sample",
                               "next_direction": "tune k", "reusable_finding": "filtering helps"})
        raise AssertionError("unexpected schema")


def make_loop(tmp_path, llm, ts):
    def evaluate(candidate_dir):
        improved = "re.findall" in (Path(candidate_dir) / "agent/tool_router.py").read_text()
        return [{"id": t["id"], "correct": improved or i < 3, "failure": None if (improved or i < 3) else "wrong_tool",
                 "n_tools_shown": 2, "calls": [["weather", {}]]} for i, t in enumerate(ts)]
    return ResearchLoop(llm, evaluate, CandidateSandbox, ts[:3], tmp_path / "run")


def parent(tmp_path):
    shutil.copytree(TEMPLATE, tmp_path / "parent")
    return tmp_path / "parent"


def parent_results(ts):
    return [{"id": t["id"], "correct": i < 3, "failure": None if i < 3 else "wrong_tool", "n_tools_shown": 2,
             "calls": [["weather", {}]]} for i, t in enumerate(ts)]


def test_full_cycle_records_checks_and_improvement(tmp_path):
    ts = tasks()
    loop = make_loop(tmp_path, FakeLLM({"wrong_tool": 7}), ts)
    rec = loop.run_cycle(1, parent(tmp_path), parent_results(ts), {t["id"]: t for t in ts}, [])
    assert rec["status"] == "evaluated", rec
    assert rec["checks"] == {"diagnose_evidence_matches_records": True, "hypothesis_testable": True,
                             "patch_valid": True, "improved_on_dev": True}
    assert rec["metrics"]["parent_accuracy"] == 0.3 and rec["metrics"]["child_accuracy"] == 1.0
    assert (tmp_path / "run" / "cycle_001" / "record.json").exists()


def test_made_up_evidence_is_flagged(tmp_path):
    ts = tasks()
    loop = make_loop(tmp_path, FakeLLM({"wrong_tool": 7}, diagnose_counts=99), ts)
    rec = loop.run_cycle(1, parent(tmp_path), parent_results(ts), {t["id"]: t for t in ts}, [])
    assert rec["checks"]["diagnose_evidence_matches_records"] is False


def test_forbidden_import_is_repaired_and_bad_json_retried(tmp_path):
    ts = tasks()
    bad = "import os\n" + GOOD_ROUTER
    loop = make_loop(tmp_path, FakeLLM({"wrong_tool": 7}, router=[bad, GOOD_ROUTER], bad_json_first=True), ts)
    rec = loop.run_cycle(1, parent(tmp_path), parent_results(ts), {t["id"]: t for t in ts}, [])
    assert rec["stages"]["implement"]["attempts"] == 2 and rec["checks"]["patch_valid"]
    assert rec["stages"]["diagnose"]["attempts"] == 2


def test_duplicate_code_is_not_evaluated_twice(tmp_path):
    ts = tasks()
    loop = make_loop(tmp_path, FakeLLM({"wrong_tool": 7}), ts)
    p = parent(tmp_path)
    assert loop.run_cycle(1, p, parent_results(ts), {t["id"]: t for t in ts}, [])["status"] == "evaluated"
    rec = loop.run_cycle(2, p, parent_results(ts), {t["id"]: t for t in ts}, [])
    assert rec["status"] == "duplicate"
    assert "not evaluated" in rec["note"] and "same router" in rec["note"]


def test_observe_reports_hidden_tools_and_rotates_examples():
    ts = tasks()
    results = [{"id": t["id"], "correct": i < 3, "failure": None if i < 3 else "no_call", "n_tools_shown": 1,
                "shown": ["add"] if i < 5 else ["weather"], "calls": []} for i, t in enumerate(ts)]
    fb = observe(results, {t["id"]: t for t in ts}, seed=1)
    assert fb["failure_counts"] == {"no_call": 7} and "no_call" in fb["failure_meanings"]
    assert fb["router_recall"]["tasks_with_an_expected_tool_hidden"] == 5
    assert fb["router_recall"]["of_which_failed"] == 5
    assert all(ex["expected_tools_shown"] == (int(ex["id"][1:]) < 5) for ex in fb["failure_examples"])
    picks = {tuple(ex["id"] for ex in observe(results, {t["id"]: t for t in ts}, seed=s)["failure_examples"])
             for s in range(6)}
    assert len(picks) > 1


def test_static_checks():
    assert check_router_source(GOOD_ROUTER) == []
    assert any("forbidden import" in p for p in check_router_source("import subprocess\n" + GOOD_ROUTER))
    assert any("forbidden call" in p for p in check_router_source(GOOD_ROUTER + "\nx = eval('1')\n"))
    assert any("no top-level function" in p for p in check_router_source("def other():\n    pass\n"))
