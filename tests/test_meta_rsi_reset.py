"""RESET-META loop and the improvers' fixed parts (rive/meta_rsi). The sandbox test needs docker (WSL)."""
import shutil
from pathlib import Path

import pytest

from rive.meta_rsi.improvers import (M1_POLICY, ORACLE_WORKFLOW, T1_CONTRACT, EffectCheck, OracleImprover,
                                       task_example)
from rive.meta_rsi.reset_meta import Evaluator, agent_hash, reset_meta
from rive.research.loop import check_python_source
from rive.sandbox.container import CandidateSandbox

TEMPLATE = Path(__file__).resolve().parents[1] / "candidate_template"
docker = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")


def tasks(n, prefix):
    return [{"id": f"{prefix}{i}", "domain": "code"} for i in range(n)]


PACKS = {"dev": tasks(10, "d"), "select": tasks(20, "s"), "test": tasks(30, "t")}


def score(agent_dir, task_list):
    """A fake agent's results: its agent/level.txt says how many tasks of each pack it solves."""
    level = int((Path(agent_dir) / "agent" / "level.txt").read_text())
    return [{"id": t["id"], "domain": "code", "correct": i < level, "failure": None if i < level else "wrong_answer"}
            for i, t in enumerate(task_list)]


class Scripted:
    """Proposes agents of the given levels in turn; None stands for a change that failed validation."""
    name = "scripted"

    def __init__(self, levels):
        self.levels, self.seen = levels, []

    def start(self, run_dir):
        self.run_dir = Path(run_dir)

    def propose(self, child_id, parent_dir, results, tasks_by_id, notes, seed, context=None):
        self.seen.append((Path(parent_dir).name, sum(r["correct"] for r in results), list(notes)))
        level = self.levels[len(self.seen) - 1]
        if level is None:
            return {"id": child_id, "status": "invalid_patch", "note": f"{child_id}: rejected"}
        child = self.run_dir / child_id
        shutil.copytree(parent_dir, child)
        (child / "agent" / "level.txt").write_text(str(level))
        return {"id": child_id, "status": "valid", "dir": str(child), "module": "agent/workflow.py",
                "record": {"hypothesis": {"hypothesis": f"level {level}"}}}

    def usage(self):
        return {}


def test_a_change_is_kept_only_with_enough_more_selection_tasks(tmp_path):
    a0 = tmp_path / "a0"
    shutil.copytree(TEMPLATE, a0)
    (a0 / "agent" / "level.txt").write_text("4")
    calls = []
    evaluate = Evaluator(lambda ts, d: calls.append(d.name) or score(d, ts), tmp_path / "evals")
    improver = Scripted([9, None, 6, 11])  # +5 kept; rejected; -3 dropped; +2 below the margin of 3
    out = reset_meta(improver, a0, PACKS, tmp_path / "run", evaluate, rounds=4, margin=3, seed=7)
    assert [s.get("kept") for s in out["rounds"]] == [True, None, False, False]
    assert [s.get("select_gain") for s in out["rounds"]] == [5, None, -3, 2]
    assert out["final_agent"].endswith("r1") and out["test_solved"] == 9
    # Each round starts from the agent kept so far and hears how the earlier ones went, as counts.
    assert [p for p, _, _ in improver.seen] == ["a0", "r1", "r1", "r1"]
    assert improver.seen[1][1] == 9 and "kept: +5 selection tasks" in improver.seen[1][2][0]
    assert improver.seen[2][2][1] == "r2: rejected"
    # Every valid child is scored on the selection pack; on the development pack only the kept one (its results are
    # the next feedback); on the test pack only the final agent.
    assert sorted(calls) == sorted(["a0", "a0", "r1", "r1", "r3", "r4", "r1"])


def test_the_evaluator_scores_equal_code_once(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(TEMPLATE, a)
    shutil.copytree(TEMPLATE, b)
    (b / "memory").mkdir(exist_ok=True)
    (b / "memory" / "notes.md").write_text("- a note\n")  # notes are not part of the agent
    calls = []
    evaluate = Evaluator(lambda ts, d: calls.append(d) or [{"id": "x", "correct": True}], tmp_path / "evals")
    assert agent_hash(a) == agent_hash(b)
    evaluate(a, "test", [])
    evaluate(b, "test", [])
    evaluate(b, "test_repeat", [])
    assert calls == [a, b]


def test_the_t1_contract_describes_bug_fix_tasks():
    assert "failing_test" in T1_CONTRACT and "buggy_code" in T1_CONTRACT
    assert "rtl: spec" not in T1_CONTRACT and T1_CONTRACT.rstrip().endswith("is one of them.")


def test_the_oracle_is_a_valid_workflow_installed_once(tmp_path):
    assert check_python_source(ORACLE_WORKFLOW, "solve", 200, "workflow") == []
    oracle = OracleImprover()
    oracle.start(tmp_path)
    first = oracle.propose("r1", TEMPLATE, [], {}, [], seed=1)
    assert first["status"] == "valid" and (Path(first["dir"]) / "agent" / "workflow.py").read_text() == ORACLE_WORKFLOW
    assert oracle.propose("r2", first["dir"], [], {}, [], seed=2)["status"] == "no_proposal"


class Solver:
    def generate(self, messages, max_tokens=1024, temperature=0.0, thinking=False, seed=0):
        return {"text": "an answer without code", "finish_reason": "stop", "prompt_tokens": 5, "completion_tokens": 5}


def fake_run(fixes):
    """A stand-in for run_workflow_tasks: one model call per task; the tasks in `fixes` come out right."""
    seen = []

    def run(tasks, cand, llm, workers, seconds, judge_fn):
        task = tasks[0]
        seen.append(task["id"])
        llm.generate([{"role": "user", "content": task["public"]["prompt"]}])
        ok = task["id"] in fixes
        return [{"id": task["id"], "correct": ok, "failure": None if ok else "wrong_answer", "tool_calls": [],
                 "detail": {"tests": ["pass", "pass", "pass" if ok else "fail"]}}]
    return run, seen


def test_the_effect_check_sends_back_a_change_that_fixes_none_of_the_failing_tasks():
    parent = [{"id": f"d{i}", "correct": i < 3} for i in range(10)]
    by_id = {f"d{i}": {"id": f"d{i}", "public": {"prompt": f"fix task {i}"}} for i in range(10)}
    run, seen = fake_run(fixes=set())
    record = {"checks": {}, "stages": {}}
    problems = EffectCheck(Solver(), k=6, run=run)("child", parent, by_id, record, seed=1)
    assert len(seen) == 6 and not {"d0", "d1", "d2"} & set(seen)  # only tasks the current agent gets wrong
    assert record["checks"]["effect_fixed"] == 0 and "fixed none" in problems[0]
    assert f"fix task {seen[0][1]}" in problems[0] and "passed 2 of 3 hidden tests" in problems[0]
    run, _ = fake_run(fixes={f"d{i}" for i in range(3, 10)})
    assert EffectCheck(Solver(), run=run)("child", parent, by_id, record, seed=1) == []
    assert record["checks"]["effect_fixed"] == 6


def test_m1_shows_a_real_task_with_long_values_cut():
    task = {"public": {"domain": "code", "prompt": "x" * 1000, "buggy_code": "def f(): pass",
                       "failing_test": "assert 1"}}
    text = M1_POLICY.format(example=task_example(task))
    assert '"failing_test": "assert 1"' in text and "x" * 301 not in text and "never rebuild" in text


@docker
def test_the_oracle_runs_the_failing_test_and_asks_again(tmp_path):
    cand = tmp_path / "cand"
    shutil.copytree(TEMPLATE, cand)
    (cand / "agent" / "workflow.py").write_text(ORACLE_WORKFLOW)
    for p in [cand, *cand.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    fence = "`" * 3
    answers = [f"{fence}python\ndef inc(x):\n    return x\n{fence}",
               f"{fence}python\ndef inc(x):\n    return x + 1\n{fence}"]
    seen = []

    def on_llm(req):
        seen.append(req["messages"])
        return {"ok": True, "text": answers[len(seen) - 1]}
    task = {"domain": "code", "source": "t1_mbpp", "prompt": "fix inc", "failing_test": "assert inc(1) == 2"}
    with CandidateSandbox(cand, memory="768m") as sb:
        done = sb.solve(task, 60, on_llm)
    assert done["ok"] and "x + 1" in done["answer"] and len(seen) == 2
    assert "AssertionError" in seen[1][-1]["content"]
    assert [c["name"] for c in done["tool_calls"]] == ["python_exec", "python_exec"]
