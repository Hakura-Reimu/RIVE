"""One Phase C generation end to end: controller, glue, research step, sandboxes and runtime, with a fake model
and judge in which the researched child really is better. Needs docker."""
import shutil
from pathlib import Path

import pytest

from rive.archive.db import Archive
from rive.evolution.controller import Evolution, Settings
from rive.evolution.phase_c import PhaseC
from rive.research.agent_loop import AgentResearch
from test_agent_research import FakeLLM

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
TEMPLATE = Path(__file__).resolve().parents[1] / "candidate_template"


def tasks(prefix, n):
    return [{"id": f"{prefix}{i:02d}", "domain": "math", "source": "aime2025", "item": {},
             "public": {"domain": "math", "source": "aime2025", "prompt": f"{prefix} problem {i}", "problem": "?"}}
            for i in range(n)]


class Model(FakeLLM):
    """Research stages as in FakeLLM; as the solver it answers well only when told to be brief."""

    def generate(self, messages, max_tokens, temperature, thinking, seed=0):
        brief = any(m["content"] == "Be brief." for m in messages)
        return {"text": "\\boxed{2}" if brief else "I am not sure.", "finish_reason": "stop", "prompt_tokens": 5,
                "completion_tokens": 50 if brief else 4000}


def judge(task, answer):
    ok = "\\boxed{2}" in answer
    return ok, None if ok else "no_final_answer", {}


def test_one_generation_promotes_a_real_improvement(tmp_path):
    dev, selection, easy = tasks("d", 24), tasks("s", 24), tasks("e", 6)
    archive = Archive(tmp_path / "archive.sqlite")
    shutil.copytree(TEMPLATE, tmp_path / "candidates" / "seed")
    archive.add_candidate("seed", None, 0, tmp_path / "candidates" / "seed", "seed", status="evaluated")
    llm = Model()
    research = AgentResearch(llm, tmp_path / "candidates", dev[:1])
    glue = PhaseC(llm, tmp_path, archive, research, easy, selection, workers=4, judge_fn=judge)
    evo = Evolution(archive, glue.research, glue.evaluate, glue.easy_guard, glue.selection, dev, {"math": 16},
                    Settings(slots=2), seed=0, report_dir=tmp_path / "reports", reflect=glue.reflect)
    report = evo.run_generation(1)
    first, *rest = report["candidates"]
    assert first["status"] == "evaluated" and first["vs_champion"] == pytest.approx(1.0)
    # Same fake model, same change: caught before evaluation, in the second slot and in both of its retries.
    assert [c["status"] for c in rest] == ["duplicate"] * 3
    assert archive.champion() == first["id"]
    assert sorted(p.name for p in (tmp_path / "evals" / first["id"]).glob("dev_*.json")) == [
        "dev_01.json", "dev_02.json"]  # the winner ran twice before the selection retest
    notes = (tmp_path / "candidates" / first["id"] / "memory" / "notes.md").read_text()
    assert "brevity helps" in notes and archive.candidate(first["id"])["note"] in notes
    assert (tmp_path / "evals" / first["id"] / "selection.json").exists()


class LocalStep:
    """A research step whose change was checked to stay in RTL (agent_loop's locality check); the child's directory
    is its parent's."""

    def propose(self, child_id, parent_dir, results, tasks_by_id, notes, seed, blocked=(), problem=None):
        return {"id": child_id, "dir": str(parent_dir), "status": "valid", "module": "agent/workflow.py",
                "record": {}, "note": None, "code_hash": child_id, "local_to": "rtl"}


def test_a_change_kept_to_its_domain_is_judged_there_when_its_parent_is_the_champion(tmp_path):
    archive = Archive(tmp_path / "archive.sqlite")
    for cid in ("seed", "other"):
        shutil.copytree(TEMPLATE, tmp_path / cid)
        archive.add_candidate(cid, None if cid == "seed" else "seed", 0, tmp_path / cid,
                              "seed" if cid == "seed" else "research", status="evaluated")
    (tmp_path / "other" / "agent" / "workflow.py").write_text(
        "def solve(task, ctx):\n    return {'answer': ctx.llm([{'role': 'user', 'content': task['prompt']}])}\n")
    glue = PhaseC(None, tmp_path, archive, LocalStep(), [], [], allocate=False)
    glue.latest = {"seed": [], "other": []}
    assert glue.research("seed", 1)["domains"] == ["rtl"]
    # A child of another candidate also carries that candidate's workflow, which reaches every domain.
    assert glue.research("other", 1)["domains"] is None
