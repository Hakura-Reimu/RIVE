"""SQLite evolution archive: lineage, evaluations against the champion, clade statistics, tools, reports."""
import pytest

from rive.archive.db import Archive, summarize
from rive.archive.report import generation_report, to_markdown


def results(correct_by_domain):
    """{"math": [1, 0], "code": [1]} -> per-item results with stable ids."""
    return [{"id": f"{d}_{i}", "domain": d, "correct": c, "tokens": 10}
            for d, flags in correct_by_domain.items() for i, c in enumerate(flags)]


@pytest.fixture
def archive(tmp_path):
    a = Archive(tmp_path / "archive.sqlite")
    a.add_candidate("seed", None, 0, tmp_path / "seed", "seed")
    a.add_evaluation("seed", "gen-001", "dev", results({"math": [1, 0, 0, 0], "code": [1, 1, 0, 0]}))
    a.add_candidate("c1", "seed", 1, tmp_path / "c1", "research", "trunk", module="agent/workflow.py")
    a.add_candidate("c2", "seed", 1, tmp_path / "c2", "research", "trunk", module="tools/check.py")
    a.add_candidate("c3", "c1", 2, tmp_path / "c3", "research", "archive", module="agent/prompts/math.md")
    yield a
    a.close()


def test_summary_is_domain_balanced():
    s = summarize(results({"math": [1, 0, 0, 0], "code": [1]}))
    assert s["accuracy"] == 0.4 and s["macro"] == pytest.approx(0.625) and s["by_domain"] == {"code": 1.0,
                                                                                              "math": 0.25}


def test_lineage_and_clade(archive):
    assert archive.ancestors("c3") == ["c3", "c1", "seed"]
    assert sorted(archive.clade("c1")) == ["c1", "c3"]
    assert sorted(archive.clade("seed")) == ["c1", "c2", "c3", "seed"]


def test_evaluation_against_champion_and_clade_stats(archive):
    archive.add_evaluation("c1", "gen-001", "dev", results({"math": [1, 1, 0, 0], "code": [1, 1, 0, 0]}),
                           reference_id="seed")
    archive.add_evaluation("c2", "gen-001", "dev", results({"math": [0, 0, 0, 0], "code": [1, 1, 0, 0]}),
                           reference_id="seed")
    archive.add_evaluation("seed", "gen-002", "dev", results({"math": [1, 0], "code": [0, 0]}))
    archive.add_evaluation("c3", "gen-002", "dev", results({"math": [1, 1], "code": [0, 0]}), reference_id="seed")
    assert archive.evaluation("c1", "gen-001")["vs_reference"] == pytest.approx(0.125)
    assert archive.clade_stats("c1") == {"members": 2, "evaluated": 2, "wins": 2, "losses": 0}
    assert archive.clade_stats("c2") == {"members": 1, "evaluated": 1, "wins": 0, "losses": 1}
    with pytest.raises(ValueError):
        archive.add_evaluation("c2", "gen-009", "dev", results({"math": [1]}), reference_id="seed")


def test_champion_follows_promotions_and_survives_reopen(archive, tmp_path):
    assert archive.champion() == "seed"
    archive.promote("c1", 1, {"selection_p": 0.01})
    archive.close()
    again = Archive(tmp_path / "archive.sqlite")
    assert again.champion() == "c1" and again.candidate("c3")["parent_source"] == "archive"
    again.close()


def test_tool_versions_outlive_their_candidate(archive):
    assert archive.register_tool("check", "c2", "h1", {"in": "code"}, True) == 1
    archive.set_status("c2", "retired")
    assert archive.register_tool("check", "c3", "h2", {"in": "code"}, True) == 2


def test_generation_report(archive):
    archive.add_evaluation("c1", "gen-001", "dev", results({"math": [1, 1, 0, 0], "code": [1, 1, 0, 0]}),
                           reference_id="seed")
    archive.set_status("c1", "evaluated")
    archive.promote("c1", 1, {"selection_p": 0.01})
    rep = generation_report(archive, 1, "gen-001")
    assert rep["champion_after"] == "c1" and [r["id"] for r in rep["candidates"]] == ["c1", "c2"]
    md = to_markdown(rep)
    assert "+12.5 个百分点" in md and "promotion" in md and "—" in md
