"""Picking a Phase C run back up after an interruption (rive.evolution.phase_c.resume), without docker."""
import json

from rive.archive.db import Archive
from rive.evolution.controller import Evolution
from rive.evolution.phase_c import PhaseC, resume


def result(tid, ok):
    return {"id": tid, "domain": "math", "correct": ok, "failure": None if ok else "wrong_answer", "tokens": 10}


def test_resume_marks_the_unfinished_generation_and_rebuilds_state(tmp_path):
    archive = Archive(tmp_path / "archive.sqlite")
    for cid, parent, g, origin, status in (("seed", None, 0, "seed", "evaluated"),
                                           ("g01c001", "seed", 1, "research", "evaluated"),
                                           ("g02c002", "seed", 2, "research", "created")):
        archive.add_candidate(cid, parent, g, tmp_path / "candidates" / cid, origin, status=status,
                              code_hash=None if cid == "seed" else f"hash-{cid}")
    evals = {"seed": {"dev_01.json": [result("d1", True)], "dev_02.json": [result("d1", False)],
                      "easy.json": [result("e1", True)]},
             "g01c001": {"dev_01.json": [result("d1", True)]}}
    for cid, files in evals.items():
        (tmp_path / "evals" / cid).mkdir(parents=True)
        for name, rows in files.items():
            (tmp_path / "evals" / cid / name).write_text(json.dumps(rows))
    dev = [{"id": "d1", "domain": "math"}]

    class Research:
        seen_hashes = set()

    glue = PhaseC(None, tmp_path, archive, Research(), [], [])
    evo = Evolution(archive, glue.research, glue.evaluate, glue.easy_guard, glue.selection, dev, {"math": 1})
    done = [{"generation": 1, "champion_before": "seed", "champion_after": "seed"}]

    assert resume(archive, glue, evo, done, dev) == 2
    assert archive.candidate("g02c002")["status"] == "interrupted"
    assert archive.candidate("g01c001")["status"] == "evaluated"
    assert glue.made == 2  # the next child is g02c003, so no id is reused
    assert glue.latest["seed"] == [result("d1", False)] and "g01c001" in glue.latest
    assert glue.tasks_by_id == {"d1": dev[0]}
    assert glue.step.seen_hashes == {"hash-g01c001", "hash-g02c002"}
    assert evo.stagnant == 1 and evo._cache == {("easy", "seed"): [result("e1", True)]}


def test_stagnation_counts_back_to_the_last_promotion(tmp_path):
    archive = Archive(tmp_path / "archive.sqlite")
    archive.add_candidate("seed", None, 0, tmp_path / "seed", "seed", status="evaluated")
    glue = PhaseC(None, tmp_path, archive, None, [], [])
    evo = Evolution(archive, None, None, None, None, [], {})
    done = [{"champion_before": "seed", "champion_after": "seed"}, {"champion_before": "seed", "champion_after": "a"},
            {"champion_before": "a", "champion_after": "a"}, {"champion_before": "a", "champion_after": "a"}]
    assert resume(archive, glue, evo, done, []) == 5 and evo.stagnant == 2
