"""Evolution controller with a scripted fake world: promotion gates, selection retest, regression control,
parent choice and the stagnation rule; since C5, the champion's runs as the reference, judged domains, the second
run of a passing child and refilled slots."""
import hashlib
import random

import pytest

from rive.archive.db import Archive
from rive.benchmark.stats import paired_bootstrap_macro
from rive.evolution.controller import (Evolution, Settings, choose_parents, history_share, judge, sample_task_set,
                                         t_sf)

DOMAINS = ("math", "code", "rtl")
DEV = [{"id": f"{d}{i:03d}", "domain": d} for d in DOMAINS for i in range(60)]
SELECTION = [{"id": f"s{d}{i:03d}", "domain": d} for d in DOMAINS for i in range(60)]
SIZES = {"math": 20, "code": 20, "rtl": 20}
ALL = {"math": 60, "code": 60, "rtl": 60}


def h(task_id):
    """Stable pseudo-random number in [0, 1) per task."""
    return int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) / 2 ** 32


class World:
    """A candidate with quality q solves exactly the tasks with h < q, so better ones solve supersets;
    candidates in `flip` solve the tasks with h > 1 - q instead (different items, same accuracy). `runs` gives a
    candidate another quality on each development run, as run-to-run noise does."""

    def __init__(self, tmp_path, plan, flip=(), selection_quality=None, runs=None):
        self.tmp, self.plan, self.flip = tmp_path, list(plan), set(flip)
        self.quality, self.selection_quality = {"seed": 0.3}, dict(selection_quality or {})
        self.runs = {cid: list(qs) for cid, qs in (runs or {}).items()}
        self.n, self.easy_calls, self.evaluated = 0, [], []

    def solved(self, cid, task_id, q):
        return (1 - h(task_id)) < q if cid in self.flip else h(task_id) < q

    def research(self, parent, g):
        self.n += 1
        cid = f"g{g}c{self.n}"
        delta, status, *extra = self.plan.pop(0)
        self.quality[cid] = self.quality[parent] + delta
        return {"id": cid, "dir": str(self.tmp / cid), "status": status, "module": "agent/workflow.py",
                "record": {"parent": parent}, **(extra[0] if extra else {})}

    def evaluate(self, cid, tasks):
        q = self.runs[cid].pop(0) if self.runs.get(cid) else self.quality[cid]
        self.evaluated.append((cid, sorted({t["domain"] for t in tasks})))
        return [{"id": t["id"], "domain": t["domain"], "correct": self.solved(cid, t["id"], q)} for t in tasks]

    def easy(self, cid):
        self.easy_calls.append(cid)
        return [{"id": f"e{i}", "domain": "easy", "correct": True} for i in range(150)]

    def selection(self, cid):
        q = self.selection_quality.get(cid, self.quality[cid])
        return [{"id": t["id"], "domain": t["domain"], "correct": self.solved(cid, t["id"], q)} for t in SELECTION]


def evolution(tmp_path, world, sizes=SIZES, **settings):
    a = Archive(tmp_path / "archive.sqlite")
    a.add_candidate("seed", None, 0, tmp_path / "seed", "seed", status="evaluated")
    return Evolution(a, world.research, world.evaluate, world.easy, world.selection, DEV, sizes,
                     Settings(**settings), seed=1, report_dir=tmp_path / "reports")


def run(q, tasks=DEV):
    """Results of a candidate of quality q on `tasks`."""
    return [{"id": t["id"], "domain": t["domain"], "correct": h(t["id"]) < q} for t in tasks]


def test_clear_improvement_is_promoted_after_selection_retest(tmp_path):
    world = World(tmp_path, [(0.25, "valid"), (0.0, "valid"), (-0.1, "valid"), (0.0, "invalid_patch")])
    evo = evolution(tmp_path, world, refills=0)
    report = evo.run_generation(1)
    a = evo.archive
    assert a.champion() == "g1c1" and evo.stagnant == 0
    assert {c["id"]: c["status"] for c in a.candidates(1)} == {
        "g1c1": "evaluated", "g1c2": "evaluated", "g1c3": "evaluated", "g1c4": "invalid_patch"}
    assert [e["detail"]["passed"] for e in a.events("selection_retest")] == [True]
    # The winner ran twice on the development tasks before its selection retest, the others once.
    assert [cid for cid, _ in world.evaluated] == ["seed", "g1c1", "g1c1", "g1c2", "g1c3"]
    assert any(e["kind"] == "promotion" for e in report["events"])
    assert (tmp_path / "reports" / "gen_001.md").exists()


def test_selection_retest_can_block_a_dev_winner(tmp_path):
    world = World(tmp_path, [(0.25, "valid")] + [(0.0, "invalid_patch")] * 3, selection_quality={"g1c1": 0.3})
    evo = evolution(tmp_path, world, refills=0)
    evo.run_generation(1)
    assert evo.archive.champion() == "seed" and evo.stagnant == 1
    assert [e["detail"]["passed"] for e in evo.archive.events("selection_retest")] == [False]


def test_regression_gate_blocks_a_child_that_solves_different_items(tmp_path):
    world = World(tmp_path, [(0.25, "valid")] + [(0.0, "invalid_patch")] * 3, flip={"g1c1"})
    evo = evolution(tmp_path, world, refills=0)
    evo.run_generation(1)
    gate = evo.archive.events("gates")[0]["detail"]
    assert gate["diff"] > 0.1 and gate["pass_dev"]  # better on average...
    assert gate["regression"] > 0.5 and not gate["pass_regression"]  # ...but loses what the champion solved
    assert "pass_easy" not in gate and world.easy_calls == [] and evo.archive.champion() == "seed"


def test_judge_takes_the_noise_from_the_champions_runs():
    s, child = Settings(), [run(0.36)]
    steady = [run(0.30)] * 6
    noisy = [run(q) for q in (0.20, 0.40, 0.22, 0.38, 0.24, 0.36)]
    calm, shaky = judge(steady, child, list(DOMAINS), s), judge(noisy, child, list(DOMAINS), s)
    assert calm["sd"] < shaky["sd"] and calm["pass_dev"] and not shaky["pass_dev"]
    # A run of the champion itself loses about what its runs lose among themselves.
    assert judge(noisy, [run(0.30)], list(DOMAINS), s)["pass_regression"]
    assert t_sf(0.0, 20) == pytest.approx(0.5) and t_sf(1.812, 10) == pytest.approx(0.05, abs=0.002)


def test_a_prompt_only_child_is_run_and_judged_on_its_own_domains(tmp_path):
    world = World(tmp_path, [(0.0, "valid", {"domains": ["code"]})] + [(0.0, "invalid_patch")] * 3)
    evo = evolution(tmp_path, world, refills=0)
    evo.run_generation(1)
    assert world.evaluated == [("seed", ["code", "math", "rtl"]), ("g1c1", ["code"])]
    gate = evo.archive.events("gates")[0]["detail"]
    assert gate["domains"] == ["code"] and gate["diff"] == 0.0 and not gate["pass_dev"]


def test_a_passing_child_must_pass_again_on_a_second_run(tmp_path):
    # Lucky once: the first run is far above the champion, the second below it.
    world = World(tmp_path, [(0.0, "valid")] + [(0.0, "invalid_patch")] * 3, runs={"g1c1": [0.45, 0.22]})
    evo = evolution(tmp_path, world, refills=0)
    evo.run_generation(1)
    gate = evo.archive.events("gates")[0]["detail"]
    assert gate["first_run"]["pass_dev"] and not gate["pass_dev"] and gate["child_runs"] == 2
    assert evo.archive.events("selection_retest") == [] and evo.archive.champion() == "seed"


def test_slots_whose_child_was_not_evaluated_get_another_try(tmp_path):
    world = World(tmp_path, [(0.0, "invalid_patch"), (0.0, "duplicate"), (0.0, "invalid_patch")]
                  + [(0.0, "valid")] * 3)
    evo = evolution(tmp_path, world)
    evo.run_generation(1)
    assert [c["status"] for c in evo.archive.candidates(1)] == [
        "invalid_patch", "duplicate", "invalid_patch", "evaluated", "evaluated", "evaluated"]


def test_champion_runs_add_up_until_a_promotion(tmp_path):
    world = World(tmp_path, [(0.0, "valid")] * 4 + [(0.25, "valid")] + [(0.0, "invalid_patch")] * 3
                  + [(0.0, "valid")] * 4)
    evo = evolution(tmp_path, world, sizes=ALL, refills=0)
    for g in (1, 2, 3):
        evo.run_generation(g)
    # Its runs as a challenger do not count for the new champion.
    assert evo.archive.champion() == "g2c5"
    assert [e["detail"]["champion_runs"] for e in evo.archive.events("generation_start")] == [1, 2, 1]


def test_history_share_rises_with_stagnation():
    s = Settings(patience=5)
    assert [history_share(n, s) for n in (0, 4, 5, 10, 14, 15)] == [
        (0.2, 0), (0.2, 0), (0.35, 1), (0.5, 1), (0.5, 1), (0.5, 2)]


def test_parent_slots_follow_the_share(tmp_path):
    world = World(tmp_path, [(0.0, "valid")] * 4)
    evo = evolution(tmp_path, world)
    evo.run_generation(1)  # four evaluated children, none better: they form the archive pool
    rng = random.Random(0)
    assert {src for _, src in choose_parents(evo.archive, "seed", evo.settings, 0.0, rng)} == {"trunk"}
    picks = choose_parents(evo.archive, "seed", evo.settings, 1.0, rng)
    assert {src for _, src in picks} == {"archive"} and all(p != "seed" for p, _ in picks)


def test_task_batches_are_stratified_and_reproducible():
    a = sample_task_set(DEV, {"math": 5, "code": 7}, random.Random(3))
    b = sample_task_set(DEV, {"math": 5, "code": 7}, random.Random(3))
    assert a == b and [t["domain"] for t in a].count("math") == 5 and len(a) == 12
    assert [t["id"] for t in a] == sorted(t["id"] for t in a)


def test_macro_bootstrap_weighs_domains_equally():
    # domain A: 100 items, child gains 10; domain B: 10 items, child loses 3 -> micro +, macro -
    parent = [0] * 100 + [1] * 10
    child = [1] * 10 + [0] * 90 + [0] * 3 + [1] * 7
    diff, p = paired_bootstrap_macro(parent, child, ["A"] * 100 + ["B"] * 10, iterations=2000)
    assert diff == pytest.approx((0.1 - 0.3) / 2) and p > 0.5
