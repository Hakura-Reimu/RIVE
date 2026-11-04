import sys
from itertools import permutations, product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rive_rl_tasks as rt  # noqa: E402


def test_knights_have_one_solution_and_the_judge_reads_it():
    for i in range(30):
        t = rt.logic_task(f"test:{i}", "knights")
        line = ", ".join(f"{n}={k}" for n, k in zip(t["names"], t["solution"]))
        assert rt.judge(t, f"Some reasoning.\nANSWER: {line}") == (True, None)
        wrong = ", ".join(f"{n}={'knave' if k == 'knight' else 'knight'}" for n, k in zip(t["names"], t["solution"]))
        assert not rt.judge(t, f"ANSWER: {wrong}")[0]
        assert rt.judge(t, "no answer line")[1] == "no_final_answer"


def test_seating_has_one_solution_and_the_judge_reads_it():
    for i in range(30):
        t = rt.logic_task(f"test:{i}", "seating")
        assert rt.judge(t, "**ANSWER:** " + ", ".join(t["solution"])) == (True, None)
        assert not rt.judge(t, "ANSWER: " + ", ".join(reversed(t["solution"])))[0]


def test_target_accepts_any_correct_expression_and_checks_the_numbers():
    for i in range(30):
        t = rt.logic_task(f"test:{i}", "target")
        assert rt.judge(t, f"ANSWER: {t['expression']}") == (True, None)
        assert rt.judge(t, f"answer: {t['target']} = {t['expression']}")[0]
        assert not rt.judge(t, f"ANSWER: {t['target']}")[0]  # the numbers must all be used
    t = {"domain": "logic", "family": "target", "numbers": [2, 3, 4, 6], "target": 36}
    assert rt.judge(t, "ANSWER: (6 / 2) * 3 * 4")[0]
    assert not rt.judge(t, "ANSWER: 6 * 6")[0]
    assert not rt.judge(t, "ANSWER: __import__('os')")[0]


def test_writing_checks():
    t = {"domain": "writing", "constraints": ["lowercase", "no_commas"],
         "_checks": [rt._constraint("lowercase", rt._rng(0), set())[1], rt._constraint("no_commas", rt._rng(0), set())[1]]}
    assert rt.judge(t, "all lower case and no commas here") == (True, None)
    assert rt.judge(t, "Not lower case") == (False, "failed:lowercase")
    text, ok = "<<A Title>>\n\nfirst part.\n\nsecond part.\n\nP.S. bye", rt._constraint("title", rt._rng(0), set())[1]
    assert ok(text)
    assert rt._constraint("postscript", rt._rng(0), set())[1](text)


def test_sets_are_fixed_and_training_avoids_them():
    a = [t["prompt"] for t in rt.fixed_set("logic", "dev", 12)]
    assert a == [t["prompt"] for t in rt.fixed_set("logic", "dev", 12)]
    banned = set(a) | {t["prompt"] for t in rt.fixed_set("writing", "dev", 12)}
    for k in range(10):
        assert rt.training_task("logic", 3, k, banned)["prompt"] not in banned
        assert rt.training_task("writing", 3, k, banned)["prompt"] not in banned


def test_easy_levels_are_smaller_and_still_judged():
    for i in range(20):
        k = rt.logic_task(f"test-easy:{i}", "knights", "easy")
        assert len(k["names"]) <= 5 and "If " not in k["prompt"]
        line = ", ".join(f"{n}={s}" for n, s in zip(k["names"], k["solution"]))
        assert rt.judge(k, f"ANSWER: {line}")[0]
        s = rt.logic_task(f"test-easy:{i}", "seating", "easy")
        assert len(s["names"]) <= 5 and rt.judge(s, "ANSWER: " + ", ".join(s["solution"]))[0]
        t = rt.logic_task(f"test-easy:{i}", "target", "easy")
        assert len(t["numbers"]) <= 5 and rt.judge(t, f"ANSWER: {t['expression']}")[0]
        w = rt.writing_task(f"test-easy:{i}", "easy")
        assert 1 <= len(w["constraints"]) <= 2 and not {"capitals", "letter"} & set(w["constraints"])
        stored = {k: v for k, v in w.items() if k != "_checks"}  # as reloaded: the checks are made again
        assert rt.judge(stored, "x") == rt.judge(w, "x")


def test_writing_prompts_have_no_clashing_constraints():
    for i in range(200):
        t = rt.writing_task(f"test:{i}")
        names = t["constraints"]
        assert not any(frozenset((x, y)) in rt.CLASH for x in names for y in names if x != y)
