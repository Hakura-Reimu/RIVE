"""Phase D data selection and pass criterion (docs/phase-d-design.md)."""
from rive.benchmark.hard import public_task
from rive.phase_d import Overlap, bootstrap_p, criterion, last_boxed, macro, pick_examples

BOILERPLATE = ("The input is given from Standard Input in the following format: N M. Print the answer. "
               "All input values are integers.")


def test_last_boxed_takes_the_last_answer_with_nested_braces():
    assert last_boxed("so \\boxed{1} then \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"
    assert last_boxed("\\fbox{x+1}") == "x+1" and last_boxed("no answer") is None
    assert last_boxed("cut off \\boxed{12") is None


def test_overlap_finds_repeats_but_not_shared_boilerplate():
    protected = {"sel-1": "Takahashi has N apples and gives M of them to Aoki every day until none are left. "
                          + BOILERPLATE,
                 "sel-2": "Find the number of ordered pairs of positive integers whose sum is 2026.",
                 "short": "Compute 2+2."}
    overlap = Overlap(protected)
    assert overlap.match("Takahashi has N apples and gives M of them to Aoki every day until none are left. "
                         "Print the answer.") == "sel-1"
    assert overlap.match("compute 2 + 2") == "short"
    # Another problem with the same input-format lines is not a repeat.
    assert overlap.match("Snuke paints a grid of H rows and W columns black and white in a checkered pattern. "
                         + BOILERPLATE) is None


def sample(text, correct=True, tokens=10, finish="stop"):
    return {"text": text, "correct": correct, "completion_tokens": tokens, "finish_reason": finish}


def test_pick_examples_keeps_complete_correct_answers_shortest_first():
    samples = [sample("long", tokens=900), sample("wrong", correct=False, tokens=5), sample("cut", tokens=4096,
               finish="length"), sample("short", tokens=100), sample(" short ", tokens=101), sample("mid", tokens=500)]
    assert [s["text"] for s in pick_examples(samples)] == ["short", "mid"]
    assert pick_examples([sample("x", correct=False)]) == []


def run(correct_by_domain, n=10):
    """One run: per domain, the first k of n items correct."""
    return [{"id": f"{d}{i}", "domain": d, "correct": i < k} for d, k in correct_by_domain.items() for i in range(n)]


def test_criterion_needs_the_mean_gain_every_seed_positive_and_no_easy_drop():
    v0 = [run({"math": 3, "code": 3}), run({"math": 3, "code": 3})]
    easy0 = run({"math": 9}, n=100)
    better = [run({"math": 4, "code": 3})]  # +5 points macro
    v = criterion(v0, {"a": better, "b": better}, easy0, {"a": easy0, "b": easy0})
    assert v["passed"] and abs(v["mean_gain"] - 0.05) < 1e-9
    assert not criterion(v0, {"a": better, "b": v0}, easy0, {"a": easy0, "b": easy0})["passed"]  # b gains 0
    worse_easy = run({"math": 7}, n=100)  # 2 points down on the easy sets
    assert not criterion(v0, {"a": better}, easy0, {"a": worse_easy})["passed"]
    assert macro(run({"math": 10, "code": 0})) == 0.5


def test_bootstrap_p_is_small_only_for_a_real_gain():
    v0 = [run({"math": 3, "code": 3}, n=40)] * 2
    assert bootstrap_p(v0, v0) == 1.0
    assert bootstrap_p(v0, [run({"math": 20, "code": 20}, n=40)]) < 0.01


def test_training_sources_use_the_evaluation_prompts():
    item = {"problem": "What is 1+1?", "content": "Print 2.", "starter_code": "", "testtype": "stdin",
            "func_name": None}
    assert public_task("math_train", item)["prompt"] == public_task("aime2025", item)["prompt"]
    assert public_task("math500", item)["domain"] == "math"
    assert public_task("lcb_train", item)["prompt"] == public_task("livecodebench", item)["prompt"]
