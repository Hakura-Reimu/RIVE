"""T1 bug-fix tasks (rive/meta_rsi/bugfix.py). The sandbox checks need docker and the alter-judge image (WSL)."""
import ast
import shutil

import pytest

from rive.meta_rsi.bugfix import candidates, check, failing, judge, make_task, mutate, sites

ITEM = {"id": "mbpp-x", "prompt": "Count the positive numbers below ten.", "test_imports": [],
        "reference": "def count_small(xs):\n    n = 0\n    for i in range(len(xs)):\n"
                     "        if xs[i] > 0 and xs[i] < 10:\n            n += 1\n    return n\n",
        "tests": ["assert count_small([1, 5, 10, -2]) == 2", "assert count_small([]) == 0",
                  "assert count_small([0, 9, 3]) == 2"]}
docker = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")


def test_every_site_mutates_into_different_valid_code():
    tree = ast.parse(ITEM["reference"])
    found = sites(tree)
    assert {kind for _, kind, _ in found} >= {"compare", "boolop", "constant", "augassign", "range_stop"}
    original = ast.unparse(tree)
    for site in found:
        code = ast.unparse(mutate(tree, site))
        assert code != original and ast.parse(code)
    assert ast.unparse(tree) == original  # mutate works on a copy


def test_mutations_are_the_intended_bugs():
    tree = ast.parse("def f(s, x):\n    if not x:\n        return min(s)\n    return range(len(s))\n")
    by_kind = {kind: site for site in sites(tree) for kind in [site[1]]}
    assert "if x:" in ast.unparse(mutate(tree, by_kind["drop_not"]))
    assert "max(s)" in ast.unparse(mutate(tree, by_kind["swap_call"]))
    assert "range(len(s) - 1)" in ast.unparse(mutate(tree, by_kind["range_stop"]))


def test_candidates_are_fixed_by_seed_and_distinct():
    one, again = candidates(ITEM, 1), candidates(ITEM, 1)
    assert one == again and len({c["code"] for c in one}) == len(one) == 6
    for c in candidates(ITEM, 2):
        assert len(c["mutations"]) == 2 and len({m[0] for m in c["mutations"]}) == 2


def test_task_shows_what_the_setting_says_and_keeps_tests_private():
    mutant = {"code": "def count_small(xs):\n    return 0", "mutations": [], "failing": [0]}
    shown = make_task(ITEM, mutant, "spec_fail_1")["public"]["prompt"]
    assert ITEM["prompt"] in shown and ITEM["tests"][0] in shown and mutant["code"] in shown
    hidden = make_task(ITEM, mutant, "nospec_fail_2")["public"]
    assert ITEM["prompt"] not in hidden["prompt"] and ITEM["tests"][1] not in hidden["prompt"]
    assert "tests" not in hidden


@docker
def test_the_sandbox_tells_the_reference_from_a_bug():
    assert failing(check(ITEM, ITEM["reference"])) == []
    buggy = ITEM["reference"].replace("xs[i] < 10", "xs[i] <= 10")
    assert failing(check(ITEM, buggy)) == [0]
    assert failing(check(ITEM, "def count_small(xs):\n    while True:\n        pass\n", timeout=5)) is None
    task = make_task(ITEM, {"code": buggy, "mutations": [], "failing": [0]}, "spec_fail_1")
    assert judge(task, f"Fixed:\n```python\n{ITEM['reference']}```") == (True, None, {"tests": ["pass"] * 3})
    ok, failure, _ = judge(task, f"```python\n{buggy}```")
    assert not ok and failure == "wrong_answer"
    assert judge(task, "```python\ndef count_small(xs) return 1\n```")[1] == "load_error"
