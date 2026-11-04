"""R-011's teaching data (rive/meta_rsi/teach.py): problem specs, the writer's reply as a LiveCodeBench item, dedup."""
import json

import pytest

from rive.benchmark.hard import public_task
from rive.meta_rsi import teach

STDIN = {"title": "Bakery Boxes", "statement": "Count the boxes.\n\nInput\n\nN\n\nOutput\n\nThe count.",
         "examples": [{"input": "3\n", "output": "3"}, {"input": "5", "output": 5}, {"input": "0", "output": "0"}]}
FUNCTIONAL = {"title": "Pair Count", "statement": "Count pairs (i, j) with nums[i] + nums[j] == k.",
              "starter_code": "class Solution:\n    def countPairs(self, nums: List[int], k: int) -> int:\n        ",
              "func_name": "countPairs",
              "examples": [{"args": {"nums": [1, 2, 3], "k": 4}, "output": 1},
                           {"args": {"nums": [], "k": 1}, "output": 0},
                           {"args": {"nums": [2, 2], "k": 4}, "output": 1}]}


def reply(obj):
    return "Here it is.\n```json\n" + json.dumps(obj) + "\n```"


def test_specs_differ_and_mix_test_types_and_difficulties():
    specs = [teach.spec(i) for i in range(300)]
    assert len({teach.writer_prompt(s) for s in specs}) == 300
    for key in ("testtype", "difficulty"):
        by_topic = {}
        for s in specs:
            by_topic.setdefault(s["topic"], set()).add(s[key])
        assert all(len(v) > 1 for v in by_topic.values())  # not tied to the topic


def test_a_stdin_problem_becomes_an_item_with_its_examples_shown():
    item, tests = teach.to_item({**teach.spec(0), "testtype": "stdin"}, reply(STDIN))
    assert item["testtype"] == "stdin" and item["func_name"] == "" and item["tests_dir"] == teach.TESTS_DIR
    assert [t["input"] for t in tests] == ["3\n", "5\n", "0\n"] and [t["output"] for t in tests] == ["3\n", "5\n", "0\n"]
    assert "Sample Input 2\n\n5\n\nSample Output 2\n\n5" in item["content"]
    assert "standard input" in public_task("lcb_train", item)["prompt"]


def test_a_functional_problem_keeps_the_parameter_order():
    item, tests = teach.to_item({**teach.spec(1), "testtype": "functional"}, reply(FUNCTIONAL))
    assert item["func_name"] == "countPairs" and item["starter_code"].endswith("\n        ")
    assert tests[0] == {"input": "[1, 2, 3]\n4", "output": "1", "testtype": "functional"}
    assert "Input: nums = [1, 2, 3], k = 4\nOutput: 1" in item["content"]
    assert "starter code" in public_task("lcb_train", item)["prompt"]


@pytest.mark.parametrize("bad", [
    "no json here",
    reply({**FUNCTIONAL, "examples": FUNCTIONAL["examples"][:2]}),
    reply({**FUNCTIONAL, "examples": [{"args": {"k": 4, "nums": [1]}, "output": 1}] * 3}),  # wrong order
    reply({**FUNCTIONAL, "starter_code": "class Solution:\n    def other(self, x):\n        "}),
    reply({**FUNCTIONAL, "title": ""}),
])
def test_unusable_replies_are_refused(bad):
    with pytest.raises(ValueError):
        teach.to_item({**teach.spec(1), "testtype": "functional"}, bad)


def test_dedup_drops_held_out_overlaps_titles_and_repeats():
    long = " ".join(f"word{i}" for i in range(40))
    items = [{"id": "a", "title": "New One", "content": long},
             {"id": "b", "title": "Other", "content": long + " extra words here"},
             {"id": "c", "title": "Held Out Title", "content": "fresh text " * 10},
             {"id": "d", "title": "Clean", "content": " ".join(f"other{i}" for i in range(40))}]
    refs = [("held-out title", "unrelated statement about something else entirely and more words to fill it")]
    kept, dropped = teach.dedup(items, refs)
    assert [i["id"] for i in kept] == ["a", "d"]
    assert dict(dropped) == {"b": "repeats an earlier problem", "c": "title of a held-out task"}
    kept, dropped = teach.dedup(items[:1], [("x", long)])
    assert kept == [] and dropped == [("a", "overlaps a held-out task")]
