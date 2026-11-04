"""Hard-set tasks and judges on the real Phase 0 data (needs ~/alter/data and docker; skipped otherwise)."""
import re
import shutil
from collections import Counter

import pytest

from rive.benchmark import hard

pytestmark = pytest.mark.skipif(not (hard.DATA / "aime2025.jsonl").exists() or shutil.which("docker") is None,
                                reason="needs the Phase 0 data and docker")

HIDDEN = {"answer", "testbench", "reference", "support", "n_tests", "question_id"}


def test_public_tasks_hide_answers_and_tests():
    tasks = hard.load_tasks(splits=("dev", "selection", "anchor", "sealed"))
    assert len(tasks) == 465 and {t["domain"] for t in tasks} == {"math", "code", "rtl"}
    assert all(not HIDDEN & set(t["public"]) and t["item"]["id"] == t["id"] for t in tasks)


def test_dev_split_matches_the_design_doc():
    assert Counter(t["domain"] for t in hard.load_tasks()) == {"math": 39, "code": 66, "rtl": 80}


def test_math_judge():
    t = hard.load_tasks(("math",))[0]
    assert hard.judge(t, f"so \\boxed{{{t['item']['answer']}}}")[:2] == (True, None)
    assert hard.judge(t, "\\boxed{999999}")[1] == "wrong_answer"
    assert hard.judge(t, "it is large")[1] == "no_final_answer"


def test_rtl_judges_accept_reference_designs():
    rtl = hard.load_tasks(("rtl",))
    ve = next(t for t in rtl if t["source"] == "verilog_eval" and "Prob099" not in t["id"])
    ref = ve["item"]["reference"].replace("RefModule", "TopModule")
    assert hard.judge(ve, f"```verilog\n{ref}\n```")[0]
    assert hard.judge(ve, "```verilog\nmodule TopModule(\n```")[1] == "compile_error"
    rt = next(t for t in rtl if t["source"] == "rtllm")
    # Reference files declare `module verified_<name>`; the testbench instantiates the name the spec asks for.
    ref = re.sub(r"\bmodule\s+\w+", f"module {rt['public']['module']}", rt["item"]["reference"], count=1)
    assert hard.judge(rt, f"```verilog\n{ref}\n```")[0]


def test_easy_guard_tasks_and_judges():
    easy = hard.load_easy_tasks(per_source=5)
    assert Counter(t["source"] for t in easy) == {"gsm8k": 5, "mbpp": 5} and easy == hard.load_easy_tasks(5)
    g = next(t for t in easy if t["source"] == "gsm8k")
    gold = g["item"]["solution"].split("####")[-1].strip()
    assert hard.judge(g, f"so\n#### {gold}")[0] and not hard.judge(g, "#### 123456789")[0]
    m = next(t for t in easy if t["source"] == "mbpp")
    assert hard.judge(m, f"```python\n{m['item']['reference']}\n```")[0]
    assert hard.judge(m, "```python\ndef nothing():\n    pass\n```")[1] == "wrong_answer"
    assert not HIDDEN & set(m["public"]) and "solution" not in g["public"]


def test_code_judge_labels_failures():
    code = hard.load_tasks(("code",))
    stdin = next(t for t in code if t["item"]["testtype"] == "stdin")
    assert hard.judge(stdin, "```python\nprint(\n```")[1] == "runtime_error"
    functional = next(t for t in code if t["item"]["testtype"] == "functional")
    assert hard.judge(functional, "```python\nclass Solution:\n    pass\n```")[1] == "load_error"
