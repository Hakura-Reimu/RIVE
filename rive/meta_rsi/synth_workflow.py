"""Component workflow for T2, program synthesis (RIVE, docs/rive-design.md sections 6.11 and 6.17). Fixed and tested:
improvers change only agent/prompts/config.md (rive/meta_rsi/synth.py writes this file into each agent)."""
import json
import math

DEFAULTS = {"system_prompt": "", "instruction": "", "thinking": False, "attempts": 1, "temperature": 0.7,
            "select": "first", "example_retry": 0, "feedback": "got", "review": False, "max_tokens": 4096}
FENCE = "`" * 3
# LiveCodeBench prepends these imports to every solution (rive/sandbox/lcb_runner.py).
PRELUDE = (
    "from typing import *\nfrom functools import *\nfrom collections import *\nfrom itertools import *\n"
    "from heapq import *\nfrom bisect import *\nfrom string import *\nfrom operator import *\nfrom math import *\n"
    "import math, string, re, heapq, bisect, itertools, collections, functools, operator, random, sys\n"
    "sys.setrecursionlimit(50000)\n"
)
ASK_AGAIN = "Fix the solution and return the complete corrected solution in a single " + FENCE + "python code block."
REVIEW = ("Check the solution against the statement: edge cases, and whether it is fast enough for the largest "
          "inputs the constraints allow. If something is wrong, fix it. Return the complete final solution in a "
          "single " + FENCE + "python code block.")


def code_of(text):
    start = text.find(FENCE + "python")
    if start < 0:
        return ""
    start = text.find("\n", start) + 1
    end = text.find(FENCE, start)
    return text[start:end] if end >= 0 else text[start:]


def tokens_equal(got, want):
    a, b = got.split(), want.split()
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y:
            try:
                if not math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-6):
                    return False
            except ValueError:
                return False
    return True


def values_equal(got, want):
    if isinstance(got, float) or isinstance(want, float):
        try:
            return math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    if isinstance(got, (list, tuple)) and isinstance(want, list):
        return len(got) == len(want) and all(values_equal(g, w) for g, w in zip(got, want))
    return got == want


def run_example(ctx, task, code, example):
    """(passed, what the solution gave) on one shown example."""
    if task["testtype"] == "stdin":
        r = ctx.tool("python_exec", {"code": PRELUDE + code, "stdin": example["input"], "seconds": 6})
        out = r["stdout"]
    else:
        args = [json.loads(line) for line in example["input"].split("\n") if line.strip()]
        call = "\nimport json as _json\nprint(_json.dumps(Solution()." + task["func_name"] + "(*" + repr(args) + ")))"
        r = ctx.tool("python_exec", {"code": PRELUDE + code + call, "seconds": 6})
        out = r["stdout"].strip().splitlines()[-1] if r["stdout"].strip() else ""
    if r["timed_out"]:
        return False, "time limit exceeded"
    if r["returncode"] != 0:
        return False, "runtime error:\n" + r["stderr"][-800:]
    if task["testtype"] == "stdin":
        return tokens_equal(out, example["output"]), out[-800:]
    try:
        return values_equal(json.loads(out), json.loads(example["output"])), out[-800:]
    except ValueError:
        return False, out[-800:]


def first_failure(ctx, task, code, examples):
    """None when the code passes every shown example; else (example, what it gave)."""
    for example in examples:
        ok, got = run_example(ctx, task, code, example)
        if not ok:
            return example, got
    return None


def call(ctx, messages, config):
    """The reply to `messages`, its thinking already removed by the runner; None once the budget is used up. Every
    call, retries and the review included, samples at the configured temperature (section 6.17)."""
    try:
        return ctx.llm(messages, max_tokens=config["max_tokens"], thinking=config["thinking"],
                       temperature=config["temperature"])
    except RuntimeError:  # the task's token budget is used up
        return None


def solve(task, ctx):
    try:
        config = {**DEFAULTS, **json.loads(ctx.prompt("config") or "{}")}
    except ValueError:
        config = dict(DEFAULTS)
    system, examples = config["system_prompt"], task.get("examples") or []
    user = task["prompt"] + ("\n\n" + config["instruction"] if config["instruction"] else "")
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    answers = []
    for _ in range(config["attempts"]):
        text = call(ctx, messages, config)
        if text is None:
            break
        answers.append(text)
    if not answers:
        return {"answer": ""}
    answer = answers[0]
    if config["select"] == "passes_examples" and examples:
        answer = next((a for a in answers if code_of(a) and first_failure(ctx, task, code_of(a), examples) is None),
                      answer)
    elif config["select"] == "majority":
        codes = [" ".join(code_of(a).split()) for a in answers]
        answer = answers[codes.index(max(codes, key=codes.count))]
    history = list(messages)
    for _ in range(config["example_retry"]):
        code = code_of(answer)
        if not examples or not code:
            break
        failed = first_failure(ctx, task, code, examples)
        if failed is None:
            break
        example, got = failed
        shown = "On one of the examples in the statement, that solution gave:\n" + got
        if config["feedback"] == "got_and_expected":
            shown += "\nThe example's input:\n" + example["input"][:1500] + "\nIts expected output:\n" + \
                example["output"][:1500]
        history = history + [{"role": "assistant", "content": answer},
                             {"role": "user", "content": shown + "\n" + ASK_AGAIN}]
        text = call(ctx, history, config)
        if text is None:
            break
        answer = text
    if config["review"]:
        text = call(ctx, history + [{"role": "assistant", "content": answer}, {"role": "user", "content": REVIEW}],
                    config)
        if text and code_of(text):
            answer = text
    return {"answer": answer}
