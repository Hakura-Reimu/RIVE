"""Phase C hard-set tasks (docs/phase-c-design.md section 2): maths, code and RTL items from Phase 0, split
into what a candidate may see (the public task) and what only the trusted judge sees. Prompts and judges are
the Phase 0 ones (phase0/baseline_eval.py), so the initial candidate reproduces the Phase 0 baseline."""
import json
import os
import random
import re
from pathlib import Path

from rive.benchmark.splits import split_of
from rive.sandbox.judge import run as judge_run

DATA = Path(os.path.expanduser("~/alter/data"))
LCB_RUNNER = Path(__file__).resolve().parents[1] / "sandbox" / "lcb_runner.py"
SOURCES = {"math": ("aime2025", "comp2026"), "code": ("livecodebench",), "rtl": ("verilog_eval", "rtllm")}
# Near-ceiling Phase 0 sets used as the easy-set guard (docs/phase-c-design.md section 4, gate 3). They carry the
# same domain labels, so a workflow treats them like the hard tasks of that domain.
EASY_SOURCES = (("math", "gsm8k"), ("code", "mbpp"))

# Found by phase0/check_hard_harness.py: these cannot score a design reliably under iverilog 12.
RTLLM_EXCLUDE = {
    "radix2_div": "reference times out in simulation",
    "ring_counter": "testbench uses whole-array assignment, unsupported by iverilog",
    "asyn_fifo": "reference fails under iverilog",
    "freq_divbyeven": "testbench syntax error under iverilog",
    "clkgenerator": "testbench prints no pass message",
    "comparator_3bit": "logic-free module passes (testbench compares with != so X never counts)",
    "comparator_4bit": "logic-free module passes",
    "sequence_detector": "logic-free module passes",
    "square_wave": "logic-free module passes",
}


def code_block(text, langs):
    for lang in langs:
        m = re.search(rf"```{lang}\s*\n(.*?)```", text, re.S | re.I)
        if m:
            return m.group(1)
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def rtllm_module(item):
    """The module name the description asks for (the reference file's name sometimes differs)."""
    m = re.search(r"Module name:\s*(\w+)", item["prompt"])
    return m.group(1) if m else item["module"]


def public_task(source, item):
    """What a candidate's workflow receives. "prompt" is the Phase 0 prompt; answers, tests, testbenches and
    reference designs stay with the judge."""
    # math500 and math_train (Phase D training problems, docs/phase-d-design.md) take the competition-maths prompt.
    if source in ("aime2025", "comp2026", "math500", "math_train"):
        return {"domain": "math", "source": source, "problem": item["problem"],
                "prompt": item["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
    if source == "gsm8k":
        return {"domain": "math", "source": source, "problem": item["question"],
                "prompt": item["question"] + "\nSolve step by step, then give the final answer on its own line as "
                                             "'#### <number>'."}
    if source == "mbpp":
        statement = f"{item['prompt']}\nYour code should pass this test:\n{item['tests'][0]}"
        return {"domain": "code", "source": source, "statement": statement, "starter_code": "", "testtype": "mbpp",
                "func_name": "", "prompt": statement + "\nWrite the Python function in a single ```python code block."}
    if source in ("livecodebench", "lcb_train"):  # lcb_train: LiveCodeBench release v1, Phase D training only
        if item["testtype"] == "functional":
            prompt = (f"{item['content']}\n\nComplete the following starter code in Python 3 and return the complete "
                      f"solution in a single ```python code block:\n```python\n{item['starter_code']}\n```")
        else:
            prompt = (f"{item['content']}\n\nWrite a Python 3 program that reads the input from standard input and "
                      "writes the answer to standard output. Return the complete program in a single ```python code "
                      "block.")
        return {"domain": "code", "source": source, "statement": item["content"], "starter_code": item["starter_code"],
                "testtype": item["testtype"], "func_name": item["func_name"], "prompt": prompt}
    if source == "verilog_eval":
        return {"domain": "rtl", "source": source, "spec": item["prompt"], "module": "TopModule",
                "language": "systemverilog",
                "prompt": item["prompt"].strip() + "\n\nWrite the complete SystemVerilog implementation of module "
                          "TopModule (synthesizable, IEEE 1800-2012). Return only the code in a single ```verilog "
                          "code block."}
    if source == "rtllm":
        module = rtllm_module(item)
        return {"domain": "rtl", "source": source, "spec": item["prompt"], "module": module, "language": "verilog",
                "prompt": item["prompt"].strip() + f"\n\nWrite the complete synthesizable Verilog implementation of "
                          f"module {module}. Return only the code in a single ```verilog code block."}
    raise ValueError(f"unknown source {source}")


def load_tasks(domains=("math", "code", "rtl"), splits=("dev",)):
    tasks = []
    for domain in domains:
        for source in SOURCES[domain]:
            for line in open(DATA / f"{source}.jsonl"):
                item = json.loads(line)
                if (source == "rtllm" and item["id"] in RTLLM_EXCLUDE) or split_of(item["id"]) not in splits:
                    continue
                tasks.append({"id": item["id"], "domain": domain, "source": source,
                              "public": public_task(source, item), "item": item})
    return tasks


def load_easy_tasks(per_source=50, seed=0):
    """A fixed sample of the easy guard sets (same items for every candidate)."""
    tasks = []
    for domain, source in EASY_SOURCES:
        items = [json.loads(line) for line in open(DATA / f"{source}.jsonl")]
        for item in sorted(random.Random(seed).sample(items, min(per_source, len(items))), key=lambda i: i["id"]):
            tasks.append({"id": item["id"], "domain": domain, "source": source,
                          "public": public_task(source, item), "item": item})
    return tasks


def _gsm8k_correct(item, text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.split("####")[-1] if "####" in text else text)
    pred = nums[-1].replace(",", "").rstrip(".") if nums else None
    gold = item["solution"].split("####")[-1].strip().replace(",", "")
    try:  # "16.00" and "16" are the same answer
        return pred is not None and float(pred) == float(gold)
    except ValueError:
        return pred == gold


def _sim_failure(r):
    if r.timed_out:
        return "time_limit"
    return "compile_error" if r.exit_code != 0 and not r.stdout.strip() else "mismatch"


def judge(task, answer):
    """(correct, failure, detail); failure is a short label (None when correct) for development feedback.
    Maths uses math-verify, which is not thread-safe (signal.alarm): call it from the main thread."""
    item, source = task["item"], task["source"]
    if source == "gsm8k":
        ok = _gsm8k_correct(item, answer)
        return ok, None if ok else "wrong_answer", {}
    if source == "mbpp":
        program = "\n".join(item["test_imports"]) + "\n" + code_block(answer, ["python", "py"]) + "\n\n" + "\n".join(
            item["tests"]) + "\n"
        r = judge_run({"main.py": program}, ["python", "main.py"], timeout=20)
        ok = r.exit_code == 0
        return ok, None if ok else "time_limit" if r.timed_out else "wrong_answer", {"stderr": r.stderr[-200:]}
    if task["domain"] == "math":
        from math_verify import parse, verify
        try:
            ok = bool(verify(parse("$" + item["answer"] + "$"), parse(answer)))
        except Exception:  # noqa: BLE001 - an unparseable answer is a wrong answer
            ok = False
        return ok, None if ok else "wrong_answer" if "\\boxed" in answer else "no_final_answer", {}
    if task["domain"] == "code":
        tests = (DATA / item.get("tests_dir", "lcb_tests") / f"{item['question_id']}.json").read_text()
        cmd = ["python", "/work/lcb_runner.py", item["testtype"]] + ([item["func_name"]] if item["func_name"] else [])
        r = judge_run({"lcb_runner.py": LCB_RUNNER.read_text(), "solution.py": code_block(answer, ["python", "py"]),
                       "tests.json": tests}, cmd, timeout=240, memory="1g")
        try:
            verdict = json.loads(r.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            verdict = {"passed": 0, "total": item["n_tests"],
                       "reason": "runner failed: " + (r.stderr or "timeout")[-200:]}
        if verdict["passed"] == verdict["total"] > 0:
            return True, None, verdict
        reason = verdict["reason"]
        failure = next((label for prefix, label in (("time limit", "time_limit"), ("runtime error", "runtime_error"),
                                                     ("load error", "load_error"), ("wrong answer", "wrong_answer"))
                        if reason.startswith(prefix)), "judge_error")
        return False, failure, verdict
    code = code_block(answer, ["verilog", "systemverilog", "sv"])
    if source == "verilog_eval":
        # Simulate from /tmp: the testbench writes wave.vcd to the cwd and /work is read-only.
        cmd = ["sh", "-c", "cd /tmp && iverilog -Wall -Winfloop -Wno-timescale -g2012 -s tb -o /tmp/test "
                           "/work/test.sv /work/TopModule.sv /work/ref.sv && vvp -n /tmp/test"]
        r = judge_run({"test.sv": item["testbench"], "ref.sv": item["reference"], "TopModule.sv": code}, cmd,
                      timeout=60)
        ok = bool(re.search(r"Mismatches: 0 in [1-9]\d* samples", r.stdout))
    else:
        # Some testbenches read data files from the cwd, so everything is copied to writable /tmp first.
        cmd = ["sh", "-c", "cp -r /work/. /tmp/ && cd /tmp && iverilog -g2012 -o /tmp/sim testbench.v design.v "
                           "&& vvp -n /tmp/sim"]
        r = judge_run({"testbench.v": item["testbench"], "design.v": code, **item["support"]}, cmd, timeout=120)
        ok = bool(re.search(r"Your Design Passed", r.stdout, re.I)) and not re.search(r"={3,}\s*Error", r.stdout)
    detail = {"exit": r.exit_code, "timed_out": r.timed_out, "tail": (r.stdout + r.stderr)[-300:]}
    return ok, None if ok else _sim_failure(r), detail
