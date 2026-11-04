"""Phase C research step with a scripted fake model (needs docker for the smoke test and tool tests)."""
import json
import random
import re
import shutil
import threading
from pathlib import Path

import pytest

from rive.archive.db import Archive
from rive.benchmark.leakscan import LeakScanner
from rive.evolution.phase_c import experiment_history
from rive.research.agent_loop import (MAX_TOOL_LINES, TOOL_EXAMPLE, AgentResearch, affected_domains, args_passed,
                                        assign_problem, blocked_kinds, blocked_modules, blocks, called_tools,
                                        fields_read, observe_agent, over_call_cap, prediction_check, schema_keys,
                                        tool_shape)
from rive.research.loop import check_python_source
from rive.sandbox.container import CandidateSandbox

TEMPLATE = Path(__file__).resolve().parents[1] / "candidate_template"
docker = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")

WORKFLOW = ("def solve(task, ctx):\n"
            "    fence = '`' * 3\n"
            "    text = ctx.llm([{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': task['prompt']}],\n"
            "                   max_tokens=2048)\n"
            "    return {'answer': text}\n")
TOOL = ("SCHEMA = {'args': {'code': 'str'}, 'result': {'ok': 'bool'}}\n"
        "def run(args, public):\n"
        "    return {'ok': public.python_exec(args['code'])['returncode'] == 0}\n"
        "def test(public):\n"
        "    assert run({'code': 'print(1)'}, public)['ok']\n"
        "    assert not run({'code': 'raise SystemExit(3)'}, public)['ok']\n")
TOOL_WORKFLOW = ("def solve(task, ctx):\n"
                 "    text = ctx.llm([{'role': 'user', 'content': task['prompt']}], max_tokens=4096)\n"
                 "    if task['domain'] == 'code':\n"
                 "        ctx.tool('runs_ok', {'code': 'print(0)'})\n"
                 "    return {'answer': text}\n")


def tasks():
    out = {}
    for d, src in (("math", "aime2025"), ("code", "livecodebench"), ("rtl", "verilog_eval")):
        for i in range(6):
            tid = f"{d}{i}"
            out[tid] = {"id": tid, "domain": d, "source": src, "item": {},
                        "public": {"domain": d, "source": src, "prompt": f"{d} task {i}", "module": "TopModule"}}
    return out


def parent_results(ts):
    fail = {"math": "no_final_answer", "code": "wrong_answer", "rtl": "mismatch"}
    return [{"id": t["id"], "domain": t["domain"], "correct": t["id"].endswith(("0", "1")),
             "failure": None if t["id"].endswith(("0", "1")) else fail[t["domain"]], "tokens": 4096,
             "truncated_calls": 1, "tool_calls": [], "answer": "...", "detail": {"tail": "Mismatches: 3 in 20 samples"}}
            for t in ts.values()]


class FakeLLM:
    """Answers by stage; `implement` is the list of IMPLEMENT answers, one file each (the last one repeats).
    Stages are told apart by their schema's fields: a blocked kind of module narrows the schema to a copy."""

    def __init__(self, target="agent/workflow.py", tool_name="", implement=None):
        self.target, self.tool_name = target, tool_name
        self.implement = list(implement or [f"```python\n{WORKFLOW}```"])
        self.implement_prompts, self.schemas, self.stage_prompts = [], [], []

    def chat(self, messages, schema=None, max_tokens=0, temperature=0.0, thinking=False, seed=0):
        last = messages[-1]["content"]
        if schema is None:
            if "IMPLEMENT" in last:
                self.implement_prompts.append(messages)
                return self.implement.pop(0) if len(self.implement) > 1 else self.implement[0]
            return "Some analysis."
        self.schemas.append(schema)
        self.stage_prompts.append(messages)
        fields = schema["properties"]
        if "selected_domain" in fields:
            # An assigned problem narrows the schema to one domain and one failure type.
            domains, problems = fields["selected_domain"]["enum"], fields["selected_problem"]["enum"]
            d = domains[0] if len(domains) == 1 else "math"
            f = problems[0] if len(problems) == 1 else "no_final_answer"
            text = " ".join(m["content"] for m in messages)
            ids = re.findall(rf'"id": "({d}\d)", "domain": "{d}", "failure": "{f}"', text)
            return json.dumps({"selected_domain": d, "selected_problem": f,
                               "evidence": [{"domain": d, "failure_type": f, "count": 4, "example_ids": ids[:1]}],
                               "suspected_cause": "token_budget", "explanation": "cut off",
                               "recommended_next_action": "shorter reasoning"})
        h = {"hypothesis": "ask for brevity", "target": self.target, "tool_name": self.tool_name,
             "proposed_change": "system prompt", "prediction": {"metric": "math_accuracy", "direction": "increase"},
             "control": "current agent", "potential_regression": "less careful reasoning"}
        if "hypotheses" in fields:
            return json.dumps({"hypotheses": [{"hypothesis_id": "H1", **h}, {"hypothesis_id": "H2", **h}],
                               "selected": "H1", "why_selected": "cheap"})
        if "change_steps" in fields:
            return json.dumps({"target": self.target, "tool_name": self.tool_name, "change_steps": ["do it"],
                               "primary_metric": "math_accuracy", "expected_direction": "increase",
                               "rollback_condition": "math accuracy drops"})
        if "reusable_finding" in fields:  # Phase C's REFLECT leaves the conclusion to the controller
            out = {"summary": "ok", "limitations": "small", "next_direction": "more",
                   "reusable_finding": "brevity helps"}
            return json.dumps({"conclusion": "supported", **out} if "conclusion" in fields else out)
        raise AssertionError("unexpected schema")


class NoSandbox:
    """Stands in for the container where a test checks the research flow, not the code: every run passes."""

    def __init__(self, cand, memory=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def tool_test(self, name):
        return {"ok": True}

    def solve(self, task, seconds, on_llm):
        return {"ok": True, "answer": "x", "tool_calls": []}


def research(tmp_path, llm, **kw):
    ts = tasks()
    smoke = [next(t for t in ts.values() if t["domain"] == d) for d in ("math", "code", "rtl")]
    parent = tmp_path / "parent"
    shutil.copytree(TEMPLATE, parent)
    return AgentResearch(llm, tmp_path / "run", smoke, **kw), parent, ts


def test_blocks_ignore_fences_inside_code():
    text = "Here:\n```python\nx = re.search(r\"```python(.*?)```\", s)\n```\nand\n```markdown\nhi\n```\n"
    assert blocks(text, {"python"}) == ['x = re.search(r"```python(.*?)```", s)\n']
    assert blocks(text, {"markdown"}) == ["hi\n"]


def test_feedback_is_per_domain_and_from_records():
    ts = tasks()
    fb = observe_agent(parent_results(ts), ts, seed=3)
    assert fb["failure_counts"]["rtl"] == {"mismatch": 4} and fb["accuracy_by_domain"]["math"] == 0.3333
    assert {e["domain"] for e in fb["failure_examples"]} == {"math", "code", "rtl"}
    assert fb["answers_cut_off_by_token_limit"]["code"] == 6 and "mismatch" in fb["failure_meanings"]


@docker
def test_workflow_change_is_validated_and_hashed(tmp_path):
    loop, parent, ts = research(tmp_path, FakeLLM())
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    assert child["status"] == "valid" and child["module"] == "agent/workflow.py" and child["code_hash"]
    assert child["record"]["checks"]["diagnose_evidence_matches_records"]
    assert "Be brief." in (tmp_path / "run" / "c1" / "agent" / "workflow.py").read_text()
    again = loop.propose("c2", parent, parent_results(ts), ts, [child.get("note") or "c1 tried brevity"], seed=2)
    assert again["status"] == "duplicate" and "repeated" in again["note"]


@docker
def test_prompt_change(tmp_path):
    loop, parent, ts = research(tmp_path, FakeLLM("agent/prompts/math.md",
                                                  implement=["```markdown\nKeep the reasoning short.\n```"]))
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    assert child["status"] == "valid"
    assert (tmp_path / "run" / "c1" / "agent" / "prompts" / "math.md").read_text() == "Keep the reasoning short.\n"


@docker
def test_tool_with_failing_test_is_repaired(tmp_path):
    broken = TOOL.replace("returncode'] == 0", "returncode'] != 0")
    llm = FakeLLM("tool", "Runs OK!", implement=[f"```python\n{broken}```", f"```python\n{TOOL_WORKFLOW}```",
                                                f"```python\n{TOOL}```", f"```python\n{TOOL_WORKFLOW}```"])
    loop, parent, ts = research(tmp_path, llm)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    assert child["status"] == "valid" and child["module"] == "tools/runs_ok.py"
    assert child["record"]["stages"]["implement"]["attempts"] == 2
    assert "ctx.tool('runs_ok'" in (tmp_path / "run" / "c1" / "agent" / "workflow.py").read_text()
    # The working example tool is part of the request, once.
    assert llm.implement_prompts[0][1]["content"].count(TOOL_EXAMPLE) == 1
    assert "ctx.tool: python_exec, runs_ok, verilog_sim." in llm.implement_prompts[0][1]["content"]
    # The repair sees the failed attempt's two files once, not the whole earlier conversation.
    repair_step1, repair_step2 = llm.implement_prompts[2:]
    assert [m["role"] for m in repair_step2] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "returncode'] != 0" in repair_step1[2]["content"] and "failed validation" in repair_step1[3]["content"]


def test_implement_prompts_stay_bounded_and_refusals_cost_one_child(tmp_path):
    import httpx
    from openai import BadRequestError
    bad = "def solve(task, ctx):\n    return {'answer': 1}\nimport os\n"
    llm = FakeLLM(implement=[f"```python\n{bad}```"])
    loop, parent, ts = research(tmp_path, llm)
    child = loop.propose("c1", parent, parent_results(ts), ts, ["an earlier note"], seed=1)
    assert child["status"] == "invalid_patch" and len(llm.implement_prompts) == 3
    assert {len(p) for p in llm.implement_prompts[1:]} == {4}  # system, plan, last attempt, what failed
    assert "Development feedback" not in llm.implement_prompts[0][1]["content"]

    class Refusing(FakeLLM):
        def chat(self, messages, schema=None, **kw):
            if schema is None and "IMPLEMENT" in messages[-1]["content"]:
                raise BadRequestError("prompt too long", response=httpx.Response(
                    400, request=httpx.Request("POST", "http://server")), body=None)
            return super().chat(messages, schema, **kw)

    loop, parent, ts = research(tmp_path / "second", Refusing())
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    assert child["status"] == "stage_failed" and "too long" in child["record"]["error"]


@docker
def test_workflow_that_breaks_on_stand_in_answers_is_rejected(tmp_path):
    crash = "def solve(task, ctx):\n    return {'answer': str(int(ctx.llm([{'role': 'user', 'content': 'x'}])))}\n"
    loop, parent, ts = research(tmp_path, FakeLLM(implement=[f"```python\n{crash}```"]))
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    assert child["status"] == "invalid_patch" and "stand-in" in child["note"]


@docker
def test_a_change_outside_the_assigned_domain_is_sent_back(tmp_path):
    # WORKFLOW asks every domain for brevity; the repair keeps the initial agent's calls outside RTL.
    local = ("def solve(task, ctx):\n"
             "    system = ctx.prompt(task['domain'])\n"
             "    messages = ([{'role': 'system', 'content': system}] if system else []) + [\n"
             "        {'role': 'user', 'content': task['prompt']}]\n"
             "    if task['domain'] == 'rtl':\n"
             "        messages.insert(0, {'role': 'system', 'content': 'Check every port.'})\n"
             "    return {'answer': ctx.llm(messages, max_tokens=4096)}\n")
    llm = FakeLLM(implement=[f"```python\n{WORKFLOW}```", f"```python\n{local}```"])
    loop, parent, ts = research(tmp_path, llm)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1, problem=("rtl", "mismatch"))
    assert child["status"] == "valid" and child["local_to"] == "rtl"
    assert child["record"]["stages"]["implement"]["attempts"] == 2
    assert child["record"]["checks"]["other_domains_unchanged"]
    assert 'under if task["domain"] == "rtl"' in llm.implement_prompts[0][1]["content"]
    assert ("also alters how other domains' tasks are solved (on code task code0 its model calls differ"
            in llm.implement_prompts[1][-1]["content"])


def screened(answers):
    """A stand-in for run_workflow_tasks: the n-th pre-screen gets answers[n] for every task ("" is wrong)."""
    calls = []

    def run_tasks(tasks, cand, solver, workers):
        answer = answers[min(len(calls), len(answers) - 1)]
        calls.append(sorted(t["id"] for t in tasks))
        return [{"id": t["id"], "domain": t["domain"], "correct": bool(answer), "failure": None if answer else
                 "no_final_answer", "answer": answer, "llm_calls": 1, "seconds": 1.0} for t in tasks]
    return run_tasks, calls


def test_prescreen_sends_back_a_change_that_loses_solved_tasks(tmp_path):
    run_tasks, calls = screened(["", "\\boxed{1}"])
    llm = FakeLLM()
    loop, parent, ts = research(tmp_path, llm, make_sandbox=NoSandbox, solver=object(), run_tasks=run_tasks)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1, problem=("rtl", "mismatch"))
    assert child["status"] == "valid" and child["record"]["stages"]["implement"]["attempts"] == 2
    assert calls == [["rtl0", "rtl1"]] * 2  # the RTL tasks the parent solved
    assert child["record"]["checks"]["prescreen_passed"] and len(child["record"]["stages"]["prescreen"]) == 2
    repair = llm.implement_prompts[1][-1]["content"]
    assert "the pre-screen ran this version" in repair and "got 2 wrong" in repair and "empty answer" in repair
    assert "development tasks that the current agent solves" in llm.implement_prompts[0][1]["content"]


def test_prescreen_covers_the_domains_a_change_can_affect(tmp_path):
    run_tasks, calls = screened(["x"])
    loop, parent, ts = research(tmp_path, FakeLLM(), make_sandbox=NoSandbox, solver=object(), run_tasks=run_tasks)
    loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)  # no assigned problem: the workflow, everywhere
    prompt = FakeLLM("agent/prompts/math.md", implement=["```markdown\nKeep the reasoning short.\n```"])
    loop, parent, ts = research(tmp_path / "p", prompt, make_sandbox=NoSandbox, solver=object(), run_tasks=run_tasks)
    loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    assert calls == [["code0", "code1", "math0", "math1", "rtl0", "rtl1"], ["math0", "math1"]]


def test_an_extra_check_sends_a_change_back_like_the_others(tmp_path):
    verdicts = [["it fixed nothing"], []]
    seen = []

    def extra_check(child_dir, parent_results, tasks_by_id, record, seed):
        seen.append(seed)
        return verdicts[len(seen) - 1]
    llm = FakeLLM()
    loop, parent, ts = research(tmp_path, llm, make_sandbox=NoSandbox, extra_check=extra_check)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    assert child["status"] == "valid" and child["record"]["stages"]["implement"]["attempts"] == 2
    assert seen == [1, 2] and "it fixed nothing" in llm.implement_prompts[1][-1]["content"]


def test_prescreen_runs_only_on_the_main_thread(tmp_path):
    run_tasks, _ = screened(["x"])
    loop, parent, ts = research(tmp_path, FakeLLM(), make_sandbox=NoSandbox, solver=object(), run_tasks=run_tasks)
    errors = []

    def go():
        try:
            loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
        except RuntimeError as e:
            errors.append(str(e))
    worker = threading.Thread(target=go)
    worker.start()
    worker.join()
    assert errors and "main thread" in errors[0]


def verdict(**kw):
    """A controller verdict (evolution.controller.judge) for REFLECT."""
    return {"domains": ["math", "code", "rtl"], "diff": 0.0, "p": 0.5, "p_worse": 0.5, "sd": 0.02,
            "champion_runs": 3, "child_runs": 1, "regression": 0.0, "expected_regression": 0.1, "pass_dev": False,
            "pass_regression": True, "alpha": 0.05, **kw}


def test_reflect_writes_the_note_for_descendants(tmp_path):
    loop, parent, ts = research(tmp_path, FakeLLM(), make_sandbox=NoSandbox)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    res = parent_results(ts)
    note = loop.reflect(child, verdict(domains=["math"], diff=0.05, p=0.02, p_worse=0.98, pass_dev=True), [res], [res])
    assert "brevity helps" in note and "+0.050 on math" in note
    assert note in (tmp_path / "run" / "c1" / "memory" / "notes.md").read_text()


def test_reflect_conclusion_follows_the_measurements(tmp_path):
    """C5's model wrote "refuted" for 24 of 26 results; the conclusion now comes from the comparison."""
    loop, parent, ts = research(tmp_path, FakeLLM(), make_sandbox=NoSandbox)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    champ = parent_results(ts)
    better = [dict(r, correct=True, failure=None) if r["domain"] == "math" else r for r in champ]
    record = child["record"]
    loop.reflect(child, verdict(diff=0.2, p=0.001, p_worse=0.999, pass_dev=True), [champ], [better])
    assert record["verdict"] == "supported" and record["prediction_check"]["hit"] is True
    loop.reflect(child, verdict(diff=-0.01, p=0.7, p_worse=0.3), [champ], [champ])
    assert record["verdict"] == "inconclusive" and record["prediction_check"]["hit"] is None
    loop.reflect(child, verdict(diff=-0.2, p=0.999, p_worse=0.001), [champ], [champ])
    assert record["verdict"] == record["stages"]["reflect"]["output"]["conclusion"] == "refuted"
    assert "within run-to-run noise" not in loop.llm.stage_prompts[-1][1]["content"]


def test_prediction_about_a_domain_the_change_cannot_touch_is_not_testable():
    res = parent_results(tasks())
    check = prediction_check({"metric": "math_accuracy", "direction": "increase"}, [res], [res], ["code"])
    assert check["hit"] is None and "cannot affect" in check["why"]
    fewer = prediction_check({"metric": "wrong_answer_count", "direction": "decrease"}, [res],
                             [[dict(r, failure=None, correct=True) for r in res]], ["code"])
    assert fewer["hit"] is True and (fewer["champion"], fewer["child"]) == (4, 0)


def test_problems_are_spread_over_a_generation():
    res = parent_results(tasks())  # 4 failing tasks per domain: maths cut off, code wrong, RTL mismatch
    taken = []
    for seed in range(3):
        taken.append(assign_problem(res, [], taken, random.Random(seed)))
    assert sorted(taken) == [("code", "wrong_answer"), ("math", "no_final_answer"), ("rtl", "mismatch")]
    assert assign_problem(res, [], taken, random.Random(0)) in taken  # all taken: one of them again
    closed = [{"domain": "rtl", "failure": "mismatch", "module": None, "failures": 6, "example": "x"}]
    assert all(assign_problem(res, closed, [], random.Random(s)) != ("rtl", "mismatch") for s in range(20))
    assert assign_problem(res, [], [], random.Random(0), min_count=5) is None


def test_an_assigned_problem_narrows_the_diagnosis(tmp_path):
    llm = FakeLLM()
    loop, parent, ts = research(tmp_path, llm, make_sandbox=NoSandbox)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1, problem=("rtl", "mismatch"))
    assert child["status"] == "valid" and child["record"]["kind"][:2] == ("rtl", "mismatch")
    assert llm.schemas[0]["properties"]["selected_problem"]["enum"] == ["mismatch"]
    assert "assigned one problem: rtl mismatch (4 failing tasks)" in llm.stage_prompts[0][2]["content"]
    assert child["record"]["checks"]["diagnose_evidence_matches_records"]


def test_affected_domains(tmp_path):
    reference = tmp_path / "reference"
    shutil.copytree(TEMPLATE, reference)

    def variant(name, base=reference, **files):
        out = tmp_path / name
        shutil.copytree(base, out)
        for path, text in files.items():
            (out / path).parent.mkdir(parents=True, exist_ok=True)
            (out / path).write_text(text)
        return out
    code_prompt = {"agent/prompts/code.md": "Be brief.\n", "memory/notes.md": "- a note\n"}
    assert affected_domains(variant("prompt", **code_prompt), reference) == ["code"]
    assert affected_domains(variant("flow", **code_prompt, **{"agent/workflow.py": WORKFLOW}), reference) is None
    # When the workflow reads another domain's prompt, a prompt change can reach every domain.
    shared = variant("shared", **{"agent/workflow.py": (
        "def solve(task, ctx):\n    rules = ctx.prompt('math')\n"
        "    return {'answer': ctx.llm([{'role': 'system', 'content': rules}, "
        "{'role': 'user', 'content': task['prompt']}])}\n")})
    assert affected_domains(variant("shared_math", base=shared, **{"agent/prompts/math.md": "x\n"}), shared) is None


def test_a_tool_given_arguments_it_does_not_take_is_rejected(tmp_path):
    """C5's g08c033: TOOL_EXAMPLE, a Python runner, under a Verilog name and given a design and a testbench."""
    misuse = ("def solve(task, ctx):\n    text = ctx.llm([{'role': 'user', 'content': task['prompt']}])\n"
              "    ctx.tool('verilog_sim_runner', {'design': text, 'testbench': ''})\n    return {'answer': text}\n")
    llm = FakeLLM("tool", "verilog_sim_runner",
                  implement=[f"```python\n{TOOL_EXAMPLE}```", f"```python\n{misuse}```"] * 3)
    loop, parent, ts = research(tmp_path, llm, make_sandbox=NoSandbox)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    assert child["status"] == "invalid_patch" and child["record"]["stages"]["implement"]["problems"] == [
        "agent/workflow.py passes design, testbench to ctx.tool('verilog_sim_runner'), which takes only code, stdin"]
    # The same example tool used as what it is, a Python runner, is a valid tool; its likeness is only recorded.
    use = misuse.replace("verilog_sim_runner", "run_python").replace("{'design': text, 'testbench': ''}",
                                                                      "{'code': text}")
    llm = FakeLLM("tool", "run_python", implement=[f"```python\n{TOOL_EXAMPLE}```", f"```python\n{use}```"])
    loop, parent, ts = research(tmp_path / "used", llm, make_sandbox=NoSandbox)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    assert child["status"] == "valid" and child["record"]["checks"]["tool_example_similarity"] == 1.0
    assert args_passed(use) == {"run_python": {"code"}} and schema_keys(TOOL_EXAMPLE, "args") == {"code", "stdin"}


def test_a_workflow_reading_fields_its_tool_does_not_return_is_rejected(tmp_path):
    """C5's g08c033: its RTL check read "valid" from a tool that returns crashed, stdout and error."""
    reads = ("def solve(task, ctx):\n    text = ctx.llm([{'role': 'user', 'content': task['prompt']}])\n"
             "    check = ctx.tool('runs_ok', {'code': 'print(0)'})\n"
             "    if not check.get('valid', True) or check['ok'] is False:\n"
             "        text = ctx.llm([{'role': 'user', 'content': 'again'}])\n    return {'answer': text}\n")
    llm = FakeLLM("tool", "runs_ok", implement=[f"```python\n{TOOL}```", f"```python\n{reads}```"] * 3)
    loop, parent, ts = research(tmp_path, llm, make_sandbox=NoSandbox)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    assert child["status"] == "invalid_patch" and child["record"]["stages"]["implement"]["problems"] == [
        "agent/workflow.py reads valid from ctx.tool('runs_ok'), whose result has only ok"]
    assert fields_read(reads) == {"runs_ok": {"valid", "ok"}} and schema_keys(TOOL, "result") == {"ok"}
    assert tool_shape("SCHEMA = {'args': {}}\ndef test(public):\n    pass\n") == [
        'the tool\'s SCHEMA has no literal "result" dict naming the fields run() returns']


def test_leak_scan_flags_protected_statements_but_not_shared_boilerplate():
    boiler = "I would like you to implement a module named TopModule with the following interface"
    item = lambda tid, spec: {"id": tid, "domain": "rtl", "item": {"prompt": spec, "reference": "module RefModule;"}}
    protected = [item("p1", boiler + " a seven segment decoder driven by a rotating priority arbiter for four users")]
    visible = [item("d1", boiler + " an adder")]
    scanner = LeakScanner().add_hard_tasks(protected, visible)
    assert not scanner.scan({"agent/prompts/rtl.md": boiler})["violations"]
    copied = "a seven segment decoder driven by a rotating priority arbiter for four users"
    hit = scanner.scan({"agent/prompts/rtl.md": copied})
    assert hit["violations"] and hit["violations"][0]["item"] == "p1"


def test_leak_scan_lets_common_rtl_idioms_through_but_not_copied_designs():
    reference = ("module multi_booth_8bit(input clk, input reset, output reg [15:0] p);\n"
                 "  always @(posedge clk or posedge reset) begin\n    if (reset) begin\n      p <= 16'd0;\n"
                 "      count <= 5'd0;\n    end else if (count < 5'd16) begin\n      p <= p + multiplicand;\n"
                 "      count <= count + 1;\n    end\n  end\nendmodule\n")
    protected = [{"id": "multi_booth_8bit", "domain": "rtl",
                  "item": {"prompt": "Implement a booth multiplier. Use always @(posedge clk or posedge reset) begin "
                                     "if (reset) to clear the product.", "reference": reference}}]
    scanner = LeakScanner().add_hard_tasks(protected, [])
    # The C4 pilot's rejected RTL prompt (g01c004) had this idiom in an example.
    idiom = "always @(posedge clk or posedge reset) begin\n  if (reset) q <= 10'd0;\n  else q <= d;\nend\n"
    assert not scanner.scan({"agent/prompts/rtl.md": idiom})["violations"]
    assert scanner.scan({"agent/prompts/rtl.md": reference})["violations"]


def test_blocking_follows_failures_in_a_row():
    def h(kind, success, text="an idea"):
        return {"kind": kind, "success": success, "hypothesis": text}
    tool, workflow, prompt = (("rtl", "compile_error", m) for m in ("tool", "workflow", "prompt"))
    tried = [h(tool, False), h(tool, False)]
    assert blocked_kinds(tried) == []
    tried.append(h(tool, False, "a third checker"))
    assert [(b["module"], b["failures"], b["example"]) for b in blocked_kinds(tried)] == [
        ("tool", 3, "a third checker")]
    # Passing the development gate resets the count; an experiment without a result neither counts nor resets.
    assert blocked_kinds(tried + [h(tool, True)]) == []
    assert blocked_kinds(tried + [h(tool, None)])
    # Six failures in a row through any modules block the problem itself.
    mixed = [h(tool, False), h(tool, False), h(workflow, False), h(workflow, False), h(prompt, False),
             h(workflow, False)]
    blocked = blocked_kinds(mixed)
    assert [b["module"] for b in blocked] == [None, "workflow"]
    assert blocked_modules(blocked, "rtl", "compile_error") == {"workflow", "prompt", "tool"}
    assert blocked_modules(blocked, "rtl", "mismatch") == set()
    # A block lasts until BLOCK_FOR (8) more experiments have been made; one more failure then blocks again.
    later = [h(None, None)] * 7
    assert blocked_kinds(tried + later) and not blocked_kinds(tried + later + [h(None, None)])
    assert blocked_kinds(tried + later + [h(None, None), h(tool, False)])


def test_history_comes_from_the_archive(tmp_path):
    archive = Archive(tmp_path / "archive.sqlite")
    archive.add_candidate("seed", None, 0, tmp_path, "seed", status="evaluated")

    def record(target):
        return {"hypothesis": {"hypothesis": f"change {target}"}, "stages": {
            "diagnose": {"output": {"selected_domain": "rtl", "selected_problem": "compile_error"}},
            "design": {"output": {"target": target}}}}
    for cid, status, target in (("c1", "evaluated", "tool"), ("c2", "invalid_patch", "agent/workflow.py"),
                                ("c3", "interrupted", "agent/prompts/rtl.md"), ("c4", "evaluated", "tool"),
                                ("c5", "stage_failed", None)):
        archive.add_candidate(cid, "seed", 1, tmp_path, "research", status=status)
        archive.add_experiment(cid, record(target) if target else {"stages": {}})
    archive.log("gates", {"candidate": "c1", "pass_dev": False})
    archive.log("gates", {"candidate": "c4", "pass_dev": True})
    hist = experiment_history(archive)
    assert [(h["id"], h["kind"], h["success"]) for h in hist] == [
        ("c1", ("rtl", "compile_error", "tool"), False), ("c2", ("rtl", "compile_error", "workflow"), False),
        ("c3", ("rtl", "compile_error", "prompt"), None), ("c4", ("rtl", "compile_error", "tool"), True),
        ("c5", None, False)]
    assert hist[0]["hypothesis"] == "change tool"


def test_blocked_problem_is_refused_and_blocked_modules_are_left_out(tmp_path):
    class Stubborn(FakeLLM):
        """Picks the blocked RTL compile errors first, as the C5 run did."""
        diagnoses = 0

        def chat(self, messages, schema=None, **kw):
            if schema is not None and "selected_domain" in schema["properties"]:
                self.diagnoses += 1
                if self.diagnoses == 1:
                    return json.dumps({"selected_domain": "rtl", "selected_problem": "compile_error", "evidence": [
                        {"domain": "rtl", "failure_type": "compile_error", "count": 0, "example_ids": []}],
                        "suspected_cause": "missing_tool", "explanation": "x", "recommended_next_action": "x"})
            return super().chat(messages, schema, **kw)

    blocked = [{"domain": "rtl", "failure": "compile_error", "module": None, "failures": 9, "example": "a checker"},
               {"domain": "math", "failure": "no_final_answer", "module": "tool", "failures": 3, "example": "a tool"}]
    llm = Stubborn()
    loop, parent, ts = research(tmp_path, llm, make_sandbox=NoSandbox)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1, blocked=blocked)
    assert child["status"] == "valid"
    diagnose = child["record"]["stages"]["diagnose"]
    assert diagnose["attempts"] == 2 and "blocked for every module" in diagnose["errors"][0]
    assert child["record"]["kind"] == ("math", "no_final_answer", "workflow")
    assert child["record"]["blocked"] == blocked
    hyp_schema, design_schema = (s for s in llm.schemas if "hypotheses" in s["properties"] or "change_steps" in
                                 s["properties"])
    assert hyp_schema["properties"]["hypotheses"]["items"]["properties"]["target"]["enum"] == [
        "agent/workflow.py", "agent/prompts/math.md", "agent/prompts/code.md", "agent/prompts/rtl.md"]
    assert "tool" not in design_schema["properties"]["target"]["enum"]
    first = llm.stage_prompts[0]
    assert "rtl compile_error through any module: failed 9 times in a row" in first[1]["content"]
    assert "cannot be selected now: rtl compile_error." in first[2]["content"]
    hyp_request = next(p for p, s in zip(llm.stage_prompts, llm.schemas) if s is hyp_schema)[2]["content"]
    assert "these kinds of module are blocked: a tool; the target must be one of" in hyp_request


def test_workflow_calling_a_tool_nobody_wrote_is_rejected(tmp_path):
    calls = ("def solve(task, ctx):\n    ctx.tool('python_exec', {'code': 'print(1)'})\n"
             "    ctx.tool('verilog_checker', {'design': ''})\n"
             "    return {'answer': ctx.llm([{'role': 'user', 'content': task['prompt']}])}\n")
    llm = FakeLLM(implement=[f"```python\n{calls}```"])
    loop, parent, ts = research(tmp_path, llm, make_sandbox=NoSandbox)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    implement = child["record"]["stages"]["implement"]
    assert child["status"] == "invalid_patch" and len(implement["problems"]) == 1
    assert "ctx.tool('verilog_checker'), but there is no such tool" in implement["problems"][0]
    assert "verilog_checker" in implement["last_answer"]
    request = llm.implement_prompts[0][1]["content"]
    assert "may call only these tools with ctx.tool: python_exec, verilog_sim." in request


def test_called_tools_reads_literal_names():
    src = ("def solve(task, ctx):\n    ctx.tool('python_exec', {})\n    name = 'x'\n    ctx.tool(name, {})\n"
           "    return ctx.tool(\"mine\", {})\n")
    assert called_tools(src) == {"python_exec", "mine"} and called_tools("def solve(:") == set()


def test_tool_example_passes_the_static_checks():
    assert not check_python_source(TOOL_EXAMPLE, "run", MAX_TOOL_LINES, "tool") and not tool_shape(TOOL_EXAMPLE)


@docker
def test_tool_example_and_the_verilog_hint_pass_in_the_sandbox(tmp_path):
    cand = tmp_path / "cand"
    shutil.copytree(TEMPLATE, cand)
    (cand / "tools" / "run_python.py").write_text(TOOL_EXAMPLE)
    (cand / "tools" / "and2.py").write_text(
        "SCHEMA = {}\ndef run(args, public):\n    return {}\ndef test(public):\n"
        "    r = public.verilog_sim('module and2(input a, input b, output y); assign y = a & b; endmodule',\n"
        "                           'module tb; endmodule')\n    assert r['compiled'], r\n")
    with CandidateSandbox(cand, memory="768m") as sb:
        assert sb.tool_test("run_python") == {"ok": True}
        assert sb.tool_test("and2") == {"ok": True}


def test_max_tokens_above_the_call_cap_is_rejected(tmp_path):
    literal = "def solve(task, ctx):\n    return {'answer': ctx.llm([{'role': 'user', 'content': 'x'}], max_tokens=8192)}\n"
    named = ("def solve(task, ctx):\n    if task['domain'] == 'math':\n        n = 8192\n    else:\n        n = 4096\n"
             "    return {'answer': ctx.llm([{'role': 'user', 'content': 'x'}], max_tokens=n)}\n")
    budget = ("BUDGET = 8192\n\ndef solve(task, ctx):\n    left = BUDGET\n"
              "    return {'answer': ctx.llm([{'role': 'user', 'content': 'x'}], max_tokens=min(4096, left))}\n")
    assert over_call_cap(literal) == {8192} and over_call_cap(named) == {8192} and over_call_cap(budget) == set()
    loop, parent, ts = research(tmp_path, FakeLLM(implement=[f"```python\n{named}```"]), make_sandbox=NoSandbox)
    child = loop.propose("c1", parent, parent_results(ts), ts, [], seed=1)
    assert child["status"] == "invalid_patch"
    assert "ctx.llm(max_tokens=8192) changes nothing" in child["record"]["stages"]["implement"]["problems"][0]
