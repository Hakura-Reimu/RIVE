"""Phase C workflow sandbox: model calls served by the host, public and candidate tools, timeouts, metering.
Needs docker and the alter-judge:phase0 image (WSL)."""
import shutil
import time
from pathlib import Path

import pytest

from rive.runtime.workflow import run_workflow_tasks
from rive.sandbox.container import CandidateSandbox

TEMPLATE = Path(__file__).resolve().parents[1] / "candidate_template"
TASK = {"domain": "math", "source": "aime2025", "problem": "1+1?", "prompt": "1+1? Put the answer in \\boxed{}."}

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")


def candidate(tmp_path, workflow=None, tools=None):
    shutil.copytree(TEMPLATE, tmp_path / "cand")
    if workflow is not None:
        (tmp_path / "cand" / "agent" / "workflow.py").write_text(workflow)
    for name, src in (tools or {}).items():
        (tmp_path / "cand" / "tools" / f"{name}.py").write_text(src)
    for p in [tmp_path / "cand", *(tmp_path / "cand").rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    return tmp_path / "cand"


def echo(log):
    def on_llm(req):
        log.append(req)
        return {"ok": True, "text": f"reply {req['call']}"}
    return on_llm


def test_initial_workflow_sends_the_phase0_prompt(tmp_path):
    log = []
    with CandidateSandbox(candidate(tmp_path)) as sb:
        done = sb.solve(TASK, 30, echo(log))
    assert done["ok"] and done["answer"] == "reply 1"
    assert log[0]["messages"] == [{"role": "user", "content": TASK["prompt"]}] and log[0]["max_tokens"] == 4096


def test_several_calls_and_public_tools(tmp_path):
    wf = ("def solve(task, ctx):\n"
          "    a = ctx.llm([{'role': 'user', 'content': 'draft'}])\n"
          "    b = ctx.llm([{'role': 'user', 'content': a + ' check'}], max_tokens=10)\n"
          "    py = ctx.tool('python_exec', {'code': 'print(6*7)'})\n"
          "    sim = ctx.tool('verilog_sim', {'design': 'module m; initial $display(\"hi\"); endmodule',\n"
          "                                   'testbench': 'module tb; m u(); endmodule', 'top': 'tb'})\n"
          "    return {'answer': '|'.join([b, py['stdout'].strip(), sim['stdout'].strip()])}\n")
    log = []
    with CandidateSandbox(candidate(tmp_path, wf), memory="768m") as sb:
        done = sb.solve(TASK, 60, echo(log))
    assert done["ok"], done
    assert done["answer"] == "reply 2|42|hi"
    assert [c["name"] for c in done["tool_calls"]] == ["python_exec", "verilog_sim"]
    assert log[1]["messages"][0]["content"] == "reply 1 check"


def test_candidate_tool_builds_on_public_tools(tmp_path):
    tool = ("SCHEMA = {'x': 'int'}\n"
            "def run(args, public):\n"
            "    return {'y': int(public.python_exec('print(%d * 2)' % args['x'])['stdout'])}\n")
    wf = "def solve(task, ctx):\n    return {'answer': str(ctx.tool('double', {'x': 21})['y'])}\n"
    with CandidateSandbox(candidate(tmp_path, wf, {"double": tool}), memory="768m") as sb:
        done = sb.solve(TASK, 30, echo([]))
    assert done["ok"] and done["answer"] == "42"


def test_a_workflow_that_never_stops_calling_the_model_is_cut_off(tmp_path):
    # Stand-in answers come back at once; the loops swallow every error. The first is stopped by the worker's own
    # time limit, which `except Exception` cannot catch; the second, which catches even that, by the host.
    for handler, limit in (("except Exception", 10), ("except BaseException", 40)):
        wf = ("def solve(task, ctx):\n    while True:\n        try:\n"
              f"            ctx.llm([{{'role': 'user', 'content': 'x'}}])\n        {handler}:\n            pass\n")
        start = time.monotonic()
        with CandidateSandbox(candidate(tmp_path / handler.split()[1], wf)) as sb:
            done = sb.solve(TASK, 2, lambda req: {"ok": True, "text": "again"})
        assert not done["ok"] and time.monotonic() - start < limit, done


def test_candidate_output_stays_off_the_channel(tmp_path):
    # A tool test that printed "Tool test passed." once reached the host as the sandbox's reply.
    tool = "def run(args, public):\n    return {}\ndef test(public):\n    print('Tool test passed.')\n"
    wf = ("import os\n"
          "def solve(task, ctx):\n"
          "    print('thinking')\n"
          "    os.system('echo from a shell')\n"
          "    return {'answer': ctx.llm([{'role': 'user', 'content': 'x'}])}\n")
    with CandidateSandbox(candidate(tmp_path, wf, {"loud": tool})) as sb:
        assert sb.tool_test("loud") == {"ok": True}
        done = sb.solve(TASK, 30, echo([]))
    assert done["ok"] and done["answer"] == "reply 1", done


def test_timeout_then_late_reply_keeps_the_channel_in_sync(tmp_path):
    wf = "def solve(task, ctx):\n    return {'answer': ctx.llm([{'role': 'user', 'content': 'slow'}])}\n"

    def slow(req):
        time.sleep(3)
        return {"ok": True, "text": "late"}
    with CandidateSandbox(candidate(tmp_path, wf)) as sb:
        first = sb.solve(TASK, 1, slow)
        second = sb.solve(TASK, 10, echo([]))
    assert not first["ok"] and "exceeded" in first["error"]
    assert second["ok"] and second["answer"] == "reply 1"


def test_contract_errors_are_reported(tmp_path):
    with CandidateSandbox(candidate(tmp_path, "def solve(task, ctx):\n    return 'just text'\n")) as sb:
        r = sb.solve(TASK, 10, echo([]))
    assert not r["ok"] and "must return" in r["error"]
    bad_role = "def solve(task, ctx):\n    return {'answer': ctx.llm([{'role': 'tool', 'content': 'x'}])}\n"
    with CandidateSandbox(candidate(tmp_path / "b", bad_role)) as sb:
        r = sb.solve(TASK, 10, echo([]))
    assert not r["ok"] and "messages must be" in r["error"]


class FakeLLM:
    def __init__(self, tokens_per_call=3000):
        self.tokens = tokens_per_call

    def generate(self, messages, max_tokens, temperature, thinking, seed=0):
        n = min(self.tokens, max_tokens)
        return {"text": "<think>trace</think>\\boxed{2}" if thinking else "\\boxed{2}", "finish_reason":
                "length" if n == max_tokens else "stop", "prompt_tokens": 7, "completion_tokens": n}


def test_runtime_meters_tokens_and_enforces_the_budget(tmp_path):
    wf = ("def solve(task, ctx):\n"
          "    out = []\n"
          "    for _ in range(5):\n"
          "        try:\n"
          "            out.append(ctx.llm([{'role': 'user', 'content': 'x'}], max_tokens=4096, thinking=True))\n"
          "        except RuntimeError as e:\n"
          "            out.append('stop')\n"
          "            break\n"
          "    return {'answer': out[0] + ' ' + str(len(out))}\n")
    tasks = [{"id": f"t{i}", "domain": "math", "source": "aime2025", "public": TASK, "item": {}} for i in range(3)]
    judged = []

    def judge(task, answer):
        judged.append(answer)
        return answer.startswith("\\boxed{2}"), None, {}
    res = run_workflow_tasks(tasks, candidate(tmp_path, wf), FakeLLM(), workers=2, token_budget=8192, judge_fn=judge)
    assert [r["id"] for r in res] == ["t0", "t1", "t2"] and all(r["correct"] for r in res)
    # 3000 + 3000 + 2192 (capped by what is left) = 8192, then the fourth call is refused
    assert all(r["tokens"] == 8192 and r["llm_calls"] == 3 and r["truncated_calls"] == 1 for r in res)
    assert judged[0] == "\\boxed{2} 4"
