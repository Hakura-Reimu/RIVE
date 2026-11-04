"""T2 program synthesis (rive/meta_rsi/synth.py and synth_workflow.py): which tests are shown, the packs, the
workflow's selection and retries on the shown examples (run with the real python_exec, without the sandbox), and the
improver on T2's configuration space."""
import json
import sys
from pathlib import Path

import jsonschema
import pytest

from rive.meta_rsi import synth
from rive.meta_rsi import synth_workflow as wf
from rive.meta_rsi.components import read_config
from rive.meta_rsi.config_improver import ConfigImprover
from rive.research.loop import check_python_source
from rive.sandbox.public_tools import python_exec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rive_t2  # noqa: E402

FENCE = "`" * 3
STDIN = {"testtype": "stdin", "func_name": "None", "prompt": "add the two numbers",
         "examples": [{"input": "1 2\n", "output": "3\n"}, {"input": "5 5\n", "output": "10\n"}]}
FUNCTIONAL = {"testtype": "functional", "func_name": "twice", "prompt": "double x",
              "examples": [{"input": "3", "output": "6"}, {"input": "[1, 2]", "output": "[1, 2, 1, 2]"}]}


def code(body):
    return f"Here it is.\n{FENCE}python\n{body}\n{FENCE}"


class Ctx:
    """The sandbox's ctx: scripted replies, the real python_exec, and the configuration as the prompt file."""

    def __init__(self, replies, config):
        self.replies, self.config, self.seen = list(replies), config, []

    def llm(self, messages, max_tokens=1024, thinking=False, temperature=0.0):
        self.seen.append({"messages": messages, "max_tokens": max_tokens, "thinking": thinking,
                          "temperature": temperature})
        if not self.replies:
            raise RuntimeError("model call failed: token budget used up")
        return self.replies.pop(0)

    def tool(self, name, args):
        assert name == "python_exec"
        return python_exec(**args)

    def prompt(self, name):
        return json.dumps(self.config) if name == "config" else ""


def solve(task, replies, **changes):
    ctx = Ctx(replies, {**synth.DEFAULT_CONFIG, **changes})
    return wf.solve(task, ctx)["answer"], ctx


def test_examples_are_the_leading_tests_found_in_the_statement():
    item = {"content": "Read a and b.\nSample Input 1:\n1 2\nSample Output 1:\n3", "testtype": "stdin"}
    assert synth.is_example(item, {"input": "1 2\n", "output": "3\n"})
    assert not synth.is_example(item, {"input": "7 8\n", "output": "15\n"})
    func = {"content": "Example 1:\nInput: nums = [1, 2], k = 3\nOutput: 4", "testtype": "functional"}
    assert synth.is_example(func, {"input": "[1,2]\n3", "output": "4"})
    assert not synth.is_example(func, {"input": "[5]\n3", "output": "4"})


def test_packs_split_every_stratum_and_keep_tasks_apart():
    tasks = [{"id": f"t{i}", "source": "lcb_train", "item": {"testtype": t, "difficulty": d}}
             for i, (t, d) in enumerate([(t, d) for t in ("stdin", "functional") for d in ("easy", "hard")] * 23)]
    packs = synth.make_packs(tasks)
    ids = [t["id"] for p in packs.values() for t in p]
    assert sorted(ids) == sorted(t["id"] for t in tasks)  # every task once
    assert {k: len(v) for k, v in packs.items()} == {"dev": 12, "select": 40, "test": 40}
    assert packs == synth.make_packs(tasks)  # the same split every time


def test_workflow_is_checked_and_stored(tmp_path):
    assert check_python_source(synth.WORKFLOW, "solve", 200, "workflow") == []
    agent = synth.make_agent(tmp_path / "a", {**synth.DEFAULT_CONFIG, "example_retry": 2})
    assert read_config(agent)["example_retry"] == 2
    assert (agent / "agent" / "workflow.py").read_text() == synth.WORKFLOW
    with pytest.raises(jsonschema.ValidationError):
        synth.validate({**synth.DEFAULT_CONFIG, "max_tokens": 3000})
    for changes, _ in rive_t2.SCAN.values():
        synth.validate({**synth.DEFAULT_CONFIG, **changes})


def test_examples_run_for_stdin_and_functional_tasks():
    right = "a, b = map(int, input().split())\nprint(a + b)"
    assert wf.first_failure(Ctx([], {}), STDIN, right, STDIN["examples"]) is None
    example, got = wf.first_failure(Ctx([], {}), STDIN, "print(3)", STDIN["examples"])
    assert example["input"] == "5 5\n" and got.strip() == "3"
    twice = "class Solution:\n    def twice(self, x):\n        return x * 2"
    assert wf.first_failure(Ctx([], {}), FUNCTIONAL, twice, FUNCTIONAL["examples"]) is None
    crash = "class Solution:\n    def twice(self, x):\n        return x / 0"
    assert wf.first_failure(Ctx([], {}), FUNCTIONAL, crash, FUNCTIONAL["examples"])[1].startswith("runtime error")
    slow = "import time\ntime.sleep(30)"
    assert wf.first_failure(Ctx([], {}), STDIN, slow, STDIN["examples"])[1] == "time limit exceeded"


def test_a0_keeps_its_one_answer():
    answer, ctx = solve(STDIN, [code("print(3)")])
    assert answer == code("print(3)") and len(ctx.seen) == 1 and ctx.seen[0]["max_tokens"] == 4096


def test_select_takes_the_first_answer_that_passes_the_examples():
    replies = [code("print(3)"), code("a, b = map(int, input().split())\nprint(a + b)"), code("print(0)")]
    answer, _ = solve(STDIN, replies, attempts=3, temperature=0.7, select="passes_examples")
    assert "a + b" in answer


def test_retry_shows_what_the_code_gave_and_stops_once_it_passes():
    replies = [code("print(3)"), code("a, b = map(int, input().split())\nprint(a + b)"), code("never asked")]
    answer, ctx = solve(STDIN, replies, example_retry=3, feedback="got_and_expected")
    assert "a + b" in answer and len(ctx.seen) == 2
    shown = ctx.seen[1]["messages"][-1]["content"]
    assert "gave:\n3" in shown and "5 5" in shown and "10" in shown
    _, ctx = solve(STDIN, [code("print(3)"), code("print(4)")], example_retry=1)
    assert "expected output" not in ctx.seen[1]["messages"][-1]["content"]
    assert [s["temperature"] for s in ctx.seen] == [0.7, 0.7]  # retries sample as the first answer does (6.17)


def test_review_and_a_spent_budget():
    answer, ctx = solve(FUNCTIONAL, [code("class Solution:\n    def twice(self, x):\n        return x + x"),
                                     "no code here"], review=True, thinking=True, max_tokens=8192)
    assert "x + x" in answer and ctx.seen[1]["thinking"] and ctx.seen[1]["max_tokens"] == 8192
    assert solve(STDIN, [])[0] == ""  # the budget ran out before any answer


class FakeServer:
    """The parts of the OpenAI client T2Model uses: chat replies cut off as scripted, completions that answer."""

    def __init__(self, chat_text, chat_finish):
        self.chat_text, self.chat_finish, self.calls = chat_text, chat_finish, []
        self.chat = type("Chat", (), {"completions": type("C", (), {"create": self.chat_create})()})()
        self.completions = type("Completions", (), {"create": self.complete})()

    @staticmethod
    def reply(**fields):
        usage = type("Usage", (), {"prompt_tokens": 10, "completion_tokens": fields.pop("tokens")})()
        choice = type("Choice", (), fields)()
        return type("Response", (), {"choices": [choice], "usage": usage})()

    def chat_create(self, **kw):
        self.calls.append(("chat", kw))
        message = type("Message", (), {"content": self.chat_text})()
        return self.reply(message=message, finish_reason=self.chat_finish, tokens=kw["max_tokens"])

    def complete(self, **kw):
        self.calls.append(("completion", kw))
        return self.reply(text=code("print(1)"), finish_reason="stop", tokens=50)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kw):
        assert kw["enable_thinking"] and kw["add_generation_prompt"] and not kw["tokenize"]
        return "<prompt>"


def model_with(server):
    model = synth.T2Model()
    model.client, model._tokenizer = server, FakeTokenizer()
    return model


def test_thinking_calls_use_the_adapter_and_a_thinking_budget():
    server = FakeServer("still thinking", "length")
    r = model_with(server).generate([{"role": "user", "content": "p"}], max_tokens=8192, temperature=0.6,
                                    thinking=True, seed=3)
    (_, chat), (_, done) = server.calls
    assert chat["model"] == done["model"] == synth.THINKING_ADAPTER
    assert chat["max_tokens"] == 6144 and done["max_tokens"] == 2048
    assert chat["top_p"] == 0.95 and chat["presence_penalty"] == 0.0 and chat["extra_body"]["top_k"] == 20
    assert chat["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert done["prompt"] == "<prompt>still thinking" + synth.FORCE and done["seed"] == 3
    assert r["completion_tokens"] == 6144 + 50 and r["finish_reason"] == "stop"
    assert r["text"].split("</think>")[-1] == "\n\n" + code("print(1)")
    # Thinking that closed in time is not forced, even when the answer is then cut off.
    server = FakeServer("done</think>\n" + code("print(2)"), "length")
    r = model_with(server).generate([{"role": "user", "content": "p"}], max_tokens=4096, thinking=True)
    assert len(server.calls) == 1 and server.calls[0][1]["max_tokens"] == 3072 and r["finish_reason"] == "length"


def test_calls_without_thinking_use_the_model_and_the_instruct_sampling():
    server = FakeServer(code("print(2)"), "length")
    model = model_with(server)
    r = model.generate([{"role": "user", "content": "p"}], max_tokens=4096, temperature=0.7)
    (_, chat), = server.calls
    assert chat["model"] == "qwen3.5-4b" and chat["max_tokens"] == 4096 and chat["temperature"] == 0.7
    assert chat["top_p"] == 0.8 and chat["presence_penalty"] == 1.5 and r["finish_reason"] == "length"
    assert model.usage["forced"] == 0


class ProposingLLM:
    def __init__(self, config):
        self.config, self.prompts, self.usage = config, [], {"completion_tokens": 0}

    def chat(self, messages, schema=None, **kw):
        self.prompts.append((messages, schema))
        return json.dumps({"diagnosis": "wrong answers on the examples", "hypothesis": "retry on the examples",
                           "prediction": "more solved", "predicted_gain": 9, "keep_probability": 0.5,
                           "config": self.config})


def test_the_improver_works_in_t2s_configuration_space(tmp_path):
    proposal = {**synth.DEFAULT_CONFIG, "example_retry": 2}
    llm = ProposingLLM(proposal)
    improver = ConfigImprover("m0c", lambda: llm, docs=synth.component_docs(), schema=synth.CONFIG_SCHEMA,
                              make=synth.make_agent, system=synth.SYSTEM)
    improver.start(tmp_path / "run")
    a0 = synth.make_agent(tmp_path / "a0", synth.DEFAULT_CONFIG)
    results = [{"id": "d1", "domain": "code", "correct": False, "failure": "wrong_answer", "answer": "x"}]
    start = {"select_solved": 100, "dev_solved": 30, "select_tokens": 900, "pack_sizes": {"select": 249, "dev": 74}}
    child = improver.propose("r1", a0, results, {"d1": {"id": "d1", "public": {"prompt": "p"}}}, [], seed=1,
                             context={"round": 1, "rounds": 4, "start": start, "history": [], "margin": 8})
    assert child["status"] == "valid" and read_config(child["dir"]) == proposal
    assert (Path(child["dir"]) / "agent" / "workflow.py").read_text() == synth.WORKFLOW
    messages, schema = llm.prompts[-1]
    assert schema["properties"]["config"] == synth.CONFIG_SCHEMA
    assert messages[0]["content"] == synth.SYSTEM and "needs at least 8 more" in messages[1]["content"]
    assert "example_retry (0-3)" in messages[1]["content"]
