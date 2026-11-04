"""The component agent (rive/meta_rsi/components.py). The sandbox tests need docker and the alter-judge image."""
import json
import shutil

import jsonschema
import pytest

from rive.meta_rsi.components import DEFAULT_CONFIG, WORKFLOW, make_agent, read_config, validate
from rive.meta_rsi.config_improver import ConfigImprover
from rive.research.loop import check_python_source
from rive.sandbox.container import CandidateSandbox

docker = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
FENCE = "`" * 3
TASK = {"domain": "code", "source": "t1_mbpp", "prompt": "fix inc", "failing_test": "assert inc(1) == 2"}
WRONG = f"{FENCE}python\ndef inc(x):\n    return x\n{FENCE}"
RIGHT = f"{FENCE}python\ndef inc(x):\n    return x + 1\n{FENCE}"


def test_configs_are_checked_and_stored_with_the_fixed_workflow(tmp_path):
    assert check_python_source(WORKFLOW, "solve", 200, "workflow") == []
    agent = make_agent(tmp_path / "a", {**DEFAULT_CONFIG, "test_retry": 2})
    assert read_config(agent)["test_retry"] == 2 and (agent / "agent" / "workflow.py").read_text() == WORKFLOW
    for bad in ({**DEFAULT_CONFIG, "attempts": 9}, {**DEFAULT_CONFIG, "extra": 1}, {"attempts": 2}):
        with pytest.raises(jsonschema.ValidationError):
            validate(bad)


class ProposingLLM:
    """Analysis as plain text; as JSON, the given configurations in turn."""

    def __init__(self, configs):
        self.configs, self.prompts, self.usage = list(configs), [], {"completion_tokens": 0}

    def chat(self, messages, schema=None, **kw):
        self.prompts.append(messages)
        if schema is None:
            return "The agent never checks its answer."
        return json.dumps({"diagnosis": "unchecked answers", "hypothesis": "run the failing test",
                           "prediction": "more tasks solved", "predicted_gain": 5, "keep_probability": 0.6,
                           "config": self.configs.pop(0)})


def test_the_improver_proposes_untried_configurations(tmp_path):
    retry = {**DEFAULT_CONFIG, "test_retry": 2}
    llm = ProposingLLM([DEFAULT_CONFIG, retry, retry, {**retry, "review": True}])
    improver = ConfigImprover("m0c", lambda: llm)
    improver.start(tmp_path / "run")
    a0 = make_agent(tmp_path / "a0", DEFAULT_CONFIG)
    results = [{"id": "d1", "domain": "code", "correct": False, "failure": "wrong_answer", "answer": "x"}]
    tasks = {"d1": {"id": "d1", "public": {"prompt": "fix f"}}}
    first = improver.propose("r1", a0, results, tasks, [], seed=1)
    # The current configuration is refused, the next one is taken.
    assert first["status"] == "valid" and read_config(first["dir"]) == retry
    assert first["record"]["stages"]["propose"]["attempts"] == 2
    second = improver.propose("r2", first["dir"], results, tasks, ["r1 (config): kept"], seed=2)
    assert read_config(second["dir"]) == {**retry, "review": True}  # retry again is refused as tried
    user = llm.prompts[-1][1]["content"]
    assert "test_retry (0-3)" in user and "r1 (config): kept" in user and '"test_retry": 2' in user


def run(tmp_path, config, replies):
    """The workflow on TASK; the model gives `replies` in turn (None: the token budget is used up)."""
    agent = make_agent(tmp_path / "agent", config)
    for p in [agent, *agent.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    sent = []

    def on_llm(req):
        sent.append(req)
        reply = replies[min(len(sent), len(replies)) - 1]
        return {"ok": False, "error": "token budget used up"} if reply is None else {"ok": True, "text": reply}
    with CandidateSandbox(agent, memory="768m") as sb:
        done = sb.solve(TASK, 60, on_llm)
    assert done["ok"], done
    return done, sent


@docker
def test_the_default_config_answers_once_like_a0(tmp_path):
    done, sent = run(tmp_path, DEFAULT_CONFIG, [WRONG])
    assert done["answer"] == WRONG and len(sent) == 1 and done["tool_calls"] == []
    assert sent[0]["messages"] == [{"role": "user", "content": "fix inc"}] and sent[0]["max_tokens"] == 4096
    assert sent[0]["temperature"] == 0.0


@docker
def test_select_and_retry_use_the_failing_test(tmp_path):
    done, sent = run(tmp_path / "select", {**DEFAULT_CONFIG, "attempts": 3, "temperature": 0.7,
                                           "select": "passes_test"}, [WRONG, RIGHT, WRONG])
    assert "x + 1" in done["answer"] and len(sent) == 3 and sent[0]["temperature"] == 0.7
    done, sent = run(tmp_path / "retry", {**DEFAULT_CONFIG, "test_retry": 2, "feedback": "stderr_and_test"},
                     [WRONG, RIGHT])
    assert "x + 1" in done["answer"] and len(sent) == 2
    shown = sent[1]["messages"][-1]["content"]
    assert "AssertionError" in shown and "assert inc(1) == 2" in shown
    done, sent = run(tmp_path / "majority", {**DEFAULT_CONFIG, "attempts": 3, "select": "majority"},
                     [WRONG, RIGHT, RIGHT])
    assert "x + 1" in done["answer"]


@docker
def test_agreement_keeps_an_answer_whose_results_most_answers_share(tmp_path):
    config = {**DEFAULT_CONFIG, "attempts": 3, "temperature": 0.7, "select": "agreement"}
    also_right = f"{FENCE}python\ndef inc(x):\n    return 1 + x\n{FENCE}"
    calls = f"{FENCE}python\ninc(1)\ninc(-3)\nnot a call(\nx = 2\n{FENCE}"
    # Three different texts, so majority would keep the first; two of them give the same results.
    done, sent = run(tmp_path / "agree", config, [WRONG, RIGHT, also_right, calls])
    assert "x + 1" in done["answer"] and len(sent) == 4 and "example calls" in sent[3]["messages"][-1]["content"]
    assert [c["name"] for c in done["tool_calls"]] == ["python_exec"] * 3
    done, sent = run(tmp_path / "no_calls", config, [WRONG, RIGHT, also_right, "no code"])
    assert done["answer"] == WRONG and done["tool_calls"] == []
    broken = f"{FENCE}python\ndef inc(x)\n    return x\n{FENCE}"
    done, sent = run(tmp_path / "broken", {**config, "attempts": 2}, [broken, RIGHT, calls])
    assert "x + 1" in done["answer"]  # code that does not run is in no group


@docker
def test_review_and_a_spent_budget(tmp_path):
    done, sent = run(tmp_path / "review", {**DEFAULT_CONFIG, "review": True}, [WRONG, RIGHT])
    assert "x + 1" in done["answer"] and "edge cases" in sent[1]["messages"][-1]["content"]
    done, sent = run(tmp_path / "budget", {**DEFAULT_CONFIG, "test_retry": 3}, [WRONG, None])
    assert done["answer"] == WRONG and len(sent) == 2  # the failed call ends the retries; the answer stays
