"""Component agent for Meta-RSI (docs/meta-rsi-design.md section 6): one fixed, tested workflow whose behaviour a
configuration sets. Improvers change only the configuration, which they generate under CONFIG_SCHEMA, so every
agent they make runs tested code: the calibration showed the 4B proposing the right changes and writing them
wrongly.

The configuration lives in agent/prompts/config.md as JSON; the workflow reads it with ctx.prompt("config").
"""
import json
import shutil
from pathlib import Path

import jsonschema

TEMPLATE = Path(__file__).resolve().parents[2] / "candidate_template"

CONFIG_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["system_prompt", "instruction", "attempts", "temperature", "select", "self_test", "test_retry",
                 "feedback", "review", "max_tokens"],
    "properties": {
        "system_prompt": {"type": "string", "maxLength": 1500},
        "instruction": {"type": "string", "maxLength": 500},
        "attempts": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "temperature": {"type": "number", "enum": [0.0, 0.3, 0.5, 0.7, 1.0]},
        "select": {"type": "string", "enum": ["first", "passes_test", "majority", "agreement"]},
        "self_test": {"type": "boolean"},
        "test_retry": {"type": "integer", "enum": [0, 1, 2, 3]},
        "feedback": {"type": "string", "enum": ["stderr", "stderr_and_test"]},
        "review": {"type": "boolean"},
        "max_tokens": {"type": "integer", "enum": [512, 1024, 2048, 4096]},
    },
}
# A0: one answer to the task's prompt, nothing else (the same model call as candidate_template's workflow).
DEFAULT_CONFIG = {"system_prompt": "", "instruction": "", "attempts": 1, "temperature": 0.0, "select": "first",
                  "self_test": False, "test_retry": 0, "feedback": "stderr", "review": False, "max_tokens": 4096}


def component_docs(budget=8192):
    """What the improver is told each setting does; `budget` is the task set's token budget per task."""
    return f"""The agent fixes a buggy Python function; it is run with this configuration:
- system_prompt: system message for every model call ("" for none), at most 1500 characters.
- instruction: text appended to the task's prompt ("" for none), at most 500 characters.
- attempts (1-5): independent answers to the prompt. With temperature 0.0 they are all the same answer.
- temperature (0.0, 0.3, 0.5, 0.7, 1.0): sampling temperature of those answers.
- select: which answer to keep: "first"; "passes_test", the first whose code passes the test (run with
  python_exec; "first" when none does); "majority", the code given most often; "agreement": one call asks the
  model for example calls of the function (arguments only, no expected results), every answer's code is run on
  them, and an answer whose results most answers share is kept ("first" when there are no such calls). majority
  and agreement need attempts of 2 or more and a temperature above 0.0.
- self_test (true/false): when the task shows no failing test, one call first asks the model to write a few
  asserts from what the task shows; passes_test and test_retry then use those as the test. No effect when the task
  shows a failing test.
- test_retry (0-3): after selecting, run the test against the answer's code; while it fails, send the model its
  answer and the error and ask again, at most this many times.
- feedback: what such a retry shows: "stderr" (the end of the error output) or "stderr_and_test" (also the test).
- review (true/false): one last call asking the model to check its code for edge cases and correct it.
- max_tokens (512, 1024, 2048, 4096): limit of each model call. All calls of a task share {budget} tokens; a call
  past that is cut off or fails, and the agent keeps what it has.
The test is the failing test the task shows, or the agent's own asserts with self_test; without one, passes_test
acts as "first" and test_retry does nothing. Some task sets show the problem statement and a failing test, some
only one of them or neither: look at the task examples in the feedback. The answer is judged on hidden assert
tests (a failing test the task shows is one of them)."""


COMPONENT_DOCS = component_docs()

WORKFLOW = '''"""Component workflow (Meta-RSI, docs/meta-rsi-design.md section 6). Fixed and tested:
improvers change only agent/prompts/config.md (rive/meta_rsi/components.py)."""
import ast
import json

DEFAULTS = {"system_prompt": "", "instruction": "", "attempts": 1, "temperature": 0.0, "select": "first",
            "self_test": False, "test_retry": 0, "feedback": "stderr", "review": False, "max_tokens": 4096}
FENCE = "`" * 3
ASK_AGAIN = "Fix it and return the complete corrected code in a single " + FENCE + "python code block."
REVIEW = ("Check the corrected code for edge cases the hidden tests may cover (empty input, boundaries, types). If "
          "something is wrong, fix it. Return the complete final code in a single " + FENCE + "python code block.")
SELF_TEST = ("Do not fix it yet. Write 3 to 5 assert statements that check what the function should do, with "
             "concrete inputs and expected outputs. Return only the asserts, one per line, in a single " + FENCE
             + "python code block.")
INPUTS = ("Do not fix it yet. Write 5 different example calls of the function that a correct version must handle, "
          "with concrete arguments (include edge cases), one call per line and without expected results, in a "
          "single " + FENCE + "python code block.")


def code_of(text):
    start = text.find(FENCE + "python")
    if start < 0:
        return ""
    start = text.find("\\n", start) + 1
    end = text.find(FENCE, start)
    return text[start:end] if end >= 0 else text[start:]


def passes(ctx, code, test):
    result = ctx.tool("python_exec", {"code": code + "\\n" + test, "seconds": 10})
    return result["returncode"] == 0 and not result["timed_out"], result["stderr"][-1500:]


def call(ctx, messages, config, temperature=0.0):
    try:
        return ctx.llm(messages, max_tokens=config["max_tokens"], temperature=temperature)
    except RuntimeError:  # the task's token budget is used up
        return None


def ask_shown(ctx, task, system, config, request):
    """The model's reply to `request` about what the task shows (the statement, if shown, and the code)."""
    shown = ("Task: " + task["statement"] + "\\n\\n") if task.get("statement") else ""
    shown += "This Python code may have a bug:\\n" + FENCE + "python\\n" + task.get("buggy_code", "") + "\\n" + FENCE
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": shown + "\\n" + request}]
    return code_of(call(ctx, messages, config) or "")


def own_inputs(ctx, task, system, config):
    """Example calls of the function the model writes from what the task shows; lines that are not a single call
    are dropped."""
    keep = []
    for line in ask_shown(ctx, task, system, config, INPUTS).splitlines():
        line = line.strip()
        try:
            node = ast.parse(line, mode="eval").body
        except SyntaxError:
            continue
        if isinstance(node, ast.Call):
            keep.append(line)
    return keep[:8]


def behaviour(ctx, code, inputs):
    """What `code` gives for each call in `inputs` (the start of the result's repr, or the exception's name); None
    when it does not run."""
    lines = [code, "import json as _json", "_out = []", "for _src in " + repr(inputs) + ":", "    try:",
             "        _out.append(repr(eval(_src))[:200])", "    except Exception as _e:",
             "        _out.append('error ' + type(_e).__name__)", "print(_json.dumps(_out))"]
    result = ctx.tool("python_exec", {"code": "\\n".join(lines), "seconds": 10})
    if result["returncode"] != 0 or result["timed_out"]:
        return None
    try:
        return tuple(json.loads(result["stdout"].strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return None


def own_tests(ctx, task, system, config):
    """Asserts the model writes from what the task shows; lines that are not an assert that parses are dropped."""
    keep = []
    for line in ask_shown(ctx, task, system, config, SELF_TEST).splitlines():
        line = line.strip()
        if line.startswith("assert"):
            try:
                ast.parse(line)
            except SyntaxError:
                continue
            keep.append(line)
    return "\\n".join(keep)


def solve(task, ctx):
    try:
        config = {**DEFAULTS, **json.loads(ctx.prompt("config") or "{}")}
    except ValueError:
        config = dict(DEFAULTS)
    system, test, own = config["system_prompt"], task.get("failing_test"), False
    if not test and config["self_test"]:
        test, own = own_tests(ctx, task, system, config), True
    user = task["prompt"] + ("\\n\\n" + config["instruction"] if config["instruction"] else "")
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    answers = []
    for _ in range(config["attempts"]):
        text = call(ctx, messages, config, config["temperature"])
        if text is None:
            break
        answers.append(text)
    if not answers:
        return {"answer": ""}
    answer = answers[0]
    if config["select"] == "passes_test" and test:
        answer = next((a for a in answers if code_of(a) and passes(ctx, code_of(a), test)[0]), answer)
    elif config["select"] == "majority":
        codes = [" ".join(code_of(a).split()) for a in answers]
        answer = answers[codes.index(max(codes, key=codes.count))]
    elif config["select"] == "agreement" and len(answers) > 1:
        inputs = own_inputs(ctx, task, system, config)
        seen = [behaviour(ctx, code_of(a), inputs) if inputs and code_of(a) else None for a in answers]
        ran = [b for b in seen if b is not None]
        if ran:  # on a tie, the group of the earliest answer
            answer = answers[seen.index(max(ran, key=ran.count))]
    history = list(messages)
    for _ in range(config["test_retry"]):
        code = code_of(answer)
        if not test or not code:
            break
        ok, error = passes(ctx, code, test)
        if ok:
            break
        shown = "Running that code with " + ("these tests" if own else "the failing test") + " gave:\\n" + error
        if config["feedback"] == "stderr_and_test":
            shown += ("\\nThe tests:\\n" if own else "\\nThe test:\\n") + test
        history = history + [{"role": "assistant", "content": answer},
                             {"role": "user", "content": shown + "\\n" + ASK_AGAIN}]
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
'''


def validate(config):
    jsonschema.validate(config, CONFIG_SCHEMA)
    return config


def make_agent(path, config):
    """A candidate directory running WORKFLOW with `config`."""
    path = Path(path)
    if not path.exists():
        shutil.copytree(TEMPLATE, path)
    (path / "agent" / "workflow.py").write_text(WORKFLOW)
    (path / "agent" / "prompts").mkdir(parents=True, exist_ok=True)
    (path / "agent" / "prompts" / "config.md").write_text(json.dumps(validate(config), sort_keys=True))
    return path


def read_config(path):
    return json.loads((Path(path) / "agent" / "prompts" / "config.md").read_text())
