"""T2 program-synthesis tasks for RIVE (docs/rive-design.md sections 6.11 and 6.17): LiveCodeBench problems, release
v1 and test6 together. The examples in a problem's statement are the tests the agent is shown; the answer is judged
on every test, the examples and the hidden ones, by the Phase C judge (rive/benchmark/hard.py).

The component agent runs synth_workflow.py with a configuration under CONFIG_SCHEMA, as components.py does for T1:
improvers change only the configuration. T2Model is the model as T2 serves it from section 6.17 on.
"""
import json
import random
import shutil
import threading
from pathlib import Path

import jsonschema

from rive.benchmark.hard import DATA, judge, public_task  # noqa: F401 - judge is T2's judge too
from rive.llm import LLM
from rive.meta_rsi.components import TEMPLATE

SOURCES = ("lcb_train", "livecodebench")
WORKFLOW = Path(__file__).with_name("synth_workflow.py").read_text()
TOKEN_BUDGET = 8192  # completion tokens per task over all calls; one call may use all of it
SHOWN_EXAMPLES = 3
CYCLE = {"dev": 3, "select": 10, "test": 10}  # each stratum is dealt to the packs in these shares
MODEL_DIR = Path.home() / "alter/models/Qwen3.5-4B"
# The model card's sampling besides the temperature: without thinking (instruct) and with it (coding), section 6.12.
CARD = {False: {"top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
        True: {"top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}}
THINKING_ADAPTER = "nowait-e1"  # the e1 edit as a rank-1 LoRA adapter (sections 6.13-6.15)
FORCE = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly "
         "now.\n</think>\n\n")  # Qwen's wording for a thinking budget (section 6.13)

CONFIG_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["system_prompt", "instruction", "thinking", "attempts", "temperature", "select", "example_retry",
                 "feedback", "review", "max_tokens"],
    "properties": {
        "system_prompt": {"type": "string", "maxLength": 1500},
        "instruction": {"type": "string", "maxLength": 500},
        "thinking": {"type": "boolean"},
        "attempts": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "temperature": {"type": "number", "enum": [0.0, 0.3, 0.5, 0.6, 0.7, 1.0]},
        "select": {"type": "string", "enum": ["first", "passes_examples", "majority"]},
        "example_retry": {"type": "integer", "enum": [0, 1, 2, 3]},
        "feedback": {"type": "string", "enum": ["got", "got_and_expected"]},
        "review": {"type": "boolean"},
        "max_tokens": {"type": "integer", "enum": [1024, 2048, 4096, 8192]},
    },
}
# A0: one answer to the Phase C prompt, without thinking, at the model card's temperature (section 6.17).
DEFAULT_CONFIG = {"system_prompt": "", "instruction": "", "thinking": False, "attempts": 1, "temperature": 0.7,
                  "select": "first", "example_retry": 0, "feedback": "got", "review": False, "max_tokens": 4096}
SYSTEM = (
    "You are the research decision-maker for an AI agent that writes Python solutions to programming-contest "
    "problems with a small language model. You improve the agent only by changing its configuration, based on "
    "evidence from development tests. You cannot change the model, the tests or the scoring. Be concrete: cite "
    "failure types, counts and example ids from the feedback you are given."
)


def component_docs(budget=TOKEN_BUDGET):
    """What the improver is told each setting does."""
    return f"""The agent writes a Python solution to a programming-contest problem: a program that reads standard
input, or a method of class Solution. It is run with this configuration:
- system_prompt: system message for every model call ("" for none), at most 1500 characters.
- instruction: text appended to the task's prompt ("" for none), at most 500 characters.
- thinking (true/false): the model thinks before it answers, in every call. The thinking may use three quarters
  of the call's limit; then it is closed and the model answers in the rest. Thinking uses the same token budget.
- attempts (1-5): independent answers to the prompt. With temperature 0.0 they are nearly the same answer.
- temperature (0.0, 0.3, 0.5, 0.6, 0.7, 1.0): sampling temperature of every call. The model's card advises 0.7
  without thinking and 0.6 with it, and advises against 0.0.
- select: which answer to keep: "first"; "passes_examples", the first whose code gives the expected output on
  every example of the statement (run with python_exec; "first" when none does); "majority", the code given most
  often (needs attempts of 2 or more and a temperature above 0.0).
- example_retry (0-3): after selecting, run the answer's code on the statement's examples; while one fails, send
  the model its answer and what the code gave, and ask again, at most this many times.
- feedback: what such a retry shows: "got" (the code's output or error) or "got_and_expected" (also that
  example's input and expected output).
- review (true/false): one last call asking the model to check its solution for edge cases and speed, and correct
  it.
- max_tokens (1024, 2048, 4096, 8192): limit of each model call. All calls of a task share {budget} tokens; a call
  past that is cut off or fails, and the agent keeps what it has.
The examples are the sample inputs and outputs in the statement; at most {SHOWN_EXAMPLES} are run. The answer is
judged on hidden tests (the examples are among them), with a time limit of 6 seconds per test."""


def squash(text):
    return "".join(text.split())


def is_example(item, test):
    """Whether a test is one of the statement's examples: its input and its output both appear in the statement."""
    text = squash(item["content"])
    parts = ([line for line in test["input"].splitlines() if line.strip()] if item["testtype"] == "functional"
             else [test["input"]])
    return all(squash(p) in text for p in parts) and squash(test["output"]) in text


def load_tasks(sources=SOURCES):
    """T2 tasks. A problem's leading tests that are examples are shown (at most SHOWN_EXAMPLES); a problem with no
    such example, or with no test besides them, is left out."""
    tasks = []
    for source in sources:
        for line in open(DATA / f"{source}.jsonl"):
            item = json.loads(line)
            tests = json.loads((DATA / item.get("tests_dir", "lcb_tests") / f"{item['question_id']}.json").read_text())
            k = 0
            while k < len(tests) and is_example(item, tests[k]):
                k += 1
            if k == 0 or k == len(tests):
                continue
            shown = [{"input": t["input"], "output": t["output"]} for t in tests[:min(k, SHOWN_EXAMPLES)]]
            tasks.append({"id": f"t2-{item['id']}", "domain": "code", "source": source,
                          "public": {**public_task(source, item), "examples": shown}, "item": item})
    return tasks


def make_packs(tasks, seed=0):
    """dev, select and test packs. Each source, test type and difficulty is shuffled and split in CYCLE's shares
    (rounded; the test pack takes the rest), so the selection and test packs have the same mix."""
    packs = {name: [] for name in CYCLE}
    total = sum(CYCLE.values())
    strata = {}
    for t in sorted(tasks, key=lambda t: t["id"]):
        strata.setdefault((t["source"], t["item"]["testtype"], t["item"]["difficulty"]), []).append(t)
    for key in sorted(strata):
        group = strata[key]
        random.Random(f"{seed}:{':'.join(key)}").shuffle(group)
        dev, select = round(len(group) * CYCLE["dev"] / total), round(len(group) * CYCLE["select"] / total)
        packs["dev"] += group[:dev]
        packs["select"] += group[dev:dev + select]
        packs["test"] += group[dev + select:]
    return packs


def validate(config):
    jsonschema.validate(config, CONFIG_SCHEMA)
    return config


def make_agent(path, config):
    """A candidate directory running the T2 workflow with `config`."""
    path = Path(path)
    if not path.exists():
        shutil.copytree(TEMPLATE, path)
    (path / "agent" / "workflow.py").write_text(WORKFLOW)
    (path / "agent" / "prompts").mkdir(parents=True, exist_ok=True)
    (path / "agent" / "prompts" / "config.md").write_text(json.dumps(validate(config), sort_keys=True))
    return path


class T2Model(LLM):
    """The model as T2 serves it from section 6.17 on. Every call samples with the model card's settings for its mode;
    the configuration sets only the temperature. A thinking call goes through the e1 adapter with a thinking budget:
    it thinks in three quarters of its limit; if the thinking is still open there, it is closed with FORCE and the
    model answers in the last quarter (sections 6.13-6.15). An answer whose thinking closed in time but which runs
    past the three quarters is cut off there, as in section 6.14."""

    def __init__(self, model="qwen3.5-4b", adapter=THINKING_ADAPTER, **kwargs):
        super().__init__(model=model, **kwargs)
        self.adapter, self._tokenizer = adapter, None
        self.usage["forced"] = 0

    def tokenizer(self):
        with self._lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        return self._tokenizer

    def generate(self, messages, max_tokens=1024, temperature=0.0, thinking=False, seed=0):
        card = CARD[bool(thinking)]
        sampling = {"temperature": temperature, "top_p": card["top_p"], "presence_penalty": card["presence_penalty"],
                    "seed": seed}
        extra = {"top_k": card["top_k"], "min_p": card["min_p"]}
        model = self.adapter if thinking else self.model
        answer = max_tokens // 4 if thinking else 0
        r = self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens - answer,
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(thinking)}, **extra}, **sampling)
        self._count(r)
        text, finish = r.choices[0].message.content or "", r.choices[0].finish_reason
        prompt_tokens = r.usage.prompt_tokens if r.usage else 0
        completion_tokens = r.usage.completion_tokens if r.usage else 0
        if thinking and finish == "length" and "</think>" not in text and answer > 0:
            prompt = self.tokenizer().apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True,
                                                          tokenize=False)
            f = self.client.completions.create(model=model, prompt=prompt + text + FORCE, max_tokens=answer,
                                               extra_body=extra, **sampling)
            self._count(f)
            with self._lock:
                self.usage["forced"] += 1
            text, finish = text + FORCE + (f.choices[0].text or ""), f.choices[0].finish_reason
            prompt_tokens += f.usage.prompt_tokens if f.usage else 0
            completion_tokens += f.usage.completion_tokens if f.usage else 0
        return {"text": text, "finish_reason": finish, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens}
