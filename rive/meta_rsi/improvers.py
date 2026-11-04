"""Improvers for RESET-META (docs/meta-rsi-design.md section 4). Each one is restarted for every run (start) and
then asked for one change per round (propose), given the current agent and its development results.

- ResearchImprover: M0, the Phase C research step with its two guards, told about the T1 bug-fix tasks. `hint`
  is its policy layer (section 2, change 4): text added to what it is told. The outer loop will edit it; the
  positive control sets it by hand.
- OracleImprover: a scripted positive control for the ruler itself. It installs ORACLE_WORKFLOW once and makes
  no other change.
- M1 (section 5, after the calibration): M0 that knows how its changes went wrong. It is told to keep the first
  model call and is shown a real task (M1_POLICY), and EffectCheck sends back a change that fixes none of a few
  development tasks the current agent gets wrong, showing what the workflow did on one of them.
"""
import json
import random
import shutil
from functools import partial
from pathlib import Path

from rive.meta_rsi.bugfix import judge
from rive.research.agent_loop import CONTRACT, AgentResearch, _judge_note, assign_problem
from rive.runtime.workflow import run_workflow_tasks

T1_SYSTEM = (
    "You are the research decision-maker for an AI agent that fixes bugs in short Python functions with a small "
    "language model. You improve the agent only by changing one of its modules, based on evidence from "
    "development tests. You cannot change the model, the tests or the scoring. Be concrete: cite failure types, "
    "counts and example ids from the feedback you are given."
)


def _t1_contract():
    fields = ("task holds domain, source, prompt (the full original prompt) and, for maths:\n  problem; code: "
              "statement, starter_code, testtype, func_name; rtl: spec, module, language.")
    prompts = "agent/prompts/<domain>.md for domain math, code or rtl"
    assert fields in CONTRACT and prompts in CONTRACT and "Judging:" in CONTRACT
    text = CONTRACT.replace(fields, (
        'task holds domain ("code"), source, prompt (the full original prompt: code that\n  does not work, one '
        "test it fails and the instructions), buggy_code (that code alone) and failing_test\n  (that test, an "
        "assert statement)."))
    text = text.replace(prompts, "agent/prompts/code.md (every task here is a code task)")
    return text[:text.index("Judging:")] + (
        "Judging: the first ```python block of the answer is run against hidden assert tests of the function; a "
        "task\nis solved only when every hidden test passes. The failing test the task shows is one of them.")


T1_CONTRACT = _t1_contract()

# The positive control's policy: a strategy the outer loop could have written into the policy layer.
TEST_HINT = (
    "A strategy known to work on these tasks: the workflow can append task[\"failing_test\"] to the code it took "
    "from\nthe model's answer and run it with ctx.tool(\"python_exec\", {\"code\": ..., \"seconds\": 10}); when "
    "the result's\nreturncode is not 0, it sends the model its answer and the end of stderr and asks again, at "
    "most twice,\nkeeping the latest answer when a call fails.")

# A*: A0 plus that strategy, written by hand. A0's first call is unchanged.
ORACLE_WORKFLOW = '''"""Workflow A* (Meta-RSI positive control, docs/meta-rsi-design.md section 4): answer as A0 does,
then run the failing test the task shows against the answer's code; while it still fails, give the model the error
and ask again, twice at most. Hand-written, not researched."""


def code_of(text):
    fence = "`" * 3
    start = text.find(fence + "python")
    if start < 0:
        return ""
    start = text.find("\\n", start) + 1
    end = text.find(fence, start)
    return text[start:end] if end >= 0 else text[start:]


def solve(task, ctx):
    system = ctx.prompt(task["domain"])
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": task["prompt"]}]
    answer = ctx.llm(messages, max_tokens=4096)
    test = task.get("failing_test")
    for _ in range(2):
        code = code_of(answer)
        if not test or not code:
            break
        result = ctx.tool("python_exec", {"code": code + "\\n" + test, "seconds": 10})
        if result["returncode"] == 0 and not result["timed_out"]:
            break
        messages = messages + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Running that code with the failing test gave:\\n" + result["stderr"][-1500:]
             + "\\nFix it and return the complete corrected code in a single " + "`" * 3 + "python code block."}]
        try:
            answer = ctx.llm(messages, max_tokens=2048)
        except RuntimeError:
            break
    return {"answer": answer}
'''


# M1's policy, from what the calibration's changes got wrong (docs/meta-rsi-design.md section 5): a missing
# field read as starter_code, the task prompt rebuilt without the code, the test never appended.
M1_POLICY = (
    "How to change the workflow safely (from earlier experiments on these tasks):\n"
    "- Keep the agent's first model call as it is: the system prompt from ctx.prompt(task[\"domain\"]) and "
    "task[\"prompt\"]\n  as the user message, max_tokens=4096. Add steps after it; never rebuild or replace "
    "task[\"prompt\"].\n"
    "- Read only fields the task has. A real task as the workflow receives it (long values cut):\n{example}\n"
    "- A change must make a difference: before it is evaluated, it runs on development tasks the current agent "
    "gets\n  wrong, and it comes back to you, with what the workflow did, when it fixes none of them.")


def task_example(task, cut=300):
    return json.dumps({k: v[:cut] + "..." if isinstance(v, str) and len(v) > cut else v
                       for k, v in task["public"].items()}, ensure_ascii=False, indent=1)


class Recorder:
    """The solver, keeping each model call a task makes: the messages sent and the text that came back."""

    def __init__(self, llm):
        self.llm, self.calls = llm, []

    def generate(self, messages, max_tokens=1024, temperature=0.0, thinking=False, seed=0):
        r = self.llm.generate(messages, max_tokens=max_tokens, temperature=temperature, thinking=thinking, seed=seed)
        self.calls.append({"messages": len(messages), "sent": messages[-1]["content"], "reply": r["text"]})
        return r


class EffectCheck:
    """M1's check: a change must fix at least one of `k` development tasks the current agent gets wrong. When it
    fixes none, the problem shows what the workflow did on one of them: the model calls it made and the judge's
    verdict. Each task runs alone (so its calls are its own) with a short time limit."""

    def __init__(self, solver, k=6, seconds=120, run=run_workflow_tasks):
        self.solver, self.k, self.seconds, self.run = solver, k, seconds, run

    def __call__(self, child_dir, parent_results, tasks_by_id, record, seed):
        failing = sorted(r["id"] for r in parent_results if not r["correct"] and r["id"] in tasks_by_id)
        if not failing:
            return []
        chosen = random.Random(seed).sample(failing, min(self.k, len(failing)))
        traces = []
        for tid in chosen:
            rec = Recorder(self.solver)
            res = self.run([tasks_by_id[tid]], Path(child_dir), rec, workers=1, seconds=self.seconds,
                           judge_fn=judge)[0]
            traces.append((tid, res, rec.calls))
        fixed = sum(res["correct"] for _, res, _ in traces)
        record["checks"]["effect_fixed"] = fixed
        record["stages"].setdefault("effect", []).append({"tasks": chosen, "fixed": fixed})
        if fixed:
            return []
        tid, res, calls = traces[0]
        shown = [f"call {i}: {c['messages']} messages, the last one ending {c['sent'][-500:]!r}; the model's "
                 f"answer began {c['reply'][:300]!r}" for i, c in enumerate(calls[:3], 1)]
        return [f"the effect check ran this version on {len(chosen)} development tasks the current agent gets "
                f"wrong, and it fixed none. On {tid} ({res['failure']}) the workflow made {len(calls)} model "
                f"call(s) and {len(res['tool_calls'])} tool call(s):\n" + "\n".join(shown)
                + f"\nThe judge: {_judge_note(res)}"]


class ResearchImprover:
    def __init__(self, name, make_llm, smoke_tasks, solver, hint="", extra_check=None):
        """make_llm: a new model client for each run, so its token use is the run's own."""
        self.name, self.make_llm, self.smoke_tasks, self.solver, self.hint = name, make_llm, smoke_tasks, solver, hint
        self.extra_check = extra_check

    def start(self, run_dir):
        self.llm, self.taken = self.make_llm(), []
        self.step = AgentResearch(self.llm, Path(run_dir) / "candidates", self.smoke_tasks, system=T1_SYSTEM,
                                  contract=T1_CONTRACT + (f"\n{self.hint}" if self.hint else ""), solver=self.solver,
                                  run_tasks=partial(run_workflow_tasks, judge_fn=judge), extra_check=self.extra_check)

    def propose(self, child_id, parent_dir, results, tasks_by_id, notes, seed, context=None):
        problem = assign_problem(results, [], self.taken, random.Random(seed))
        if problem:
            self.taken.append(problem)
        return self.step.propose(child_id, Path(parent_dir), results, tasks_by_id, notes, seed=seed, problem=problem)

    def usage(self):
        return dict(self.llm.usage)


class OracleImprover:
    name = "oracle"

    def start(self, run_dir):
        self.run_dir, self.done = Path(run_dir), False

    def propose(self, child_id, parent_dir, results, tasks_by_id, notes, seed, context=None):
        if self.done:
            return {"id": child_id, "status": "no_proposal", "dir": None, "module": None, "record": {},
                    "note": f"{child_id}: no further change"}
        child = self.run_dir / "candidates" / child_id
        shutil.copytree(parent_dir, child)
        (child / "agent" / "workflow.py").write_text(ORACLE_WORKFLOW)
        self.done = True
        return {"id": child_id, "status": "valid", "dir": str(child), "module": "agent/workflow.py",
                "record": {"hypothesis": {"hypothesis": "run the failing test and retry (hand-written)"}}, "note": None}

    def usage(self):
        return {}
