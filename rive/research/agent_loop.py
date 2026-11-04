"""Phase C research step (docs/phase-c-design.md section 4, step 3): one experiment on one module of a maths, code
and RTL agent: its workflow, a domain prompt, or a tool together with the workflow change that calls it. Same state
machine as Phase A (handoff doc 3.7.3): the model proposes; the controller supplies the facts, checks everything
and decides whether the child is evaluated. Evaluation belongs to the evolution controller; reflect() runs after it.
"""
import ast
import copy
import difflib
import hashlib
import json
import random
import re
import shutil
import threading
from collections import Counter
from pathlib import Path

from openai import BadRequestError

from rive.research import schemas
from rive.research.loop import StageFailed, Stages, check_python_source, normalized_hash, strip_docstrings
from rive.runtime.workflow import MAX_CALL_TOKENS, TOKEN_BUDGET, run_workflow_tasks
from rive.sandbox.container import CandidateSandbox

DOMAINS = schemas.AGENT_DOMAINS
WORKFLOW = "agent/workflow.py"
MAX_WORKFLOW_LINES = 200
MAX_TOOL_LINES = 150
MAX_PROMPT_CHARS = 4000
SMOKE_SECONDS = 60
# Stand-in model answers for the smoke test: it checks that the code runs, not how good it is.
CANNED = {"math": "So the answer is \\boxed{0}.", "code": "```python\nprint(0)\n```",
          "rtl": "```verilog\nmodule TopModule();\nendmodule\n```"}
PUBLIC_TOOLS = ("python_exec", "verilog_sim")
# What the public tools take and return (rive/sandbox/public_tools.py)
PUBLIC_ARGS = {"python_exec": {"code", "stdin", "seconds"}, "verilog_sim": {"design", "testbench", "top", "seconds"}}
PUBLIC_RESULTS = {"python_exec": {"returncode", "stdout", "stderr", "timed_out"},
                  "verilog_sim": {"compiled", "log", "stdout", "timed_out"}}
MODULES = {"workflow": "agent/workflow.py", "prompt": "a prompt file", "tool": "a tool"}
UNFIXABLE = {"judge_error"}  # failures of the judge, not of the agent
# Two checks added after the research ability test (docs/phase-c-design.md section 8). Each of the 4B's 11 changes
# altered how all three domains were solved although it was assigned one problem; 4 of the 25 changes measured lost
# 78-98% of the tasks their parent solved, which evaluation found only after 7-33 minutes each.
# Locality: on a few tasks of each other domain, with each set of stand-in answers, a child must make the same model
# calls, tool calls and answers as its parent. The second set has no answer in any format, for retry paths.
STAND_INS = (CANNED, {d: "I am not sure." for d in DOMAINS})
LOCALITY_TASKS = 3  # per other domain
# Pre-screen: a few development tasks the parent solved, answered with the real model; more than a third wrong sends
# the change back. Tasks the parent solved within PRESCREEN_SECONDS come first, so the pre-screen stays short.
PRESCREEN_TASKS = 3  # per domain the change can affect
PRESCREEN_SECONDS = 120
# Blocking (docs/phase-c-design.md section 8, after C5 generation 3): all 13 experiments of the first three C5
# generations worked on RTL compile errors, 7 of them with a Verilog checker tool that failed validation. A problem
# (domain and failure type) tried through one kind of module and failing BLOCK_KIND_AFTER times in a row, or failing
# BLOCK_PROBLEM_AFTER times in a row through any module, is blocked for the next BLOCK_FOR experiments (two
# generations of four) after its latest attempt.
BLOCK_KIND_AFTER = 3
BLOCK_PROBLEM_AFTER = 6
BLOCK_FOR = 8

# Same labels as rive.benchmark.hard.judge, in words.
FAILURE_MEANINGS = {
    "wrong_answer": "an answer was given but it is wrong (maths: the final answer; code: a hidden test failed)",
    "no_final_answer": "maths: there is no \\boxed{} answer in the output, often because it was cut off",
    "compile_error": "RTL: the design did not compile together with the testbench",
    "mismatch": "RTL: the design compiled, but its outputs differ from the reference in simulation",
    "time_limit": "the program or the simulation ran out of time",
    "runtime_error": "code: the program crashed on a test",
    "load_error": "code: the solution could not be loaded (syntax error, or the class or function is missing)",
    "judge_error": "the judge could not run the answer",
    "workflow_error": "the agent's workflow raised an error, timed out or returned no answer",
}

SYSTEM = (
    "You are the research decision-maker for an AI agent that solves maths, programming and hardware design (RTL) "
    "tasks with a small language model. You improve the agent only by changing one of its modules, based on "
    "evidence from development tests. You cannot change the model, the tests or the scoring. Be concrete: cite "
    "domains, failure types, counts and example ids from the feedback you are given."
)

CONTRACT = f"""Modules you may change, one per experiment:
- agent/workflow.py defines solve(task, ctx) -> {{"answer": str}}.
  ctx.llm(messages, max_tokens=1024, thinking=False, temperature=0.0) -> str is one model call. All calls of one
  task share {TOKEN_BUDGET} completion tokens; one call returns at most {MAX_CALL_TOKENS}, whatever max_tokens says,
  so using more of the budget takes more than one call. Past the budget a call raises RuntimeError.
  thinking=True lets the model reason first and returns only the text after its reasoning; the
  reasoning counts as tokens, and a call cut off while reasoning returns "". Measured on this model: on hard maths,
  code and RTL tasks its reasoning had not finished after 12288 tokens on 79% of tasks.
  ctx.tool(name, args) -> dict runs a tool. ctx.prompt(name) -> str returns agent/prompts/<name>.md.
  A task may take 600 seconds. task holds domain, source, prompt (the full original prompt) and, for maths:
  problem; code: statement, starter_code, testtype, func_name; rtl: spec, module, language.
  The workflow must cope with any model text: the smoke test feeds it stand-in answers.
- agent/prompts/<domain>.md for domain math, code or rtl: text for the workflow to use (the initial workflow sends
  it as the system message when it is not empty), at most {MAX_PROMPT_CHARS} characters.
- tool: a new or changed tools/<tool_name>.py together with the agent/workflow.py change that calls it. A tool
  defines SCHEMA (a dict with "args" and a "result" dict naming every field run() returns; the workflow may read
  only those fields), run(args, public) returning a JSON-serializable dict, and test(public), which raises
  AssertionError when the tool is wrong. Available to it:
  public.python_exec(code, stdin="", seconds=10) -> {{"returncode", "stdout", "stderr", "timed_out"}} and
  public.verilog_sim(design, testbench, top=None, seconds=20) -> {{"compiled", "log", "stdout", "timed_out"}}.
Python files use only the standard library, without os, sys, subprocess, socket, file access, eval or exec. Inside
Python code never write three backticks literally; build them as "`" * 3.
Judging: maths takes the final \\boxed{{}} answer; code takes the first ```python block and runs hidden tests;
RTL takes the first ```verilog block and simulates it against a hidden testbench."""

# Shown when the experiment writes a tool. In the first three C5 generations every one of 7 tools failed validation:
# over the line limit, no SCHEMA or test(), a call to a public function that does not exist, unclosed strings, and
# self-tests on Verilog the model had written wrongly. tests/test_agent_research.py checks that this example passes.
TOOL_EXAMPLE = r'''"""Runs a Python program on one input and says whether it crashed."""

SCHEMA = {
    "description": "Run a Python program on one stdin; report whether it crashed and what it printed.",
    "args": {"code": "str, the program", "stdin": "str, its input (optional)"},
    "result": {"crashed": "bool", "stdout": "str", "error": "str, the end of stderr"},
}


def run(args, public):
    out = public.python_exec(args.get("code", ""), stdin=args.get("stdin", ""), seconds=5)
    crashed = out["timed_out"] or out["returncode"] != 0
    return {"crashed": crashed, "stdout": out["stdout"], "error": out["stderr"][-500:]}


def test(public):
    double = "\n".join(["n = int(input())", "print(n * 2)"])
    r = run({"code": double, "stdin": "21\n"}, public)
    assert not r["crashed"] and r["stdout"].strip() == "42", r
    r = run({"code": "raise ValueError('bad input')"}, public)
    assert r["crashed"] and "ValueError" in r["error"], r
'''
TOOL_GUIDE = f"""A complete tool that passes validation. It only shows the required shape; yours does what the plan
says.
```python
{TOOL_EXAMPLE}```
Earlier tools failed validation, so: stay well under {MAX_TOOL_LINES} lines; define SCHEMA, run(args, public) and
test(public) at the top level as above; SCHEMA["args"] names the arguments run() uses and SCHEMA["result"] every
field it returns, and the workflow passes only those arguments and reads only those fields; public has only
python_exec and verilog_sim; in test() use plain assert on the smallest inputs whose result you are sure of, such as
a two-line program or the Verilog module "module and2(input a, input b, output y); assign y = a & b; endmodule";
build multi-line strings with "\\n".join([...]) and close every string."""


def _judge_note(r):
    d = r.get("detail") or {}
    if r["failure"] == "workflow_error":
        return d.get("error", "")[:200]
    if "passed" in d:
        return f"passed {d['passed']} of {d['total']} hidden tests; {d.get('reason', '')}"[:200]
    if "tests" in d:  # Meta-RSI T1 tasks (rive/meta_rsi/bugfix.py): one verdict per hidden assert
        return f"passed {d['tests'].count('pass')} of {len(d['tests'])} hidden tests; results {d['tests']}"[:200]
    if "load" in d:
        return f"the code did not load: {d['load']}"[:200]
    return "timed out" if d.get("timeout") else d.get("tail", "")[-200:]


def summary_counts(results):
    by_domain = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r)
    acc = {d: round(sum(r["correct"] for r in rs) / len(rs), 4) for d, rs in sorted(by_domain.items())}
    fails = {d: dict(Counter(r["failure"] for r in rs if r["failure"]).most_common()) for d, rs in
             sorted(by_domain.items())}
    return by_domain, acc, fails


def observe_agent(results, tasks_by_id, seed=0, max_examples=9):
    """Development feedback built by the program (handoff doc 3.7.3 OBSERVE), per domain. Examples are drawn with
    `seed`: two for the most common failure of each domain and one for the next most common."""
    rng = random.Random(seed)
    by_domain, acc, counts = summary_counts(results)
    examples = []
    for d, rs in by_domain.items():
        for (failure, _), k in zip(Counter(r["failure"] for r in rs if r["failure"]).most_common(2), (2, 1)):
            pool = [r for r in rs if r["failure"] == failure]
            for r in rng.sample(pool, min(k, len(pool))):
                examples.append({"id": r["id"], "domain": d, "failure": failure,
                                 "request": tasks_by_id[r["id"]]["public"]["prompt"][:500],
                                 "answer_tail": (r.get("answer") or "")[-400:], "judge": _judge_note(r),
                                 "completion_tokens": r.get("tokens"),
                                 "cut_off_by_token_limit": r.get("truncated_calls", 0) > 0})
    used = {f for c in counts.values() for f in c}
    calls = [c for r in results for c in r.get("tool_calls", [])]
    return {"n_tasks": len(results), "macro_accuracy": round(sum(acc.values()) / len(acc), 4),
            "accuracy_by_domain": acc, "failure_counts": counts,
            "failure_meanings": {f: FAILURE_MEANINGS[f] for f in sorted(used) if f in FAILURE_MEANINGS},
            "mean_completion_tokens": {d: round(sum(r.get("tokens", 0) for r in rs) / len(rs))
                                       for d, rs in by_domain.items()},
            "answers_cut_off_by_token_limit": {d: sum(r.get("truncated_calls", 0) > 0 for r in rs)
                                               for d, rs in by_domain.items()},
            "tool_calls": dict(Counter(c["name"] for c in calls)), "failed_tool_calls": sum(not c["ok"] for c in calls),
            "failure_examples": examples[:max_examples]}


def describe(cand):
    """The candidate's current modules, as the research model sees them."""
    parts = [f"agent/workflow.py:\n```python\n{(cand / WORKFLOW).read_text()}```"]
    for d in DOMAINS:
        p = cand / "agent" / "prompts" / f"{d}.md"
        text = p.read_text() if p.exists() else ""
        parts.append(f"agent/prompts/{d}.md:" + (f"\n```markdown\n{text}\n```" if text.strip() else " (empty)"))
    tools = sorted((cand / "tools").glob("*.py"))
    parts.append("tools: " + (", ".join(p.stem for p in tools) if tools else "(none yet)"))
    for p in tools[:3]:
        parts.append(f"tools/{p.name}:\n```python\n{p.read_text()[:2500]}```")
    notes = cand / "memory" / "notes.md"
    if notes.exists():
        parts.append("memory/notes.md (this branch's research notes):\n" + notes.read_text()[-1500:])
    return "\n\n".join(parts)


def blocks(text, langs):
    """Fenced blocks in `langs` (or unlabelled). Fences count only at the start of a line, so ``` inside code,
    as in a regex that extracts code from an answer, is left alone."""
    out, cur, lang = [], None, None
    for line in text.splitlines(keepends=True):
        if cur is None:
            if line.startswith("```"):
                lang, cur = line.strip()[3:].strip().lower(), []
        elif line.startswith("```") and line.strip() == "```":
            if not lang or lang in langs:
                out.append("".join(cur))
            cur = None
        else:
            cur.append(line)
    if cur is not None and (not lang or lang in langs):
        out.append("".join(cur))  # cut off before the closing fence; the checks will say what is wrong
    return out


def tool_name_of(raw):
    name = re.sub(r"[^a-z0-9_]", "_", (raw or "").strip().lower()).strip("_")[:30]
    return name if re.fullmatch(r"[a-z][a-z0-9_]*", name) else "tool"


def tool_shape(source):
    tree = ast.parse(source)
    problems = []
    if not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SCHEMA" for t in n.targets)
               for n in tree.body):
        problems.append("the tool defines no top-level SCHEMA")
    elif schema_keys(source, "result") is None:
        problems.append('the tool\'s SCHEMA has no literal "result" dict naming the fields run() returns')
    if not any(isinstance(n, ast.FunctionDef) and n.name == "test" for n in tree.body):
        problems.append("the tool defines no top-level test(public)")
    return problems


def schema_keys(source, part):
    """The keys of SCHEMA[part] ("args" or "result") in a tool's source; None when SCHEMA is not a literal dict with
    that part as a dict."""
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return None
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SCHEMA" for t in n.targets):
            try:
                schema = ast.literal_eval(n.value)
            except (ValueError, TypeError, SyntaxError):
                return None
            value = schema.get(part) if isinstance(schema, dict) else None
            return set(value) if isinstance(value, dict) else None
    return None


def tool_keys(name, part, cand, files):
    """What tool `name` declares as its "args" or "result": the public tools' own, else the SCHEMA of the
    tools/<name>.py being written (`files`) or already in candidate directory `cand`; None when unknown."""
    if name in PUBLIC_TOOLS:
        return (PUBLIC_ARGS if part == "args" else PUBLIC_RESULTS)[name]
    own = Path(cand) / "tools" / f"{name}.py"
    return schema_keys(files.get(f"tools/{name}.py") or (own.read_text() if own.exists() else None), part)


def args_passed(source):
    """{tool: the argument names a workflow passes it}, where it calls ctx.tool("tool", {...}) with a literal dict."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    out = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "tool" and len(n.args) > 1
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)
                and isinstance(n.args[1], ast.Dict)):
            out.setdefault(n.args[0].value, set()).update(
                k.value for k in n.args[1].keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
    return out


def fields_read(source):
    """{tool: the result fields a workflow reads from it}, where it keeps x = ctx.tool("tool", ...) and reads
    x["field"] or x.get("field", ...)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    holder = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == "tool" and n.value.args and isinstance(n.value.args[0], ast.Constant)
                and isinstance(n.value.args[0].value, str)):
            holder[n.targets[0].id] = n.value.args[0].value
    out = {}
    for n in ast.walk(tree):
        name = key = None
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            name, key = n.value.id, n.slice
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
              and isinstance(n.func.value, ast.Name) and n.args):
            name, key = n.func.value.id, n.args[0]
        if name in holder and isinstance(key, ast.Constant) and isinstance(key.value, str):
            out.setdefault(holder[name], set()).add(key.value)
    return out


def example_similarity(source):
    """How close a tool is to TOOL_EXAMPLE, ignoring docstrings and formatting (0 to 1)."""
    def norm(src):
        return ast.unparse(strip_docstrings(ast.parse(src)))
    return difflib.SequenceMatcher(None, norm(TOOL_EXAMPLE), norm(source), autojunk=False).ratio()


def content_hash(files):
    """Hash of the changed files, ignoring comments, docstrings and formatting in Python (duplicate detection)."""
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(path.encode())
        h.update((normalized_hash(files[path]) if path.endswith(".py") else " ".join(files[path].split())).encode())
    return h.hexdigest()[:16]


def called_tools(source):
    """Tool names that a workflow passes literally, as in ctx.tool("name", args)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {n.args[0].value for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "tool"
            and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)}


def over_call_cap(source):
    """max_tokens values above MAX_CALL_TOKENS that a workflow gives ctx.llm, written in the call or assigned to the
    name the call passes. The host cuts every call to MAX_CALL_TOKENS, so these change nothing."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    values, names = [], set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "llm":
            for v in [k.value for k in n.keywords if k.arg == "max_tokens"] + n.args[1:2]:
                values.append(v)
                if isinstance(v, ast.Name):
                    names.add(v.id)
    values += [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    return {c.value for v in values for c in ast.walk(v)
            if isinstance(c, ast.Constant) and type(c.value) is int and c.value > MAX_CALL_TOKENS}


def callable_tools(cand, tool=None):
    """What the candidate's workflow may call: the public tools, its tools/*.py, and the tool being written."""
    own = {p.stem for p in (Path(cand) / "tools").glob("*.py")}
    return sorted(set(PUBLIC_TOOLS) | own | ({tool} if tool else set()))


def module_kind(target):
    return "tool" if target == "tool" else "workflow" if target == WORKFLOW else "prompt"


def kind_of(record):
    """(domain, failure type, kind of module) an experiment worked on, or None if it stopped before DESIGN."""
    stages = record.get("stages", {})
    diag, design = stages.get("diagnose", {}).get("output"), stages.get("design", {}).get("output")
    if not diag or not design:
        return None
    return diag["selected_domain"], diag["selected_problem"], module_kind(design["target"])


def blocked_kinds(history, kind_after=BLOCK_KIND_AFTER, problem_after=BLOCK_PROBLEM_AFTER, block_for=BLOCK_FOR):
    """What the next experiment may not work on. history: the earlier experiments in order, each {"kind": kind_of()
    or None, "success": whether it passed the development gate (None: no result, as after an interruption),
    "hypothesis"}. Returns [{"domain", "failure", "module" (None: every module), "failures", "example"}]."""
    streak, last, example = {}, {}, {}
    for i, h in enumerate(history):
        if h["kind"] is None or h["success"] is None:
            continue
        domain, failure, module = h["kind"]
        for key in ((domain, failure, module), (domain, failure, None)):
            streak[key] = 0 if h["success"] else streak.get(key, 0) + 1
            last[key], example[key] = i, h["hypothesis"]
    return [{"domain": k[0], "failure": k[1], "module": k[2], "failures": n, "example": example[k][:150]}
            for k, n in sorted(streak.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or ""))
            if n >= (kind_after if k[2] else problem_after) and len(history) - 1 - last[k] < block_for]


def blocked_modules(blocked, domain, failure):
    """Kinds of module that the next experiment may not change for this problem."""
    out = set()
    for b in blocked:
        if (b["domain"], b["failure"]) == (domain, failure):
            out |= set(MODULES) if b["module"] is None else {b["module"]}
    return out


def describe_blocked(blocked):
    lines = [f"- {b['domain']} {b['failure']}" + (f" through {MODULES[b['module']]}" if b["module"] else
                                                   " through any module")
             + f": failed {b['failures']} times in a row; the latest attempt: {b['example']}" for b in blocked]
    return ("Blocked for now, because these failed repeatedly (an experiment on a blocked problem, or through a "
            "blocked kind of module, is not accepted):\n" + "\n".join(lines)) if lines else ""


def with_targets(schema, targets):
    """A copy of a HYPOTHESIZE or DESIGN schema whose target may only be one of `targets`."""
    out = copy.deepcopy(schema)
    props = out["properties"]
    (props["hypotheses"]["items"]["properties"] if "hypotheses" in props else props)["target"]["enum"] = list(targets)
    return out


def with_problem(schema, domain, failure):
    """A copy of the DIAGNOSE schema that can only select this problem."""
    out = copy.deepcopy(schema)
    out["properties"]["selected_domain"]["enum"] = [domain]
    out["properties"]["selected_problem"]["enum"] = [failure]
    return out


def assign_problem(results, blocked, taken, rng, min_count=3):
    """The problem (domain, failure type) for the next experiment of a generation (docs/phase-c-design.md section
    8; MiTa's allocation): drawn in proportion to its failing tasks in `results`, among the problems with at least
    min_count of them that are not blocked for every module and not yet `taken` in this generation, or among all of
    them once every one is taken. In C5, 4 of 7 generations put 3 of their 4 experiments on one problem. None when
    no problem qualifies."""
    counts = Counter((r["domain"], r["failure"]) for r in results if r["failure"] and r["failure"] not in UNFIXABLE)
    closed = {(b["domain"], b["failure"]) for b in blocked
              if blocked_modules(blocked, b["domain"], b["failure"]) == set(MODULES)}
    open_ = sorted((k, n) for k, n in counts.items() if n >= min_count and k not in closed)
    pool = [(k, n) for k, n in open_ if k not in taken] or open_
    if not pool:
        return None
    x = rng.random() * sum(n for _, n in pool)
    for k, n in pool:
        x -= n
        if x < 0:
            return k
    return pool[-1][0]


def reads_own_prompt_only(source):
    """Whether every ctx.prompt(...) in a workflow asks for the task's own domain: task["domain"] or
    task.get("domain")."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    def own(arg):
        if isinstance(arg, ast.Subscript):
            return (isinstance(arg.value, ast.Name) and arg.value.id == "task" and isinstance(arg.slice, ast.Constant)
                    and arg.slice.value == "domain")
        return (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.attr == "get"
                and isinstance(arg.func.value, ast.Name) and arg.func.value.id == "task" and bool(arg.args)
                and isinstance(arg.args[0], ast.Constant) and arg.args[0].value == "domain")
    return all(n.args and own(n.args[0]) for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "prompt")


def candidate_files(root):
    """A candidate directory's files as {relative path: bytes}. Research notes do not count."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and p.relative_to(root).parts[0] != "memory" and "__pycache__" not in p.parts}


def affected_domains(cand, reference):
    """The domains whose tasks can go differently in candidate directory `cand` than in `reference`: the domains
    of the prompt files when nothing but agent/prompts/<domain>.md differs and the shared workflow reads prompts
    only for the task's own domain; otherwise None, meaning every domain."""
    a, b = candidate_files(cand), candidate_files(reference)
    changed = {p for p in a.keys() | b.keys() if a.get(p) != b.get(p)}
    prompts = {f"agent/prompts/{d}.md": d for d in DOMAINS}
    if not changed or not changed <= prompts.keys() or not reads_own_prompt_only(a[WORKFLOW].decode()):
        return None
    return sorted(prompts[p] for p in changed)


def behaviour(sb, tasks):
    """What a candidate's workflow does on `tasks` with each set of stand-in model answers (STAND_INS): its model
    calls, the tools it used and its answer or error. Code that does the same gives the same record."""
    out = []
    for t in tasks:
        for answers in STAND_INS:
            calls = []

            def on_llm(req, text=answers[t["domain"]], calls=calls):
                calls.append([req["messages"], req["max_tokens"], req["thinking"], req["temperature"]])
                return {"ok": True, "text": text}
            done = sb.solve(t["public"], SMOKE_SECONDS, on_llm)
            out.append({"task": t["id"], "domain": t["domain"], "calls": calls, "answer": done.get("answer"),
                        "error": done.get("error"),
                        "tools": [[c.get("name"), c.get("ok")] for c in done.get("tool_calls", [])]})
    return out


def first_difference(before, after):
    """Where two behaviour() records first differ, in words; None when they do not."""
    for b, a in zip(before, after):
        for key, what in (("calls", "model calls"), ("tools", "tool calls"), ("answer", "answer"), ("error", "error")):
            if b[key] != a[key]:
                return f"on {a['domain']} task {a['task']} its {what} differ from the current agent's"
    return None


def locality_tasks(tasks_by_id, domain, per_domain=LOCALITY_TASKS):
    """The first `per_domain` tasks by id of every domain other than `domain`."""
    by = {}
    for tid in sorted(tasks_by_id):
        t = tasks_by_id[tid]
        if t["domain"] != domain and len(by.setdefault(t["domain"], [])) < per_domain:
            by[t["domain"]].append(t)
    return [t for d in sorted(by) for t in by[d]]


def prescreen_tasks(parent_results, tasks_by_id, rng, per_domain=PRESCREEN_TASKS, seconds=PRESCREEN_SECONDS):
    """{domain: tasks}: per domain, `per_domain` development tasks the parent solved in its latest run, drawn from
    those it solved within `seconds` when there are enough of them."""
    out = {}
    for d in DOMAINS:
        solved = sorted((r for r in parent_results if r["domain"] == d and r["correct"] and r["id"] in tasks_by_id),
                        key=lambda r: r["id"])
        quick = [r for r in solved if (r.get("seconds") or 0) <= seconds]
        pool = quick if len(quick) >= per_domain else solved
        out[d] = [tasks_by_id[r["id"]] for r in rng.sample(pool, min(per_domain, len(pool)))]
    return out


def metric_value(results, name):
    """A prediction metric (schemas.AGENT_METRICS) over per-item results."""
    if name == "macro_accuracy":
        domains = sorted({r["domain"] for r in results})
        return sum(metric_value(results, f"{d}_accuracy") for d in domains) / len(domains)
    if name == "completion_tokens":
        return sum(r.get("tokens") or 0 for r in results) / len(results)
    if name.endswith("_accuracy"):
        rs = [r for r in results if r["domain"] == name[:-len("_accuracy")]]
        return sum(r["correct"] for r in rs) / len(rs)
    return sum(r.get("failure") == name[:-len("_count")] for r in results)


def prediction_check(prediction, champion_runs, child_runs, domains):
    """Whether the predicted metric moved the predicted way: each side averaged over its runs, on the judged
    domains' tasks only. A prediction about a domain the change cannot affect is not testable (hit None)."""
    name, direction = prediction["metric"], prediction["direction"]
    out = {"metric": name, "direction": direction}
    if name.endswith("_accuracy") and name != "macro_accuracy" and name[:-len("_accuracy")] not in domains:
        return {**out, "hit": None, "why": "the change cannot affect that domain"}

    def value(runs):
        return sum(metric_value([r for r in run if r["domain"] in domains], name) for run in runs) / len(runs)
    before, after = value(champion_runs), value(child_runs)
    return {**out, "champion": before, "child": after,
            "hit": None if after == before else (after > before) == (direction == "increase")}


def implement_steps(target, tool):
    """IMPLEMENT requests as (file, request) pairs, one file per answer. Asked for a tool and its workflow as two
    code blocks in one answer, the 4B model wrote one block in all 3 attempts, twice in the C4 pilot."""
    if target == "tool":
        return [(f"tools/{tool}.py", f"IMPLEMENT, step 1 of 2: write the complete tools/{tool}.py. Output it in one "
                                     "```python code block."),
                (WORKFLOW, f"IMPLEMENT, step 2 of 2: write the complete new agent/workflow.py that calls this tool "
                           f"with ctx.tool(\"{tool}\", args). Output the whole file in one ```python code block.")]
    if target == WORKFLOW:
        return [(WORKFLOW, "IMPLEMENT: write the complete new agent/workflow.py implementing this plan. Output the "
                           "whole file in one ```python code block.")]
    return [(target, f"IMPLEMENT: write the complete new {target} (at most {MAX_PROMPT_CHARS} characters). Output it "
                     "in one ```markdown code block.")]


def fenced(path, src):
    lang = "python" if path.endswith(".py") else "markdown"
    return f"{path}:\n```{lang}\n{src.rstrip()}\n```"


def parse_file(text, path):
    """The content an IMPLEMENT answer gives for `path`, or the problems that stop it."""
    if path.endswith(".py"):
        py = blocks(text, {"python", "py"})
        return (py[0], []) if py else (None, [f"no ```python code block for {path}"])
    md = (blocks(text, {"markdown", "md", "text"}) or [text])[0].strip()
    if not md or len(md) > MAX_PROMPT_CHARS:
        return None, [f"the prompt must have 1 to {MAX_PROMPT_CHARS} characters, it has {len(md)}"]
    return md + "\n", []


class AgentResearch(Stages):
    def __init__(self, llm, run_dir, smoke_tasks, leak_scanner=None, make_sandbox=CandidateSandbox, thinking=False,
                 retries=2, temperature=0.7, locality=True, solver=None, run_tasks=run_workflow_tasks,
                 system=SYSTEM, contract=CONTRACT, extra_check=None):
        """smoke_tasks: a few dev tasks (one per domain) to check that a new workflow runs;
        leak_scanner: LeakScanner over protected items, checked before any evaluation;
        locality: a change for an assigned problem must not alter how other domains' tasks are solved;
        solver: the model client the agent answers with; when given, a change first answers a few development
        tasks its parent solved (the pre-screen, through run_tasks, which judges maths on the main thread only);
        system, contract: what the research model is told about its role and the agent (Meta-RSI tells it about
        its bug-fix tasks, rive/meta_rsi/improvers.py);
        extra_check(child_dir, parent_results, tasks_by_id, record, seed) -> problems: one more check after all
        others, whose problems send the change back like theirs (Meta-RSI's effect check)."""
        super().__init__(llm, thinking, retries, temperature)
        self.run_dir, self.smoke_tasks = Path(run_dir), smoke_tasks
        self.leak_scanner, self.make_sandbox = leak_scanner, make_sandbox
        self.locality, self.solver, self.run_tasks = locality, solver, run_tasks
        self.system, self.contract, self.extra_check = system, contract, extra_check
        self.seen_hashes = set()
        self._behaviour = {}  # behaviour() of a parent: {(its files, task ids): record}

    def propose(self, child_id, parent_dir, parent_results, tasks_by_id, history_notes, seed, blocked=(), problem=None):
        """DIAGNOSE to VALIDATE for one child of parent_dir. Returns the child dict the evolution controller expects;
        status "valid" means it may be evaluated. blocked: blocked_kinds() of the earlier experiments; problem: the
        (domain, failure type) assigned to this experiment (assign_problem()), or None to let DIAGNOSE choose."""
        self.seed = seed
        child_dir = self.run_dir / child_id
        shutil.rmtree(child_dir, ignore_errors=True)
        shutil.copytree(parent_dir, child_dir)
        record = {"stages": {}, "checks": {}, "blocked": list(blocked), "assigned": list(problem) if problem else None}
        child = {"id": child_id, "dir": str(child_dir), "status": "started", "module": None, "record": record}
        try:
            self._propose(child, child_dir, parent_results, tasks_by_id, history_notes, list(blocked), problem,
                          Path(parent_dir))
        except (StageFailed, BadRequestError) as e:  # a request the server refused (e.g. too long) costs one child
            child["status"], record["error"] = "stage_failed", str(e)[:500]
        if child["status"] != "valid":
            tried = record.get("hypothesis", {}).get("hypothesis", "no hypothesis reached")
            reason = {"duplicate": "it repeated an earlier change exactly",
                      "leak_violation": "its files contained protected test content",
                      "invalid_patch": "it failed validation: " + "; ".join(
                          record["stages"].get("implement", {}).get("problems", []))[:200],
                      "stage_failed": "a stage produced no valid output or the model server refused a request"
                      }.get(child["status"], child["status"])
            child["note"] = f"{child_id} (not evaluated): {tried[:160]} -> rejected because {reason}."
        elif record["checks"].get("other_domains_unchanged"):
            child["local_to"] = problem[0]  # checked: only the assigned problem's domain can go differently
        return child

    def _propose(self, child, child_dir, parent_results, tasks_by_id, history_notes, blocked, problem=None,
                 parent_dir=None):
        record = child["record"]
        local = (f" The change may alter only how {problem[0]} tasks are solved; tasks of the other domains must go "
                 "exactly as now." if problem and self.locality else "")
        feedback = observe_agent(parent_results, tasks_by_id, seed=self.seed)
        record["stages"]["observe"] = feedback
        older = [n[:200] for n in history_notes[:-5]]
        history = "\n".join(f"- {n}" for n in older + list(history_notes[-5:])) or "(none yet)"
        base = [{"role": "system", "content": self.system},
                {"role": "user", "content": f"{self.contract}\n\nCurrent agent:\n{describe(child_dir)}\n\nDevelopment "
                                            f"feedback (JSON):\n{json.dumps(feedback, ensure_ascii=False)}\n\n"
                                            f"Conclusions of earlier experiments:\n{history}"
                                            + (f"\n\n{describe_blocked(blocked)}" if blocked else "")}]

        # DIAGNOSE. The note "do not return to an approach that failed" alone did not stop the C5 run from
        # proposing a Verilog checker tool 7 times, so problems blocked for every module are refused here.
        closed = sorted({(b["domain"], b["failure"]) for b in blocked
                         if blocked_modules(blocked, b["domain"], b["failure"]) == set(MODULES)})

        def open_problem(d):
            if (d["selected_domain"], d["selected_problem"]) in closed:
                return (f"{d['selected_domain']} {d['selected_problem']} is blocked for every module, select another "
                        "domain or failure type")

        evidence = ("Evidence must use the exact domains, failure types, counts and example ids from the feedback; an "
                    "example id belongs only to the domain and failure type it is listed with. Do not return to an "
                    "approach that an earlier experiment tried without improvement unless you change it "
                    "substantially.")
        if problem:
            # The controller spreads one generation's experiments over different problems (assign_problem).
            n = feedback["failure_counts"].get(problem[0], {}).get(problem[1], 0)
            ask = (f"DIAGNOSE: this experiment is assigned one problem: {problem[0]} {problem[1]} ({n} failing "
                   "tasks); the other experiments of this generation work on other problems. Select it and diagnose "
                   f"its cause. {evidence}")
            diag, meta = self._structured(base + [{"role": "user", "content": ask}],
                                          with_problem(schemas.AGENT_DIAGNOSE, *problem))
        else:
            diag, meta = self._structured(base + [{"role": "user", "content": (
                f"DIAGNOSE: pick the single most valuable problem to work on now: one domain and one failure type. "
                f"{evidence}"
                + (" These problems are blocked and cannot be selected now: " + ", ".join(f"{d} {f}" for d, f in closed)
                   + "." if closed else ""))}],
                schemas.AGENT_DIAGNOSE, check=open_problem)
        counts = feedback["failure_counts"]
        shown = {}
        for ex in feedback["failure_examples"]:
            shown.setdefault((ex["domain"], ex["failure"]), set()).add(ex["id"])
        record["stages"]["diagnose"] = {"output": diag, **meta}
        record["checks"]["diagnose_evidence_matches_records"] = (
            counts.get(diag["selected_domain"], {}).get(diag["selected_problem"], 0) > 0 and all(
                e["count"] == counts.get(e["domain"], {}).get(e["failure_type"], 0)
                and set(e["example_ids"]) <= shown.get((e["domain"], e["failure_type"]), set())
                for e in diag["evidence"]))

        # HYPOTHESIZE and DESIGN may only target kinds of module that are not blocked for this problem; the schema
        # enforces it, the request says why.
        shut = blocked_modules(blocked, diag["selected_domain"], diag["selected_problem"])
        targets = [t for t in schemas.AGENT_TARGETS if module_kind(t) not in shut]
        limit = (" For this problem these kinds of module are blocked: " + ", ".join(MODULES[m] for m in sorted(shut))
                 + "; the target must be one of: " + ", ".join(targets) + ".") if shut else ""
        hyp, meta = self._structured(base + [{"role": "user", "content": (
            f"Your diagnosis:\n{json.dumps(diag, ensure_ascii=False)}\n\nHYPOTHESIZE: propose exactly two different, "
            "testable hypotheses for fixing it, each changing one module (target), with a measurable prediction and "
            "the unchanged current agent as control; then select one. Give tool_name only when the target is tool, "
            "otherwise an empty string." + limit + local)}],
            with_targets(schemas.AGENT_HYPOTHESIZE, targets) if shut else schemas.AGENT_HYPOTHESIZE)
        chosen = next(h for h in hyp["hypotheses"] if h["hypothesis_id"] == hyp["selected"])
        record["stages"]["hypothesize"] = {"output": hyp, **meta}
        record["hypothesis"] = chosen
        record["checks"]["hypothesis_testable"] = bool(chosen["prediction"]["metric"] and chosen["control"].strip())

        # DESIGN
        design, meta = self._structured(base + [{"role": "user", "content": (
            f"Selected hypothesis:\n{json.dumps(chosen, ensure_ascii=False)}\n\nDESIGN: write the experiment plan: "
            "the target module, concrete change steps, the primary metric and when to roll back." + limit + local)}],
            with_targets(schemas.AGENT_DESIGN, targets) if shut else schemas.AGENT_DESIGN)
        record["stages"]["design"] = {"output": design, **meta}
        record["kind"] = kind_of(record)
        target = design["target"]
        tool = tool_name_of(design["tool_name"] or chosen["tool_name"]) if target == "tool" else None
        child["module"] = f"tools/{tool}.py" if tool else target

        # IMPLEMENT + VALIDATE, one file per answer. The feedback and history stay out of this prompt (DESIGN has
        # turned them into the plan), and a repair sees only the latest attempt: in the C5 run, attempts piled up in
        # one conversation went past the server's 12288-token context.
        steps = implement_steps(target, tool)
        others = locality_tasks(tasks_by_id, problem[0]) if problem and self.locality else []
        screen = (prescreen_tasks(parent_results, tasks_by_id, random.Random(self.seed))
                  if self.solver is not None else {})
        rules = ""
        if problem and self.locality:
            rest = " and ".join(d for d in DOMAINS if d != problem[0])
            rules += (f"This experiment is assigned {problem[0]} {problem[1]}: the change must leave how {rest} tasks "
                      f"are solved exactly as it is. The checker runs the current agent and your version on {rest} "
                      "tasks with stand-in model answers and compares every model call, tool call and answer; in "
                      f"agent/workflow.py put new behaviour under if task[\"domain\"] == \"{problem[0]}\".\n")
        if screen:
            rules += (f"Before it is evaluated, your version answers {PRESCREEN_TASKS} development tasks that the "
                      "current agent solves in each domain it can affect, with the real model; more than a third of "
                      "them wrong sends it back to you with what went wrong.\n")
        intro = (f"{self.contract}\n\nCurrent agent:\n{describe(child_dir)}\n\nSelected hypothesis: "
                 f"{chosen['hypothesis']}\n\nPlan:\n{json.dumps(design, ensure_ascii=False)}\n\n"
                 + (f"{TOOL_GUIDE}\n\n" if tool else "")
                 + (f"agent/workflow.py may call only these tools with ctx.tool: "
                    f"{', '.join(callable_tools(child_dir, tool))}. Any other name fails validation.\n\n"
                    if target in ("tool", WORKFLOW) else "")
                 + (f"{rules}\n" if rules else ""))
        files, problems, last = {}, [], ""
        for attempt in range(self.retries + 1):
            prompt = [{"role": "system", "content": self.system}, {"role": "user", "content": intro + steps[0][1]}]
            if attempt:
                prompt = prompt + [{"role": "assistant", "content": last},
                                   {"role": "user", "content": "That failed validation:\n- " + "\n- ".join(problems)
                                                               + "\nFix it. " + steps[0][1]}]
            files, problems = {}, []
            for key in ("other_domains_unchanged", "prescreen_passed", "effect_fixed"):  # the latest attempt's
                record["checks"].pop(key, None)
            for n, (path, request) in enumerate(steps):
                if n:
                    prompt = prompt + [{"role": "user", "content": request}]
                text = self.llm.chat(prompt, max_tokens=3500, thinking=self.thinking, temperature=self.temperature,
                                     seed=self.seed + attempt)
                src, problems = parse_file(text, path)
                if problems:
                    last = text[-6000:]
                    break
                files[path] = src
                prompt = prompt + [{"role": "assistant", "content": fenced(path, src)}]
            if not problems:
                problems = self._write_and_check(child_dir, files, tool)
                last = "\n\n".join(fenced(p, s) for p, s in files.items())
            if not problems and problem and self.locality:
                problems = self._check_locality(child_dir, parent_dir, others, problem, record)
            if not problems and screen:
                domains = ([problem[0]] if problem and self.locality else
                           affected_domains(child_dir, parent_dir) or list(DOMAINS))
                problems = self._prescreen(child_dir, [t for d in domains for t in screen[d]], record)
            if not problems and self.extra_check is not None:
                problems = self.extra_check(child_dir, parent_results, tasks_by_id, record, self.seed + attempt)
            if not problems:
                break
        record["stages"]["implement"] = {"attempts": attempt + 1, "problems": problems, "files": sorted(files)}
        record["checks"]["patch_valid"] = not problems
        if tool and f"tools/{tool}.py" in files:
            try:  # recorded, not judged: a plain Python runner like the example can be the right tool
                record["checks"]["tool_example_similarity"] = round(example_similarity(files[f"tools/{tool}.py"]), 3)
            except SyntaxError:
                pass
        if problems:
            record["stages"]["implement"]["last_answer"] = last[-4000:]  # files that fail static checks are not kept
            child["status"] = "invalid_patch"
            return
        child["code_hash"] = content_hash(files)
        if child["code_hash"] in self.seen_hashes:
            child["status"] = "duplicate"
            return
        self.seen_hashes.add(child["code_hash"])
        if self.leak_scanner is not None:
            texts = {str(p.relative_to(child_dir)): p.read_text(errors="replace")
                     for p in child_dir.rglob("*") if p.is_file() and p.suffix in (".py", ".md")}
            scan = self.leak_scanner.scan(texts)
            record["leak_scan"] = scan
            record["checks"]["leak_scan_clean"] = not scan["violations"]
            if scan["violations"]:
                child["status"] = "leak_violation"
                return
        child["status"] = "valid"

    def _write_and_check(self, child_dir, files, tool):
        problems = []
        for path, src in files.items():
            if path == WORKFLOW:
                problems += check_python_source(src, "solve", MAX_WORKFLOW_LINES, "workflow")
                # C5's g01c001 and g03c012 called tools nobody had written; ctx.tool raised, and the smoke test
                # passed because the workflow caught the error or never reached the call.
                known = callable_tools(child_dir, tool)
                problems += [f"agent/workflow.py calls ctx.tool({name!r}), but there is no such tool; it may call only "
                             f"{', '.join(known)}" for name in sorted(called_tools(src) - set(known))]
                # In the check before generation 4, 3 of 4 research steps "fixed" maths truncation this way.
                problems += [f"ctx.llm(max_tokens={n}) changes nothing: one call returns at most {MAX_CALL_TOKENS} "
                             f"completion tokens; the {TOKEN_BUDGET} of a task are used through more calls"
                             for n in sorted(over_call_cap(src))[:1]]
                # C5's g08c033 gave design and testbench to a tool that takes code and stdin, and read "valid" from
                # its crashed/stdout/error result, so its check never fired.
                for name, passed in sorted(args_passed(src).items()):
                    have = tool_keys(name, "args", child_dir, files)
                    if have is not None and passed - have:
                        problems.append(f"agent/workflow.py passes {', '.join(sorted(passed - have))} to "
                                        f"ctx.tool({name!r}), which takes only {', '.join(sorted(have))}")
                for name, wanted in sorted(fields_read(src).items()):
                    have = tool_keys(name, "result", child_dir, files)
                    if have is not None and wanted - have:
                        problems.append(f"agent/workflow.py reads {', '.join(sorted(wanted - have))} from "
                                        f"ctx.tool({name!r}), whose result has only {', '.join(sorted(have))}")
            elif path.endswith(".py"):
                found = check_python_source(src, "run", MAX_TOOL_LINES, "tool")
                problems += found or tool_shape(src)
        if problems:
            return problems
        for path, src in files.items():
            (child_dir / path).parent.mkdir(parents=True, exist_ok=True)
            (child_dir / path).write_text(src)
        with self.make_sandbox(child_dir, memory="768m") as sb:
            if tool:
                r = sb.tool_test(tool)
                if not r.get("ok"):
                    return [f"test() of tools/{tool}.py failed: {r.get('error', '')[:300]}"]
            for t in self.smoke_tasks:
                done = sb.solve(t["public"], SMOKE_SECONDS, lambda req, d=t["domain"]: {"ok": True, "text": CANNED[d]})
                if not done.get("ok"):
                    return [f"the workflow failed on {t['domain']} task {t['id']} with stand-in model answers: "
                            f"{done.get('error', '')[:300]}"]
        return []

    def _check_locality(self, child_dir, parent_dir, tasks, problem, record):
        """Problems when the change alters how `tasks` (of the domains other than the assigned problem's) are
        solved: their behaviour() must be the parent's."""
        h = hashlib.sha256()
        for path, data in sorted(candidate_files(parent_dir).items()):
            h.update(path.encode() + b"\0" + hashlib.sha256(data).digest())
        key = (h.hexdigest(), tuple(t["id"] for t in tasks))
        if key not in self._behaviour:
            with self.make_sandbox(parent_dir, memory="768m") as sb:
                self._behaviour[key] = behaviour(sb, tasks)
        with self.make_sandbox(child_dir, memory="768m") as sb:
            diff = first_difference(self._behaviour[key], behaviour(sb, tasks))
        record["checks"]["other_domains_unchanged"] = diff is None
        if diff is None:
            return []
        return [f"this experiment is assigned {problem[0]} {problem[1]}, but the change also alters how other "
                f"domains' tasks are solved ({diff}, with stand-in model answers): leave them exactly as they are, "
                f"for example by putting the new behaviour under if task[\"domain\"] == \"{problem[0]}\""]

    def _prescreen(self, child_dir, tasks, record):
        """Problems when the change, answering with the real model, gets more than a third of `tasks` (development
        tasks its parent solved) wrong."""
        if not tasks:
            return []
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("the pre-screen judges maths with math-verify, which works only on the main thread")
        results = self.run_tasks(tasks, child_dir, self.solver, workers=len(tasks))
        wrong = [r for r in results if not r["correct"]]
        record["stages"].setdefault("prescreen", []).append(
            [{k: r.get(k) for k in ("id", "domain", "correct", "failure", "llm_calls", "seconds")} for r in results])
        record["checks"]["prescreen_passed"] = 3 * len(wrong) <= len(results)
        if record["checks"]["prescreen_passed"]:
            return []
        shown = "; ".join(f"{r['id']} ({r['domain']}) {r['failure']}, " + (
            "empty answer" if not (r.get("answer") or "").strip() else f"answer ends {r['answer'][-120:]!r}")
            for r in wrong[:3])
        return [f"the pre-screen ran this version with the real model on {len(results)} development tasks that the "
                f"current agent solves, and it got {len(wrong)} wrong: {shown}"]

    def reflect(self, child, verdict, champion_runs, child_runs):
        """After evaluation: REFLECT on facts the controller measured. verdict: the controller's judge() result plus
        "alpha"; champion_runs and child_runs: the per-item results it compared.

        The conclusion follows from the measurements by rule (docs/phase-c-design.md section 8): supported when the
        child is significantly better and the predicted metric moved the predicted way, refuted when it is
        significantly worse, otherwise inconclusive. In C5 the model itself wrote "refuted" for 24 of 26 results,
        20 of them within run-to-run noise. The model now explains the conclusion. Returns the note, which is also
        appended to the child's memory/notes.md for its descendants."""
        record, domains = child["record"], verdict["domains"]
        check = prediction_check(record["hypothesis"]["prediction"], champion_runs, child_runs, domains)
        conclusion = ("supported" if verdict["pass_dev"] and check["hit"] else
                      "refuted" if verdict["p_worse"] < verdict["alpha"] else "inconclusive")
        record["prediction_check"], record["verdict"] = check, conclusion

        def on(run):
            return [r for r in run if r["domain"] in domains]

        def mean_accuracy(runs):
            return {d: round(sum(summary_counts(on(run))[1][d] for run in runs) / len(runs), 4) for d in domains}
        facts = {"judged_domains": domains, "difference_vs_champion": round(verdict["diff"], 4),
                 "p_better": round(verdict["p"], 4), "p_worse": round(verdict["p_worse"], 4),
                 "run_to_run_sd": round(verdict["sd"], 4),
                 "runs": {"champion": verdict["champion_runs"], "child": verdict["child_runs"]},
                 "share_of_champion_solved_tasks_lost": round(verdict["regression"], 4),
                 "share_lost_expected_from_noise": round(verdict["expected_regression"], 4),
                 "prediction": {k: round(v, 4) if isinstance(v, float) else v for k, v in check.items()},
                 "accuracy_by_domain": {"champion_mean": mean_accuracy(champion_runs),
                                        "child_mean": mean_accuracy(child_runs)},
                 "failure_counts": {"champion_latest": summary_counts(on(champion_runs[-1]))[2],
                                    "child_latest": summary_counts(on(child_runs[-1]))[2]}}
        meaning = {"supported": "the child is significantly better and the predicted metric moved as predicted",
                   "refuted": "the child is significantly worse than the champion",
                   "inconclusive": "the difference is within run-to-run noise, so the experiment neither supports nor "
                                   "refutes the hypothesis; do not conclude that the approach does not work"}
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": (
            f"Hypothesis:\n{json.dumps(record['hypothesis'], ensure_ascii=False)}\n\nPlan:\n"
            f"{json.dumps(record['stages']['design']['output'], ensure_ascii=False)}\n\nMeasured result (recorded by "
            f"the controller; the child against the current champion's runs on the same tasks):\n{json.dumps(facts)}"
            f"\n\nThe conclusion follows from these measurements by rule: {conclusion} ({meaning[conclusion]}). "
            "REFLECT: explain the result, its limitation and the next direction.")}]
        try:
            out, meta = self._structured(messages, schemas.AGENT_REFLECT)
            record["stages"]["reflect"] = {"output": {"conclusion": conclusion, **out}, **meta}
            finding = out["reusable_finding"][:160]
        except StageFailed as e:
            record["stages"]["reflect"] = {"error": str(e), "output": {"conclusion": conclusion}}
            finding = "no reflection"
        held = {True: "held", False: "failed", None: "unchanged or not testable"}[check["hit"]]
        note = (f"{child['id']} ({child['module']}): {record['hypothesis']['hypothesis'][:160]} -> {conclusion}: "
                f"{verdict['diff']:+.3f} on {', '.join(domains)} vs champion (p={verdict['p']:.3f}); predicted "
                f"{check['metric']} {check['direction']}: {held}; {finding}")
        (Path(child["dir"]) / "memory").mkdir(exist_ok=True)
        with open(Path(child["dir"]) / "memory" / "notes.md", "a") as f:
            f.write(f"\n- {note}\n")
        return note
