"""T1 bug-fix tasks for Meta-RSI (docs/meta-rsi-design.md section 4): MBPP reference functions with bugs injected
by AST mutation.

A mutant is kept when the reference passes every MBPP assert and the mutant fails at least one of them without
timing out; both are checked in the judge sandbox. A setting sets the difficulty: whether the problem statement is
shown, whether one failing test is shown, and how many bugs are injected. The hidden tests are the MBPP asserts.
"""
import ast
import copy
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor

from rive.benchmark.hard import code_block
from rive.sandbox.judge import run as judge_run

SOURCE = "t1_mbpp"
# name: (statement shown, one failing test shown, bugs injected)
SETTINGS = {"spec_fail_1": (True, True, 1), "nospec_fail_1": (False, True, 1),
            "nospec_fail_2": (False, True, 2), "spec_nofail_1": (True, False, 1), "spec_nofail_2": (True, False, 2),
            "nospec_nofail_1": (False, False, 1), "nospec_nofail_2": (False, False, 2)}
# The pilot put A0 at 45% and 41% on these (docs/meta-rsi-design.md section 5): the task packs use them.
PACK_SETTINGS = {1: "nospec_fail_1", 2: "nospec_fail_2"}

COMPARE = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
           ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
BINOP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.FloorDiv: ast.Mod, ast.Mod: ast.FloorDiv,
         ast.Div: ast.FloorDiv}
BOOL = {ast.And: ast.Or, ast.Or: ast.And}
SWAP_CALL = {"min": "max", "max": "min", "any": "all", "all": "any"}

# Runs inside the judge container: the code, then each assert on its own, so every test gets a verdict.
CHECK = r'''
import json, sys
tests = json.load(open("/work/tests.json"))
env = {}
try:
    exec(compile(open("/work/code.py").read(), "code.py", "exec"), env)
except BaseException as e:
    print(json.dumps({"load": type(e).__name__})); sys.exit(0)
results = []
for t in tests:
    try:
        exec(t, dict(env)); results.append("pass")
    except AssertionError:
        results.append("fail")
    except BaseException as e:
        results.append(type(e).__name__)
print(json.dumps({"tests": results}))
'''


def preorder(node):
    yield node
    for child in ast.iter_child_nodes(node):
        yield from preorder(child)


def sites(tree):
    """Where a bug can go, as (node index in pre-order, kind, detail)."""
    out = []
    for i, node in enumerate(preorder(tree)):
        if isinstance(node, ast.Compare):
            out += [(i, "compare", k) for k, op in enumerate(node.ops) if type(op) in COMPARE]
        elif isinstance(node, ast.BinOp) and type(node.op) in BINOP:
            out.append((i, "binop", None))
        elif isinstance(node, ast.AugAssign) and type(node.op) in (ast.Add, ast.Sub):
            out.append((i, "augassign", None))
        elif isinstance(node, ast.BoolOp):
            out.append((i, "boolop", None))
        elif isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10:
            out += [(i, "constant", 1), (i, "constant", -1)]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            out.append((i, "drop_not", None))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in SWAP_CALL:
                out.append((i, "swap_call", None))
            elif node.func.id == "range" and node.args:
                out.append((i, "range_stop", None))
    return out


def mutate(tree, site):
    """A copy of `tree` with a bug at `site`."""
    tree = copy.deepcopy(tree)
    nodes = list(preorder(tree))
    i, kind, detail = site
    node = nodes[i]
    if kind == "compare":
        node.ops[detail] = COMPARE[type(node.ops[detail])]()
    elif kind == "binop":
        node.op = BINOP[type(node.op)]()
    elif kind == "augassign":
        node.op = ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()
    elif kind == "boolop":
        node.op = BOOL[type(node.op)]()
    elif kind == "constant":
        node.value += detail
    elif kind == "swap_call":
        node.func.id = SWAP_CALL[node.func.id]
    elif kind == "range_stop":
        k = 0 if len(node.args) == 1 else 1
        node.args[k] = ast.BinOp(left=node.args[k], op=ast.Sub(), right=ast.Constant(1))
    elif kind == "drop_not":
        for parent in nodes:
            for field, value in ast.iter_fields(parent):
                if value is node:
                    setattr(parent, field, node.operand)
                elif isinstance(value, list):
                    value[:] = [node.operand if v is node else v for v in value]
    return ast.fix_missing_locations(tree)


def candidates(item, bugs, n=6, seed=0):
    """Up to n distinct mutants of the item's reference with `bugs` bugs each, in a fixed order."""
    tree = ast.parse(item["reference"])
    options, original = sites(tree), ast.unparse(tree)
    rng = random.Random(f"{item['id']}:{bugs}:{seed}")
    out, seen = [], {original}
    for _ in range(n * 6):
        if len(options) < bugs or len(out) == n:
            break
        chosen = rng.sample(options, bugs)
        if len({s[0] for s in chosen}) < bugs:
            continue
        mutant = tree
        for site in sorted(chosen, key=lambda s: -s[0]):  # later nodes first: the earlier indices stay valid
            mutant = mutate(mutant, site)
        code = ast.unparse(mutant)
        if code not in seen:
            seen.add(code)
            out.append({"code": code, "mutations": [list(s) for s in chosen]})
    return out


def check(item, code, timeout=20):
    """Each MBPP assert against `code` in the judge sandbox: {"tests": ["pass" | "fail" | error name]},
    {"load": error name} or {"timeout": True}."""
    source = "\n".join(item["test_imports"]) + "\n" + code
    r = judge_run({"check.py": CHECK, "code.py": source, "tests.json": json.dumps(item["tests"])},
                  ["python", "check.py"], timeout=timeout)
    if r.timed_out:
        return {"timeout": True}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"load": "no output: " + r.stderr[-200:]}


def failing(result):
    """Indices of the tests a check() result failed; None when the code did not load or timed out."""
    return [i for i, v in enumerate(result["tests"]) if v != "pass"] if "tests" in result else None


def prompt(item, buggy, failing_test, spec):
    lines = [f"Task: {item['prompt']}\n"] if spec else []
    lines.append(f"This Python code does not work correctly:\n```python\n{buggy}\n```")
    if failing_test:
        lines.append(f"It fails this test:\n```python\n{failing_test}\n```")
    lines.append("Find what is wrong and fix it. Keep the function names and signatures. Return the complete "
                 "corrected code in a single ```python code block.")
    return "\n".join(lines)


def make_task(item, mutant, setting):
    """A task for the agent; the tests stay in "item", which only the judge reads."""
    spec, show_failing, _ = SETTINGS[setting]
    shown = item["tests"][mutant["failing"][0]] if show_failing else None
    tid = f"t1-{item['id']}-{setting}-{hashlib.sha256(mutant['code'].encode()).hexdigest()[:6]}"
    return {"id": tid, "domain": "code", "source": SOURCE,
            "public": {"domain": "code", "source": SOURCE, "prompt": prompt(item, mutant["code"], shown, spec),
                       "buggy_code": mutant["code"], "failing_test": shown,
                       "statement": item["prompt"] if spec else ""},
            "item": {**item, "setting": setting, "buggy_code": mutant["code"], "mutations": mutant["mutations"],
                     "shown_test": shown}}


def retarget(task, setting):
    """The task's buggy code in another setting with as many bugs: only what the agent is shown changes."""
    old = task["item"]
    item = {k: v for k, v in old.items() if k not in ("setting", "buggy_code", "mutations", "shown_test")}
    if SETTINGS[setting][2] != len(old["mutations"]):
        raise ValueError(f"{setting} has {SETTINGS[setting][2]} bugs; the task has {len(old['mutations'])}")
    failing = [item["tests"].index(old["shown_test"])] if old.get("shown_test") in item["tests"] else []
    if SETTINGS[setting][1] and not failing:
        raise ValueError(f"{task['id']} shows no failing test to keep showing in {setting}")
    return make_task(item, {"code": old["buggy_code"], "mutations": old["mutations"], "failing": failing}, setting)


def make_pack(items, per_problem, n=6, seed=0, threads=8):
    """Checked tasks from `items`: per problem, up to per_problem[bugs] mutants with that many bugs, in the
    PACK_SETTINGS setting for that count. A problem whose reference fails its own tests is left out."""
    cands = {(i["id"], b): candidates(i, b, n=n, seed=seed) for i in items for b in per_problem}
    jobs = [(i, None) for i in items] + [(i, c) for i in items for b in per_problem for c in cands[(i["id"], b)]]
    with ThreadPoolExecutor(threads) as pool:
        results = list(pool.map(lambda j: check(j[0], j[0]["reference"] if j[1] is None else j[1]["code"]), jobs))
    reference_ok = {}
    for (item, cand), r in zip(jobs, results):
        if cand is None:
            reference_ok[item["id"]] = failing(r) == []
        else:
            cand["failing"] = failing(r)
    return [make_task(i, c, PACK_SETTINGS[b]) for i in items if reference_ok[i["id"]]
            for b, k in per_problem.items() for c in [c for c in cands[(i["id"], b)] if c["failing"]][:k]]


def judge(task, answer):
    """(correct, failure, detail) for an answer to a T1 task: every MBPP assert must pass."""
    r = check(task["item"], code_block(answer, ["python", "py"]))
    if r.get("timeout"):
        return False, "time_limit", r
    if "load" in r:
        return False, "load_error", r
    if not failing(r):
        return True, None, r
    return False, "wrong_answer" if "fail" in r["tests"] else "runtime_error", r
