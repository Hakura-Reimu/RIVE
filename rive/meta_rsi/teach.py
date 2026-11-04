"""R-011's teaching data (docs/rive-design.md section 6.18): the teacher writes new LiveCodeBench-style problems and
solves them as T2's agent is asked to; a solution is kept when it reproduces the examples the problem was written
with. Problems are turned into items in the LiveCodeBench format of ~/alter/data, so T2's judge checks them.
"""
import ast
import json
import random
import re

TOPICS = ["strings", "greedy", "sorting", "two pointers", "prefix sums", "binary search on the answer",
          "hashing and counting", "simulation", "number theory", "combinatorics modulo a prime", "one-dimensional DP",
          "two-dimensional DP", "knapsack", "breadth-first search", "connected components", "shortest paths",
          "trees", "union-find", "bit manipulation", "monotonic stacks", "heaps", "intervals", "grids",
          "sliding windows"]
SETTINGS = ["a library", "trains", "a farm", "a robot on a line", "an online shop", "a card game", "a chess club",
            "weather records", "a music playlist", "a city map", "a factory line", "a school timetable",
            "a garden", "a bank queue", "a space station", "a bakery", "a race", "a museum", "a river crossing",
            "a tournament", "a parking lot", "a message log", "a mountain hike", "a puzzle box", "a vending machine"]
DIFFICULTIES = ["easy", "medium", "medium", "hard"]
TESTTYPES = ["stdin", "functional"]
TESTS_DIR = "rive_teach_tests"  # under ~/alter/data, next to LiveCodeBench's own tests

WRITER = (
    "You write original programming contest problems for training data. A problem must be new (not a known contest "
    "problem), precisely stated, with constraints, and solvable in Python within a few seconds at its constraints. "
    "Every example output must be exactly right: work each one out carefully.")
STYLE = {
    "stdin": ("AtCoder style: the program reads standard input and writes standard output. The statement has "
              "Input, Output and Constraints sections. Each example gives the exact input text and the exact "
              "expected output text."),
    "functional": ("LeetCode style: the solver completes a method of class Solution. Give the starter code, e.g. "
                   "'class Solution:\\n    def countPairs(self, nums: List[int], k: int) -> int:\\n        ', "
                   "and for each example the arguments by parameter name as JSON values and the expected return "
                   "value as a JSON value."),
}
FIELDS = {
    "stdin": '{"title": ..., "statement": ..., "examples": [{"input": ..., "output": ...}, x3]}',
    "functional": ('{"title": ..., "statement": ..., "starter_code": ..., "func_name": ..., '
                   '"examples": [{"args": {"name": value, ...}, "output": value}, x3]}'),
}
SOLVER = (
    "You are a strong competitive programmer writing model answers for a smaller model to learn from. Write the "
    "answer the way a strong contestant writes up a solution: first at most five sentences on the key idea and its "
    "time complexity, then the complete solution in a single ```python code block, and nothing after the block.")


def spec(i):
    """The i-th problem's topic, setting, test type and difficulty. Every (topic, setting) pair differs for i < 600,
    so no two requests are the same; test type and difficulty are drawn apart from them (seeded by i)."""
    rng = random.Random(f"teach:{i}")
    return {"index": i, "topic": TOPICS[i % len(TOPICS)], "setting": SETTINGS[(i // len(TOPICS)) % len(SETTINGS)],
            "testtype": rng.choice(TESTTYPES), "difficulty": rng.choice(DIFFICULTIES)}


def writer_prompt(s):
    return (f"Write one new {s['difficulty']} problem (difficulty as on LeetCode / AtCoder ABC) on the topic "
            f"\"{s['topic']}\", with a story about {s['setting']}. {STYLE[s['testtype']]} Do not put the examples "
            "in the statement; give exactly 3 in the examples field. Reply with only a JSON object in a ```json "
            f"block, with these keys: {FIELDS[s['testtype']]}")


def json_block(text):
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    return json.loads(m.group(1) if m else text)


def params(starter_code, func_name):
    """The parameter names of Solution.func_name, in order, from the starter code."""
    tree = ast.parse(starter_code.rstrip() + "\n        pass\n")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [a.arg for a in node.args.args[1:]]
    raise ValueError(f"no method {func_name} in the starter code")


def to_item(s, text):
    """(item, tests) from the writer's reply, in LiveCodeBench's format; ValueError if the reply is unusable."""
    try:
        p = json_block(text)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"not JSON: {e}") from e
    if not isinstance(p, dict) or not all(isinstance(p.get(k), str) and p[k].strip() for k in ("title", "statement")):
        raise ValueError("title or statement missing")
    examples = p.get("examples")
    if not isinstance(examples, list) or len(examples) != 3:
        raise ValueError("not 3 examples")
    qid = f"teach-{s['index']:04d}"
    if s["testtype"] == "stdin":
        def text_of(v):  # a number written as JSON is the same output text
            return str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        if not all(isinstance(e, dict) for e in examples):
            raise ValueError("an example is not an object")
        examples = [{"input": text_of(e.get("input")), "output": text_of(e.get("output"))} for e in examples]
        if not all(isinstance(e["input"], str) and isinstance(e["output"], str) and e["input"].strip()
                   and e["output"].strip() for e in examples):
            raise ValueError("an example lacks input or output text")
        tests = [{"input": e["input"].rstrip("\n") + "\n", "output": e["output"].rstrip("\n") + "\n",
                  "testtype": "stdin"} for e in examples]
        shown = "".join(f"\n\nSample Input {k}\n\n{t['input'].rstrip()}\n\nSample Output {k}\n\n{t['output'].rstrip()}"
                        for k, t in enumerate(tests, 1))
        starter, func = "", ""
    else:
        starter, func = p.get("starter_code"), p.get("func_name")
        if not (isinstance(starter, str) and isinstance(func, str) and func):
            raise ValueError("starter code or method name missing")
        try:
            names = params(starter, func)
        except SyntaxError as e:
            raise ValueError(f"starter code does not parse: {e}") from e
        if not all(isinstance(e, dict) and isinstance(e.get("args"), dict) and list(e["args"]) == names
                   and "output" in e for e in examples):
            raise ValueError(f"example arguments do not match the parameters {names}")
        tests = [{"input": "\n".join(json.dumps(e["args"][n]) for n in names), "output": json.dumps(e["output"]),
                  "testtype": "functional"} for e in examples]
        shown = "".join(f"\n\nExample {k}:\n\nInput: " + ", ".join(f"{n} = {json.dumps(e['args'][n])}" for n in names)
                        + f"\nOutput: {json.dumps(e['output'])}" for k, e in enumerate(examples, 1))
        starter = starter.rstrip() + "\n        "
    item = {"id": qid, "question_id": qid, "title": p["title"].strip(), "content": p["statement"].strip() + shown,
            "starter_code": starter, "func_name": func, "testtype": s["testtype"], "difficulty": s["difficulty"],
            "platform": "synthetic", "contest_date": "", "n_tests": len(tests), "tests_dir": TESTS_DIR}
    return item, tests


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(text, n=8):
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def overlap(a, b):
    """The share of a's 8-grams that are also in b (0 when a has none)."""
    return len(a & b) / len(a) if a else 0.0


def dedup(items, references, against_refs=0.3, among=0.5):
    """(kept, dropped): items whose statement shares at least `against_refs` of its 8-grams with a reference
    statement, or has a reference's title, are dropped; of items sharing `among` with an earlier kept item, only the
    first is kept. references: [(title, statement)]."""
    ref_grams = [ngrams(text) for _, text in references]
    titles = {" ".join(words(t)) for t, _ in references}
    kept, dropped, kept_grams = [], [], []
    for item in items:
        g = ngrams(item["content"])
        if " ".join(words(item["title"])) in titles:
            dropped.append((item["id"], "title of a held-out task"))
        elif max((overlap(g, r) for r in ref_grams), default=0) >= against_refs:
            dropped.append((item["id"], "overlaps a held-out task"))
        elif max((overlap(g, k) for k in kept_grams), default=0) >= among:
            dropped.append((item["id"], "repeats an earlier problem"))
        else:
            kept.append(item)
            kept_grams.append(g)
    return kept, dropped
