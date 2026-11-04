"""The non-programming problems of the reinforcement learning (docs/rive-design.md section 6.25, revised): maths from
the MATH sets on disk, and logic and writing problems made here, each with an exact check.

  - maths: MATH's level-5 training problems, split once at random into evaluation (250), development (74) and
    training (1,978) problems. Judged with math-verify (rive/benchmark/hard.py), which must run in the main thread.
  - logic: knights and knaves, seating orders (both with exactly one solution, checked by brute force) and target
    numbers (any expression that uses the given numbers once and reaches the target).
  - writing: a short text with two or three checkable instructions (length, paragraphs, keywords, no commas, ...),
    judged like IFEval's strict prompt-level accuracy: every instruction must hold.

Problems are made from a string seed: training ("train:..."), development ("dev:...") and evaluation ("eval:...")
seeds never meet, and a training problem whose prompt equals a development or evaluation one is made again.
"""
import ast
import json
import operator
import random
import re
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path

DATA = Path.home() / "alter/data"
NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Henry", "Ivy", "Jack", "Kate", "Leo"]
ANSWER = "Reason step by step, then give the final answer on the last line in the form: "


def _rng(seed):
    return random.Random(seed)


# ---- maths ------------------------------------------------------------------------------------------------------

def _math_task(item, split):
    from rive.benchmark.hard import public_task
    return {"id": f"math-{item['id']}", "domain": "math", "source": "math_train", "split": split,
            "public": public_task("math_train", item), "item": item}


MATH_EVAL, MATH_DEV, MATH_DEV_EASY = 250, 74, 30
# The level-5 subjects the base 4B solved 30-70% of (25 training problems each, two answers): precalculus 60%,
# counting and probability 58%, intermediate algebra 58%, geometry 38%; algebra 94%, prealgebra 88% and number theory
# 80% are left out. All of level 5 together was 76%.
HARD_SUBJECTS = ("counting_and_probability", "geometry", "intermediate_algebra", "precalculus")
MATH500_ALL = Path.home() / (".cache/huggingface/hub/datasets--HuggingFaceH4--MATH-500/snapshots/"
                             "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl")  # the revision in MANIFEST.json


def math_tasks(split):
    """MATH's level-5 training problems of HARD_SUBJECTS (MATH-500's levels 1-5 were 90% solved by the base 4B),
    split at random once: the first MATH_EVAL for evaluation, the next MATH_DEV for development, the rest for
    training: the hard sets. Easy (levels 1-2) and medium (levels 3-4) sets come from MATH-500: "eval_easy" and
    "eval_medium" are those levels of the problems kept in math500.jsonl (93 and 100), "dev_easy" and
    "dev_medium" MATH_DEV_EASY problems of those levels outside it."""
    if split.endswith(("_easy", "_medium")):
        levels = (1, 2) if split.endswith("_easy") else (3, 4)
        kept = [json.loads(line) for line in open(DATA / "math500.jsonl")]
        if split.startswith("eval"):
            return [_math_task(i, split) for i in kept if i["level"] in levels]
        held = {i["id"] for i in kept}
        rest = [json.loads(line) for line in open(MATH500_ALL)]
        rest = sorted(({"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"], "level": r["level"],
                        "subject": r["subject"]} for r in rest if r["unique_id"] not in held and r["level"] in levels),
                      key=lambda r: r["id"])
        _rng(f"rl:math-{split}").shuffle(rest)
        return [_math_task(i, split) for i in rest[:MATH_DEV_EASY]]
    items = sorted((json.loads(line) for line in open(DATA / "math_train.jsonl")), key=lambda i: i["id"])
    items = [i for i in items if i["level"] == 5 and i["subject"] in HARD_SUBJECTS]
    _rng("rl:math-split").shuffle(items)
    part = {"eval": items[:MATH_EVAL], "dev": items[MATH_EVAL:MATH_EVAL + MATH_DEV],
            "train": items[MATH_EVAL + MATH_DEV:]}[split]
    return [_math_task(i, split) for i in part]


def judge_math(task, text):
    from rive.benchmark.hard import judge
    ok, failure, _ = judge({"item": task["item"], "source": task["source"], "domain": "math"}, text)
    return ok, failure


# ---- logic: knights and knaves ----------------------------------------------------------------------------------

def _kind(knight):
    return "knight" if knight else "knave"


def _statements(names):
    """(text, truth function of an assignment) for the statements a person can make about the others."""
    def is_(j, t):
        return f"{names[j]} is a {_kind(t)}.", lambda a: a[j] == t

    def both(j, k, t):
        return f"{names[j]} and {names[k]} are both {_kind(t)}s.", lambda a: a[j] == t and a[k] == t

    def either(j, k, t):
        return f"At least one of {names[j]} and {names[k]} is a {_kind(t)}.", lambda a: a[j] == t or a[k] == t

    def same(j, k):
        return f"{names[j]} and {names[k]} are of the same kind.", lambda a: a[j] == a[k]

    def count(c):
        text = "Exactly 1 of us is a knight." if c == 1 else f"Exactly {c} of us are knights."
        return text, lambda a: sum(a) == c

    def implies(j, k, t):
        return (f"If {names[j]} is a knight, then {names[k]} is a {_kind(t)}.",
                lambda a: (not a[j]) or a[k] == t)

    def iff(j, k):
        return f"{names[j]} is a knight if and only if {names[k]} is a knave.", lambda a: a[j] != a[k]

    def of_three(j, k, m, c):
        word = "is a knight" if c == 1 else "are knights"
        return (f"Exactly {c} of {names[j]}, {names[k]} and {names[m]} {word}.",
                lambda a: a[j] + a[k] + a[m] == c)
    return is_, both, either, same, count, implies, iff, of_three


def knights(seed, level="hard"):
    """Easy: 3-5 people, simple statements (the base 4B solved all on the dev set). Medium: 5-7 people (92%).
    Hard: 6-8 people, conditional and counting statements too (76%)."""
    rng = _rng(seed)
    simple = ["is", "both", "either", "same", "count"]
    kinds = simple if level != "hard" else simple + ["implies", "iff", "of_three", "implies", "of_three"]
    for _ in range(500):
        n = rng.choice({"easy": [3, 4, 5], "medium": [5, 6, 7], "hard": [6, 7, 8]}[level])
        names = rng.sample(NAMES, n)
        truth = [rng.random() < 0.5 for _ in range(n)]
        is_, both, either, same, count, implies, iff, of_three = _statements(names)
        said = []
        for i in range(n):
            others = [j for j in range(n) if j != i]
            for _ in range(100):
                kind = rng.choice(kinds)
                j, k = rng.sample(others, 2)
                m = rng.choice([x for x in others if x not in (j, k)] or [j])
                if kind == "is":
                    s = is_(j, rng.random() < 0.5)
                elif kind == "count":
                    s = count(rng.randint(0, n))
                elif kind == "implies":
                    s = implies(j, k, rng.random() < 0.5)
                elif kind == "iff":
                    s = iff(j, k)
                elif kind == "of_three":
                    s = of_three(j, k, m, rng.randint(0, 3))
                else:
                    s = (same(j, k) if kind == "same" else
                         (both if kind == "both" else either)(j, k, rng.random() < 0.5))
                if s[1](truth) == truth[i]:
                    said.append(s)
                    break
            else:
                break
        if len(said) < n:
            continue
        fits = [a for a in product([True, False], repeat=n) if all(said[i][1](a) == a[i] for i in range(n))]
        if len(fits) == 1:
            lines = "\n".join(f"{names[i]} says: \"{said[i][0]}\"" for i in range(n))
            form = ", ".join(f"{x}=<kind>" for x in names)
            prompt = ("On an island, every inhabitant is either a knight, who always tells the truth, or a knave, who "
                      f"always lies. You meet {n} inhabitants: {', '.join(names)}.\n{lines}\n"
                      f"Who is a knight and who is a knave? {ANSWER}ANSWER: {form}, where each <kind> is knight or "
                      "knave.")
            return {"family": "knights", "prompt": prompt, "names": names, "solution": [_kind(t) for t in truth]}
    raise RuntimeError(f"no knights puzzle for {seed}")


def judge_knights(task, text):
    line = _answer_line(text)
    if line is None:
        return False, "no_final_answer"
    got = dict((m.group(1).lower(), m.group(2).lower()) for m in
               re.finditer(r"([A-Za-z]+)\s*(?:=|:|is an?)\s*(knight|knave)", line))
    want = {n.lower(): k for n, k in zip(task["names"], task["solution"])}
    return got == want, None if got == want else "wrong_answer"


# ---- logic: seating orders --------------------------------------------------------------------------------------

def seating(seed, level="hard"):
    """Easy: 4-5 people, at most one clue gives a seat away (92% solved by the base 4B on the dev set). Medium:
    5-6 people, no clue gives a seat away (62%). Hard: 6-7 people (46%)."""
    rng = _rng(seed)
    kinds = ["left", "left", "next_right", "adjacent", "not_adjacent", "not_end", "between"]
    for _ in range(500):
        n = rng.choice({"easy": [4, 5], "medium": [5, 6], "hard": [6, 7]}[level])
        names = rng.sample(NAMES, n)
        order = names[:]
        rng.shuffle(order)
        pos = {p: i + 1 for i, p in enumerate(order)}
        seat_given = level == "easy" and rng.random() < 0.5

        def clue():
            a, b, c = rng.sample(names, 3)
            kind = rng.choice(kinds)
            if kind == "left" and pos[a] < pos[b]:
                return f"{a} sits somewhere to the left of {b}.", lambda q: q[a] < q[b]
            if kind == "next_right" and pos[a] == pos[b] + 1:
                return f"{a} sits immediately to the right of {b}.", lambda q: q[a] == q[b] + 1
            if kind == "adjacent" and abs(pos[a] - pos[b]) == 1:
                return f"{a} and {b} sit next to each other.", lambda q: abs(q[a] - q[b]) == 1
            if kind == "not_adjacent" and abs(pos[a] - pos[b]) > 1:
                return f"{a} and {b} do not sit next to each other.", lambda q: abs(q[a] - q[b]) > 1
            if kind == "not_end" and 1 < pos[a] < n:
                return f"{a} does not sit at either end of the row.", lambda q: 1 < q[a] < n
            if kind == "between" and min(pos[b], pos[c]) < pos[a] < max(pos[b], pos[c]):
                return (f"{a} sits somewhere between {b} and {c}.",
                        lambda q: min(q[b], q[c]) < q[a] < max(q[b], q[c]))
            return None  # no clue gives a seat away

        seatings = [{p: i + 1 for i, p in enumerate(perm)} for perm in permutations(names)]
        clues, left = [], seatings
        if seat_given:
            a = rng.choice(names)
            clues = [(f"{a} sits in seat {pos[a]}.", lambda q, a=a: q[a] == pos[a])]
            left = [q for q in seatings if clues[0][1](q)]
        for _ in range(300):  # add true clues until one seating is left
            c = clue()
            if c is None or c[0] in {t for t, _ in clues}:
                continue
            narrower = [q for q in left if c[1](q)]
            if len(narrower) < len(left):
                clues, left = clues + [c], narrower
            if len(left) == 1:
                break
        if len(left) != 1:
            continue
        for c in list(clues):  # drop clues the answer does not need, so that each one matters
            rest = [x for x in clues if x is not c]
            if sum(all(f(q) for _, f in rest) for q in seatings) == 1:
                clues = rest
        rng.shuffle(clues)
        lines = "\n".join(f"- {t}" for t, _ in clues)
        prompt = (f"{n} people, {', '.join(sorted(names))}, sit in a row of {n} seats numbered 1 to {n} from left to "
                  f"right.\n{lines}\nWho sits in each seat? {ANSWER}ANSWER: <the names from seat 1 to seat {n}, "
                  "separated by commas>")
        return {"family": "seating", "prompt": prompt, "names": names, "solution": order}
    raise RuntimeError(f"no seating puzzle for {seed}")


def judge_seating(task, text):
    line = _answer_line(text)
    if line is None:
        return False, "no_final_answer"
    known = {n.lower(): n for n in task["names"]}
    got = [known[w.lower()] for w in re.findall(r"[A-Za-z]+", line) if w.lower() in known]
    return got == task["solution"], None if got == task["solution"] else "wrong_answer"


# ---- logic: target numbers --------------------------------------------------------------------------------------

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def target(seed, level="hard"):
    """Easy: 4-5 numbers from 1-13 to a target of 10-300 (94% solved by the base 4B on the dev set). Medium: 5-6
    numbers from 1-20 (80%). Hard: 6 numbers from 1-30 to 100-999 (44%)."""
    rng = _rng(seed)
    low, high = (100, 999) if level == "hard" else (10, 300)
    for _ in range(500):
        k = rng.choice({"easy": [4, 5], "medium": [5, 6], "hard": [6]}[level])
        nums = [rng.randint(1, {"easy": 13, "medium": 20, "hard": 30}[level]) for _ in range(k)]
        values = [(Fraction(x), str(x)) for x in nums]
        while len(values) > 1:  # combine two at random until one value is left
            (a, ea), (b, eb) = values.pop(rng.randrange(len(values))), values.pop(rng.randrange(len(values)))
            op = rng.choice("+-*/")
            if op == "-" and a < b:
                a, b, ea, eb = b, a, eb, ea
            if op == "/" and (b == 0 or (a / b).denominator != 1):
                op = "+"
            v = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else a + b}[op]
            values.append((v, f"({ea} {op} {eb})"))
        value = values[0][0]
        if value.denominator == 1 and low <= value <= high and int(value) not in nums:
            shown = sorted(nums)
            prompt = (f"Use each of the numbers {', '.join(map(str, shown))} exactly once, with +, -, *, / and "
                      f"parentheses, to make {int(value)}. {ANSWER}ANSWER: <an expression, for example (1 + 2) * 3>")
            return {"family": "target", "prompt": prompt, "numbers": shown, "target": int(value),
                    "expression": values[0][1]}  # one solution, for the tests; not shown to the model
    raise RuntimeError(f"no target puzzle for {seed}")


def _value(node):
    if isinstance(node, ast.Expression):
        return _value(node.body)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left, right = _value(node.left), _value(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError
        return OPS[type(node.op)](left, right)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return Fraction(node.value)
    raise ValueError("not an arithmetic expression")


def judge_target(task, text):
    line = _answer_line(text)
    if line is None:
        return False, "no_final_answer"
    line = line.replace("×", "*").replace("÷", "/").replace("−", "-").replace("\\times", "*").replace("\\div", "/")
    ok = any(_reaches(side.strip().strip("$`.").strip(), task) for side in line.split("="))  # "50 = (3+7)*5" too
    return ok, None if ok else "wrong_answer"


def _reaches(expr, task):
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return False
    try:
        return (_value(ast.parse(expr, mode="eval")) == task["target"]
                and sorted(int(x) for x in re.findall(r"\d+", expr)) == task["numbers"])
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return False


def _answer_line(text):
    hits = re.findall(r"ANSWER\s*:\s*(.+)", text.replace("**", ""), re.I)
    return hits[-1].strip() if hits else None


LOGIC = {"knights": (knights, judge_knights), "seating": (seating, judge_seating), "target": (target, judge_target)}


def logic_task(seed, family, level="hard"):
    t = LOGIC[family][0](seed, level)
    return {"id": f"logic-{seed}", "domain": "logic", "level": level, **t, "public": {"prompt": t["prompt"]}}


# ---- writing ----------------------------------------------------------------------------------------------------

KINDS = ["a short story", "a blog post", "an email to a colleague", "a product description", "a short essay",
         "a letter to a friend", "a news report", "a speech for a school assembly", "a diary entry", "a travel guide"]
TOPICS = ["a rainy day in a small town", "learning to cook", "a lost umbrella", "the first day at a new job",
          "a trip to the mountains", "a community garden", "an old bookshop", "a broken bicycle", "a winter festival",
          "a robot that learns to paint", "saving water at home", "a visit to a museum", "a night train",
          "a new coffee machine", "the history of tea", "a power cut", "a lighthouse keeper", "a school science fair",
          "moving to a new city", "a forgotten birthday", "a chess tournament", "the sound of the sea",
          "a neighbour's dog", "a summer market", "an unexpected letter", "a snowstorm", "a street musician",
          "a solar-powered car", "a lost key", "a long-distance friendship"]
KEYWORDS = ["lantern", "harbor", "velvet", "compass", "orchard", "thunder", "marble", "whisper", "glacier", "copper",
            "meadow", "puzzle", "silver", "candle", "ribbon", "feather"]
AVOID = ["very", "really", "just", "thing", "good", "nice"]
ENDINGS = ["Thank you for reading.", "That is all for now.", "Until next time.", "And that is how it ended."]
OPENERS = ["Today", "Yesterday", "Imagine", "Once", "Honestly"]


def _words(text):
    return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text)


def _paragraphs(text):
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _count(text, word):
    return len(re.findall(rf"\b{re.escape(word)}\b", text, re.I))


def _constraint(name, rng, used, level="hard"):
    """(instruction, check) for one constraint; `used` keeps keywords apart."""
    if name == "length":
        low = rng.choice([60, 90, 120, 150])
        high = low + rng.choice([25, 35, 45] if level == "hard" else [40, 60, 80])
        return f"Your response must be between {low} and {high} words long.", lambda t: low <= len(_words(t)) <= high
    if name == "capitals":
        n = rng.choice([3, 4, 5])
        return (f"Write at least {n} words in ALL CAPITAL LETTERS.",
                lambda t: sum(len(w) > 1 and w.isupper() for w in _words(t)) >= n)
    if name == "letter":
        ch, n = rng.choice("zqxjk"), rng.choice([4, 6, 8])
        return f"The letter \"{ch}\" must appear at least {n} times in your response.", \
            lambda t: t.lower().count(ch) >= n
    if name == "paragraphs":
        n = rng.choice([2, 3, 4])
        return (f"Write exactly {n} paragraphs, separated from each other by one blank line.",
                lambda t: len(_paragraphs(t)) == n)
    if name == "keywords":
        a, b = rng.sample([w for w in KEYWORDS if w not in used], 2)
        used.update({a, b})
        return f"Include the words \"{a}\" and \"{b}\".", lambda t: _count(t, a) > 0 and _count(t, b) > 0
    if name == "keyword_count":
        w = rng.choice([w for w in KEYWORDS if w not in used])
        used.add(w)
        k = rng.choice([2, 3])
        return f"Use the word \"{w}\" exactly {k} times.", lambda t: _count(t, w) == k
    if name == "avoid":
        w = rng.choice(AVOID)
        return f"Do not use the word \"{w}\" anywhere.", lambda t: _count(t, w) == 0
    if name == "lowercase":
        return ("Write your entire response in lowercase letters; no capital letters at all.",
                lambda t: not re.search(r"[A-Z]", t))
    if name == "no_commas":
        return "Do not use any commas in your response.", lambda t: "," not in t
    if name == "ending":
        s = rng.choice(ENDINGS)
        return (f"End your response with this exact sentence: \"{s}\"",
                lambda t: t.strip().strip("\"'").endswith(s))
    if name == "opener":
        w = rng.choice(OPENERS)
        return f"Begin your response with the word \"{w}\".", lambda t: _words(t)[:1] == [w]
    if name == "title":
        return ("Give your response a title wrapped in double angular brackets, such as <<my title>>.",
                lambda t: bool(re.search(r"<<[^<>\n]+>>", t)))
    if name == "bullets":
        n = rng.choice([3, 4, 5])
        return (f"Include exactly {n} bullet points, each on its own line starting with \"- \".",
                lambda t: len(re.findall(r"^\s*- ", t, re.M)) == n)
    if name == "postscript":
        return ("At the end of your response, add a postscript that starts with \"P.S.\".",
                lambda t: "P.S." in _paragraphs(t)[-1] if _paragraphs(t) else False)
    raise ValueError(name)


CONSTRAINTS = ["length", "paragraphs", "keywords", "keyword_count", "avoid", "lowercase", "no_commas", "ending",
               "opener", "title", "bullets", "postscript", "capitals", "letter"]
CLASH = {frozenset(p) for p in [("lowercase", "ending"), ("lowercase", "opener"), ("bullets", "paragraphs"),
                                 ("ending", "postscript"), ("lowercase", "postscript"), ("lowercase", "title"),
                                 ("opener", "title"), ("postscript", "paragraphs"), ("lowercase", "capitals")]}


def writing_task(seed, level="hard"):
    """Easy: 1-2 instructions, wide word ranges, no capitals or letter instructions. Medium: 2-3 such
    instructions (80% followed by the base 4B on the dev set). Hard: 3-4 instructions of all kinds, narrow word
    ranges (54%)."""
    rng = _rng(seed)
    pool = [c for c in CONSTRAINTS if level == "hard" or c not in ("capitals", "letter")]
    for _ in range(100):
        names = rng.sample(pool, rng.choice({"easy": [1, 2], "medium": [2, 3], "hard": [3, 4]}[level]))
        if not any(frozenset((a, b)) in CLASH for a in names for b in names if a != b):
            break
    used = set()
    parts = [_constraint(n, rng, used, level) for n in names]
    prompt = (f"Write {rng.choice(KINDS)} about {rng.choice(TOPICS)}.\n" + "\n".join(f"- {p[0]}" for p in parts)
              + "\nAnswer with the text only.")
    return {"id": f"writing-{seed}", "domain": "writing", "level": level, "family": "+".join(names),
            "prompt": prompt, "public": {"prompt": prompt}, "_checks": [p[1] for p in parts], "constraints": names,
            "_seed": seed}


def judge_writing(task, text):
    checks = task.get("_checks") or writing_task(task["_seed"], task.get("level", "hard"))["_checks"]
    failed = [n for n, c in zip(task["constraints"], checks) if not c(text)]
    return not failed, None if not failed else "failed:" + ",".join(failed)


# ---- sets -------------------------------------------------------------------------------------------------------

FAMILIES = ("knights", "seating", "target")


def fixed_set(domain, split, n, level="hard"):
    """The development or evaluation problems of a generated domain, the same every time. The easy sets guard
    against losing what the base model could already do; training uses hard problems only."""
    tag = split if level == "hard" else f"{split}-{level}"
    if domain == "logic":
        return [logic_task(f"{tag}:logic:{i}", FAMILIES[i % 3], level) for i in range(n)]
    return [writing_task(f"{tag}:writing:{i}", level) for i in range(n)]


def training_task(domain, it, k, banned):
    """Problem k of iteration `it`, never one whose prompt is in `banned` (the development and evaluation prompts)."""
    for attempt in range(20):
        seed = f"train:{domain}:{it}:{k}:{attempt}"
        t = logic_task(seed, FAMILIES[(it + k) % 3]) if domain == "logic" else writing_task(seed)
        if t["prompt"] not in banned:
            return t
    raise RuntimeError("no fresh training problem")


def judge(task, text):
    """(correct, failure) for a problem of any non-programming domain."""
    if task["domain"] == "math":
        return judge_math(task, text)
    if task["domain"] == "logic":
        return LOGIC[task["family"]][1](task, text)
    return judge_writing(task, text)
