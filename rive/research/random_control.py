"""Control group for Phase A (handoff doc 3.7.8): router changes drawn at random from fixed templates,
with no research loop. Same evaluation, same promotion rule, same number of attempts as the loop."""
import random

TEMPLATES = {
    "keyword_topk": '''import re

def _words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))

def select(task, tools, history=None):
    q = _words(task)
    def score(t):
        return len(q & _words(t["name"].replace("_", " ") + " " + t.get("description", "")))
    ranked = sorted(tools, key=score, reverse=True)
    return {{"tools": [t["name"] for t in ranked[:{k}]], "hint": {hint!r}}}
''',
    "jaccard_topk": '''import re

def _words(s):
    return set(re.findall(r"[a-z]+", s.lower()))

def select(task, tools, history=None):
    q = _words(task)
    def score(t):
        d = _words(t["name"].replace("_", " ") + " " + t.get("description", ""))
        return len(q & d) / (len(q | d) or 1)
    ranked = sorted(tools, key=score, reverse=True)
    return {{"tools": [t["name"] for t in ranked[:{k}]], "hint": {hint!r}}}
''',
    "first_k": '''def select(task, tools, history=None):
    return {{"tools": [t["name"] for t in tools[:{k}]], "hint": {hint!r}}}
''',
    "shuffle_all": '''import random

def select(task, tools, history=None):
    names = [t["name"] for t in tools]
    random.Random(len(task)).shuffle(names)
    return {{"tools": names, "hint": {hint!r}}}
''',
    "threshold": '''import re

def _words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))

def select(task, tools, history=None):
    q = _words(task)
    keep = [t["name"] for t in tools
            if len(q & _words(t["name"].replace("_", " ") + " " + t.get("description", ""))) >= {m}]
    return {{"tools": keep, "hint": {hint!r}}}
''',
}

HINTS = [
    "",
    "Only call a tool if it clearly matches the user's request; otherwise answer directly.",
    "Read every tool description before choosing. Call each required tool once per requested item.",
    "If the request contains several independent requests, make one tool call for each of them.",
]


def random_router(rng: random.Random):
    """Returns (template name, params, source)."""
    name = rng.choice(sorted(TEMPLATES))
    params = {"k": rng.choice([3, 5, 8, 12]), "m": rng.choice([1, 2, 3]), "hint": rng.choice(HINTS)}
    return name, params, TEMPLATES[name].format(**params)


RULE_BASELINE = TEMPLATES["keyword_topk"].format(k=8, hint="")
