"""Leak scan (handoff doc 3.4 "防背题"): candidate code/memory must not contain protected test content.

Protected = selection, anchor and sealed items. Run before every evaluation; hits go into the report.
"""
import re

from rive.benchmark.bfcl import expected_calls

NGRAM = 8
# RTL statements and reference designs are largely code, where 8-word runs are common idioms: the C4 pilot rejected
# an RTL prompt for "always @(posedge clk or posedge reset) begin if (reset)", which one protected statement also
# contains. A copied design or statement still matches many 13-word runs.
NGRAM_CODE = 13
MIN_VALUE_CHARS = 8
# Single-word answer values ("mitochondria", "strategy") are ordinary vocabulary that also shows up in
# dev examples; flagging them caused false positives in the first pilot. Only multi-word values count.
MIN_VALUE_WORDS = 2


def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(words, n=NGRAM):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _answer_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _answer_strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _answer_strings(v)


class LeakScanner:
    def __init__(self, protected_items=(), dev_function_names=()):
        """protected_items: BFCL-style items whose content must never appear in a candidate.
        Function names that also occur in dev items are visible to candidates anyway and are not flagged."""
        self.text_ngrams = {}
        self.code_ngrams = {}
        self.answer_values = {}
        self.function_names = {}
        allowed = set(dev_function_names)
        for item in protected_items:
            request = " ".join(m["content"] for m in item["messages"] if m["role"] == "user")
            for g in _ngrams(_words(request)):
                self.text_ngrams.setdefault(g, item["id"])
            for _, params in expected_calls(item["ground_truth"]):
                for v in _answer_strings(params):
                    if len(v) >= MIN_VALUE_CHARS and len(_words(v)) >= MIN_VALUE_WORDS:
                        self.answer_values.setdefault(v.lower(), item["id"])
            for f in item["functions"]:
                if f["name"] not in allowed:
                    self.function_names.setdefault(f["name"], item["id"])

    def add_hard_tasks(self, protected, visible=()):
        """Phase C maths, code and RTL tasks (rive.benchmark.hard): statements and reference designs, as 8-word
        runs, or 13-word runs for RTL. Runs that also occur in visible (development) tasks are boilerplate, e.g.
        VerilogEval's interface sentence or LiveCodeBench's input-format lines, and are not flagged."""
        def texts(t):
            item = t["item"]
            statement = item.get("problem") or item.get("content") or item.get("prompt") or ""
            return [statement] + ([item["reference"]] if t["domain"] == "rtl" else [])
        shared = {n: {g for t in visible for text in texts(t) for g in _ngrams(_words(text), n)}
                  for n in (NGRAM, NGRAM_CODE)}
        for t in protected:
            n, table = (NGRAM_CODE, self.code_ngrams) if t["domain"] == "rtl" else (NGRAM, self.text_ngrams)
            for text in texts(t):
                for g in _ngrams(_words(text), n) - shared[n]:
                    table.setdefault(g, t["id"])
            answer = t["item"].get("answer")
            if isinstance(answer, str) and len(answer) >= MIN_VALUE_CHARS and len(_words(answer)) >= MIN_VALUE_WORDS:
                self.answer_values.setdefault(answer.lower(), t["id"])
        return self

    def scan(self, files: dict[str, str]):
        """files: {path: text}. Returns {"violations": [...], "warnings": [...]}."""
        violations, warnings = [], []
        for path, text in files.items():
            words = _words(text)
            for n, table in ((NGRAM, self.text_ngrams), (NGRAM_CODE, self.code_ngrams)):
                for g in _ngrams(words, n) & table.keys():
                    violations.append({"file": path, "kind": "request_text", "item": table[g], "match": g})
            lowered = text.lower()
            for v, item in self.answer_values.items():
                if v in lowered:
                    violations.append({"file": path, "kind": "answer_value", "item": item, "match": v[:80]})
            for name, item in self.function_names.items():
                if re.search(rf"\b{re.escape(name)}\b", text):
                    warnings.append({"file": path, "kind": "protected_function_name", "item": item, "match": name})
        return {"violations": violations[:20], "warnings": warnings[:20]}
