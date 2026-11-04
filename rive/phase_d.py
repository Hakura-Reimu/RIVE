"""Phase D (docs/phase-d-design.md): rejection-sampling fine-tuning data and the pass criterion, as plain functions
the Phase D scripts share and tests/test_phase_d.py checks."""
import random
import re
import statistics
from collections import Counter, defaultdict

NGRAM = 8


def last_boxed(text):
    """The content of the last \\boxed{...} or \\fbox{...} in `text`, braces matched; None when there is none."""
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return None
    depth = 0
    for j in range(text.index("{", start), len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[text.index("{", start) + 1:j]
    return None


def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(words, n=NGRAM):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


class Overlap:
    """Which protected item a training statement repeats. A statement repeats an item when at least `share` of its
    8-word runs occur in that one item, or, for statements too short to have one, when the words are the same.
    Boilerplate that many statements share (LiveCodeBench's input-format lines) stays well below that share."""

    def __init__(self, protected, share=0.5):
        """protected: {item id: statement}."""
        self.share, self.index, self.exact = share, defaultdict(set), {}
        for pid, text in protected.items():
            words = _words(text)
            self.exact.setdefault(" ".join(words), pid)
            for g in _ngrams(words):
                self.index[g].add(pid)

    def match(self, text):
        """The id of the protected item `text` repeats, or None."""
        words = _words(text)
        if " ".join(words) in self.exact:
            return self.exact[" ".join(words)]
        grams = _ngrams(words)
        counts = Counter(pid for g in grams for pid in self.index.get(g, ()))
        if not counts:
            return None
        pid, n = counts.most_common(1)[0]
        return pid if n >= self.share * len(grams) else None


def pick_examples(samples, per_task=2):
    """Training answers from one task's samples: correct and complete (not cut off by the token limit), without
    repeats, shortest first: answers that finish within the budget are what the training is meant to teach."""
    out, seen = [], set()
    for s in sorted((s for s in samples if s["correct"] and s["finish_reason"] != "length"),
                    key=lambda s: s["completion_tokens"]):
        key = " ".join(s["text"].split())
        if key not in seen:
            seen.add(key)
            out.append(s)
        if len(out) == per_task:
            break
    return out


def macro(results):
    """Accuracy averaged over the domains present, each domain weighted equally."""
    by = defaultdict(list)
    for r in results:
        by[r["domain"]].append(r["correct"])
    return statistics.mean(statistics.mean(v) for v in by.values())


def accuracy(results):
    return statistics.mean(r["correct"] for r in results)


def bootstrap_p(base_runs, new_runs, n=2000, seed=0):
    """One-sided p that `new_runs` is no better than `base_runs` in macro accuracy: items are resampled within each
    domain, each item scored by its mean over the runs of each side (all runs cover the same items)."""
    def per_item(runs):
        acc = defaultdict(list)
        for run in runs:
            for r in run:
                acc[(r["domain"], r["id"])].append(r["correct"])
        return {k: statistics.mean(v) for k, v in acc.items()}
    a, b = per_item(base_runs), per_item(new_runs)
    by = defaultdict(list)
    for key in a:
        by[key[0]].append(b[key] - a[key])
    rng, worse = random.Random(seed), 0
    for _ in range(n):
        diff = statistics.mean(statistics.mean(rng.choices(d, k=len(d))) for d in by.values())
        worse += diff <= 0
    return (worse + 1) / (n + 1)


def criterion(v0_runs, v1_runs, v0_easy, v1_easy, gain=0.02, easy_drop=0.01):
    """The Phase D pass criterion (phase0/REPORT.md section 9, chosen by the user on 2026-10-28): on held-out hard
    tasks a mean gain of at least `gain` in macro accuracy over the seeds, every seed's gain positive, and no seed
    more than `easy_drop` below V0 on the easy sets. v1_runs and v1_easy are keyed by adapter (seed)."""
    base = statistics.mean(macro(r) for r in v0_runs)
    gains = {k: statistics.mean(macro(r) for r in runs) - base for k, runs in v1_runs.items()}
    easy = {k: accuracy(v1_easy[k]) - accuracy(v0_easy) for k in v1_easy}
    passed = (statistics.mean(gains.values()) >= gain and all(g > 0 for g in gains.values())
              and all(e >= -easy_drop for e in easy.values()))
    return {"v0_macro": base, "gains": gains, "mean_gain": statistics.mean(gains.values()),
            "easy_v0": accuracy(v0_easy), "easy_changes": easy,
            "p": {k: bootstrap_p(v0_runs, runs) for k, runs in v1_runs.items()}, "passed": passed}
