"""Statistics for comparing candidates on the same items."""
import math
import random


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap(parent, child, iterations=10_000, seed=0):
    """One-sided test that child beats parent on the same items (lists of 0/1 in the same order).

    Returns (mean difference, p-value = share of resamples where the difference is <= 0).
    """
    if len(parent) != len(child) or not parent:
        raise ValueError("need two non-empty result lists over the same items")
    diffs = [c - p for p, c in zip(parent, child)]
    n = len(diffs)
    observed = sum(diffs) / n
    rng = random.Random(seed)
    not_better = 0
    for _ in range(iterations):
        if sum(diffs[rng.randrange(n)] for _ in range(n)) <= 0:
            not_better += 1
    return observed, not_better / iterations


def paired_bootstrap_macro(parent, child, domains, iterations=10_000, seed=0):
    """paired_bootstrap for the domain-balanced mean (the Phase C primary metric, docs/phase-c-design.md
    section 2): items are resampled within their domain and every domain weighs the same."""
    if not (len(parent) == len(child) == len(domains)) or not parent:
        raise ValueError("need non-empty result and domain lists over the same items")
    groups = {}
    for p, c, d in zip(parent, child, domains):
        groups.setdefault(d, []).append(c - p)
    groups = list(groups.values())
    observed = sum(sum(g) / len(g) for g in groups) / len(groups)
    rng = random.Random(seed)
    not_better = 0
    for _ in range(iterations):
        if sum(sum(g[rng.randrange(len(g))] for _ in g) / len(g) for g in groups) <= 0:
            not_better += 1
    return observed, not_better / iterations
