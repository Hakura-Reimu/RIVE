from collections import Counter

from rive.benchmark.splits import split_of
from rive.benchmark.stats import paired_bootstrap, wilson
from rive.benchmark.toolpool import build_tasks, distractor_pool


def test_wilson_bounds():
    lo, hi = wilson(95, 100)
    assert 0.88 < lo < 0.90 and 0.97 < hi < 0.99
    assert wilson(0, 0) == (0.0, 0.0)


def test_paired_bootstrap_detects_clear_gain_and_not_noise():
    parent = [0] * 50 + [1] * 50
    child = [1] * 70 + [0] * 30  # 50 items improve, 30 regress: +20 points, one-sided p about 0.01
    diff, p = paired_bootstrap(parent, child, iterations=2000)
    assert abs(diff - 0.2) < 1e-9 and p < 0.05
    same_diff, same_p = paired_bootstrap(parent, parent, iterations=2000)
    assert same_diff == 0 and same_p == 1.0


def test_splits_are_stable_and_roughly_proportional():
    ids = [f"item_{i}" for i in range(5000)]
    assert [split_of(i) for i in ids[:20]] == [split_of(i) for i in ids[:20]]
    share = Counter(split_of(i) for i in ids)
    assert abs(share["dev"] / 5000 - 0.40) < 0.03
    assert abs(share["selection"] / 5000 - 0.30) < 0.03


def _items(n):
    return [{"id": f"multiple_{i}", "category": "multiple", "messages": [{"role": "user", "content": f"q{i}"}],
             "functions": [{"name": f"fn_{i}_{j}", "parameters": {"type": "dict", "properties": {}}}
                           for j in range(3)],
             "ground_truth": [{f"fn_{i}_0": {}}]} for i in range(n)]


def test_toolpool_distractors_only_from_dev_and_never_own():
    items = _items(200)
    pool_sources = {src for src, _ in distractor_pool(items)}
    assert all(split_of(s) == "dev" for s in pool_sources)
    tasks = build_tasks(items, k=16, round_id=1, splits=("dev", "selection"))
    for t in tasks:
        names = [f["name"] for f in t["functions"]]
        assert len(names) == len(set(names)) == 3 + 16
        own_index = t["base_id"].split("_")[1]
        distractor_sources = {n.split("_")[1] for n in names} - {own_index}
        assert all(split_of(f"multiple_{s}") == "dev" for s in distractor_sources)


def test_toolpool_same_round_same_task_new_round_new_draw():
    items = _items(200)
    a = build_tasks(items, k=8, round_id=1)
    b = build_tasks(items, k=8, round_id=1)
    c = build_tasks(items, k=8, round_id=2)
    assert a == b
    assert [t["functions"] for t in a] != [t["functions"] for t in c]
