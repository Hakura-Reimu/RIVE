"""Tool-pool tasks for Phase A: a BFCL item plus K distractor tools, so choosing tools is hard.

Distractors only come from dev-split items, so no task (of any split) shows the model, or a
candidate reading dev failures, a function that belongs to a selection/anchor/sealed item.
The seed is (item, round, K): every candidate in a round sees the same task ("同代同题").
"""
import hashlib
import random

from rive.benchmark.bfcl import expected_calls
from rive.benchmark.splits import split_of


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def distractor_pool(items):
    """Unique functions (by name) from dev-split items, each tagged with the item it came from."""
    pool, seen = [], set()
    for item in sorted(items, key=lambda i: i["id"]):
        if split_of(item["id"]) != "dev":
            continue
        for f in item["functions"]:
            if f["name"] not in seen:
                seen.add(f["name"])
                pool.append((item["id"], f))
    return pool


def build_task(item, k, round_id, pool):
    own = {f["name"] for f in item["functions"]}
    own |= {name for name, _ in expected_calls(item["ground_truth"])}
    candidates = [f for source, f in pool if source != item["id"] and f["name"] not in own]
    rng = random.Random(_seed(item["id"], round_id, k))
    distractors = rng.sample(candidates, min(k, len(candidates)))
    functions = list(item["functions"]) + distractors
    rng.shuffle(functions)
    return {"id": f"{item['id']}@k{k}", "base_id": item["id"], "category": item["category"],
            "split": split_of(item["id"]), "k": k, "round": round_id, "messages": item["messages"],
            "functions": functions, "ground_truth": item["ground_truth"],
            "irrelevance": item["category"] == "irrelevance"}


def category_of(task_id):
    """BFCL category of a task id such as "parallel_multiple_12@k32"."""
    return task_id.split("@")[0].rsplit("_", 1)[0]


def build_tasks(items, k, round_id, splits=("dev",)):
    pool = distractor_pool(items)
    return [build_task(i, k, round_id, pool) for i in items if split_of(i["id"]) in splits]
