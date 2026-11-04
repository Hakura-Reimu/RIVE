"""Run a candidate's workflow on hard-set tasks (Phase C, docs/phase-c-design.md section 3). The workflow runs in
the candidate's sandbox; its model calls are served and metered here; the trusted judge scores the answers."""
import queue
import time
from concurrent.futures import ThreadPoolExecutor

import openai

from rive.benchmark.hard import judge
from rive.sandbox.container import CandidateSandbox

TOKEN_BUDGET = 8192     # completion tokens per task over all model calls (twice the Phase 0 single answer)
MAX_CALL_TOKENS = 4096  # Phase 0's non-thinking cap for one answer
SECONDS = 600


def strip_thinking(text):
    """Only what follows </think> is the answer; a trace cut off before </think> has none (as in Phase 0)."""
    return text.split("</think>")[-1] if "</think>" in text else ""


def run_workflow_tasks(tasks, candidate_dir, llm, workers=8, token_budget=TOKEN_BUDGET, seconds=SECONDS,
                       judge_fn=judge, judge_workers=6, progress=None, seed_base=0, max_call_tokens=MAX_CALL_TOKENS):
    """Results in task order: {"id", "domain", "correct", "failure", "tokens", "prompt_tokens", "llm_calls",
    "truncated_calls", "server_errors", "tool_calls", "seconds", "detail", "answer" (its last 1500 characters)}.
    A failed workflow counts as "workflow_error". server_errors: model calls the server failed (not reached, timed
    out, internal error), which say nothing about the agent. progress() is called as each task finishes. The calls
    of a task use seeds seed_base, seed_base + 1, ...: runs with different seed_base sample independently.
    max_call_tokens caps one call (T2 lets one call use the whole budget)."""
    boxes = queue.Queue()
    for _ in range(workers):
        boxes.put(CandidateSandbox(candidate_dir, memory="768m", request_timeout=10))

    def run_one(task):
        meter = {"tokens": 0, "prompt_tokens": 0, "calls": 0, "truncated": 0, "requests": 0, "server_errors": 0}

        def on_llm(req):
            left = token_budget - meter["tokens"]
            if left <= 0:
                return {"ok": False, "error": f"token budget of {token_budget} used up"}
            # Each call of a task samples with its own seed. With one seed for all, the server gave the same sample
            # for the same messages, so several answers at a temperature above 0 were copies of one another.
            seed = seed_base + meter["requests"]
            meter["requests"] += 1
            try:
                r = llm.generate(req["messages"], max_tokens=max(1, min(req["max_tokens"], max_call_tokens, left)),
                                 temperature=req["temperature"], thinking=req["thinking"], seed=seed)
            except (openai.APIConnectionError, openai.InternalServerError) as e:
                meter["server_errors"] += 1
                return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
            except Exception as e:  # noqa: BLE001 - the workflow sees a failed call; the run goes on
                return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
            meter["tokens"] += r["completion_tokens"]
            meter["prompt_tokens"] += r["prompt_tokens"]
            meter["calls"] += 1
            meter["truncated"] += r["finish_reason"] == "length"
            return {"ok": True, "text": strip_thinking(r["text"]) if req["thinking"] else r["text"]}

        box = boxes.get()
        start = time.perf_counter()
        try:
            done = box.solve(task["public"], seconds, on_llm)
        finally:
            boxes.put(box)
        if progress:
            progress()
        return done, meter, time.perf_counter() - start

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(run_one, tasks))
    finally:
        while not boxes.empty():
            boxes.get().close()

    def score(pair):
        task, (done, _, _) = pair
        if not done.get("ok"):
            return False, "workflow_error", {"error": done.get("error", "")[:300]}
        return judge_fn(task, done["answer"])

    pairs = list(zip(tasks, runs))
    verdicts = [None] * len(pairs)
    parallel = [i for i, (t, _) in enumerate(pairs) if t["domain"] != "math"]
    with ThreadPoolExecutor(max_workers=judge_workers) as pool:
        for i, v in zip(parallel, pool.map(score, [pairs[i] for i in parallel])):
            verdicts[i] = v
    for i, (t, _) in enumerate(pairs):
        if t["domain"] == "math":  # math-verify uses signal.alarm, so it stays on this thread
            verdicts[i] = score(pairs[i])

    out = []
    for (task, (done, meter, secs)), (ok, failure, detail) in zip(pairs, verdicts):
        out.append({"id": task["id"], "domain": task["domain"], "correct": bool(ok), "failure": failure,
                    "tokens": meter["tokens"], "prompt_tokens": meter["prompt_tokens"], "llm_calls": meter["calls"],
                    "truncated_calls": meter["truncated"], "server_errors": meter["server_errors"],
                    "tool_calls": done.get("tool_calls", []),
                    "seconds": round(secs, 2), "detail": detail, "answer": (done.get("answer") or "")[-1500:]})
    return out
