"""Run a candidate's tool router plus the model on tool-pool tasks and score each task."""
from concurrent.futures import ThreadPoolExecutor

from rive.benchmark.bfcl import as_tools, classify, parse_tool_calls

MAX_TOKENS = 1024


def task_text(task):
    return "\n".join(m["content"] for m in task["messages"] if m["role"] == "user")


def render(tokenizer, task, functions, hint):
    messages = ([{"role": "system", "content": hint}] if hint else []) + list(task["messages"])
    return tokenizer.apply_chat_template(messages, tools=as_tools(functions) if functions else None, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def run_tasks(tasks, sandbox, llm, tokenizer, workers=16):
    """Results in task order: {"id", "correct", "failure", "n_tools_shown", "shown", "calls"}.

    A router error on a task counts as failure "router_error" for that task.
    """
    routed = []
    for t in tasks:
        r = sandbox.select(task_text(t), t["functions"])
        if not r.get("ok"):
            routed.append((t, None, None, r.get("error", "router failed")))
            continue
        by_name = {f["name"]: f for f in t["functions"]}
        shown = [by_name[n] for n in r["tools"]]
        routed.append((t, shown, r["hint"], None))

    def solve(entry):
        t, shown, hint, error = entry
        if error is not None:
            return {"id": t["id"], "correct": False, "failure": "router_error", "n_tools_shown": 0, "shown": [],
                    "calls": [], "error": error[:300]}
        text = llm.complete(render(tokenizer, t, shown, hint), max_tokens=MAX_TOKENS)
        calls = parse_tool_calls(text, t["functions"])
        failure = classify(text, calls, t["ground_truth"], t["irrelevance"])
        return {"id": t["id"], "correct": failure is None, "failure": failure, "n_tools_shown": len(shown),
                "shown": [f["name"] for f in shown], "calls": [[n, a] for n, a in calls][:4], "output": text[-800:]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(solve, routed))
