"""Tool router: decides which tools the model sees for a task, in what order, plus an optional hint.

This file is what a Phase A candidate may rewrite. Contract (enforced by the sandbox worker):
- select(task, tools, history) -> {"tools": [names], "hint": str}
- only names from `tools`, no duplicates; hint at most 2000 characters
- pure Python standard library, at most 1 second per call, no network, no file writes

`task` is the user's request. `tools` is a list of function dicts with "name", "description" and
"parameters" (JSON Schema). `history` is reserved and currently None.

The initial version is the identity router: every tool, in the given order, no hint.
"""


def select(task, tools, history=None):
    return {"tools": [t["name"] for t in tools], "hint": ""}
