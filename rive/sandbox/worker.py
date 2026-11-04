"""Runs inside the judge container (no network, read-only). One JSON request per stdin line,
one JSON response per stdout line. The candidate's code is mounted read-only at /cand.

Requests:
  {"op": "select", "task": str, "tools": [function dicts]}  -> {"ok": true, "tools": [names], "hint": str}
  {"op": "solve", "rid": str, "task": {public task fields}, "seconds": int}
      Runs solve(task, ctx) from /cand/agent/workflow.py (Phase C). Each ctx.llm() call goes to the host as
      {"op": "llm", "rid": rid, "call": n, "messages": [...], "max_tokens": k, "thinking": bool, "temperature": t}
      and is answered with {"op": "llm_result", "rid": rid, "call": n, "ok": bool, "text": str, "error": str}.
      The last line is {"op": "done", "ok": true, "answer": str, "tool_calls": [...]}.
  {"op": "tool_test", "name": str}  -> runs test(public) of /cand/tools/<name>.py -> {"ok": true}
  {"op": "ping"}                                             -> {"ok": true}
Any exception, timeout or contract violation becomes {"ok": false, "error": ...}; the worker keeps running.
A reply to a model call abandoned by a timeout can still arrive; it is recognised by its op and dropped.
"""
import importlib.util
import json
import os
import re
import signal
import sys
import time
import traceback

import public_tools

CALL_SECONDS = 1
TOOL_TEST_SECONDS = 30
MAX_HINT_CHARS = 2000
MAX_ANSWER_CHARS = 50000
TOOL_NAME = re.compile(r"[a-z_][a-z0-9_]{0,40}")
PUBLIC_TOOLS = {"python_exec": public_tools.python_exec, "verilog_sim": public_tools.verilog_sim}
ROLES = {"system", "user", "assistant"}

_modules = {}
_channel = sys.stdout  # replies to the host; main() moves it off file descriptor 1


class CallTimeout(BaseException):
    """Not an Exception: a candidate's `except Exception` must not swallow its time limit."""


def _alarm(signum, frame):
    raise CallTimeout


def send(msg):
    _channel.write(json.dumps(msg) + "\n")
    _channel.flush()


def load(path, name):
    """A candidate module, loaded once; a load error is reported on every use."""
    if path not in _modules:
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _modules[path] = (module, None)
        except Exception:  # noqa: BLE001 - a broken candidate is reported per request
            _modules[path] = (None, traceback.format_exc(limit=3)[-800:])
    module, error = _modules[path]
    if error:
        raise RuntimeError(f"{name} failed to load:\n{error}")
    return module


def check_contract(result, tools):
    """The router may only reorder or drop the tools it was given, plus add a short hint."""
    if not isinstance(result, dict):
        raise TypeError("select() must return a dict")
    names = result.get("tools")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise TypeError("'tools' must be a list of tool names")
    offered = {t["name"] for t in tools}
    unknown = [n for n in names if n not in offered]
    if unknown:
        raise ValueError(f"unknown tools returned: {unknown[:3]}")
    if len(set(names)) != len(names):
        raise ValueError("duplicate tool names returned")
    hint = result.get("hint", "")
    if not isinstance(hint, str) or len(hint) > MAX_HINT_CHARS:
        raise ValueError(f"'hint' must be a string of at most {MAX_HINT_CHARS} characters")
    return {"ok": True, "tools": names, "hint": hint}


class Context:
    """What a workflow may use: the model (through the host, which meters it), tools, and its prompt files."""

    def __init__(self, rid):
        self.rid = rid
        self.calls = 0
        self.tool_calls = []

    def llm(self, messages, max_tokens=1024, thinking=False, temperature=0.0):
        if not isinstance(messages, list) or not messages or not all(
                isinstance(m, dict) and m.get("role") in ROLES and isinstance(m.get("content"), str)
                for m in messages):
            raise ValueError("messages must be a non-empty list of {'role', 'content'} with role system/user/assistant")
        self.calls += 1
        call = self.calls
        send({"op": "llm", "rid": self.rid, "call": call, "messages": messages, "max_tokens": int(max_tokens),
              "thinking": bool(thinking), "temperature": float(temperature)})
        while True:
            line = sys.stdin.readline()
            if not line:
                raise EOFError("host closed the channel")
            msg = json.loads(line)
            if msg.get("op") == "llm_result" and msg.get("rid") == self.rid and msg.get("call") == call:
                break
        if not msg.get("ok"):
            raise RuntimeError(f"model call failed: {msg.get('error')}")
        return msg["text"]

    def tool(self, name, args):
        """A public tool, or a tool the candidate wrote in /cand/tools/<name>.py (run(args, public))."""
        start, ok = time.monotonic(), False
        try:
            if not isinstance(name, str) or not TOOL_NAME.fullmatch(name):
                raise ValueError(f"bad tool name {name!r}")
            if name in PUBLIC_TOOLS:
                out = PUBLIC_TOOLS[name](**args)
            else:
                out = load(f"/cand/tools/{name}.py", f"tool_{name}").run(args, public_tools)
            ok = True
            return out
        finally:
            self.tool_calls.append({"name": name, "ok": ok, "seconds": round(time.monotonic() - start, 3)})

    def prompt(self, name):
        """Text of /cand/agent/prompts/<name>.md, or "" if the candidate has none."""
        if not isinstance(name, str) or not TOOL_NAME.fullmatch(name):
            raise ValueError(f"bad prompt name {name!r}")
        try:
            with open(f"/cand/agent/prompts/{name}.md") as f:
                return f.read()
        except FileNotFoundError:
            return ""


def solve(req):
    ctx = Context(req.get("rid"))
    seconds = max(1, int(req.get("seconds", 600)))
    signal.alarm(seconds)
    try:
        result = load("/cand/agent/workflow.py", "workflow").solve(req["task"], ctx)
    except CallTimeout:
        return {"op": "done", "ok": False, "error": f"solve() exceeded {seconds}s", "tool_calls": ctx.tool_calls}
    except Exception as e:  # noqa: BLE001
        return {"op": "done", "ok": False, "error": f"{type(e).__name__}: {e}"[:800], "tool_calls": ctx.tool_calls}
    finally:
        signal.alarm(0)
    answer = result.get("answer") if isinstance(result, dict) else None
    if not isinstance(answer, str) or len(answer) > MAX_ANSWER_CHARS:
        return {"op": "done", "ok": False, "error": f"solve() must return {{'answer': str}} of at most "
                                                    f"{MAX_ANSWER_CHARS} characters", "tool_calls": ctx.tool_calls}
    return {"op": "done", "ok": True, "answer": answer, "tool_calls": ctx.tool_calls}


def main():
    global _channel
    # Whatever the candidate's code prints (print, a shell, a child process) goes to stderr, which the host drops.
    # On stdout the host would read it as a reply: in the research ability test a tool test that printed "Tool
    # test passed." crashed a research step this way.
    _channel = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    signal.signal(signal.SIGALRM, _alarm)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "llm_result":
                continue  # late reply to a call abandoned by a timeout
            if op == "ping":
                try:
                    load("/cand/agent/tool_router.py", "tool_router")
                    resp = {"ok": True, "error": None}
                except RuntimeError as e:
                    resp = {"ok": False, "error": str(e)}
            elif op == "select":
                select = load("/cand/agent/tool_router.py", "tool_router").select
                signal.alarm(CALL_SECONDS)
                try:
                    result = select(req["task"], req["tools"], req.get("history"))
                finally:
                    signal.alarm(0)
                resp = check_contract(result, req["tools"])
            elif op == "solve":
                resp = solve(req)
            elif op == "tool_test":
                if not isinstance(req.get("name"), str) or not TOOL_NAME.fullmatch(req["name"]):
                    raise ValueError(f"bad tool name {req.get('name')!r}")
                tool = load(f"/cand/tools/{req['name']}.py", f"tool_{req['name']}")
                signal.alarm(TOOL_TEST_SECONDS)
                try:
                    tool.test(public_tools)
                finally:
                    signal.alarm(0)
                resp = {"ok": True}
            else:
                resp = {"ok": False, "error": f"unknown op {op!r}"}
        except CallTimeout:
            limit = TOOL_TEST_SECONDS if op == "tool_test" else CALL_SECONDS
            resp = {"ok": False, "error": f"{'test()' if op == 'tool_test' else 'select()'} exceeded {limit}s"}
        except Exception as e:  # noqa: BLE001
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"[:800]}
        send(resp)


if __name__ == "__main__":
    main()
