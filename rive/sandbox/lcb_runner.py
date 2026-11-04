"""Runs inside the judge container: check /work/solution.py against /work/tests.json (LiveCodeBench format).

  python /work/lcb_runner.py stdin
  python /work/lcb_runner.py functional <func_name>

Stops at the first failing test (pass@1 needs every test). Prints one JSON line with the verdict.
"""
import json
import math
import signal
import subprocess
import sys

PER_TEST_SECONDS = 6
# LiveCodeBench prepends these imports to every solution.
PRELUDE = (
    "from typing import *\nfrom functools import *\nfrom collections import *\nfrom itertools import *\n"
    "from heapq import *\nfrom bisect import *\nfrom string import *\nfrom operator import *\nfrom math import *\n"
    "import math, string, re, heapq, bisect, itertools, collections, functools, operator, random, sys\n"
    "sys.setrecursionlimit(50000)\n"
)


def tokens_equal(got: str, want: str) -> bool:
    a, b = got.split(), want.split()
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            if not math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-6):
                return False
        except ValueError:
            return False
    return True


def values_equal(got, want) -> bool:
    if isinstance(got, float) or isinstance(want, float):
        try:
            return math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    if isinstance(got, (list, tuple)) and isinstance(want, list):
        return len(got) == len(want) and all(values_equal(g, w) for g, w in zip(got, want))
    return got == want


def run_stdin(tests):
    with open("/tmp/solution.py", "w") as f:
        f.write(PRELUDE + open("/work/solution.py").read())
    for i, t in enumerate(tests):
        try:
            p = subprocess.run([sys.executable, "/tmp/solution.py"], input=t["input"], capture_output=True,
                               text=True, timeout=PER_TEST_SECONDS)
        except subprocess.TimeoutExpired:
            return i, "time limit"
        if p.returncode != 0:
            return i, "runtime error: " + p.stderr.strip()[-200:]
        if not tokens_equal(p.stdout, t["output"]):
            return i, "wrong answer"
    return len(tests), "ok"


def _alarm(signum, frame):
    raise TimeoutError


def run_functional(tests, func_name):
    namespace = {}
    try:
        exec(compile(PRELUDE + open("/work/solution.py").read(), "solution.py", "exec"), namespace)
        method = getattr(namespace["Solution"](), func_name)
    except Exception as e:  # noqa: BLE001 - any failure to load the candidate is a verdict
        return 0, f"load error: {type(e).__name__}: {e}"[:200]
    signal.signal(signal.SIGALRM, _alarm)
    for i, t in enumerate(tests):
        args = [json.loads(line) for line in t["input"].split("\n") if line.strip()]
        try:
            signal.alarm(PER_TEST_SECONDS)
            got = method(*args)
            signal.alarm(0)
        except TimeoutError:
            return i, "time limit"
        except Exception as e:  # noqa: BLE001
            signal.alarm(0)
            return i, f"runtime error: {type(e).__name__}: {e}"[:200]
        if not values_equal(got, json.loads(t["output"])):
            return i, "wrong answer"
    return len(tests), "ok"


if __name__ == "__main__":
    tests = json.load(open("/work/tests.json"))
    if sys.argv[1] == "stdin":
        passed, reason = run_stdin(tests)
    else:
        passed, reason = run_functional(tests, sys.argv[2])
    print(json.dumps({"passed": passed, "total": len(tests), "reason": reason}))
