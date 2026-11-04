"""Can evaluation be made reproducible? (docs/rive-design.md section 6.10.) The same agent (P2.0b's retry2 on
fail_b2048's selection pack) evaluated several times with the same seeds: in one server session as served so far,
then in two sessions with vLLM's batch-invariant kernels (VLLM_BATCH_INVARIANT=1). The script starts and stops the
server itself, so nothing else may be serving. Finished evaluations are kept and not repeated.

  python scripts/rive_determinism_check.py [--out-dir ~/alter/rive/determinism] [--out results/rive_determinism.json]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.llm import LLM  # noqa: E402
from rive.meta_rsi.bugfix import judge  # noqa: E402
from rive.meta_rsi.reset_meta import solved, write_json  # noqa: E402
from rive.runtime.workflow import run_workflow_tasks  # noqa: E402

P2B = Path.home() / "alter/meta-rsi/p2b"
TOKEN_BUDGET = 2048  # fail_b2048's
# (label, batch-invariant, evaluations in the session)
SESSIONS = (("default", False, 2), ("invariant_1", True, 2), ("invariant_2", True, 1))
URL = "http://127.0.0.1:8000/v1/models"


def up():
    try:
        urllib.request.urlopen(URL, timeout=5)
        return True
    except OSError:
        return False


def serve(invariant, log):
    """The server as scripts/serve.sh starts it, with or without the batch-invariant kernels; None if it exits or
    is not up within 15 minutes."""
    env = dict(os.environ, VLLM_BATCH_INVARIANT="1" if invariant else "0")
    with open(log, "a") as f:
        proc = subprocess.Popen(["bash", str(ROOT / "scripts" / "serve.sh")], stdout=f, stderr=subprocess.STDOUT,
                                env=env)
    for _ in range(90):
        if proc.poll() is not None:
            return None
        if up():
            return proc
        time.sleep(10)
    stop(proc)
    return None


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
    subprocess.run(["pkill", "-f", "bin/vllm serve"])
    while up():
        time.sleep(5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.expanduser("~/alter/rive/determinism"))
    p.add_argument("--out", default=str(ROOT / "results" / "rive_determinism.json"))
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if up():
        sys.exit("a model server is already up; this check starts its own")
    tasks = json.loads((P2B / "packs" / "fail_b2048.json").read_text())["select"]
    agent = P2B / "scan" / "retry2"
    llm = LLM()
    log = ROOT / "results" / "logs" / "vllm_serve_determinism.log"
    started = {}
    for label, invariant, n in SESSIONS:
        paths = [out / f"{label}_r{i + 1}.json" for i in range(n)]
        if all(path.exists() for path in paths):
            continue
        t = time.time()
        proc = serve(invariant, log)
        started[label] = time.time() - t
        if proc is None:
            print(f"{label}: the server did not start", flush=True)
            write_json(out / f"{label}_failed.json", {"seconds": started[label]})
            continue
        try:
            for path in paths:
                if path.exists():
                    continue
                t = time.time()
                results = run_workflow_tasks(tasks, agent, llm, workers=args.workers, token_budget=TOKEN_BUDGET,
                                             judge_fn=judge)
                wall = time.time() - t
                errors = sum(r.get("server_errors", 0) for r in results)
                write_json(path, {"seconds": wall, "server_errors": errors, "results": results})
                print(f"{path.stem}: solved {solved(results)}, {wall:.0f} s, server errors {errors}", flush=True)
        finally:
            stop(proc)
    report(out, Path(args.out), started)


def report(out, path, started):
    runs = {}
    for label, _, n in SESSIONS:
        for i in range(n):
            f = out / f"{label}_r{i + 1}.json"
            if f.exists():
                runs[f.stem] = json.loads(f.read_text())
    failed = [label for label, _, _ in SESSIONS if (out / f"{label}_failed.json").exists()]

    def same(a, b):
        ra, rb = runs[a]["results"], runs[b]["results"]
        return {"answers": sum(x["answer"] == y["answer"] for x, y in zip(ra, rb)),
                "verdicts": sum(x["correct"] == y["correct"] for x, y in zip(ra, rb)), "tasks": len(ra)}

    names = sorted(runs)
    pairs = {f"{a} vs {b}": same(a, b) for i, a in enumerate(names) for b in names[i + 1:]}
    seconds = {k: v["seconds"] for k, v in runs.items()}
    default = [v for k, v in seconds.items() if k.startswith("default")]
    invariant = [v for k, v in seconds.items() if k.startswith("invariant")]
    summary = {"solved": {k: solved(v["results"]) for k, v in runs.items()},
               "server_errors": {k: v["server_errors"] for k, v in runs.items()},
               "seconds": seconds, "server_start_seconds": started, "failed_to_start": failed, "same": pairs,
               "slowdown": statistics.mean(invariant) / statistics.mean(default) if default and invariant else None}
    write_json(path, summary)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
