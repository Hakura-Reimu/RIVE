"""Host side of the candidate sandbox: one long-lived, network-less container per candidate."""
import json
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

IMAGE = "alter-judge:phase0"
WORKER_DIR = Path(__file__).resolve().parent


class CandidateSandbox:
    """Talks to worker.py over stdin/stdout. Restarts the container if it dies or hangs."""

    def __init__(self, candidate_dir: Path, memory="256m", cpus="1", request_timeout=5.0):
        self.candidate_dir = Path(candidate_dir).resolve()
        self.memory, self.cpus, self.request_timeout = memory, cpus, request_timeout
        self.proc = None
        self.name = None
        self._lines = None

    def _start(self):
        self.name = f"cand-{uuid.uuid4().hex[:12]}"
        cmd = ["docker", "run", "-i", "--rm", "--name", self.name, "--network", "none",
               "--memory", self.memory, "--memory-swap", self.memory, "--cpus", self.cpus, "--pids-limit", "64",
               "--read-only", "--tmpfs", "/tmp:rw,size=64m,mode=1777", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--user", "10001:10001",
               "-v", f"{self.candidate_dir}:/cand:ro", "-v", f"{WORKER_DIR}:/runner:ro",
               IMAGE, "python", "-u", "/runner/worker.py"]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self._lines), daemon=True).start()

    @staticmethod
    def _pump(proc, lines):
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _send(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def request(self, payload: dict, timeout=None) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        try:
            self._send(payload)
            line = self._lines.get(timeout=timeout or self.request_timeout)
        except (BrokenPipeError, queue.Empty):
            line = None
        if line is None:
            self.close()
            return {"ok": False, "error": "sandbox worker died or timed out"}
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            self.close()
            return {"ok": False, "error": f"sandbox worker sent a reply that is not JSON: {line[:200]!r}"}

    def select(self, task: str, tools: list, history=None) -> dict:
        return self.request({"op": "select", "task": task, "tools": tools, "history": history})

    def tool_test(self, name: str) -> dict:
        return self.request({"op": "tool_test", "name": name}, timeout=45)

    def solve(self, task: dict, seconds: float, on_llm) -> dict:
        """Run the candidate's workflow on one public task (Phase C). on_llm(request) -> {"ok", "text"|"error"}
        serves the workflow's model calls; the controller meters them. Returns the worker's "done" message."""
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        rid = uuid.uuid4().hex[:12]
        deadline = time.monotonic() + seconds + 30  # the worker's own alarm fires at `seconds`
        try:
            self._send({"op": "solve", "rid": rid, "task": task, "seconds": int(seconds)})
            # The deadline holds even while model calls keep arriving: a Meta-RSI workflow looped on calls that
            # stand-in answers served at once, and the sandbox ran for 35 minutes.
            while (left := deadline - time.monotonic()) > 0:
                line = self._lines.get(timeout=left)
                if line is None:
                    break
                msg = json.loads(line)
                if msg.get("op") != "llm":
                    return msg
                self._send({"op": "llm_result", "rid": rid, "call": msg.get("call"), **on_llm(msg)})
        except (BrokenPipeError, queue.Empty, json.JSONDecodeError):
            pass
        self.close()
        return {"op": "done", "ok": False, "error": "sandbox worker died or timed out", "tool_calls": []}

    def close(self):
        if self.proc is not None:
            subprocess.run(["docker", "kill", self.name], capture_output=True)
            self.proc.kill()
            self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
