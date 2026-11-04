"""Public tools for Phase C workflows (docs/phase-c-design.md section 3). They run inside the candidate's
sandbox (no network, read-only root, writable /tmp only) and are what candidate-written tools build on."""
import os
import resource
import subprocess
import sys
import tempfile

MAX_OUTPUT = 4000
MAX_SECONDS = 30


def _tail(text):
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return (text or "")[-MAX_OUTPUT:]


def _limit_memory():
    resource.setrlimit(resource.RLIMIT_AS, (384 << 20, 384 << 20))


def python_exec(code, stdin="", seconds=10):
    """Run a Python program; returns its exit code, the tail of stdout/stderr and whether it timed out."""
    seconds = min(float(seconds), MAX_SECONDS)
    with tempfile.TemporaryDirectory(dir="/tmp") as work:
        path = os.path.join(work, "main.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, "-I", path], input=stdin, capture_output=True, text=True,
                               timeout=seconds, cwd=work, preexec_fn=_limit_memory)
            return {"returncode": p.returncode, "stdout": _tail(p.stdout), "stderr": _tail(p.stderr),
                    "timed_out": False}
        except subprocess.TimeoutExpired as e:
            return {"returncode": None, "stdout": _tail(e.stdout), "stderr": _tail(e.stderr), "timed_out": True}


def verilog_sim(design, testbench, top=None, seconds=20):
    """Compile a design and a testbench with iverilog 12 (-g2012) and simulate with vvp."""
    seconds = min(float(seconds), MAX_SECONDS)
    with tempfile.TemporaryDirectory(dir="/tmp") as work:
        for name, text in (("design.sv", design), ("tb.sv", testbench)):
            with open(os.path.join(work, name), "w") as f:
                f.write(text)
        cmd = ["iverilog", "-g2012", "-o", "sim", *(["-s", top] if top else []), "tb.sv", "design.sv"]
        try:
            c = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds, cwd=work)
        except subprocess.TimeoutExpired:
            return {"compiled": False, "log": "compilation timed out", "stdout": "", "timed_out": True}
        if c.returncode != 0:
            return {"compiled": False, "log": _tail(c.stdout + c.stderr), "stdout": "", "timed_out": False}
        try:
            r = subprocess.run(["vvp", "-n", "sim"], capture_output=True, text=True, timeout=seconds, cwd=work)
        except subprocess.TimeoutExpired as e:
            return {"compiled": True, "log": _tail(c.stderr), "stdout": _tail(e.stdout), "timed_out": True}
        return {"compiled": True, "log": _tail(c.stderr), "stdout": _tail(r.stdout + r.stderr), "timed_out": False}
