"""Trusted judge: runs code under test in a fresh container (no network, capped resources, read-only root).
Same as phase0/sandbox/sandbox.py; candidates never see this container or the hidden tests it is given.

  r = run({"main.py": "print(1+1)"}, ["python", "main.py"], timeout=10)
"""
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

IMAGE = "alter-judge:phase0"


@dataclass
class Result:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


def run(files: dict[str, str], cmd: list[str], timeout: float = 10.0, memory: str = "512m",
        cpus: str = "1", pids: int = 64) -> Result:
    work = Path(tempfile.mkdtemp(prefix="judge-"))
    name = f"judge-{uuid.uuid4().hex[:12]}"
    try:
        for rel, content in files.items():
            path = work / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        work.chmod(0o755)
        docker = [
            "docker", "run", "--rm", "--name", name,
            "--network", "none",
            "--memory", memory, "--memory-swap", memory,
            "--cpus", cpus, "--pids-limit", str(pids),
            "--read-only", "--tmpfs", "/tmp:rw,size=64m,mode=1777",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "10001:10001",
            "-v", f"{work}:/work:ro", "-w", "/work",
            IMAGE, *cmd,
        ]
        try:
            p = subprocess.run(docker, capture_output=True, text=True, timeout=timeout)
            return Result(p.returncode, p.stdout[-20000:], p.stderr[-20000:], False)
        except subprocess.TimeoutExpired as e:
            subprocess.run(["docker", "kill", name], capture_output=True)
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return Result(-1, out[-20000:], "timeout", True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
