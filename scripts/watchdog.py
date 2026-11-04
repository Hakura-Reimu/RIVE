"""Watchdog for long unattended runs (docs/meta-rsi-design.md section 7, after DeepSeek's Super-agent).

It starts a chain script and checks it every two minutes: is it still running, has its heartbeat file been touched
recently, does the vLLM server answer (once it has answered since the start)? A chain that failed, stalled or lost
its server is stopped together with the server and its sandbox containers, and started again; its steps go on where
they stopped. After `--tries` restarts it gives up, leaves the GPU free and says why. Events go to
results/logs/watchdog.log.

  python scripts/watchdog.py --heartbeat FILE [--stall-minutes 25] [--tries 3] -- bash scripts/x_chain.sh args
"""
import argparse
import os
import signal
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "results" / "logs" / "watchdog.log"
SERVER = "http://127.0.0.1:8000/v1/models"


def log(message):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}\n")
    print(message, flush=True)


def server_up():
    try:
        urllib.request.urlopen(SERVER, timeout=10)
        return True
    except OSError:
        return False


def stop(proc, cleanup):
    """The chain's whole process group; with `cleanup`, also the server and leftover sandbox containers."""
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    if cleanup:
        subprocess.run(["pkill", "-f", "bin/vllm serve"])
        subprocess.run("docker ps -aq --filter name=cand- --filter name=judge- | xargs -r docker rm -f",
                       shell=True, capture_output=True)
        time.sleep(30)  # the server's GPU memory is freed


def watch(cmd, heartbeat, every, stall, tries, cleanup):
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    restarts = 0
    while True:
        heartbeat.touch()
        started = time.time()
        proc = subprocess.Popen(cmd, start_new_session=True)
        log(f"started: {' '.join(cmd)} (pid {proc.pid})")
        answered, missed, reason = False, 0, None
        while reason is None:
            time.sleep(every)
            code = proc.poll()
            if code == 0:
                log("chain finished")
                return 0
            if code is not None:
                reason = f"the chain exited with {code}"
                break
            if server_up():
                answered, missed = True, 0
            elif answered:
                missed += 1
                if missed >= 3:
                    reason = "the server stopped answering"
            idle = time.time() - max(heartbeat.stat().st_mtime, started)
            if reason is None and idle > stall * 60:
                reason = f"no progress for {idle / 60:.0f} minutes"
        stop(proc, cleanup)
        if restarts >= tries:
            log(f"giving up after {tries} restarts: {reason}")
            return 1
        restarts += 1
        log(f"restart {restarts} of {tries}: {reason}")


def main():
    global LOG
    p = argparse.ArgumentParser()
    p.add_argument("--heartbeat", required=True)
    p.add_argument("--every", type=float, default=120, help="seconds between checks")
    p.add_argument("--stall-minutes", type=float, default=25)
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--no-cleanup", action="store_true", help="do not stop the server or containers (tests)")
    p.add_argument("--log", default=str(LOG))
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    args = p.parse_args()
    LOG = Path(args.log)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    raise SystemExit(watch(cmd, Path(os.path.expanduser(args.heartbeat)), args.every, args.stall_minutes,
                           args.tries, not args.no_cleanup))


if __name__ == "__main__":
    main()
