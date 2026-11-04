"""Needs docker and the alter-judge:phase0 image (WSL). Checks the candidate sandbox contract."""
import shutil
from pathlib import Path

import pytest

from rive.sandbox.container import CandidateSandbox

TEMPLATE = Path(__file__).resolve().parents[1] / "candidate_template"
TOOLS = [{"name": "a", "description": "add", "parameters": {}}, {"name": "b", "description": "sub", "parameters": {}}]

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")


def candidate(tmp_path, router_source=None):
    shutil.copytree(TEMPLATE, tmp_path / "cand")
    if router_source is not None:
        (tmp_path / "cand" / "agent" / "tool_router.py").write_text(router_source)
    for p in [tmp_path / "cand", *(tmp_path / "cand").rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    return tmp_path / "cand"


def test_identity_router(tmp_path):
    with CandidateSandbox(candidate(tmp_path)) as sb:
        assert sb.select("add 1 and 2", TOOLS) == {"ok": True, "tools": ["a", "b"], "hint": ""}
        assert sb.select("again", TOOLS)["ok"]  # same container serves many calls


def test_contract_violations_are_reported_not_fatal(tmp_path):
    src = "def select(task, tools, history=None):\n    return {'tools': ['zzz'], 'hint': ''}\n"
    with CandidateSandbox(candidate(tmp_path, src)) as sb:
        r = sb.select("x", TOOLS)
        assert not r["ok"] and "unknown tools" in r["error"]
        assert sb.request({"op": "ping"})["ok"]


def test_slow_router_times_out(tmp_path):
    src = "def select(task, tools, history=None):\n    while True:\n        pass\n"
    with CandidateSandbox(candidate(tmp_path, src)) as sb:
        r = sb.select("x", TOOLS)
        assert not r["ok"] and "exceeded" in r["error"]


def test_router_cannot_reach_network_or_write(tmp_path):
    src = ("import socket\n"
           "def select(task, tools, history=None):\n"
           "    try:\n"
           "        socket.create_connection(('1.1.1.1', 80), timeout=0.5)\n"
           "        net = 'open'\n"
           "    except OSError:\n"
           "        net = 'blocked'\n"
           "    try:\n"
           "        open('/cand/x', 'w')\n"
           "        fs = 'writable'\n"
           "    except OSError:\n"
           "        fs = 'read-only'\n"
           "    return {'tools': [], 'hint': net + ',' + fs}\n")
    with CandidateSandbox(candidate(tmp_path, src)) as sb:
        assert sb.select("x", TOOLS)["hint"] == "blocked,read-only"


def test_broken_candidate_reports_load_error(tmp_path):
    with CandidateSandbox(candidate(tmp_path, "def select(:\n")) as sb:
        r = sb.select("x", TOOLS)
        assert not r["ok"] and "failed to load" in r["error"]
