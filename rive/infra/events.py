"""Append-only event log (docs/rive-design.md section 4, R-000): one JSON line per state change of a step, run or
evaluation, with a sequence number, the time and the hash of the code that made it. Lines are only ever added;
a restarted run continues the sequence."""
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path


def code_hash(paths):
    """Hash of the given source files (sorted by path), to tie every event to the code that produced it."""
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        h.update(str(p.name).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()[:16]


class EventLog:
    def __init__(self, path, code):
        self.path, self.code = Path(path), code
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.seq = 0
        if self.path.exists():
            text = self.path.read_text()
            self.seq = len(text.splitlines())
            if text and not text.endswith("\n"):  # a line cut off by a crash: the next event starts a new line
                with self.path.open("a") as f:
                    f.write("\n")

    def log(self, event, **fields):
        with self.lock:
            self.seq += 1
            line = {"seq": self.seq, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "event": event, "code": self.code, **fields}
            with self.path.open("a") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()
            return line


def read_events(path):
    """The events in order, skipping a last line cut off by a crash."""
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
