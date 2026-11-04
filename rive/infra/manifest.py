"""Run manifest (docs/rive-design.md section 3, item 4; RIVE v2.4 section 5.4): what a set of runs depended on, so a
result can be traced and repeated. Fields that cannot be read are recorded as None rather than guessed."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def command(args, cwd=None):
    try:
        out = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_record(model_dir):
    """The model's configuration hashes and weight file sizes; the weights themselves are too large to hash each
    time, and the directory has no revision metadata of its own."""
    d = Path(model_dir).expanduser()
    if not d.exists():
        return {"dir": str(d), "present": False}
    weights = sorted(d.glob("*.safetensors"))
    return {"dir": str(d), "present": True,
            "config_sha256": sha256(d / "config.json") if (d / "config.json").exists() else None,
            "index_sha256": sha256(d / "model.safetensors.index.json")
            if (d / "model.safetensors.index.json").exists() else None,
            "weights": {p.name: p.stat().st_size for p in weights}}


def manifest(repo, model_dir, data_files, images=("alter-judge:phase0",), extra=None):
    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": command(["git", "rev-parse", "HEAD"], cwd=repo),
        "git_dirty": bool(command(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo)),
        "model": model_record(model_dir),
        "packages": {p: version(p) for p in ("vllm", "torch", "transformers", "openai", "jsonschema", "gepa")},
        "python": platform.python_version(), "os": platform.platform(),
        "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "docker_images": {i: command(["docker", "image", "inspect", "--format", "{{.Id}}", i]) for i in images},
        "data": {str(p): sha256(p) for p in map(Path, data_files) if p.exists()},
        **(extra or {}),
    }


def write_manifest(path, **kwargs):
    from rive.meta_rsi.reset_meta import write_json
    m = manifest(**kwargs)
    write_json(path, m)
    return m


if __name__ == "__main__":
    print(json.dumps(manifest(Path.cwd(), "~/alter/models/Qwen3.5-4B", []), indent=1))
