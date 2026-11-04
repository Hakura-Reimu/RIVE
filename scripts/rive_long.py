"""Longer answers (docs/rive-design.md section 6.22): 62 of reason-s1's 65 cut-offs at a0's 4096 tokens per call end
inside its reasoning. Does it catch up with the base model when both may answer in 8192 tokens? And does training fit
in memory at 4096 tokens per example, so that 6.20's long rows could be kept?

  python scripts/rive_long.py probe    # phase_d_train.py --probe 3 --max-len 4096 (needs the GPU free)
  python scripts/rive_long.py eval     # a0 with 8192 tokens per call on the selection pack: base, reason-s1
  python scripts/rive_long.py report   # results/rive_long.json

Steps keep what they finished: an evaluation already on disk is not made again.
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rive_reason  # noqa: E402
import rive_t2  # noqa: E402
import rive_teach  # noqa: E402
from rive.meta_rsi import synth  # noqa: E402
from rive.meta_rsi.reset_meta import write_json  # noqa: E402

HOME = Path.home()
OUT = HOME / "alter/rive/long"
CALL_TOKENS = 8192  # a0 has 4096; synth.TOKEN_BUDGET allows one call of 8192
PROBE_LEN = 4096
BAR = 15  # as in sections 6.18-6.20: about twice the SD of the difference of two runs (7.4)
BASE_A0 = (109, 112, 117)  # a0 at 4096 on the selection pack: sections 6.17 (two) and 6.18
REASON_A0 = (79, 85)  # reason-s1's a0 at 4096, section 6.20
EVALS = [  # (label, model, seed base); the selection pack, a0 with CALL_TOKENS
    ("base_8192_select_s0", "qwen3.5-4b", 0),
    ("reason_8192_select_s0", rive_reason.ADAPTER, 0),
]
EARLIER = {  # the same model and seed base at 4096 tokens, for the per-task comparison
    "base_8192_select_s0": ("t2b", HOME / "alter/rive/t2b/scan/a0"),
    "reason_8192_select_s0": ("file", rive_reason.OUT / "evals/reason_a0_select_s0.json"),
}


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def beat():
    (OUT / "heartbeat").touch()  # scripts/watchdog.py restarts a chain whose heartbeat goes stale


def probe_step(args):
    if (OUT / f"train_probe-{PROBE_LEN}.json").exists():
        return
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/phase_d_train.py"), "--run-dir", str(OUT),
                             "--probe", "3", "--max-len", str(PROBE_LEN)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        beat()
    code = proc.wait()
    if code:  # running out of memory is a result, not a failure of the chain
        write_json(OUT / f"train_probe-{PROBE_LEN}.json", {"failed": True, "exit_code": code})
    log(f"probe at {PROBE_LEN} tokens: exit {code}")


def t2_for(model):
    out = OUT / f"t2_{model}"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "packs.json").exists():
        shutil.copy2(rive_teach.T2B / "packs.json", out / "packs.json")
    t2 = rive_t2.T2(argparse.Namespace(out_dir=str(out), model=model, adapter=synth.THINKING_ADAPTER, workers=16,
                                       step="eval"))
    t2.heartbeat = OUT / "heartbeat"
    return t2


def eval_step(args):
    p = rive_teach.packs()
    for label, model, seed in EVALS:
        path = OUT / "evals" / f"{label}.json"
        if path.exists():
            continue
        t2 = t2_for(model)
        agent = synth.make_agent(t2.out / "scan" / f"a0_{CALL_TOKENS}", {**synth.DEFAULT_CONFIG,
                                                                         "max_tokens": CALL_TOKENS})
        results = t2.run_tasks(p["select"], agent, seed_base=seed)
        if any(r.get("server_errors", 0) for r in results):
            raise SystemExit(f"the model server failed calls in {label}")
        write_json(path, results)
        log(f"eval {label}: {json.dumps(rive_teach.summary(results))}")


def earlier(label):
    kind, where = EARLIER[label]
    if kind == "t2b":
        where = HOME / "alter/rive/t2b/evals" / f"{rive_t2.agent_hash(where)}_select.json"
    return {r["id"]: r for r in json.loads(where.read_text())} if where.exists() else None


def reading(base, reason):
    """Section 6.22's reading from the two 8192-token runs."""
    gap = base - reason
    return {"base_8192": base, "reason_8192": reason, "gap": gap, "case": "L" if gap < BAR else "N",
            "gap_at_4096": statistics.mean(BASE_A0) - statistics.mean(REASON_A0),
            "base_gain_from_length": base - statistics.mean(BASE_A0)}


def report_step(args):
    out = {"evals": {}, "cut_at_4096": {}}
    for label, _, _ in EVALS:
        results = json.loads((OUT / "evals" / f"{label}.json").read_text())
        out["evals"][label] = rive_teach.summary(results)
        before = earlier(label)
        if before:
            cut = [r for r in results if before.get(r["id"], {}).get("truncated_calls")]
            out["cut_at_4096"][label] = {"cut": len(cut), "solved_now": sum(r["correct"] for r in cut),
                                         "still_cut": sum(bool(r["truncated_calls"]) for r in cut),
                                         "solved_before": sum(r["correct"] for r in before.values())}
    probe = OUT / f"train_probe-{PROBE_LEN}.json"
    out["probe"] = json.loads(probe.read_text()) if probe.exists() else None
    out["reading"] = reading(out["evals"]["base_8192_select_s0"]["solved"],
                             out["evals"]["reason_8192_select_s0"]["solved"])
    write_json(ROOT / "results" / "rive_long.json", out)
    log(f"report {json.dumps(out['reading'])}; cut at 4096 {json.dumps(out['cut_at_4096'])}")


STEPS = {"probe": probe_step, "eval": eval_step, "report": report_step}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=STEPS)
    args = p.parse_args()
    (OUT / "evals").mkdir(parents=True, exist_ok=True)
    beat()
    STEPS[args.step](args)


if __name__ == "__main__":
    main()
