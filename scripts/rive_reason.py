"""DeepSeek teaches with its reasoning (docs/rive-design.md section 6.20): section 6.18 again, but the teacher's
reasoning, which the gateway now keeps, goes into the training answer before the reply.

  python scripts/rive_reason.py solve    # the teacher solves 6.18's problems and the dev pack again, reasoning kept
  python scripts/rive_reason.py data     # dedup as in 6.18 -> train.jsonl, answer = reasoning + reply
  python scripts/rive_reason.py train    # QLoRA (scripts/phase_d_train.py) -> ~/alter/models/lora-reason-s1
  python scripts/rive_reason.py canary   # the adapter changes answers (needs the server with the adapter)
  python scripts/rive_reason.py eval     # section 6.20's evaluations
  python scripts/rive_reason.py report   # results/rive_reason.json

Steps keep what they finished: a request or evaluation already on disk is not made again.
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rive_t2  # noqa: E402
import rive_teach  # noqa: E402
from rive.benchmark.hard import DATA, public_task  # noqa: E402
from rive.infra.teacher import BudgetExceeded, TeacherGateway  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.meta_rsi import synth, teach  # noqa: E402
from rive.meta_rsi.reset_meta import write_json  # noqa: E402

HOME = Path.home()
OUT = HOME / "alter/rive/reason"
ADAPTER = "reason-s1"
ADAPTER_DIR = HOME / "alter/models" / f"lora-{ADAPTER}"
SPEND = 4.5  # CNY for the batch; no request is made once it could pass this
DAILY = 12.5  # the gateway's daily cap for this batch only: the user allowed going a little over 10 (2026-11-04)
WORKERS = 8
EFFORT = rive_teach.EFFORT  # 6.18's thinking and output limits
BASE_A0 = (109, 112, 117)  # the base model's a0 on the selection pack: sections 6.17 (two) and 6.18
TEACH_A0 = (56, 63)  # section 6.18's adapter, trained on the replies alone
BAR = 15


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def beat():
    (OUT / "heartbeat").touch()  # scripts/watchdog.py restarts a chain whose heartbeat goes stale


def spent():
    """CNY this experiment's requests have cost so far."""
    path = rive_teach.TEACHER_LOG / "usage.jsonl"
    return sum(e["cost_cny"] for e in map(json.loads, path.read_text().splitlines())
               if e["purpose"].startswith("reason_")) if path.exists() else 0.0


def ask(gw, kind, text, forbidden, stop):
    """(reasoning, reply), or None once the budget is used up (then `stop` is set and no request follows)."""
    if stop.is_set():
        return None
    effort, limit = EFFORT[kind]
    if spent() + limit * 8.0 / 1e6 > SPEND:  # this experiment's own cap, at the worst case (peak output price)
        log(f"budget: this experiment has spent {spent():.3f} of {SPEND} CNY")
        stop.set()
        return None
    try:
        return gw.ask(teach.SOLVER, text, f"reason_{kind}", max_output_tokens=limit, effort=effort,
                      forbidden=forbidden, with_reasoning=True)
    except BudgetExceeded as e:
        log(f"budget: {e}")
        stop.set()
        return None


def solve_one(path, gw, forbidden, stop):
    out = OUT / "solve" / path.name
    if out.exists():
        return
    made = json.loads(path.read_text())
    try:
        item, tests = teach.to_item(made["spec"], made["text"])
    except ValueError:
        return  # 6.18's bad problems
    tests_path = DATA / teach.TESTS_DIR / f"{item['question_id']}.json"
    if not tests_path.exists():
        tests_path.parent.mkdir(parents=True, exist_ok=True)
        tests_path.write_text(json.dumps(tests))
    prompt = public_task("lcb_train", item)["prompt"]
    got = ask(gw, "solve", prompt, forbidden, stop)
    if got is None:
        return
    task = {"id": item["id"], "domain": "code", "source": "lcb_train", "item": item}
    write_json(out, {"id": item["id"], "item": item, "prompt": prompt, "reasoning": got[0], "answer": got[1],
                     **rive_teach.judged(task, got[1])})


def dev_one(task, gw, forbidden, stop):
    out = OUT / "dev" / f"{task['id']}.json"
    if out.exists():
        return
    prompt = task["public"]["prompt"]
    got = ask(gw, "dev", prompt, forbidden, stop)
    if got is not None:
        write_json(out, {"id": task["id"], "prompt": prompt, "reasoning": got[0], "answer": got[1],
                         **rive_teach.judged(task, got[1])})


def run_all(fn, items, *rest):
    with ThreadPoolExecutor(WORKERS) as pool:
        for f in [pool.submit(fn, x, *rest) for x in items]:
            f.result()
            beat()


def solve_step(args):
    p, stop = rive_teach.packs(), threading.Event()
    gw = TeacherGateway(rive_teach.TEACHER_LOG, daily_cny=DAILY, shadow=False)
    forbidden = rive_teach.held_out(p)
    run_all(solve_one, sorted((rive_teach.OUT / "make").glob("*.json")), gw, forbidden, stop)
    run_all(dev_one, p["dev"], gw, forbidden, stop)
    for kind in ("solve", "dev"):
        rows = [json.loads(f.read_text()) for f in (OUT / kind).glob("*.json")]
        log(f"{kind}: {sum(r['ok'] for r in rows)} of {len(rows)} kept; failures "
            f"{dict(Counter(r.get('failure') for r in rows if not r['ok']))}; spent {spent():.3f} CNY")


def joined(r):
    """The training answer: the teacher's reasoning, then its reply."""
    return (r["reasoning"].strip() + "\n\n" + r["answer"].strip()) if r["reasoning"].strip() else r["answer"]


def data_step(args):
    p = rive_teach.packs()
    refs = [(t["item"].get("title", ""), t["public"]["statement"]) for k in ("select", "test") for t in p[k]]
    synthetic = [json.loads(f.read_text()) for f in sorted((OUT / "solve").glob("*.json"))]
    passed = {r["id"]: r for r in synthetic if r["ok"]}
    kept, dropped = teach.dedup([r["item"] for r in passed.values()], refs)
    dev = [json.loads(f.read_text()) for f in sorted((OUT / "dev").glob("*.json"))]
    used = [passed[i["id"]] for i in kept] + [r for r in dev if r["ok"]]
    rows = [{"id": r["id"], "source": "synthetic" if r["id"].startswith("teach-") else "dev", "prompt": r["prompt"],
             "answer": joined(r)} for r in used]
    with open(OUT / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"synthetic_solved": len(synthetic), "synthetic_passed": len(passed), "synthetic_kept": len(kept),
             "dropped": dict(Counter(r for _, r in dropped)), "dev_solved": len(dev),
             "dev_passed": sum(r["ok"] for r in dev), "rows": len(rows),
             "kept_by_type": dict(Counter(f"{i['testtype']}/{i['difficulty']}" for i in kept)),
             "empty_reasoning": sum(not r["reasoning"].strip() for r in used),
             "reasoning_chars": round(statistics.median(len(r["reasoning"]) for r in used)) if used else 0,
             "answer_chars": round(statistics.median(len(r["answer"]) for r in rows)) if rows else 0,
             "teacher_cost_cny": round(spent(), 4)}
    write_json(OUT / "data.json", stats)
    log(f"data {json.dumps(stats)}")


def train_step(args):
    adapter = OUT / "adapters" / "seed1-vllm"
    if not adapter.exists():
        # Section 6.18's settings, --max-len 3072 included; longer examples are left out and counted.
        proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/phase_d_train.py"), "--run-dir", str(OUT),
                                 "--seed", "1", "--max-len", "3072"], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:  # one line per optimizer step
            print(line, end="", flush=True)
            beat()
        if proc.wait():
            raise SystemExit(f"training failed with exit code {proc.returncode}")
    shutil.copytree(adapter, ADAPTER_DIR, dirs_exist_ok=True)
    log(f"adapter in {ADAPTER_DIR}")


def canary_step(args):
    """The adapter must change the answers (phase0/REPORT.md section 4: a misnamed adapter is skipped silently)."""
    tasks = rive_teach.packs()["dev"][:3]
    base, taught = LLM(model="qwen3.5-4b"), LLM(model=ADAPTER)
    pairs = [[m.chat([{"role": "user", "content": t["public"]["prompt"]}], max_tokens=200) for m in (base, taught)]
             for t in tasks]
    differ = sum(a != b for a, b in pairs)
    write_json(OUT / "canary.json", {"differ": differ, "of": len(pairs), "starts": [[a[:200], b[:200]] for a, b in pairs]})
    log(f"canary: {differ} of {len(pairs)} answers differ with the adapter")
    if differ < 2:
        raise SystemExit("the adapter does not change the answers: not scored")


def t2_for(model, name):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "packs.json").exists():
        shutil.copy2(rive_teach.T2B / "packs.json", out / "packs.json")
    t2 = rive_t2.T2(argparse.Namespace(out_dir=str(out), model=model, adapter=synth.THINKING_ADAPTER, workers=16,
                                       step="eval"))
    t2.heartbeat = OUT / "heartbeat"  # the chain's heartbeat, touched as tasks finish
    return t2


EVALS = [  # (label, configuration, pack, seed base), all with the adapter
    ("reason_a0_select_s0", "a0", "select", 0),
    ("reason_a0_select_s1", "a0", "select", 100000),
    ("reason_a0_test_s0", "a0", "test", 0),
    ("reason_combo_select_s0", "retry2_expected_review", "select", 0),
    ("reason_a0_dev_s0", "a0", "dev", 0),
]


def eval_step(args):
    p, t2 = rive_teach.packs(), None
    for label, config, pack, seed in EVALS:
        path = OUT / "evals" / f"{label}.json"
        if path.exists():
            continue
        t2 = t2 or t2_for(ADAPTER, "t2_reason")
        results = t2.run_tasks(p[pack], t2.agent(config), seed_base=seed)
        if any(r.get("server_errors", 0) for r in results):
            raise SystemExit(f"the model server failed calls in {label}")
        write_json(path, results)
        log(f"eval {label}: {json.dumps(rive_teach.summary(results))}")


def report_step(args):
    evals = {f.stem: rive_teach.summary(json.loads(f.read_text())) for f in sorted((OUT / "evals").glob("*.json"))}
    taught = [evals[k]["solved"] for k in ("reason_a0_select_s0", "reason_a0_select_s1")]
    gain = statistics.mean(taught) - statistics.mean(BASE_A0)
    out = {"data": json.loads((OUT / "data.json").read_text()),
           "training": json.loads((OUT / "train_seed1.json").read_text()),
           "canary": json.loads((OUT / "canary.json").read_text()), "evals": evals,
           "reading": {"base_a0_select": list(BASE_A0), "reason_a0_select": taught, "gain": gain, "bar": BAR,
                       "useful": gain >= BAR, "over_6_18": statistics.mean(taught) - statistics.mean(TEACH_A0)},
           "teacher_cost_cny": round(spent(), 4)}
    write_json(ROOT / "results" / "rive_reason.json", out)
    log(f"report {json.dumps(out['reading'])}; teacher cost {out['teacher_cost_cny']} CNY")


STEPS = {"solve": solve_step, "data": data_step, "train": train_step, "canary": canary_step, "eval": eval_step,
         "report": report_step}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=STEPS)
    args = p.parse_args()
    for sub in ("solve", "dev", "evals"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    beat()
    STEPS[args.step](args)


if __name__ == "__main__":
    main()
