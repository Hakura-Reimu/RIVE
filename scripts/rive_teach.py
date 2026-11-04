"""R-011's pilot (docs/rive-design.md section 6.18): DeepSeek teaches T2's non-thinking agent through a LoRA.

  python scripts/rive_teach.py smoke    # 4 problems written and solved, 2 dev tasks solved; sets the number of problems
  python scripts/rive_teach.py make     # the teacher writes the problems
  python scripts/rive_teach.py solve    # the teacher solves them and the dev pack; kept if the solutions pass
  python scripts/rive_teach.py data     # dedup against the selection and test packs -> train.jsonl
  python scripts/rive_teach.py train    # QLoRA (scripts/phase_d_train.py) -> ~/alter/models/lora-teach-s1
  python scripts/rive_teach.py canary   # the adapter changes answers (needs the server with the adapter)
  python scripts/rive_teach.py eval     # section 6.18's evaluations
  python scripts/rive_teach.py report   # results/rive_teach.json

Steps keep what they finished: a request or evaluation already on disk is not made again.
"""
import argparse
import json
import math
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
from rive.benchmark.hard import DATA, public_task  # noqa: E402
from rive.infra.teacher import BudgetExceeded, TeacherGateway  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.meta_rsi import synth, teach  # noqa: E402
from rive.meta_rsi.reset_meta import mean_tokens, solved, write_json  # noqa: E402
from rive.meta_rsi.synth_workflow import code_of  # noqa: E402

HOME = Path.home()
OUT = HOME / "alter/rive/teach"
T2B = HOME / "alter/rive/t2b"
TEACHER_LOG = HOME / "alter/rive/teacher"  # the gateway's log, shared with every other use of the teacher
ADAPTER = "teach-s1"
ADAPTER_DIR = HOME / "alter/models" / f"lora-{ADAPTER}"
SPEND = 8.0  # CNY for the whole batch (section 6.18); no request is made once it could pass this
MAX_PROBLEMS = 450  # 300 before the run, raised for the off-peak start (section 6.18)
WORKERS = 8  # teacher requests in flight
# (thinking, max output tokens). The writer's 4000 of the first smoke test left a hard problem's reply empty.
EFFORT = {"make": ("low", 6000), "solve": ("low", 6000), "dev": ("high", 8000)}
BASE_A0 = (109, 112)  # section 6.17's selection evaluations of the base model's a0 (seed bases 0 and 100000)
BAR = 15


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def packs():
    return json.loads((T2B / "packs.json").read_text())


def gateway():
    return TeacherGateway(TEACHER_LOG, shadow=False)  # R-011 uses the answers: they are not shadow answers


def held_out(p):
    """The selection and test packs' statements: never sent, never trained on."""
    return [t["public"]["statement"] for k in ("select", "test") for t in p[k]]


def ask(gw, kind, instructions, text, forbidden, stop):
    """The teacher's reply, or None once the budget is used up (then `stop` is set and no request follows)."""
    if stop.is_set():
        return None
    effort, limit = EFFORT[kind]
    if spent() + limit * 8.0 / 1e6 > SPEND:  # this experiment's own cap, at the worst case (peak output price)
        log(f"budget: this experiment has spent {spent():.3f} of {SPEND} CNY")
        stop.set()
        return None
    try:
        return gw.ask(instructions, text, f"teach_{kind}", max_output_tokens=limit, effort=effort,
                      forbidden=forbidden)
    except BudgetExceeded as e:
        log(f"budget: {e}")
        stop.set()
        return None


def judged(task, answer):
    ok, failure, verdict = synth.judge(task, answer) if code_of(answer) else (False, "no_code", {})
    return {"ok": bool(ok), "failure": failure, "verdict": verdict}


def make_one(i, gw, forbidden, stop):
    path = OUT / "make" / f"{i:04d}.json"
    if path.exists():
        return
    s = teach.spec(i)
    text = ask(gw, "make", teach.WRITER, teach.writer_prompt(s), forbidden, stop)
    if text is not None:
        write_json(path, {"spec": s, "text": text})


def solve_one(path, gw, forbidden, stop):
    out = OUT / "solve" / path.name
    if out.exists():
        return
    made = json.loads(path.read_text())
    try:
        item, tests = teach.to_item(made["spec"], made["text"])
    except ValueError as e:
        write_json(out, {"id": path.stem, "ok": False, "failure": "bad_problem", "reason": str(e)[:300]})
        return
    tests_path = DATA / teach.TESTS_DIR / f"{item['question_id']}.json"
    tests_path.parent.mkdir(parents=True, exist_ok=True)
    tests_path.write_text(json.dumps(tests))
    prompt = public_task("lcb_train", item)["prompt"]
    answer = ask(gw, "solve", teach.SOLVER, prompt, forbidden, stop)
    if answer is None:
        return
    task = {"id": item["id"], "domain": "code", "source": "lcb_train", "item": item}
    write_json(out, {"id": item["id"], "item": item, "prompt": prompt, "answer": answer, **judged(task, answer)})


def dev_one(task, gw, forbidden, stop):
    out = OUT / "dev" / f"{task['id']}.json"
    if out.exists():
        return
    prompt = task["public"]["prompt"]
    answer = ask(gw, "dev", teach.SOLVER, prompt, forbidden, stop)
    if answer is not None:
        write_json(out, {"id": task["id"], "prompt": prompt, "answer": answer, **judged(task, answer)})


def beat():
    (OUT / "heartbeat").touch()  # scripts/watchdog.py restarts a chain whose heartbeat goes stale


def run_all(fn, items, *rest):
    with ThreadPoolExecutor(WORKERS) as pool:
        for f in [pool.submit(fn, x, *rest) for x in items]:
            f.result()
            beat()


def spent():
    """CNY this experiment's requests have cost so far."""
    path = TEACHER_LOG / "usage.jsonl"
    return sum(e["cost_cny"] for e in map(json.loads, path.read_text().splitlines())
               if e["purpose"].startswith("teach_")) if path.exists() else 0.0


def costs():
    """Mean CNY per request of each kind of this experiment in the teacher's log (cache hits left out)."""
    lines = (TEACHER_LOG / "usage.jsonl").read_text().splitlines()
    by = {}
    for e in map(json.loads, lines):
        if e["purpose"].startswith("teach_") and not e["cache_hit"]:
            by.setdefault(e["purpose"][6:], []).append(e)
    # The batch starts off-peak (04:00 UK, 12:00 Beijing), so the plan uses off-peak prices; ask() keeps the cap.
    return {k: {"requests": len(v), "cny": statistics.mean(x["cost_cny"] for x in v),
                "cny_offpeak": statistics.mean(x["cost_cny"] / (2 if x["peak"] else 1) for x in v),
                "output_tokens": statistics.mean(x["output_tokens"] for x in v)} for k, v in by.items()}


def smoke_step(args):
    p, gw, stop = packs(), gateway(), threading.Event()
    forbidden = held_out(p)
    run_all(make_one, range(4), gw, forbidden, stop)
    run_all(solve_one, sorted((OUT / "make").glob("000[0-3].json")), gw, forbidden, stop)
    run_all(dev_one, p["dev"][:2], gw, forbidden, stop)
    c = costs()
    per_problem = c["make"]["cny_offpeak"] + c["solve"]["cny_offpeak"]
    n = min(MAX_PROBLEMS, math.floor((SPEND - spent() - len(p["dev"]) * c["dev"]["cny_offpeak"]) / per_problem))
    for f in sorted((OUT / "solve").glob("*.json")) + sorted((OUT / "dev").glob("*.json")):
        r = json.loads(f.read_text())
        log(f"smoke {f.parent.name}/{f.stem}: ok {r['ok']} ({r.get('failure')}, {r.get('reason', '')[:120]})")
    write_json(OUT / "plan.json", {"costs": c, "problems": n, "spend_cny": SPEND})
    log(f"smoke costs {json.dumps(c)}; problems {n}")


def make_step(args):
    p, gw, stop = packs(), gateway(), threading.Event()
    n = json.loads((OUT / "plan.json").read_text())["problems"]
    run_all(make_one, range(n), gw, held_out(p), stop)
    log(f"make: {len(list((OUT / 'make').glob('*.json')))} of {n} problems written")


def solve_step(args):
    p, gw, stop = packs(), gateway(), threading.Event()
    forbidden = held_out(p)
    run_all(dev_one, p["dev"], gw, forbidden, stop)
    run_all(solve_one, sorted((OUT / "make").glob("*.json")), gw, forbidden, stop)
    for kind in ("dev", "solve"):
        rows = [json.loads(f.read_text()) for f in (OUT / kind).glob("*.json")]
        log(f"{kind}: {sum(r['ok'] for r in rows)} of {len(rows)} kept; failures "
            f"{dict(Counter(r.get('failure') for r in rows if not r['ok']))}")


def data_step(args):
    p = packs()
    refs = [(t["item"].get("title", ""), t["public"]["statement"]) for k in ("select", "test") for t in p[k]]
    synthetic = [json.loads(f.read_text()) for f in sorted((OUT / "solve").glob("*.json"))]
    passed = {r["id"]: r for r in synthetic if r["ok"]}
    kept, dropped = teach.dedup([r["item"] for r in passed.values()], refs)
    dev = [json.loads(f.read_text()) for f in sorted((OUT / "dev").glob("*.json"))]
    rows = ([{"id": i["id"], "source": "synthetic", "prompt": passed[i["id"]]["prompt"],
              "answer": passed[i["id"]]["answer"]} for i in kept]
            + [{"id": r["id"], "source": "dev", "prompt": r["prompt"], "answer": r["answer"]} for r in dev if r["ok"]])
    with open(OUT / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"problems_written": len(list((OUT / "make").glob("*.json"))), "synthetic_solved": len(synthetic),
             "synthetic_passed": len(passed), "synthetic_kept": len(kept), "dropped": dict(Counter(r for _, r in dropped)),
             "dev_solved": len(dev), "dev_passed": sum(r["ok"] for r in dev), "rows": len(rows),
             "kept_by_type": dict(Counter(f"{i['testtype']}/{i['difficulty']}" for i in kept)),
             "answer_chars": round(statistics.mean(len(r["answer"]) for r in rows)) if rows else 0}
    write_json(OUT / "data.json", stats)
    log(f"data {json.dumps(stats)}")


def train_step(args):
    adapter = OUT / "adapters" / "seed1-vllm"
    if not adapter.exists():
        # --max-len 3072: the 4608 default now fails in the backward pass ("CUDA driver error: device not ready"),
        # 2048 and 3072 train (probe before the run, peak 8.65 GiB); longer examples are left out and counted.
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
    tasks = packs()["dev"][:3]
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
        shutil.copy2(T2B / "packs.json", out / "packs.json")
    t2 = rive_t2.T2(argparse.Namespace(out_dir=str(out), model=model, adapter=synth.THINKING_ADAPTER, workers=16,
                                       step="eval"))
    t2.heartbeat = OUT / "heartbeat"  # the chain's heartbeat, touched as tasks finish
    return t2


EVALS = [  # (label, model, configuration, pack, seed base)
    ("teach_a0_select_s0", ADAPTER, "a0", "select", 0),
    ("teach_a0_select_s1", ADAPTER, "a0", "select", 100000),
    ("base_a0_select_s2", "qwen3.5-4b", "a0", "select", 200000),
    ("teach_a0_test_s0", ADAPTER, "a0", "test", 0),
    ("teach_combo_select_s0", ADAPTER, "retry2_expected_review", "select", 0),
    ("teach_a0_dev_s0", ADAPTER, "a0", "dev", 0),
]


def summary(results):
    return {"solved": solved(results), "tasks": len(results), "tokens": mean_tokens(results),
            "cut_off": sum(r.get("truncated_calls", 0) > 0 for r in results),
            "failures": dict(Counter(r["failure"] for r in results if not r["correct"]))}


def eval_step(args):
    p, t2s = packs(), {}
    for label, model, config, pack, seed in EVALS:
        path = OUT / "evals" / f"{label}.json"
        if path.exists():
            continue
        t2 = t2s.setdefault(model, t2_for(model, "t2_" + ("teach" if model == ADAPTER else "base")))
        results = t2.run_tasks(p[pack], t2.agent(config), seed_base=seed)
        if any(r.get("server_errors", 0) for r in results):
            raise SystemExit(f"the model server failed calls in {label}")
        write_json(path, results)
        log(f"eval {label}: {json.dumps(summary(results))}")


def report_step(args):
    evals = {f.stem: summary(json.loads(f.read_text())) for f in sorted((OUT / "evals").glob("*.json"))}
    base = list(BASE_A0) + [evals["base_a0_select_s2"]["solved"]]
    taught = [evals[k]["solved"] for k in ("teach_a0_select_s0", "teach_a0_select_s1")]
    gain = statistics.mean(taught) - statistics.mean(base)
    out = {"plan": json.loads((OUT / "plan.json").read_text()), "data": json.loads((OUT / "data.json").read_text()),
           "training": json.loads((OUT / "train_seed1.json").read_text()),
           "canary": json.loads((OUT / "canary.json").read_text()), "evals": evals,
           "reading": {"base_a0_select": base, "teach_a0_select": taught, "gain": gain, "bar": BAR,
                       "useful": gain >= BAR},
           "teacher_cost_cny": round(sum(e["cost_cny"] for e in map(json.loads, (TEACHER_LOG / "usage.jsonl")
                                         .read_text().splitlines()) if e["purpose"].startswith("teach_")), 4)}
    write_json(ROOT / "results" / "rive_teach.json", out)
    log(f"report {json.dumps(out['reading'])}; teacher cost {out['teacher_cost_cny']} CNY")


STEPS = {"smoke": smoke_step, "make": make_step, "solve": solve_step, "data": data_step, "train": train_step,
         "canary": canary_step, "eval": eval_step, "report": report_step}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=STEPS)
    args = p.parse_args()
    for sub in ("make", "solve", "dev", "evals"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    beat()
    STEPS[args.step](args)


if __name__ == "__main__":
    main()
