"""Self-taught (docs/rive-design.md section 6.19): the base model solves section 6.18's synthetic problems and the
dev pack with a0, and a LoRA learns its own answers that pass the tests (rejection-sampling fine-tuning).

  python scripts/rive_self.py sample   # a0 on the problems, ROUNDS samples each, the whole answers kept
  python scripts/rive_self.py data     # at most KEEP passing answers with different code per problem -> train.jsonl
  python scripts/rive_self.py train    # QLoRA (scripts/phase_d_train.py) -> ~/alter/models/lora-self-s1
  python scripts/rive_self.py canary   # the adapter changes answers (needs the server with the adapter)
  python scripts/rive_self.py eval     # section 6.19's evaluations
  python scripts/rive_self.py report   # results/rive_self.json

Steps keep what they finished: a round or evaluation already on disk is not made again.
"""
import argparse
import json
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rive_t2  # noqa: E402
import rive_teach  # noqa: E402
from rive.benchmark.hard import public_task  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.meta_rsi import synth  # noqa: E402
from rive.meta_rsi.reset_meta import write_json  # noqa: E402
from rive.meta_rsi.synth_workflow import code_of  # noqa: E402

HOME = Path.home()
OUT = HOME / "alter/rive/self"
ADAPTER = "self-s1"
ADAPTER_DIR = HOME / "alter/models" / f"lora-{ADAPTER}"
ROUNDS = (300000, 400000, 500000, 600000)  # seed bases of the samples, apart from the evaluations' 0-200000
KEEP = 2  # passing answers kept per problem, so the easy ones do not fill the data
BASE_A0 = (109, 112, 117)  # the base model's a0 on the selection pack: sections 6.17 (two) and 6.18
BAR = 15


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def beat():
    (OUT / "heartbeat").touch()  # scripts/watchdog.py restarts a chain whose heartbeat goes stale


class Recorder(synth.T2Model):
    """T2's model, keeping every whole reply: T2 stores only an answer's last 1500 characters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.replies = {}

    def generate(self, messages, max_tokens=1024, temperature=0.0, thinking=False, seed=0):
        r = super().generate(messages, max_tokens=max_tokens, temperature=temperature, thinking=thinking, seed=seed)
        with self._lock:
            self.replies[(messages[-1]["content"], seed)] = r["text"]
        return r


def t2_for(model, name):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "packs.json").exists():
        shutil.copy2(rive_teach.T2B / "packs.json", out / "packs.json")
    t2 = rive_t2.T2(argparse.Namespace(out_dir=str(out), model=model, adapter=synth.THINKING_ADAPTER, workers=16,
                                       step="eval"))
    t2.heartbeat = OUT / "heartbeat"  # the chain's heartbeat, touched as tasks finish
    return t2


def problems():
    """Section 6.18's synthetic problems whose teacher solution passed, as T2 tasks, then the dev pack."""
    tasks = []
    for f in sorted((rive_teach.OUT / "solve").glob("*.json")):
        r = json.loads(f.read_text())
        if r["ok"]:
            item = r["item"]
            tests = json.loads((synth.DATA / item["tests_dir"] / f"{item['question_id']}.json").read_text())
            shown = [{"input": t["input"], "output": t["output"]} for t in tests[:synth.SHOWN_EXAMPLES]]
            tasks.append({"id": f"t2-{item['id']}", "domain": "code", "source": "lcb_train",
                          "public": {**public_task("lcb_train", item), "examples": shown}, "item": item})
    return tasks + rive_teach.packs()["dev"]


def origin(task):
    return "synthetic" if task["item"]["id"].startswith("teach-") else "dev"


def sample_step(args):
    tasks, t2 = problems(), t2_for("qwen3.5-4b", "t2_sample")
    t2.solver = Recorder(model="qwen3.5-4b", adapter=synth.THINKING_ADAPTER)
    for seed in ROUNDS:
        path = OUT / "samples" / f"{seed}.json"
        if path.exists():
            continue
        results = t2.run_tasks(tasks, t2.agent("a0"), seed_base=seed)
        if any(r.get("server_errors", 0) for r in results):
            raise SystemExit(f"the model server failed calls in round {seed}")
        rows = []
        for task, r in zip(tasks, results):
            whole = t2.solver.replies.get((task["public"]["prompt"], seed), "")
            if whole[-1500:] != r["answer"]:  # a0 makes one call: its reply is the answer
                raise SystemExit(f"{task['id']}: the kept reply is not T2's answer")
            rows.append({**{k: r[k] for k in ("id", "correct", "failure", "tokens", "truncated_calls")},
                         "source": origin(task), "difficulty": task["item"]["difficulty"],
                         "prompt": task["public"]["prompt"], "answer": whole})
        write_json(path, rows)
        log(f"round {seed}: {sum(r['correct'] for r in rows)} of {len(rows)} solved")


def pick(samples, keep=KEEP):
    """At most `keep` passing, whole (not cut off) answers with different code, chosen at random per problem."""
    by = {}
    for s in samples:
        if s["correct"] and not s["truncated_calls"]:
            by.setdefault(s["id"], {}).setdefault(" ".join(code_of(s["answer"]).split()), s)
    out = []
    for i in sorted(by):
        distinct = [by[i][code] for code in sorted(by[i])]  # the same choice whatever the order of the rounds
        random.Random(f"self:{i}").shuffle(distinct)
        out += distinct[:keep]
    return out


def data_step(args):
    samples = [s for seed in ROUNDS for s in json.loads((OUT / "samples" / f"{seed}.json").read_text())]
    rows = pick(samples)
    with open(OUT / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "source", "prompt", "answer")}, ensure_ascii=False) + "\n")
    groups = {}  # "source/difficulty": {problem id: solved in some round}
    for s in samples:
        g = groups.setdefault(f"{s['source']}/{s['difficulty']}", {})
        g[s["id"]] = g.get(s["id"], False) or s["correct"]
    stats = {"samples": len(samples), "passed": sum(s["correct"] for s in samples),
             "passed_by_round": {seed: sum(s["correct"] for s in samples[k * len(samples) // len(ROUNDS):
                                                                         (k + 1) * len(samples) // len(ROUNDS)])
                                 for k, seed in enumerate(ROUNDS)},
             "solved_at_least_once": {g: f"{sum(v.values())} of {len(v)}" for g, v in sorted(groups.items())},
             "rows": len(rows), "rows_by_source": dict(Counter(r["source"] for r in rows)),
             "rows_by_difficulty": dict(Counter(r["difficulty"] for r in rows)),
             "answer_chars": round(statistics.mean(len(r["answer"]) for r in rows)) if rows else 0,
             "answer_tokens": round(statistics.mean(r["tokens"] for r in rows)) if rows else 0}
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


EVALS = [  # (label, configuration, pack, seed base), all with the adapter
    ("self_a0_select_s0", "a0", "select", 0),
    ("self_a0_select_s1", "a0", "select", 100000),
    ("self_a0_test_s0", "a0", "test", 0),
    ("self_combo_select_s0", "retry2_expected_review", "select", 0),
    ("self_a0_dev_s0", "a0", "dev", 0),
]


def eval_step(args):
    p, t2 = rive_teach.packs(), None
    for label, config, pack, seed in EVALS:
        path = OUT / "evals" / f"{label}.json"
        if path.exists():
            continue
        t2 = t2 or t2_for(ADAPTER, "t2_self")
        results = t2.run_tasks(p[pack], t2.agent(config), seed_base=seed)
        if any(r.get("server_errors", 0) for r in results):
            raise SystemExit(f"the model server failed calls in {label}")
        write_json(path, results)
        log(f"eval {label}: {json.dumps(rive_teach.summary(results))}")


def report_step(args):
    evals = {f.stem: rive_teach.summary(json.loads(f.read_text())) for f in sorted((OUT / "evals").glob("*.json"))}
    taught = [evals[k]["solved"] for k in ("self_a0_select_s0", "self_a0_select_s1")]
    gain = statistics.mean(taught) - statistics.mean(BASE_A0)
    out = {"data": json.loads((OUT / "data.json").read_text()),
           "training": json.loads((OUT / "train_seed1.json").read_text()),
           "canary": json.loads((OUT / "canary.json").read_text()), "evals": evals,
           "reading": {"base_a0_select": list(BASE_A0), "self_a0_select": taught, "gain": gain, "bar": BAR,
                       "useful": gain >= BAR, "harmful": gain <= -BAR}}
    write_json(ROOT / "results" / "rive_self.json", out)
    log(f"report {json.dumps(out['reading'])}")


STEPS = {"sample": sample_step, "data": data_step, "train": train_step, "canary": canary_step, "eval": eval_step,
         "report": report_step}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=STEPS)
    args = p.parse_args()
    for sub in ("samples", "evals"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    beat()
    STEPS[args.step](args)


if __name__ == "__main__":
    main()
