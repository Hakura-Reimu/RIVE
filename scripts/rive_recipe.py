"""Training-recipe check (docs/rive-design.md section 6.21): every adapter of sections 6.18-6.20 is worse than the
base model on the dev pack it was trained on. Is the recipe the cause, or does the served adapter differ from the
trained one? Section 6.19's data (the 4B's own passing answers) is trained again at a learning rate five times
smaller; the base model and self-s1 get a second dev evaluation each; the served adapters' loss on their own
training answers is compared with the base model's.

  python scripts/rive_recipe.py train      # QLoRA at lr 2e-5 on ~/alter/rive/self/train.jsonl -> lora-gentle-s1
  python scripts/rive_recipe.py canary     # the adapter changes answers (needs the server with the adapters)
  python scripts/rive_recipe.py eval       # section 6.21's evaluations
  python scripts/rive_recipe.py fidelity   # served loss on training answers: base, self-s1, gentle-s1
  python scripts/rive_recipe.py report     # results/rive_recipe.json

Steps keep what they finished: an evaluation already on disk is not made again.
"""
import argparse
import json
import random
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rive_self  # noqa: E402
import rive_t2  # noqa: E402
import rive_teach  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.meta_rsi import synth  # noqa: E402
from rive.meta_rsi.reset_meta import write_json  # noqa: E402

HOME = Path.home()
OUT = HOME / "alter/rive/recipe"
ADAPTER = "gentle-s1"
ADAPTER_DIR = HOME / "alter/models" / f"lora-{ADAPTER}"
LR = 2e-5  # 6.19 used 1e-4; nothing else changes
BASE_A0 = rive_self.BASE_A0
BAR = rive_self.BAR
GAP = 5  # dev tasks between two-run means: about twice the SD of such a difference (single runs 35-40)
FIDELITY_ROWS = 64
SERVER = "http://127.0.0.1:8000/v1/completions"


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def beat():
    (OUT / "heartbeat").touch()  # scripts/watchdog.py restarts a chain whose heartbeat goes stale


def t2_for(model):
    out = OUT / f"t2_{model}"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "packs.json").exists():
        shutil.copy2(rive_teach.T2B / "packs.json", out / "packs.json")
    t2 = rive_t2.T2(argparse.Namespace(out_dir=str(out), model=model, adapter=synth.THINKING_ADAPTER, workers=16,
                                       step="eval"))
    t2.heartbeat = OUT / "heartbeat"  # the chain's heartbeat, touched as tasks finish
    return t2


def train_step(args):
    adapter = OUT / "adapters" / "seed1-vllm"
    if not adapter.exists():
        shutil.copy2(rive_self.OUT / "train.jsonl", OUT / "train.jsonl")
        proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/phase_d_train.py"), "--run-dir", str(OUT),
                                 "--seed", "1", "--max-len", "3072", "--lr", str(LR)], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:  # one line per optimizer step
            print(line, end="", flush=True)
            beat()
        if proc.wait():
            raise SystemExit(f"training failed with exit code {proc.returncode}")
    shutil.copytree(adapter, ADAPTER_DIR, dirs_exist_ok=True)
    log(f"adapter in {ADAPTER_DIR}")


def canary_step(args):
    tasks = rive_teach.packs()["dev"][:3]
    base, taught = LLM(model="qwen3.5-4b"), LLM(model=ADAPTER)
    pairs = [[m.chat([{"role": "user", "content": t["public"]["prompt"]}], max_tokens=200) for m in (base, taught)]
             for t in tasks]
    differ = sum(a != b for a, b in pairs)
    write_json(OUT / "canary.json", {"differ": differ, "of": len(pairs), "starts": [[a[:200], b[:200]] for a, b in pairs]})
    log(f"canary: {differ} of {len(pairs)} answers differ with the adapter")
    if differ < 2:
        raise SystemExit("the adapter does not change the answers: not scored")


EVALS = [  # (label, model, pack, seed base), all a0
    ("base_dev_s0", "qwen3.5-4b", "dev", 0),
    ("base_dev_s1", "qwen3.5-4b", "dev", 100000),
    ("self_dev_s1", rive_self.ADAPTER, "dev", 100000),  # its seed-0 run is section 6.19's self_a0_dev_s0
    ("gentle_dev_s0", ADAPTER, "dev", 0),
    ("gentle_dev_s1", ADAPTER, "dev", 100000),
    ("gentle_select_s0", ADAPTER, "select", 0),
    ("gentle_select_s1", ADAPTER, "select", 100000),
]


def eval_step(args):
    p, runners = rive_teach.packs(), {}
    for label, model, pack, seed in EVALS:
        path = OUT / "evals" / f"{label}.json"
        if path.exists():
            continue
        if model not in runners:
            runners[model] = t2_for(model)
        t2 = runners[model]
        results = t2.run_tasks(p[pack], t2.agent("a0"), seed_base=seed)
        if any(r.get("server_errors", 0) for r in results):
            raise SystemExit(f"the model server failed calls in {label}")
        write_json(path, results)
        log(f"eval {label}: {json.dumps(rive_teach.summary(results))}")


def answer_loss(model, ids, labels):
    """Summed negative log-likelihood of the labelled tokens as the server scores them, and their count."""
    body = json.dumps({"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
    req = urllib.request.Request(SERVER, body.encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        scored = json.load(resp)["choices"][0]["prompt_logprobs"]
    total, n = 0.0, 0
    for k, label in enumerate(labels):
        if label == -100:
            continue
        entry = scored[k]
        total -= entry[str(ids[k])]["logprob"]
        n += 1
    return total, n


def fidelity_step(args):
    path = OUT / "fidelity.json"
    if path.exists():
        return
    from transformers import AutoTokenizer
    import phase_d_train
    tok = AutoTokenizer.from_pretrained(str(HOME / "alter/models/Qwen3.5-4B"))
    rows = [json.loads(line) for line in open(rive_self.OUT / "train.jsonl")]
    rows = [r for r in rows if len(phase_d_train.encode(tok, r["prompt"], r["answer"])[0]) <= 3072]
    rows = random.Random("fidelity").sample(rows, FIDELITY_ROWS)
    out = {}
    for model in ("qwen3.5-4b", rive_self.ADAPTER, ADAPTER):
        sums = [answer_loss(model, *phase_d_train.encode(tok, r["prompt"], r["answer"])) for r in rows]
        out[model] = round(sum(s for s, _ in sums) / sum(n for _, n in sums), 4)
        beat()
    trained = {name: statistics.mean(json.loads((d / "train_seed1.json").read_text())["loss_last"])
               for name, d in ((rive_self.ADAPTER, rive_self.OUT), (ADAPTER, OUT))}
    write_json(path, {"rows": FIDELITY_ROWS, "served_loss": out, "training_loss_last": trained})
    log(f"fidelity: served {out}; training loss at the end {trained}")


def reading(base_dev, self_dev, gentle_dev, gentle_select, served, trained):
    """Section 6.21's readings, from two-run dev means and the served losses."""
    b, s, g = (statistics.mean(x) for x in (base_dev, self_dev, gentle_dev))
    half = (served["qwen3.5-4b"] + trained[rive_self.ADAPTER]) / 2
    faithful = served[rive_self.ADAPTER] <= half
    if not faithful:
        case = "P"  # the served adapter is not the trained one
    elif b - s < GAP:
        case = "R3"  # the dev drop of 6.19 was noise
    elif b - g < GAP:
        case = "R1"  # the recipe hurts; the smaller learning rate does not
    else:
        case = "R2"  # even the smaller learning rate hurts
    gain = statistics.mean(gentle_select) - statistics.mean(BASE_A0)
    return {"base_dev": base_dev, "self_dev": self_dev, "gentle_dev": gentle_dev, "faithful": faithful,
            "half_way_loss": round(half, 4), "case": case, "gentle_select": gentle_select, "gain": gain,
            "useful": gain >= BAR}


def report_step(args):
    evals = {f.stem: rive_teach.summary(json.loads(f.read_text())) for f in sorted((OUT / "evals").glob("*.json"))}
    solved = lambda k: evals[k]["solved"]  # noqa: E731
    self0 = rive_teach.summary(json.loads((rive_self.OUT / "evals/self_a0_dev_s0.json").read_text()))["solved"]
    fid = json.loads((OUT / "fidelity.json").read_text())
    out = {"training": json.loads((OUT / "train_seed1.json").read_text()),
           "canary": json.loads((OUT / "canary.json").read_text()), "evals": evals, "fidelity": fid,
           "reading": reading([solved("base_dev_s0"), solved("base_dev_s1")], [self0, solved("self_dev_s1")],
                              [solved("gentle_dev_s0"), solved("gentle_dev_s1")],
                              [solved("gentle_select_s0"), solved("gentle_select_s1")],
                              fid["served_loss"], fid["training_loss_last"])}
    write_json(ROOT / "results" / "rive_recipe.json", out)
    log(f"report {json.dumps(out['reading'])}")


STEPS = {"train": train_step, "canary": canary_step, "eval": eval_step, "fidelity": fidelity_step,
         "report": report_step}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=STEPS)
    args = p.parse_args()
    (OUT / "evals").mkdir(parents=True, exist_ok=True)
    beat()
    STEPS[args.step](args)


if __name__ == "__main__":
    main()
