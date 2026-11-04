"""Test-reward reinforcement learning on the 4B (docs/rive-design.md section 6.25, revised): the model samples a0
answers to problems of four domains, each answer is checked exactly (reward 1 if right, else 0), and a LoRA
adapter is trained on the answers weighted by their advantage (scripts/rive_rl_train.py). The domains
(scripts/rive_rl_tasks.py for all but programming):

  - code: section 6.18's synthetic programming problems, judged by T2's sandbox (tests);
  - math: MATH's training set, judged by math-verify;
  - logic: knights and knaves, seating orders, target numbers, made fresh for every iteration;
  - writing: short texts under checkable instructions, made fresh for every iteration.

One iteration: MIX problems (32), G samples each, from the current policy (the base model, then the latest
adapter); advantage = reward - mean reward of the problem's samples; the server sleeps (vLLM sleep mode), the
adapter is trained STEPS optimizer steps from where it was, the server wakes and serves the new adapter.

Every DEV_EVERY iterations the policy is checked on the four development sets. The checkpoint with the best mean
development pass rate is evaluated on T2's selection pack (two seeds) and test pack, and on the evaluation sets of
the other three domains (two seeds), where the base model is evaluated too.

  python scripts/rive_rl.py run              # the loop, resumed where it stopped
  python scripts/rive_rl.py smoke            # 2 small iterations, in <out>/smoke
  python scripts/rive_rl.py eval             # the chosen checkpoint and the base model on the evaluation sets
  python scripts/rive_rl.py report           # results/rive_rl.json
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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rive_rl_tasks as rt  # noqa: E402
import rive_self  # noqa: E402
import rive_t2  # noqa: E402
import rive_teach  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.meta_rsi import synth  # noqa: E402
from rive.meta_rsi.reset_meta import write_json  # noqa: E402

HOME = Path.home()
OUT = HOME / "alter/rive/rl"
MODEL = HOME / "alter/models/Qwen3.5-4B"
BASE = "qwen3.5-4b"
URL = "http://127.0.0.1:8000"
SERVER_LOG = ROOT / "results/logs/vllm_serve_rive_fp8.log"
MIX = {"code": 12, "math": 8, "logic": 6, "writing": 6}  # problems of each domain per iteration
G, STEPS, LR = 8, 2, 3e-5
ITERS, DEV_EVERY = 150, 15
LOOP_HOURS = 60  # the user's cap is 72 hours of GPU time in all (about 2 went to the stopped runs and calibration,
# about 4 go to the evaluations): the iterations stop here
# Difficulty classes as LiveCodeBench's easy / medium / hard. Training uses hard problems (and code's synthetic
# ones); development and evaluation have all three. Code: T2's packs carry the labels.
DEV_SIZE = {"code": 74, "math": 134, "logic": 135, "writing": 135}  # hard + 30 medium + 30 easy
EVAL_SIZE = {"math": 250, "logic": 250, "writing": 250}  # the hard sets, where a gain is measured
LEVEL_SIZE = 100  # logic and writing's medium and easy evaluation sets; maths: MATH-500's levels 3-4 (100), 1-2 (93)
EASY_LOSS = 0.05  # an easy set may lose at most this share of its problems (mean of two runs against the base's)
EVAL_SEEDS = (0, 100000)
SEED0 = 1_000_000  # sampling seeds, apart from the evaluations' (0-200000) and section 6.19's (300000-600000)
MAX_TOKENS = 4096  # a0's cap (T2), for every domain
BASE_A0_SELECT = (109, 112, 117)  # the base model's a0 on the selection pack (sections 6.17, 6.18)
BASE_A0_TEST = 104                # and on the test pack (6.17)
BAR, CONFIRM = 15, 8


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def beat():
    (OUT / "heartbeat").touch()


class Sampler(synth.T2Model):
    """T2's model, keeping every whole reply. T2 gives every task of a run the same seed, so the copies of one
    problem in a batch would get the same answer: the k-th call with the same prompt and seed samples with
    seed + 7919 k instead."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen, self.replies = Counter(), defaultdict(list)

    def generate(self, messages, max_tokens=1024, temperature=0.0, thinking=False, seed=0):
        key = messages[-1]["content"]
        with self._lock:
            k = self.seen[(key, seed)]
            self.seen[(key, seed)] += 1
        r = super().generate(messages, max_tokens=max_tokens, temperature=temperature, thinking=thinking,
                             seed=seed + 7919 * k)
        with self._lock:
            self.replies[key].append(r["text"])
        return r


# ---- the server -----------------------------------------------------------------------------------------------

def http(method, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def served():
    """The names the server answers to (the base model and its adapters), or None if it is down."""
    try:
        return {m["id"] for m in json.loads(http("GET", "/v1/models", timeout=10))["data"]}
    except OSError:
        return None


def start_server():
    """The FP8 server (scripts/serve.sh) with runtime LoRA loading and sleep mode."""
    if served() is not None:
        return
    env = {"VLLM_SERVER_DEV_MODE": "1", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
    cmd = (f"setsid nohup bash {ROOT}/scripts/serve.sh {MODEL} {BASE} --enable-lora --max-lora-rank 16 "
           f"--max-loras 2 --enable-sleep-mode >> {SERVER_LOG} 2>&1 < /dev/null &")
    subprocess.run(["bash", "-c", " ".join(f"{k}={v}" for k, v in env.items()) + " " + cmd], check=True)
    for _ in range(120):
        time.sleep(5)
        if served() is not None:
            log("server up")
            return
    raise SystemExit("the server did not start in 10 minutes")


def sleeping():
    return json.loads(http("GET", "/is_sleeping", timeout=10)).get("is_sleeping", False)


def use(name, path=None):
    """Serve the adapter `name` (loaded from `path` if the server does not have it); drop other RL adapters."""
    start_server()
    if sleeping():
        http("POST", "/wake_up")
    have = served()
    if name != BASE and name not in have:
        http("POST", "/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
    for other in have - {name, BASE}:
        if other.startswith("rl-"):
            http("POST", "/v1/unload_lora_adapter", {"lora_name": other})


# ---- the problems -----------------------------------------------------------------------------------------------

def pool():
    """Section 6.18's synthetic problems whose teacher solution passed (rive_self.problems without the dev pack)."""
    return [t for t in rive_self.problems() if rive_self.origin(t) == "synthetic"]


def batch_for(it, ids, b):
    """The ids of iteration `it` from a fixed pool: each epoch is a fresh shuffle cut into len(ids) // b batches."""
    per_epoch = len(ids) // b
    epoch, k = divmod(it, per_epoch)
    order = sorted(ids)
    random.Random(f"rl:{epoch}").shuffle(order)
    return order[k * b:(k + 1) * b]


class Problems:
    """Every domain's training, development and evaluation problems."""

    def __init__(self):
        self.code = {t["id"]: t for t in pool()}
        self.math = {t["id"]: t for t in rt.math_tasks("train")}
        self.dev = {"code": rive_teach.packs()["dev"],
                    "math": rt.math_tasks("dev") + rt.math_tasks("dev_medium") + rt.math_tasks("dev_easy"),
                    **{d: rt.fixed_set(d, "dev", 75) + rt.fixed_set(d, "dev", 30, "medium")
                       + rt.fixed_set(d, "dev", 30, "easy") for d in ("logic", "writing")}}
        self.eval = {"math": rt.math_tasks("eval"), "math_medium": rt.math_tasks("eval_medium"),
                     "math_easy": rt.math_tasks("eval_easy")}
        for d in ("logic", "writing"):
            self.eval[d] = rt.fixed_set(d, "eval", EVAL_SIZE[d])
            for level in ("medium", "easy"):
                self.eval[f"{d}_{level}"] = rt.fixed_set(d, "eval", LEVEL_SIZE, level)
        self.banned = {t["public"]["prompt"] for s in (self.dev, self.eval) for ts in s.values() for t in ts}

    def batch(self, it, mix):
        return {"code": [self.code[i] for i in batch_for(it, sorted(self.code), mix["code"])],
                "math": [self.math[i] for i in batch_for(it, sorted(self.math), mix["math"])],
                "logic": [rt.training_task("logic", it, k, self.banned) for k in range(mix["logic"])],
                "writing": [rt.training_task("writing", it, k, self.banned) for k in range(mix["writing"])]}


# ---- sampling and judging ---------------------------------------------------------------------------------------

def advantages(rows):
    """Reward minus the mean reward of the same problem's samples."""
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(float(r["correct"]))
    mean = {k: statistics.mean(v) for k, v in by.items()}
    return [float(r["correct"]) - mean[r["id"]] for r in rows]


def t2_for(out, model):
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "packs.json").exists():
        shutil.copy2(rive_teach.T2B / "packs.json", out / "packs.json")
    t2 = rive_t2.T2(argparse.Namespace(out_dir=str(out), model=model, adapter=synth.THINKING_ADAPTER, workers=16,
                                       step="eval"))
    t2.heartbeat = OUT / "heartbeat"  # the watchdog's heartbeat, touched as tasks finish
    return t2


def sample_code(out, tasks, model, g, seed_base):
    """G a0 answers per programming problem through T2, with their whole text; an answer whose text was not kept
    gets no gradient."""
    if not tasks:
        return []
    t2 = t2_for(out / "t2", model)
    t2.solver = Sampler(model=model, adapter=synth.THINKING_ADAPTER)
    copies = [t for t in tasks for _ in range(g)]
    results = t2.run_tasks(copies, t2.agent("a0"), seed_base=seed_base)
    if any(r.get("server_errors", 0) for r in results):
        raise SystemExit("the model server failed calls")
    left = {k: list(v) for k, v in t2.solver.replies.items()}
    rows = []
    for task, r in zip(copies, results):
        kept = left.get(task["public"]["prompt"], [])
        j = next((i for i, text in enumerate(kept) if text[-1500:] == r["answer"]), None)
        rows.append({**{k: r[k] for k in ("id", "correct", "failure", "tokens", "truncated_calls")}, "domain": "code",
                     "prompt": task["public"]["prompt"], "answer": kept.pop(j) if j is not None else None})
    return rows


def sample(out, batch, model, g, seed_base):
    """All domains of one iteration, programming and the others at the same time."""
    others = [t for d in ("math", "logic", "writing") for t in batch[d]]
    with ThreadPoolExecutor(max_workers=2) as ex:
        code = ex.submit(sample_code, out, batch["code"], model, g, seed_base)
        direct = ex.submit(sample_direct, others, model, g, seed_base)
        code_rows, (jobs, replies) = code.result(), direct.result()
    return code_rows + judge_direct(jobs, replies)


def sample_direct(tasks, model, g, seed_base):
    """G answers per problem of the other domains: one non-thinking call with a0's sampling (the model card's, at
    temperature 0.7) and cap. They are judged afterwards by judge_direct, in the main thread (math-verify needs it)."""
    from openai import OpenAI
    client = OpenAI(base_url=f"{URL}/v1", api_key="EMPTY", timeout=1800)
    card = synth.CARD[False]
    jobs = [(t, j) for t in tasks for j in range(g)]

    def one(job):
        task, j = job
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": task["public"]["prompt"]}], max_tokens=MAX_TOKENS,
            temperature=0.7, top_p=card["top_p"], presence_penalty=card["presence_penalty"], seed=seed_base + 7919 * j,
            extra_body={"top_k": card["top_k"], "min_p": card["min_p"],
                        "chat_template_kwargs": {"enable_thinking": False}})
        beat()
        return r.choices[0].message.content or "", r.usage.completion_tokens, r.choices[0].finish_reason == "length"

    with ThreadPoolExecutor(max_workers=16) as ex:
        return jobs, list(ex.map(one, jobs))


def judge_direct(jobs, replies):
    rows = []
    for (task, _), (text, tokens, cut) in zip(jobs, replies):
        ok, failure = rt.judge(task, text)
        rows.append({"id": task["id"], "domain": task["domain"], "correct": ok, "failure": failure, "tokens": tokens,
                     "truncated_calls": int(cut), "prompt": task["public"]["prompt"], "answer": text})
    return rows


def evaluate(path, model, domain, tasks, seed_base):
    """a0 on `tasks` of one domain (one answer each), kept in `path`."""
    if path.exists():
        return json.loads(path.read_text())
    if domain == "code":
        t2 = t2_for(OUT / "t2_eval", model)
        t2.solver = synth.T2Model(model=model, adapter=synth.THINKING_ADAPTER)
        results = t2.run_tasks(tasks, t2.agent("a0"), seed_base=seed_base)
        if any(r.get("server_errors", 0) for r in results):
            raise SystemExit("the model server failed calls")
    else:
        results = [{k: r[k] for k in ("id", "correct", "failure", "tokens", "truncated_calls", "answer")}
                   for r in judge_direct(*sample_direct(tasks, model, 1, seed_base))]
    write_json(path, results)
    return results


def canary(name):
    """The adapter must change the answers (an adapter vLLM cannot match is skipped silently, phase0 section 4)."""
    tasks = rive_teach.packs()["dev"][:3]
    base, tuned = LLM(model=BASE), LLM(model=name)
    pairs = [[m.chat([{"role": "user", "content": t["public"]["prompt"]}], max_tokens=200) for m in (base, tuned)]
             for t in tasks]
    return sum(a != b for a, b in pairs)


# ---- the loop ---------------------------------------------------------------------------------------------------

def name_of(n):
    """The policy after n iterations: the base model, or the adapter trained in iteration n - 1."""
    return BASE if n == 0 else f"rl-{n - 1:04d}"


def adapter_dir(out, it):
    return out / "adapters" / f"{it:04d}"


def train(out, it, rows, weights, lr):
    batch = out / "iters" / f"{it:04d}" / "batch.jsonl"
    with open(batch, "w") as f:
        for r, w in zip(rows, weights):
            if w and r["answer"] is not None:
                f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "answer": r["answer"], "weight": w},
                                   ensure_ascii=False) + "\n")
    cmd = [sys.executable, str(ROOT / "scripts/rive_rl_train.py"), "--batch", str(batch), "--out",
           str(adapter_dir(out, it)), "--steps", str(STEPS), "--lr", str(lr), "--seed", str(it)]
    if it > 0:
        cmd += ["--init", str(adapter_dir(out, it - 1))]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if line.startswith("{"):
            print(line, end="", flush=True)
        beat()
    if proc.wait():
        raise SystemExit(f"training failed with exit code {proc.returncode}")
    return json.loads((adapter_dir(out, it) / "train.json").read_text())


def prune(out, it, every):
    """Adapters are kept for the dev checkpoints and the two latest iterations (about 125 MB a pair)."""
    old = it - 2
    if old >= 0 and (old + 1) % every:
        for d in (adapter_dir(out, old), Path(f"{adapter_dir(out, old)}-vllm")):
            shutil.rmtree(d, ignore_errors=True)


def dev_check(out, n, name, problems, sizes):
    """The policy on every domain's development set: {domain: pass rate}."""
    rates = {}
    for domain, tasks in problems.dev.items():
        tasks = tasks[:sizes[domain]]
        rows = evaluate(out / "dev" / f"{n:04d}_{domain}.json", name, domain, tasks, 0)
        rates[domain] = round(sum(r["correct"] for r in rows) / len(rows), 4)
    log(f"dev after {n} iterations ({name}): {json.dumps(rates)}")
    return rates


def domain_stats(rows):
    out = {}
    for d in MIX:
        rs = [r for r in rows if r["domain"] == d]
        if not rs:
            continue
        groups = defaultdict(list)
        for r in rs:
            groups[r["id"]].append(r["correct"])
        out[d] = {"samples": len(rs), "passed": sum(r["correct"] for r in rs),
                  "with_signal": sum(0 < sum(v) < len(v) for v in groups.values()),
                  "cut": sum(r["truncated_calls"] > 0 for r in rs),
                  "mean_tokens": round(statistics.mean(r["tokens"] for r in rs))}
    return out


def loop(out, iters, mix, g, every, lr, dev_sizes, strict_canary=True):
    if (out / "halted.json").exists():
        raise SystemExit(f"halted for a check: {(out / 'halted.json').read_text()}")
    problems = Problems()
    (out / "iters").mkdir(parents=True, exist_ok=True)
    (out / "dev").mkdir(parents=True, exist_ok=True)
    if not (out / "started.json").exists():
        write_json(out / "started.json", {"time": time.time(), "at": time.strftime("%F %T")})
    deadline = json.loads((out / "started.json").read_text())["time"] + LOOP_HOURS * 3600
    for it in range(iters + 1):
        if time.time() > deadline:
            log(f"{LOOP_HOURS} hours since the start: no more iterations; the checkpoints done are evaluated")
            break
        n = it  # the policy has had n iterations of training
        name, d = name_of(n), out / "iters" / f"{it:04d}"
        need_dev = (n % every == 0 or n == iters) and not all(
            (out / "dev" / f"{n:04d}_{dom}.json").exists() for dom in problems.dev)
        need_iteration = it < iters and not (d / "done.json").exists()
        if not (need_dev or need_iteration):
            continue
        if n > 0 and not (adapter_dir(out, n - 1) / "train.json").exists():
            raise SystemExit(f"the adapter of iteration {n - 1} is missing")
        use(name, Path(f"{adapter_dir(out, n - 1)}-vllm") if n else None)
        if need_dev:
            dev_check(out, n, name, problems, dev_sizes)
            if n == every:
                differ = canary(name)
                log(f"canary: {differ} of 3 answers differ from the base model's")
                if differ < 2 and strict_canary:
                    raise SystemExit("the adapter does not change the answers")
        if not need_iteration:
            continue
        d.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        if not (d / "samples.json").exists():
            write_json(d / "samples.json", sample(out, problems.batch(it, mix), name, g, SEED0 + 1000 * it))
        rows = json.loads((d / "samples.json").read_text())
        sampled = time.perf_counter() - start
        weights = advantages(rows)
        http("POST", "/sleep?level=1")
        trained = train(out, it, rows, weights, lr)
        use(name_of(it + 1), Path(f"{adapter_dir(out, it)}-vllm"))
        prune(out, it, every)
        stats = {"iteration": it, "policy": name, "samples": len(rows), "passed": sum(r["correct"] for r in rows),
                 "unmatched": sum(r["answer"] is None for r in rows), "domains": domain_stats(rows),
                 "train": {k: trained[k] for k in ("examples", "cut_to_max_len", "steps", "seconds",
                                                   "peak_reserved_gib")},
                 "seconds": {"sample": round(sampled), "total": round(time.perf_counter() - start)}}
        write_json(d / "done.json", stats)
        log(f"iteration {it}: {json.dumps(stats)}")
        rates = [s["passed"] / s["samples"] for s in
                 (json.loads((out / "iters" / f"{k:04d}" / "done.json").read_text()) for k in range(it + 1))]
        if len(rates) > 10 and all(r < rates[0] / 2 for r in rates[-10:]):  # section 6.25: maybe a pipeline error
            write_json(out / "halted.json", {"iteration": it, "first": rates[0], "last_ten": rates[-10:]})
            raise SystemExit("the training pass rate stayed below half of iteration 0's for 10 iterations")


def dev_rates(out=OUT):
    """{checkpoint: {domain: pass rate}} of the development checks done."""
    out_rates = defaultdict(dict)
    for f in (out / "dev").glob("*_*.json"):
        n, domain = f.stem.split("_", 1)
        rows = json.loads(f.read_text())
        out_rates[int(n)][domain] = sum(r["correct"] for r in rows) / len(rows)
    return dict(out_rates)


def chosen(out=OUT):
    """The checkpoint with the best mean development pass rate over the four domains (ties: the later), among those
    after at least one iteration with all four checked."""
    rates = dev_rates(out)
    full = {n: statistics.mean(r.values()) for n, r in rates.items() if n > 0 and len(r) == len(DEV_SIZE)}
    return max(full, key=lambda n: (full[n], n)), rates


# ---- steps ------------------------------------------------------------------------------------------------------

def run_step(args):
    loop(OUT, args.iters, MIX, G, DEV_EVERY, LR, DEV_SIZE)


def smoke_step(args):
    loop(OUT / "smoke", 2, {"code": 2, "math": 2, "logic": 2, "writing": 2}, 4, 1, LR,
         {"code": 4, "math": 4, "logic": 4, "writing": 4}, strict_canary=False)


def eval_step(args):
    n, rates = chosen()
    log(f"dev by checkpoint: {json.dumps(rates)}; chosen: after {n} iterations")
    problems, packs = Problems(), rive_teach.packs()
    (OUT / "evals").mkdir(exist_ok=True)
    jobs = [(who, "code", f"select_{s}", packs["select"], s) for who in ("base", "tuned") for s in EVAL_SEEDS]
    jobs += [("tuned", "code", "test_0", packs["test"], 0)]
    jobs += [(who, d, f"{d}_{s}", problems.eval[d], s) for who in ("base", "tuned") for d in problems.eval
             for s in EVAL_SEEDS]
    for who, domain, label, tasks, seed in jobs:
        name = name_of(n) if who == "tuned" else BASE
        use(name, Path(f"{adapter_dir(OUT, n - 1)}-vllm") if who == "tuned" else None)
        rows = evaluate(OUT / "evals" / f"{who}_{label}.json", name, domain.split("_")[0], tasks, seed)
        log(f"eval {who} {label}: {sum(r['correct'] for r in rows)} of {len(rows)}")


def by_class(rows, label):
    """Solved per difficulty class: {class: [solved, of]}."""
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        c = label(r["id"])
        out[c][0] += r["correct"]
        out[c][1] += 1
    return dict(out)


def verdict(gain, easy_drop, easy_n):
    """Section 6.25's reading for one domain: the gain bar, then the easy guard."""
    if gain <= -BAR:
        return "harmful"
    if gain < BAR:
        return "no use"
    return "useful" if easy_drop <= EASY_LOSS * easy_n else "fails the easy guard"


def report_step(args):
    n, rates = chosen()
    problems, packs = Problems(), rive_teach.packs()
    iters = [json.loads(f.read_text()) for f in sorted((OUT / "iters").glob("*/done.json"))]
    rows = {f.stem: json.loads(f.read_text()) for f in (OUT / "evals").glob("*.json")}
    solved = {k: sum(r["correct"] for r in v) for k, v in rows.items()}

    def mean(who, label):
        return statistics.mean(solved[f"{who}_{label}_{s}"] for s in EVAL_SEEDS)

    def have(*labels):
        return all(f"{who}_{lab}_{s}" in solved for who in ("base", "tuned") for lab in labels for s in EVAL_SEEDS)
    level = {t["id"]: t["item"]["difficulty"] for p in packs.values() for t in p}
    sets = {d: [k for k in problems.eval if k.split("_")[0] == d] for d in EVAL_SIZE}
    level.update({t["id"]: f"level {t['item']['level']}" for k in sets["math"] for t in problems.eval[k]})
    level.update({t["id"]: f"{t['level']} {t['family']}" for k in sets["logic"] for t in problems.eval[k]})
    level.update({t["id"]: f"{t['level']}, {len(t['constraints'])} instructions" for k in sets["writing"]
                  for t in problems.eval[k]})
    res = {"settings": {"mix": MIX, "G": G, "steps": STEPS, "lr": LR, "iterations": ITERS, "dev_every": DEV_EVERY,
                        "easy_loss": EASY_LOSS},
           "dev": rates, "chosen": n, "evals": solved,
           "classes": {k: by_class(v, lambda i: level.get(i, "?")) for k, v in rows.items()},
           "train_pass_rate": {d: [round(s["domains"][d]["passed"] / s["domains"][d]["samples"], 3)
                                   for s in iters if d in s["domains"]] for d in MIX},
           "readings": {}}
    if have("select"):
        tuned = mean("tuned", "select")
        gain = tuned - statistics.mean(BASE_A0_SELECT)
        easy = {who: statistics.mean(sum(r["correct"] for r in rows[f"{who}_select_{s}"] if level[r["id"]] == "easy")
                                     for s in EVAL_SEEDS) for who in ("base", "tuned")}
        n_easy = sum(level[t["id"]] == "easy" for t in packs["select"])
        r = {"tuned": tuned, "base": statistics.mean(BASE_A0_SELECT), "gain": gain,
             "base_now": mean("base", "select"), "easy": easy, "easy_drop": easy["base"] - easy["tuned"],
             "easy_n": n_easy, "verdict": verdict(gain, easy["base"] - easy["tuned"], n_easy)}
        if "tuned_test_0" in solved:
            r["test_gain"] = solved["tuned_test_0"] - BASE_A0_TEST
            r["confirmed"] = r["test_gain"] >= CONFIRM
        res["readings"]["code"] = r
    for d in EVAL_SIZE:
        if have(d, f"{d}_medium", f"{d}_easy"):
            gain = mean("tuned", d) - mean("base", d)
            drop = mean("base", f"{d}_easy") - mean("tuned", f"{d}_easy")
            n_easy = len(problems.eval[f"{d}_easy"])
            res["readings"][d] = {"hard": {"tuned": mean("tuned", d), "base": mean("base", d), "gain": gain,
                                           "base_runs": [solved[f"base_{d}_{s}"] for s in EVAL_SEEDS]},
                                  "medium": {"tuned": mean("tuned", f"{d}_medium"), "base": mean("base", f"{d}_medium"),
                                             "n": len(problems.eval[f"{d}_medium"])},
                                  "easy": {"tuned": mean("tuned", f"{d}_easy"), "base": mean("base", f"{d}_easy"),
                                           "drop": drop, "n": n_easy},
                                  "verdict": verdict(gain, drop, n_easy)}
    write_json(ROOT / "results" / "rive_rl.json", res)
    log(f"report {json.dumps({k: v for k, v in res.items() if k != 'train_pass_rate'})}")


STEPS_ = {"run": run_step, "smoke": smoke_step, "eval": eval_step, "report": report_step}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=STEPS_)
    p.add_argument("--iters", type=int, default=ITERS)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    STEPS_[args.step](args)


if __name__ == "__main__":
    main()
