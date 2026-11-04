"""Phase D training data (docs/phase-d-design.md section 3, plan B reduced, chosen by the user on 2026-10-30):
download, choose the training tasks, sample answers from the initial model (V0) and keep the correct ones.

  python scripts/phase_d_data.py fetch     # MATH train and LiveCodeBench release v1 into ~/alter/data (network)
  python scripts/phase_d_data.py pool      # 1,000 MATH level 4-5 problems and the 400 LiveCodeBench ones
  python scripts/phase_d_data.py sample    # 4 answers each from qwen3.5-4b on the vLLM server
  python scripts/phase_d_data.py build     # the training file: up to 2 correct answers per task, shortest first

No development, selection, anchor or sealed task is used, and a training task that repeats one of them or an easy
set item is dropped. Every step can be started again after an interruption.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import pickle
import random
import re
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.benchmark.hard import DATA, judge, load_tasks, public_task  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.phase_d import Overlap, last_boxed, pick_examples  # noqa: E402

# Pinned when the plan was written (2026-10-30); fetch refuses a repository that has moved since.
MATH_REPO, MATH_SHA = "EleutherAI/hendrycks_math", "21a56338"
LCB_REPO, LCB_SHA = "livecodebench/code_generation_lite", "0fe84c39"
DOMAIN = {"math_train": "math", "lcb_train": "code"}


def write_jsonl(name, rows, manifest, source, revision, license_, note):
    with open(DATA / f"{name}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest[name] = {"source": source, "revision": revision, "license": license_, "count": len(rows), "note": note}
    print(f"{name}: {len(rows)} rows from {source}@{revision[:10]}", flush=True)


class _NoGlobalsUnpickler(pickle.Unpickler):
    """LiveCodeBench pickles its private tests as a plain JSON string; refuse anything that needs a class
    (as phase0/fetch_data.py)."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def lcb_private_tests(blob):
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        raw = zlib.decompress(base64.b64decode(blob.encode()))
        return json.loads(_NoGlobalsUnpickler(io.BytesIO(raw)).load())


def fetch(args):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    manifest_path = DATA / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    sha = api.dataset_info(MATH_REPO).sha
    if not sha.startswith(MATH_SHA):
        sys.exit(f"{MATH_REPO} is now at {sha}, not {MATH_SHA}: check it before using it")
    rows = []
    files = sorted(f for f in api.list_repo_files(MATH_REPO, repo_type="dataset", revision=sha)
                   if "/train-" in f and f.endswith(".parquet"))
    for f in files:
        subject = f.split("/")[0]
        table = pq.read_table(hf_hub_download(MATH_REPO, f, repo_type="dataset", revision=sha)).to_pylist()
        for i, r in enumerate(table):
            level, answer = re.search(r"\d", str(r.get("level", ""))), last_boxed(r["solution"])
            if level and answer:  # a problem without a level or a boxed reference answer cannot be judged
                rows.append({"id": f"math-train/{subject}/{i}", "problem": r["problem"], "answer": answer,
                             "level": int(level.group()), "subject": subject})
    write_jsonl("math_train", rows, manifest, MATH_REPO, sha, "MIT",
                "MATH train split (the original hendrycks/competition_math is disabled); Phase D training only")

    sha = api.dataset_info(LCB_REPO).sha
    if not sha.startswith(LCB_SHA):
        sys.exit(f"{LCB_REPO} is now at {sha}, not {LCB_SHA}: check it before using it")
    path = hf_hub_download(LCB_REPO, "test.jsonl", repo_type="dataset", revision=sha)  # release v1
    tests_dir = DATA / "lcb_train_tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        tests = json.loads(r["public_test_cases"]) + lcb_private_tests(r["private_test_cases"])
        meta = json.loads(r.get("metadata") or "{}")
        qid = str(r["question_id"])
        (tests_dir / f"{qid}.json").write_text(json.dumps(tests))
        rows.append({"id": f"lcb-train-{qid}", "question_id": qid, "title": r["question_title"],
                     "content": r["question_content"], "platform": r["platform"],
                     "contest_date": r["contest_date"][:10], "difficulty": r["difficulty"],
                     "starter_code": r.get("starter_code") or "", "func_name": meta.get("func_name"),
                     "testtype": tests[0]["testtype"] if tests else "stdin", "n_tests": len(tests),
                     "tests_dir": "lcb_train_tests"})
    dates = sorted(r["contest_date"] for r in rows)
    write_jsonl("lcb_train", rows, manifest, f"{LCB_REPO} (test.jsonl, release v1)", sha,
                "CC (unspecified variant on the dataset card)",
                f"contest dates {dates[0]}..{dates[-1]}; tests in lcb_train_tests/; Phase D training only")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def read(name):
    return [json.loads(line) for line in open(DATA / f"{name}.jsonl")]


def protected_statements():
    """Every evaluation item a training task must not repeat: the four hard splits and the easy sets."""
    out = {}
    for t in load_tasks(("math", "code", "rtl"), ("dev", "selection", "anchor", "sealed")):
        item = t["item"]
        out[t["id"]] = item.get("problem") or item.get("content") or item.get("prompt") or ""
    for source, field in (("gsm8k", "question"), ("math500", "problem"), ("mbpp", "prompt")):
        for item in read(source):
            out[f"{source}:{item['id']}"] = item[field]
    return out


def pool(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    maths = [r for r in read("math_train") if r["level"] >= 4]
    maths = sorted(random.Random(0).sample(maths, min(args.math, len(maths))), key=lambda r: r["id"])
    codes = read("lcb_train")[:args.code]
    overlap = Overlap(protected_statements())
    keep, dropped = [], []
    for source, item in [("math_train", r) for r in maths] + [("lcb_train", r) for r in codes]:
        hit = overlap.match(item.get("problem") or item["content"])
        if hit:
            dropped.append({"id": item["id"], "repeats": hit})
        else:
            keep.append({"id": item["id"], "source": source})
    (out / "pool.json").write_text(json.dumps({"tasks": keep, "dropped": dropped}, indent=1))
    print(f"pool: {len(keep)} tasks, {len(dropped)} dropped as repeats: {dropped[:5]}", flush=True)


def tasks_of(pool_rows):
    items = {r["id"]: r for name in DOMAIN for r in read(name)}
    return [{"id": p["id"], "domain": DOMAIN[p["source"]], "source": p["source"],
             "public": public_task(p["source"], items[p["id"]]), "item": items[p["id"]]} for p in pool_rows]


def sample(args):
    from transformers import AutoTokenizer
    out = Path(args.out_dir)
    tasks = tasks_of(json.loads((out / "pool.json").read_text())["tasks"])
    path = out / "samples.jsonl"
    done = {json.loads(line)["id"] for line in open(path)} if path.exists() else set()
    # Maths and code alternate, so code is judged in the background while the model writes maths answers.
    todo = [t for pair in zip(*[[t for t in tasks if t["domain"] == d and t["id"] not in done]
                                for d in ("math", "code")]) for t in pair]
    paired = {t["id"] for t in todo}
    todo += [t for t in tasks if t["id"] not in done and t["id"] not in paired]
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(args.tokenizer))
    llm = LLM(model=args.model)

    def generate(task):
        seed = int(hashlib.sha256(task["id"].encode()).hexdigest()[:8], 16)
        for attempt in range(3):
            try:
                r = llm.client.chat.completions.create(
                    model=llm.model, messages=[{"role": "user", "content": task["public"]["prompt"]}], n=args.k,
                    max_tokens=4096, temperature=0.7, top_p=0.8, seed=seed,
                    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
                break
            except Exception:  # noqa: BLE001 - a dropped request is asked again; the third failure stops the run
                if attempt == 2:
                    raise
        llm._count(r)
        return [{"text": c.message.content or "", "finish_reason": c.finish_reason} for c in r.choices]

    def judged(task, samples):
        for s in samples:
            s["completion_tokens"] = len(tokenizer(s["text"], add_special_tokens=False)["input_ids"])
            s["correct"], s["failure"], _ = judge(task, s["text"])
        return {"id": task["id"], "source": task["source"], "domain": task["domain"], "samples": samples}

    with ThreadPoolExecutor(args.workers) as gen, ThreadPoolExecutor(args.judges) as judges, open(path, "a") as f:
        pending = []
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start + args.batch]
            for task, samples in zip(batch, gen.map(generate, batch)):
                if task["domain"] == "math":  # math-verify works only on the main thread
                    f.write(json.dumps(judged(task, samples), ensure_ascii=False) + "\n")
                else:
                    pending.append(judges.submit(judged, task, samples))
            for p in [p for p in pending if p.done()]:
                f.write(json.dumps(p.result(), ensure_ascii=False) + "\n")
                pending.remove(p)
            f.flush()
            print(f"sampled {len(done) + start + len(batch)} of {len(tasks)}; usage {llm.usage}", flush=True)
        for p in pending:
            f.write(json.dumps(p.result(), ensure_ascii=False) + "\n")


def build(args):
    out = Path(args.out_dir)
    rows = [json.loads(line) for line in open(out / "samples.jsonl")]
    prompts = {t["id"]: t["public"]["prompt"] for t in tasks_of([{"id": r["id"], "source": r["source"]}
                                                                 for r in rows])}
    examples, solved = [], {}
    for row in rows:
        picked = pick_examples(row["samples"], args.per_task)
        solved.setdefault(row["source"], [0, 0])[0] += bool(picked)
        solved[row["source"]][1] += 1
        examples += [{"id": row["id"], "source": row["source"], "domain": row["domain"], "prompt": prompts[row["id"]],
                      "answer": s["text"], "completion_tokens": s["completion_tokens"]} for s in picked]
    with open(out / "train.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    summary = {"tasks_with_a_correct_answer": solved, "examples": len(examples),
               "answer_tokens": sum(e["completion_tokens"] for e in examples),
               "by_source": {s: sum(e["source"] == s for e in examples) for s in DOMAIN}}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("fetch", "pool", "sample", "build"))
    p.add_argument("--out-dir", default=os.path.expanduser("~/alter/evo/runs/phase-d"))
    p.add_argument("--math", type=int, default=1000, help="pool: MATH level 4-5 problems")
    p.add_argument("--code", type=int, default=400, help="pool: LiveCodeBench release v1 problems")
    p.add_argument("--model", default="qwen3.5-4b", help="sample: the served name of V0")
    p.add_argument("--tokenizer", default="~/alter/models/Qwen3.5-4B")
    p.add_argument("--k", type=int, default=4, help="sample: answers per task")
    p.add_argument("--workers", type=int, default=4, help="sample: requests at a time (each asks for k answers)")
    p.add_argument("--judges", type=int, default=6, help="sample: code judgements at a time")
    p.add_argument("--batch", type=int, default=24, help="sample: tasks between checkpoints")
    p.add_argument("--per-task", type=int, default=2, help="build: answers kept per task")
    args = p.parse_args()
    {"fetch": fetch, "pool": pool, "sample": sample, "build": build}[args.step](args)


if __name__ == "__main__":
    main()
