"""A1: unmodified Qwen3.5-4B on tool-pool tasks (BFCL item + K distractor tools), all tools exposed.

This is the "identity router" baseline that Phase A candidates must beat. Greedy, non-thinking.

  python scripts/a1_toolpool_baseline.py --k 0,8,16,32 --splits dev,selection
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from rive.benchmark.bfcl import as_tools, classify, parse_tool_calls  # noqa: E402
from rive.benchmark.stats import wilson  # noqa: E402
from rive.benchmark.toolpool import build_tasks  # noqa: E402

DATA = Path(os.path.expanduser("~/alter/data"))
CATEGORIES = ["multiple", "parallel_multiple", "irrelevance"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/alter/models/Qwen3.5-4B"))
    p.add_argument("--k", default="0,8,16,32")
    p.add_argument("--splits", default="dev,selection")
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--out", default=os.path.expanduser("~/alter/evo/results/a1"))
    p.add_argument("--report-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir, report_dir = Path(args.out), Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for c in CATEGORIES for l in open(DATA / f"bfcl_{c}.jsonl")]
    splits = tuple(args.splits.split(","))
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.85,
              max_num_seqs=16, limit_mm_per_prompt={"image": 0, "video": 0})
    params = SamplingParams(temperature=0.0, max_tokens=1024)

    summary = {"model": (Path(args.model) / "REVISION").read_text().strip(), "round": args.round,
               "decoding": "greedy, non-thinking", "router": "identity (all tools, shuffled)", "by_k": {}}
    for k in [int(x) for x in args.k.split(",")]:
        tasks = build_tasks(items, k, args.round, splits)
        prompts = [tok.apply_chat_template(t["messages"], tools=as_tools(t["functions"]), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) for t in tasks]
        prompt_tokens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        start = time.perf_counter()
        outs = llm.generate(prompts, params)
        seconds = time.perf_counter() - start

        groups = defaultdict(lambda: [0, 0])
        failures = defaultdict(Counter)
        with open(out_dir / f"items_k{k}.jsonl", "w") as f:
            for t, o in zip(tasks, outs):
                text = o.outputs[0].text
                calls = parse_tool_calls(text, t["functions"])
                failure = classify(text, calls, t["ground_truth"], t["irrelevance"])
                for key in (t["split"], f"{t['split']}/{t['category']}"):
                    groups[key][0] += failure is None
                    groups[key][1] += 1
                if failure:
                    failures[t["split"]][failure] += 1
                f.write(json.dumps({"id": t["id"], "split": t["split"], "category": t["category"],
                                    "failure": failure, "calls": [[n, a] for n, a in calls][:4],
                                    "output": text[-1500:]}, ensure_ascii=False) + "\n")
        summary["by_k"][k] = {
            "mean_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens)),
            "max_prompt_tokens": max(prompt_tokens),
            "seconds": round(seconds, 1),
            "accuracy": {g: {"n": v[1], "acc": round(v[0] / v[1], 3),
                             "ci95": [round(x, 3) for x in wilson(v[0], v[1])]} for g, v in sorted(groups.items())},
            "failures": {s: dict(c.most_common()) for s, c in failures.items()},
        }
        print(f"K={k}", json.dumps(summary["by_k"][k]["accuracy"].get("dev")),
              json.dumps(summary["by_k"][k]["accuracy"].get("selection")), "prompt tokens",
              summary["by_k"][k]["mean_prompt_tokens"], flush=True)

    for d in (out_dir, report_dir):
        (d / "a1_toolpool_baseline.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("a1_toolpool_baseline: OK")


if __name__ == "__main__":
    main()
