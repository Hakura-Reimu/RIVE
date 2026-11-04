"""C1 harness check (docs/phase-c-design.md section 8), no GPU needed: is the Phase C path the same as Phase 0's?

1. Prompts: the initial workflow sends task["prompt"] as its only (user) message. For every development item it
   must equal the messages Phase 0's baseline_eval.py built, with no extra chat-template arguments, and the
   rendered prompts must add up to the prompt tokens the server counted in the C1 check.
2. Judges: Phase 0's stored outputs, judged again with rive.benchmark.hard.judge, must get Phase 0's verdicts.

If both hold, differences between the C1 check and Phase 0 come from generation (the vLLM server instead of the
offline engine, different batches), not from the harness.

  python scripts/c1_harness_check.py
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE0_CODE = ROOT.parent / "phase0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PHASE0_CODE))

import baseline_eval  # noqa: E402  (Phase 0's prompt builders)
from transformers import AutoTokenizer  # noqa: E402

from rive.benchmark.hard import judge, load_tasks  # noqa: E402

PHASE0 = Path(os.path.expanduser("~/alter/runs/phase0/baseline"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/alter/models/Qwen3.5-4B"))
    p.add_argument("--c1-check", default=str(ROOT / "results" / "c1_baseline_check.json"))
    p.add_argument("--out", default=str(ROOT / "results" / "c1_harness_check.json"))
    args = p.parse_args()

    tasks = load_tasks(("math", "code", "rtl"), ("dev",))
    tok = AutoTokenizer.from_pretrained(args.model)
    phase0 = {}
    for source in {t["source"] for t in tasks}:
        for line in open(PHASE0 / f"items_{source}.jsonl"):
            r = json.loads(line)
            phase0[r["id"]] = r

    prompt_mismatch, prompt_tokens = [], 0
    for t in tasks:
        messages, extra = baseline_eval.SPECS[t["source"]][0](t["item"])
        if messages != [{"role": "user", "content": t["public"]["prompt"]}] or extra:
            prompt_mismatch.append(t["id"])
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_tokens += len(tok(text, add_special_tokens=False)["input_ids"])
    server_prompt_tokens = json.loads(Path(args.c1_check).read_text())["llm_usage"]["prompt_tokens"]

    # Maths judging uses signal.alarm, so it runs on the main thread; the sandboxed judges run in parallel.
    verdicts = {t["id"]: judge(t, phase0[t["id"]]["output"])[0] for t in tasks if t["domain"] == "math"}
    others = [t for t in tasks if t["domain"] != "math"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for t, (ok, _, _) in zip(others, pool.map(lambda t: judge(t, phase0[t["id"]]["output"]), others)):
            verdicts[t["id"]] = ok

    by_domain = {}
    for domain in ("math", "code", "rtl"):
        ids = [t["id"] for t in tasks if t["domain"] == domain]
        differ = [i for i in ids if bool(verdicts[i]) != bool(phase0[i]["correct"])]
        by_domain[domain] = {"n": len(ids), "prompt_mismatches": [i for i in prompt_mismatch if i in ids],
                             "rejudged_same_verdict": len(ids) - len(differ),
                             "rejudged_differ": {i: {"phase0": phase0[i]["correct"], "now": verdicts[i]}
                                                 for i in differ}}
    out = {"rendered_prompt_tokens": prompt_tokens, "server_prompt_tokens": server_prompt_tokens,
           "by_domain": by_domain}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
