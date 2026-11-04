"""Cut-off diagnosis for T2 (docs/rive-design.md section 6.12): why a quarter of A0's answers, and most answers with
thinking, run to the token cap, and whether the sampling settings of Qwen3.5-4B's model card fix that. One model call
per task on T2's development pack: the call the a0 and think scan configurations make, with the full text kept.
  python scripts/rive_cutoff_diag.py run       # the runs not done yet, then the report
  python scripts/rive_cutoff_diag.py report
  python scripts/rive_cutoff_diag.py smoke     # 2 tasks per run, 512 tokens, in <out-dir>/smoke
Writes <out-dir>/<run>.json (default ~/alter/rive/diag) and results/rive_cutoff_diag.json."""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rive.meta_rsi import synth  # noqa: E402
from rive.runtime.workflow import strip_thinking  # noqa: E402

PACKS = Path.home() / "alter/rive/t2/packs.json"
SCAN_A0_DEV = Path.home() / "alter/rive/t2/evals/7e1d34ecb44239a4_dev.json"  # the scan's A0 on the same pack
GREEDY = {"temperature": 0.0}
# The model card's settings: non-thinking for general tasks; thinking for precise coding; thinking for general tasks.
CARD_INSTRUCT = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
CARD_THINK_CODE = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
CARD_THINK_GENERAL = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
# name: (thinking, max_tokens, sampling), in running order.
RUNS = {
    "a0_greedy": (False, 4096, GREEDY),
    "a0_card": (False, 4096, CARD_INSTRUCT),
    "think_greedy": (True, 8192, GREEDY),
    "think_card_code": (True, 8192, CARD_THINK_CODE),
    "think_card_general": (True, 8192, CARD_THINK_GENERAL),
}
PAIRS = (("a0_greedy", "a0_card"), ("think_greedy", "think_card_code"), ("think_greedy", "think_card_general"))
W, STEP = 64, 16
LOOP = 0.5           # repeated share from which an output counts as stuck
MAINLY, RARELY = 0.5, 0.2
SOLVED_SLACK = 3     # about 1.2 standard deviations of the difference of two runs on 74 tasks (section 6.10)
BETTER = 5


def repeated(text):
    """Share of the 64-character windows in the last third of `text` that also occur in the two thirds before it:
    near 1 for an output going round in circles, near 0 for one that keeps writing new things."""
    cut = len(text) * 2 // 3
    if len(text) < 3 * W:
        return 0.0
    seen = {text[i:i + W] for i in range(cut - W + 1)}
    windows = range(cut, len(text) - W + 1, STEP)
    return sum(text[i:i + W] in seen for i in windows) / len(windows)


def generate(client, task, thinking, max_tokens, sampling):
    sampling = dict(sampling)
    extra = {"chat_template_kwargs": {"enable_thinking": thinking}}
    for key in ("top_k", "min_p"):  # not in the OpenAI API; vLLM takes them in the body
        if key in sampling:
            extra[key] = sampling.pop(key)
    start = time.perf_counter()
    r = client.chat.completions.create(model="qwen3.5-4b", seed=0, max_tokens=max_tokens, extra_body=extra,
                                       messages=[{"role": "user", "content": task["public"]["prompt"]}], **sampling)
    text = r.choices[0].message.content or ""
    return {"id": task["id"], "tokens": r.usage.completion_tokens, "cut": r.choices[0].finish_reason == "length",
            "seconds": round(time.perf_counter() - start, 1), "closed_thinking": "</think>" in text,
            "repeated": round(repeated(text), 3), "text": text}


def judge(task, row, thinking):
    ok, failure, _ = synth.judge(task, strip_thinking(row["text"]) if thinking else row["text"])
    return {**row, "correct": bool(ok), "failure": failure}


def run(name, tasks, out, workers, max_tokens=None):
    thinking, cap, sampling = RUNS[name]
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", timeout=900)
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda t: generate(client, t, thinking, max_tokens or cap, sampling), tasks))
    # The judge after all calls and on fewer threads, as in T2 (rive/runtime/workflow.py), so tests keep their time.
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda pair: judge(pair[0], pair[1], thinking), zip(tasks, rows)))
    (out / f"{name}.json").write_text(json.dumps(rows))
    s = summary(rows, thinking)
    print(f"{time.strftime('%F %T')} run {name}: {time.time() - start:.0f} s, {json.dumps(s)}", flush=True)


def summary(rows, thinking):
    cut = [r for r in rows if r["cut"]]
    rest = [r for r in rows if not r["cut"]]
    s = {"tasks": len(rows), "solved": sum(r["correct"] for r in rows), "cut": len(cut),
         "solved_when_cut": sum(r["correct"] for r in cut), "mean_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
         "loops_when_cut": sum(r["repeated"] >= LOOP for r in cut),
         "mean_repeated_cut": round(sum(r["repeated"] for r in cut) / len(cut), 3) if cut else None,
         "mean_repeated_rest": round(sum(r["repeated"] for r in rest) / len(rest), 3) if rest else None}
    if thinking:
        s["cut_inside_thinking"] = sum(not r["closed_thinking"] for r in cut)
    return s


def report(out):
    rows = {n: json.loads((out / f"{n}.json").read_text()) for n in RUNS if (out / f"{n}.json").exists()}
    summaries = {n: summary(r, RUNS[n][0]) for n, r in rows.items()}
    greedy_cut = [x for n in ("a0_greedy", "think_greedy") if n in rows for x in rows[n] if x["cut"]]
    loops = sum(x["repeated"] >= LOOP for x in greedy_cut) / len(greedy_cut) if greedy_cut else None
    pairs = {}
    for greedy, card in PAIRS:
        if greedy in rows and card in rows:
            g, c = summaries[greedy], summaries[card]
            pairs[f"{card}_vs_{greedy}"] = {
                "solved": [g["solved"], c["solved"]], "cut": [g["cut"], c["cut"]],
                "flips": sum(a["correct"] != b["correct"] for a, b in zip(rows[greedy], rows[card])),
                "effective": c["cut"] <= g["cut"] / 2 and c["solved"] >= g["solved"] - SOLVED_SLACK,
                "better": c["solved"] >= g["solved"] + BETTER}
    scan = json.loads(SCAN_A0_DEV.read_text())
    out_json = {"runs": summaries, "pairs": pairs,
                "greedy_cut_loop_share": round(loops, 3) if loops is not None else None,
                "cut_offs_are": None if loops is None else
                ("mainly loops" if loops >= MAINLY else "mainly long reasoning" if loops < RARELY else "both"),
                "scan_a0_dev": {"solved": sum(r["correct"] for r in scan),
                                "cut": sum(r["truncated_calls"] > 0 for r in scan)}}
    if out.name != "smoke":
        (ROOT / "results/rive_cutoff_diag.json").write_text(json.dumps(out_json, indent=1) + "\n")
    print("report", json.dumps(out_json), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("run", "report", "smoke"))
    p.add_argument("--out-dir", default=str(Path.home() / "alter/rive/diag"))
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    tasks = json.loads(PACKS.read_text())["dev"]
    out = Path(args.out_dir)
    if args.step == "smoke":
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    if args.step in ("run", "smoke"):
        for name in RUNS:
            if args.step == "smoke":
                run(name, tasks[:2], out, args.workers, max_tokens=512)
            elif not (out / f"{name}.json").exists():
                run(name, tasks, out, args.workers)
    report(out)


if __name__ == "__main__":
    main()
