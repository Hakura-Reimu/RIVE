"""Held-out check of the 6.13 weight edit (docs/rive-design.md section 6.14). The edit e1 was found with outputs of
T2's development pack and judged on the same pack. Here, on T2's selection pack (249 tasks), with the FP8 server:
  - thinking at 8192 tokens with the original model (s_t0) and with e1 (s_e1);
  - from the same outputs, the thinking budget of 6.13: cut at 6144 tokens, Qwen's forcing words, at most 2048 tokens
    of answer (s_tb, s_e1b); for e1 also a 16384-token cap (cut outputs continued: s_e1_16) and the budget within it
    (14336 + 2048: s_e1_16b). The first n tokens of an output, or a cut output continued, are what a call with that
    cap would have given, so these arms share their samples and differ only where the cap matters;
  - without thinking, the model card's sampling (6.12's a0_card), with either model (s_nt, s_nt_e1).
  python scripts/rive_think_check.py run s_t0 s_tb s_nt                     # with the original model served
  python scripts/rive_think_check.py run s_e1 s_e1b s_e1_16 s_e1_16b s_nt_e1  # with e1 served
  python scripts/rive_think_check.py report
  python scripts/rive_think_check.py smoke    # every run on 2 tasks with small caps, any model, in <out-dir>/smoke
Writes <out-dir>/<run>.json (default ~/alter/rive/thinkcheck) and results/rive_think_check.json."""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rive_cutoff_diag as diag  # noqa: E402
import rive_think_fix as fix  # noqa: E402

# name: (kind, source run, model directory suffix, workers). Kinds: "think" a call at CAP; "budget" the source cut at
# `budget` tokens and the answer forced; "longer" the source's cut outputs continued to LONGER; "plain" no thinking.
# Workers for the long ones: the KV cache holds about 116k tokens, 7 requests of 16k.
RUNS = {"s_t0": ("think", None, "", 12), "s_tb": ("budget", "s_t0", "", 12), "s_nt": ("plain", None, "", 12),
        "s_e1": ("think", None, "-nowait-e1", 12), "s_e1b": ("budget", "s_e1", "-nowait-e1", 12),
        "s_e1_16": ("longer", "s_e1", "-nowait-e1", 7), "s_e1_16b": ("budget", "s_e1_16", "-nowait-e1", 7),
        "s_nt_e1": ("plain", None, "-nowait-e1", 12)}
CAPS = {"cap": 8192, "longer": 16384, "answer": 2048, "plain": 4096}
SMOKE_CAPS = {"cap": 256, "longer": 512, "answer": 64, "plain": 256}
THINKING = ("s_t0", "s_tb", "s_e1", "s_e1b", "s_e1_16", "s_e1_16b")
PLAIN = ("s_nt", "s_nt_e1")
MARGIN = 8   # independent runs on about 250 tasks, as in T2 (6.11)
PAIRED = 5   # an arm against the run whose outputs it reuses: only the tasks where the cap matters can differ
NON_THINKING_GREEDY_FP8 = 117  # FP8, no thinking, temperature 0, this pack (6.12, FP8 check): reported only


def total_cap(caps, source):
    return caps["longer"] if source and RUNS[source][0] == "longer" else caps["cap"]


def prompt_of(tokenizer, task):
    return tokenizer.apply_chat_template([{"role": "user", "content": task["public"]["prompt"]}],
                                         add_generation_prompt=True, enable_thinking=True, tokenize=False)


def finish(row, text, tokens, cut, how):
    return {**row, "text": text, "tokens": tokens, "cut": cut, "closed_thinking": "</think>" in text,
            "repeated": round(diag.repeated(text), 3), "derived": how}


def complete(client, prompt, max_tokens):
    sampling, extra = fix.split(fix.SAMPLING)
    r = client.completions.create(model="qwen3.5-4b", prompt=prompt, max_tokens=max_tokens, seed=0,
                                  extra_body=extra, **sampling)
    return r.choices[0].text, r.usage.completion_tokens, r.choices[0].finish_reason == "length"


def budget(client, tokenizer, task, row, caps, total):
    """What a call with the thinking budget (total - answer tokens of thinking, then the forced answer) gives."""
    thinking = total - caps["answer"]
    if row["tokens"] <= thinking:
        return {**row, "forced": False, "derived": "same"}
    head = tokenizer.decode(tokenizer.encode(row["text"], add_special_tokens=False)[:thinking])
    if "</think>" in head:  # the answer had begun: such a call is cut in the answer
        return {**finish(row, head, thinking, True, "cut_in_answer"), "forced": False, "correct": None}
    answer, tokens, cut = complete(client, prompt_of(tokenizer, task) + head + fix.FORCE, caps["answer"])
    return {**finish(row, head + fix.FORCE + answer, thinking + tokens, cut, "forced"), "forced": True,
            "correct": None}


def longer(client, tokenizer, task, row, caps):
    if not row["cut"]:
        return {**row, "derived": "same"}
    more, tokens, cut = complete(client, prompt_of(tokenizer, task) + row["text"], caps["longer"] - row["tokens"])
    return {**finish(row, row["text"] + more, row["tokens"] + tokens, cut, "continued"), "correct": None}


def run(name, tasks, out, smoke=False):
    from openai import OpenAI
    from transformers import AutoTokenizer
    kind, source, suffix, workers = RUNS[name]
    caps = SMOKE_CAPS if smoke else CAPS
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", timeout=1800)
    served = client.models.list().data[0].root
    if not smoke:
        assert served == str(fix.MODEL) + suffix, (name, served)
    tokenizer = AutoTokenizer.from_pretrained(fix.MODEL)
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if kind == "think":
            rows = list(pool.map(lambda t: fix.generate(client, tokenizer, t, (caps["cap"], None, workers, 0, False),
                                                        {}), tasks))
        elif kind == "plain":
            rows = list(pool.map(lambda t: {**diag.generate(client, t, False, caps["plain"], diag.CARD_INSTRUCT),
                                            "forced": False}, tasks))
        else:
            src = {r["id"]: r for r in json.loads((out / f"{source}.json").read_text())}
            if kind == "budget":
                total = total_cap(caps, source)
                rows = list(pool.map(lambda t: budget(client, tokenizer, t, src[t["id"]], caps, total), tasks))
            else:
                rows = list(pool.map(lambda t: longer(client, tokenizer, t, src[t["id"]], caps), tasks))
    # The judge after all calls and on fewer threads, as in T2; outputs taken over unchanged keep their judgment.
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda p: p[1] if p[1].get("correct") is not None and "derived" in p[1]
                             else diag.judge(p[0], p[1], kind != "plain"), zip(tasks, rows)))
    (out / f"{name}.json").write_text(json.dumps(rows))
    s = summary(rows)
    print(f"{time.strftime('%F %T')} run {name} ({served}): {time.time() - start:.0f} s, {json.dumps(s)}", flush=True)


def summary(rows):
    s = fix.summary(rows)
    s["loops_when_cut"] = sum(r["cut"] and r["repeated"] >= diag.LOOP for r in rows)
    if any("derived" in r for r in rows):
        s["changed"] = sum(r["derived"] != "same" for r in rows)
    return s


def paired(rows, source_rows):
    """Tasks solved only by the arm and only by the run it takes its outputs from."""
    a, b = ({r["id"] for r in x if r["correct"]} for x in (rows, source_rows))
    return {"gained": len(a - b), "lost": len(b - a)}


def report(out):
    rows = {n: json.loads((out / f"{n}.json").read_text()) for n in RUNS if (out / f"{n}.json").exists()}
    s = {n: summary(r) for n, r in rows.items()}
    res = {"runs": s}
    for n in rows:
        if RUNS[n][1] in rows:
            s[n]["vs_source"] = paired(rows[n], rows[RUNS[n][1]])
    if "s_t0" in s and "s_e1" in s:
        t0, e1 = s["s_t0"], s["s_e1"]
        fewer_cuts, more = e1["cut"] <= t0["cut"] / 2, e1["solved"] >= t0["solved"] + MARGIN
        res["edit_holds"] = "holds" if fewer_cuts and more else "only stops it earlier" if fewer_cuts else "does not hold"
    if "s_nt" in s and "s_nt_e1" in s:
        d = s["s_nt_e1"]["solved"] - s["s_nt"]["solved"]
        res["non_thinking_with_e1"] = {"difference": d, "reading": "harmed" if d <= -MARGIN else "better"
                                       if d >= MARGIN else "no difference shown"}
    if "s_e1" in s:
        res["combinations"] = {n: s[n]["solved"] >= s["s_e1"]["solved"] + PAIRED
                               for n in ("s_e1b", "s_e1_16", "s_e1_16b") if n in s}
    edits, plain_arms = [n for n in THINKING if n.startswith("s_e1") and n in s], [n for n in ("s_t0", "s_tb") if n in s]
    if edits and plain_arms:
        e, b = (max(x, key=lambda n: (s[n]["solved"], -s[n]["mean_tokens"])) for x in (edits, plain_arms))
        res["edit_vs_no_edit"] = {"best_edit": e, "best_without": b,
                                  "worth_it": s[e]["solved"] >= s[b]["solved"] + MARGIN}
    think, plain = [n for n in THINKING if n in s], [n for n in PLAIN if n in s]
    if think and plain:
        t, p = (max(x, key=lambda n: (s[n]["solved"], -s[n]["mean_tokens"])) for x in (think, plain))
        res["thinking_for_t2"] = {"best_thinking": t, "best_non_thinking": p, "worth_it":
                                  s[t]["solved"] >= s[p]["solved"] + MARGIN,
                                  "non_thinking_greedy_fp8": NON_THINKING_GREEDY_FP8,
                                  "token_ratio": round(s[t]["mean_tokens"] / s[p]["mean_tokens"], 1)}
    if out.name != "smoke":
        (ROOT / "results/rive_think_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print("report", json.dumps(res), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("run", "report", "smoke"))
    p.add_argument("names", nargs="*")
    p.add_argument("--out-dir", default=str(Path.home() / "alter/rive/thinkcheck"))
    args = p.parse_args()
    tasks = json.loads(diag.PACKS.read_text())["select"]
    out = Path(args.out_dir) / ("smoke" if args.step == "smoke" else "")
    out.mkdir(parents=True, exist_ok=True)
    if args.step == "run":
        for name in args.names:
            if not (out / f"{name}.json").exists():
                run(name, tasks, out)
    elif args.step == "smoke":
        for name in args.names or RUNS:
            run(name, tasks[:2], out, smoke=True)
    report(out)


if __name__ == "__main__":
    main()
