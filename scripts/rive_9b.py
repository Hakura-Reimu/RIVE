"""Qwen3.5-9B with a 64k context (docs/rive-design.md section 6.23): generation, judging, the server probe, and
6.13's e1 weight edit on the 9B.

  bash scripts/serve9b.sh [options] &            # the server
  python scripts/rive_9b.py probe [--tag NAME]   # long-context check, throughput, a few thinking answers
  python scripts/rive_9b.py run LABEL ...        # runs of RUNS on the development pack, against the served model
  python scripts/rive_9b.py direction [--limit N]   # needs the GPU, so with the server stopped
  python scripts/rive_9b.py edit [e1|e1h]        # writes ~/alter/models/Qwen3.5-9B-nowait-e1l8 (or -e1l8h, 6.24)
  python scripts/rive_9b.py talks                # does the served model answer at all (4 short answers)
  python scripts/rive_9b.py report               # results/9b/rive_9b.json
The evaluations of one edit after the direction and the edit: scripts/rive_9b_chain.sh [e1|e1h].

Answers are judged like T2's (scripts/rive_cutoff_diag.py: the thinking is removed before judging).
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import rive_cutoff_diag as diag  # noqa: E402
import rive_teach  # noqa: E402

HOME = Path.home()
MODEL = HOME / "alter/models/Qwen3.5-9B"
SERVED = "qwen3.5-9b"
URL = "http://127.0.0.1:8000"
OUT = HOME / "alter/rive/9b"
CONTEXT = 65536
FORCE = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly "
         "now.\n</think>\n\n")  # Qwen's wording for a thinking budget (section 6.13)
_tokenizer = None
_tokenizer_lock = threading.Lock()


def tokenizer():
    """Loaded once; transformers' lazy imports fail when several threads import it at the same time."""
    global _tokenizer
    with _tokenizer_lock:
        if _tokenizer is None:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(MODEL)
    return _tokenizer


def client():
    from openai import OpenAI
    return OpenAI(base_url=f"{URL}/v1", api_key="EMPTY", timeout=3600)


def split(sampling):
    sampling = dict(sampling)
    return sampling, {k: sampling.pop(k) for k in ("top_k", "min_p") if k in sampling}


def generate(cl, task, model=SERVED, thinking=True, cap=16384, answer_cap=4096, seed=0):
    """One answer. With thinking, a thinking that reaches `cap` is closed with FORCE and answered in `answer_cap`."""
    sampling, extra = split(diag.CARD_THINK_CODE if thinking else diag.CARD_INSTRUCT)
    start = time.perf_counter()
    messages = [{"role": "user", "content": task["public"]["prompt"]}]
    r = cl.chat.completions.create(model=model, seed=seed, max_tokens=cap, messages=messages, **sampling,
                                   extra_body={"chat_template_kwargs": {"enable_thinking": thinking}, **extra})
    text, tokens = r.choices[0].message.content or "", r.usage.completion_tokens
    cut = r.choices[0].finish_reason == "length"
    forced = thinking and bool(answer_cap) and cut and "</think>" not in text
    if forced:
        prompt = tokenizer().apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True,
                                                 tokenize=False)
        r = cl.completions.create(model=model, prompt=prompt + text + FORCE, max_tokens=answer_cap, seed=seed,
                                  extra_body=extra, **sampling)
        text, tokens = text + FORCE + r.choices[0].text, tokens + r.usage.completion_tokens
        cut = r.choices[0].finish_reason == "length"
    return {"id": task["id"], "tokens": tokens, "cut": cut, "forced": forced, "closed_thinking": "</think>" in text,
            "seconds": round(time.perf_counter() - start, 1), "text": text}


def run(tasks, model=SERVED, thinking=True, cap=16384, answer_cap=4096, seed=0, workers=16):
    cl = client()
    if thinking:
        tokenizer()  # before the threads start
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda t: generate(cl, t, model, thinking, cap, answer_cap, seed), tasks))
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(lambda p: diag.judge(p[0], p[1], thinking), zip(tasks, rows)))


def summary(rows):
    fails = {}
    for r in rows:
        if not r["correct"]:
            key = "cut" if r["cut"] else r["failure"] or "failed"
            fails[key] = fails.get(key, 0) + 1
    return {"tasks": len(rows), "solved": sum(r["correct"] for r in rows), "cut": sum(r["cut"] for r in rows),
            "forced": sum(r.get("forced", False) for r in rows),
            "mean_tokens": round(sum(r["tokens"] for r in rows) / max(1, len(rows))), "failures": fails}


def kv_tokens(log):
    """The KV cache size the server reported at start ("GPU KV cache size: N tokens")."""
    hits = re.findall(r"GPU KV cache size: ([\d,]+) tokens", Path(log).read_text(errors="replace"))
    return int(hits[-1].replace(",", "")) if hits else None


def long_prompt(n_tokens):
    """About n_tokens of LiveCodeBench statements, a fact hidden in the middle, and a question about it."""
    tasks = rive_teach.packs()["dev"]
    fact = "\n\nNOTE FOR THE READER: the secret code word of this document is MARMALADE-47.\n\n"
    parts, tok, used = [], tokenizer(), 0
    i = 0
    while used < n_tokens:
        text = tasks[i % len(tasks)]["public"]["prompt"]
        parts.append(text)
        used += len(tok(text, add_special_tokens=False)["input_ids"])
        i += 1
    parts.insert(len(parts) // 2, fact)
    return "\n\n---\n\n".join(parts) + "\n\nWhat is the secret code word of this document? Answer with it only."


def probe(args):
    out = OUT / "probe"
    out.mkdir(parents=True, exist_ok=True)
    cl, res = client(), {"tag": args.tag, "kv_cache_tokens": kv_tokens(args.log) if args.log else None}
    prompt = long_prompt(60000)
    start = time.perf_counter()
    r = cl.chat.completions.create(model=SERVED, max_tokens=64, temperature=0.0, messages=[
        {"role": "user", "content": prompt}], extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    res["long"] = {"prompt_tokens": r.usage.prompt_tokens, "seconds": round(time.perf_counter() - start, 1),
                   "answer": (r.choices[0].message.content or "")[:80],
                   "found": "MARMALADE-47" in (r.choices[0].message.content or "")}
    print(json.dumps(res["long"]), flush=True)
    tasks = rive_teach.packs()["dev"]
    for name, n, thinking, cap in (("plain", 16, False, 2048), ("think", 8, True, 16384)):
        start = time.perf_counter()
        rows = run(tasks[:n], thinking=thinking, cap=cap, answer_cap=4096, seed=0, workers=n)
        seconds = time.perf_counter() - start
        res[name] = {**summary(rows), "seconds": round(seconds), "tokens_per_s": round(sum(
            x["tokens"] for x in rows) / seconds)}
        print(name, json.dumps(res[name]), flush=True)
    (out / f"{args.tag}.json").write_text(json.dumps(res, indent=1))


# 6.13's e1 on the 9B, written by `edit`. Layers 8-31 only: along the direction the residual stream of layers 0-7
# holds an offset shared by every position, and with it taken out too (all 32 layers) every answer ended at once.
EDIT_LAYERS = range(8, 32)
EDITS = {"e1": (1.0, "e1l8"), "e1h": (0.5, "e1l8h")}  # name: (share taken out, copy's suffix); 6.23, 6.24
MODELS = {"base": MODEL, **{n: MODEL.parent / f"{MODEL.name}-nowait-{s}" for n, (_, s) in EDITS.items()}}
RUNS = {  # label: (model, thinking, cap, answer cap, seed); all on the development pack
    "plain_s0": ("base", False, 4096, 0, 0),
    "think16k_s0": ("base", True, 16384, 4096, 0),
    "think16k_s1": ("base", True, 16384, 4096, 1),
}
for _e in EDITS:
    RUNS.update({f"{_e}_plain_s0": (_e, False, 4096, 0, 0), f"{_e}_think16k_s0": (_e, True, 16384, 4096, 0),
                 f"{_e}_think16k_s1": (_e, True, 16384, 4096, 1)})
SOURCE = "think16k_s0"  # the outputs the direction comes from
BAND = range(8, 21)     # e1's band of layers (6.13)
BETTER, CONFIRM = 5, 3


def runs(args):
    """Every answer is kept for the analysis. Stops if the server has another copy of the weights than the run needs."""
    out = OUT / "runs"
    out.mkdir(parents=True, exist_ok=True)
    served = Path(client().models.list().data[0].root)
    for label in args.names:
        path = out / f"{label}.json"
        if path.exists():
            continue
        which, thinking, cap, answer_cap, seed = RUNS[label]
        if served != MODELS[which]:
            sys.exit(f"{label} needs {MODELS[which]}, the server has {served}")
        start = time.perf_counter()
        rows = run(rive_teach.packs()["dev"], thinking=thinking, cap=cap, answer_cap=answer_cap, seed=seed, workers=8)
        path.write_text(json.dumps(rows))
        print(f"{time.strftime('%F %T')} {label}: {json.dumps(summary(rows))}, {time.perf_counter() - start:.0f} s",
              flush=True)


def direction(args):
    """6.13's e1 direction from the 9B's own thinking (rive_think_fix.positions and .auc). The forward passes are in
    bf16; the layers that do not fit on the card stay in host memory and are moved to it when used. The residual
    stream after each layer is taken with hooks at the chosen positions only."""
    import torch
    from transformers import AutoModelForImageTextToText
    import rive_think_fix as fix
    out = OUT / ("edit/smoke" if args.limit else "edit")
    out.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto", max_memory={
        0: args.gpu_memory, "cpu": "22GiB"}).eval()
    layers = next(m for n, m in model.named_modules() if n.endswith("language_model.layers"))
    grab = {}

    def hook(layer):
        def keep(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            grab[layer] = h[0, grab["at"]].float().cpu()
        return keep
    handles = [layers[l].register_forward_hook(hook(l)) for l in fix.LAYERS]
    tasks = {t["id"]: t for t in rive_teach.packs()["dev"]}
    half_a = set(sorted(tasks)[::2])  # directions from half A, layers chosen by AUC on half B
    states = {h: {k: [] for k in ("recheck", "other")} for h in ("a", "b")}
    rows = json.loads((OUT / "runs" / f"{SOURCE}.json").read_text())
    for row in rows[:args.limit or None]:
        ids, pos = fix.positions(tok, tasks[row["id"]], row["text"].split(FORCE)[0])
        if not pos["recheck"] or not pos["other"]:
            continue
        grab["at"] = pos["recheck"] + pos["other"]
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1)
        x = torch.stack([grab[l] for l in fix.LAYERS], 1)  # (positions, layers, hidden)
        h = "a" if row["id"] in half_a else "b"
        states[h]["recheck"].append(x[:len(pos["recheck"])])
        states[h]["other"].append(x[len(pos["recheck"]):])
    for handle in handles:
        handle.remove()
    st = {h: {k: torch.cat(v) for k, v in d.items()} for h, d in states.items()}

    def mean_diff(sets, j):
        d = torch.cat([s["recheck"][:, j] for s in sets]).mean(0) - torch.cat([s["other"][:, j] for s in sets]).mean(0)
        return d / d.norm()
    aucs = {}
    for j, layer in enumerate(fix.LAYERS):
        d = mean_diff([st["a"]], j)
        aucs[layer] = round(fix.auc(st["b"]["recheck"][:, j] @ d, st["b"]["other"][:, j] @ d), 4)
    layer = max(BAND, key=lambda l: (aucs[l], -l))
    torch.save(mean_diff([st["a"], st["b"]], fix.LAYERS.index(layer)), out / "direction_e1.pt")
    info = {"source": SOURCE, "aucs": aucs, "positions": {h: {k: len(v) for k, v in s.items()} for h, s in st.items()},
            "e1": {"layer": layer, "auc": aucs[layer]}, "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (out / "direction.json").write_text(json.dumps(info, indent=1))
    print(f"{time.strftime('%F %T')} direction: {json.dumps(info)}", flush=True)


def edit(args):
    import rive_think_fix as fix
    for name in args.names or ["e1"]:
        alpha, suffix = EDITS[name]
        fix.edit("e1", OUT / "edit", model=MODEL, layers=EDIT_LAYERS, suffix=suffix, alpha=alpha)


def talks(args):
    """For the chain, before the runs: the served model answers 4 tasks without thinking in at most 256 tokens, and
    the chain stops unless every answer has at least 32 tokens."""
    cl = client()
    tokens = [generate(cl, t, thinking=False, cap=256, answer_cap=0)["tokens"] for t in rive_teach.packs()["dev"][:4]]
    print(f"{time.strftime('%F %T')} talks: tokens {tokens}", flush=True)
    if min(tokens) < 32:
        sys.exit("the served model does not talk")


def answer(text):
    return text.split("</think>", 1)[-1] if "</think>" in text else text


def details(rows):
    """The only-reported counts of the readings (section 6.23)."""
    import rive_think_fix as fix
    thinking = [r["text"].split(FORCE)[0].split("</think>")[0] for r in rows]
    lines = sum(sum(map(fix.recheck, t.split("\n"))) for t in thinking)
    return {"closed_by_itself": sum(r["closed_thinking"] and not r["forced"] for r in rows),
            "recheck_lines_per_1k": round(1000 * lines / max(1, sum(r["tokens"] for r in rows)), 2),
            "looping_cut": sum(r["cut"] and diag.repeated(r["text"]) >= 0.5 for r in rows),
            "looping_forced_thinking": sum(r["forced"] and diag.repeated(t) >= 0.5 for r, t in zip(rows, thinking)),
            "wait_comment_lines": sum(l.strip().startswith("# Wait") for r in rows for l in answer(r["text"]).split("\n"))}


def readings(s):
    """Per edit: reading 1 (6.23 for e1: the forced closings halve and 5 more solved; 6.24 for e1h: 5 more solved),
    reading 2 (seed 1: 3 more solved) and reading 3 (no thinking: 5 fewer or more)."""
    res = {}
    b = s.get("think16k_s0")
    for name in EDITS:
        e = s.get(f"{name}_think16k_s0")
        if b and e:
            more = e["solved"] >= b["solved"] + BETTER
            if name == "e1":
                halved = e["forced"] <= b["forced"] / 2
                res[name] = "works" if halved and more else "only stops it earlier" if halved else "no use"
            else:
                res[name] = "works" if more else "no use"
            if "think16k_s1" in s and f"{name}_think16k_s1" in s:
                res[f"{name}_confirmed"] = s[f"{name}_think16k_s1"]["solved"] >= s["think16k_s1"]["solved"] + CONFIRM
        if "plain_s0" in s and f"{name}_plain_s0" in s:
            d = s[f"{name}_plain_s0"]["solved"] - s["plain_s0"]["solved"]
            res[f"{name}_non_thinking"] = "harmed" if d <= -BETTER else "helped" if d >= BETTER else "no clear change"
    return res


def flips(a, b):
    """Tasks solved by `a` only and by `b` only."""
    ok = {r["id"]: r["correct"] for r in b}
    return {"first_only": sum(r["correct"] and not ok[r["id"]] for r in a),
            "second_only": sum(not r["correct"] and ok[r["id"]] for r in a)}


def report(args):
    out = OUT / "runs"
    rows = {n: json.loads((out / f"{n}.json").read_text()) for n in RUNS if (out / f"{n}.json").exists()}
    s = {n: {**summary(r), **details(r)} for n, r in rows.items()}
    res = {"runs": s, "readings": readings(s), "flips": {}}
    pairs = [(f"{e}_{run}", run) for e in EDITS for run in ("think16k_s0", "plain_s0", "think16k_s1")]
    for a, b in pairs + [("e1h_think16k_s0", "e1_think16k_s0")]:
        if a in rows and b in rows:
            res["flips"][f"{a} vs {b}"] = flips(rows[a], rows[b])
    if (OUT / "edit/direction.json").exists():
        res["direction"] = json.loads((OUT / "edit/direction.json").read_text())
    (ROOT / "results/9b").mkdir(exist_ok=True)
    (ROOT / "results/9b/rive_9b.json").write_text(json.dumps(res, indent=1) + "\n")
    print("report", json.dumps(res), flush=True)


def works(args):
    """For the chain: "yes" if the edit named works on seed 0 (reading 1), so the confirmation runs."""
    name = args.names[0]
    s = {n: summary(json.loads((OUT / "runs" / f"{n}.json").read_text())) for n in ("think16k_s0", f"{name}_think16k_s0")}
    print("yes" if readings(s).get(name) == "works" else "no")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["probe", "run", "direction", "edit", "talks", "report", "works"])
    p.add_argument("names", nargs="*", help="run: labels of RUNS; edit, works: names of EDITS")
    p.add_argument("--tag", default="mxfp4")
    p.add_argument("--log", help="the server's log, for the KV cache size")
    p.add_argument("--gpu-memory", default="6GiB", help="direction: the weights kept on the card")
    p.add_argument("--limit", type=int, help="direction: the first N outputs only (a smoke test)")
    args = p.parse_args()
    {"probe": probe, "run": runs, "direction": direction, "edit": edit, "talks": talks, "report": report,
     "works": works}[args.step](args)


if __name__ == "__main__":
    main()
