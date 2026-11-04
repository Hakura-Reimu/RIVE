"""Thinking that does not finish (docs/rive-design.md section 6.13): in the cut-off diagnosis (6.12) about 80% of the
answers with thinking were still thinking at 8192 tokens, re-checking themselves ("Wait, ..."). Ways out, each
against plain thinking on T2's development pack, with the FP8 server (scripts/serve.sh):
  - weight edits: the difference-of-means direction between the states just before a "Wait" line and before other
    lines, taken out of what every layer writes into the residual stream (directional ablation, Arditi et al. 2024);
    one direction from a middle layer, one from a late layer;
  - a ban on the word: the tokens "Wait" and " Wait" get a logit bias of -100;
  - a thinking budget: at 6144 tokens the thinking is closed by force and the model answers in at most 2048 more;
  - a longer cap, 16384 tokens.
  python scripts/rive_think_fix.py run t0 tb tw    # runs against whatever model the server has (see RUNS)
  python scripts/rive_think_fix.py direction        # needs the GPU, so with the server stopped
  python scripts/rive_think_fix.py edit e1 e2       # writes ~/alter/models/Qwen3.5-4B-nowait-<name>
  python scripts/rive_think_fix.py report
  python scripts/rive_think_fix.py smoke t0 tb tw   # 2 tasks per run, small caps, in <out-dir>/smoke
  python scripts/rive_think_fix.py direction --smoke   # 8 outputs of one source
Writes <out-dir>/<run>.json (default ~/alter/rive/thinkfix) and results/rive_think_fix.json."""
import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rive_cutoff_diag as diag  # noqa: E402

MODEL = Path.home() / "alter/models/Qwen3.5-4B"
DIAG = Path.home() / "alter/rive/diag"  # 6.12's bf16 outputs with the thinking kept: where the direction comes from
SOURCES = ("think_greedy", "think_card_code", "think_card_general")
SAMPLING = diag.CARD_THINK_CODE
# name: (cap of the first call, cap of the forced answer or None, workers, seed, ban "Wait"). Workers for t16: the KV
# cache holds about 116k tokens, 7 requests of 16k.
RUNS = {"t0": (8192, None, 12, 0, False), "tb": (6144, 2048, 12, 0, False), "tw": (8192, None, 12, 0, True),
        "e1": (8192, None, 12, 0, False), "e2": (8192, None, 12, 0, False), "t16": (16384, None, 7, 0, False),
        "t0_seed1": (8192, None, 12, 1, False), "e_seed1": (8192, None, 12, 1, False)}
THINKING_ARMS = ("t0", "tb", "tw", "e1", "e2", "t16")
FORCE = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly "
         "now.\n</think>\n\n")  # Qwen's wording for a thinking budget
BANNED = ("Wait", " Wait")
# The direction for each edit: the layer whose direction separates best (AUC on the other half of the tasks) within a
# band of layers. Late layers may only say "the next word is Wait"; the middle band is meant to catch the plan.
EDITS = {"e1": range(8, 21), "e2": range(21, 27)}
LAYERS = range(8, 27)        # the residual stream after these layers
PER_CLASS = 20               # positions of each class taken from one output at most
MAX_TOKENS = 6144            # prompt plus output for the forward passes, truncated
WRITERS = ("linear_attn.out_proj.weight", "self_attn.o_proj.weight", "mlp.down_proj.weight")
BETTER, CONFIRM = 5, 3
NON_THINKING_FP8 = 32        # FP8, no thinking, the model card's sampling, the same 74 tasks (6.12, FP8 check)


def recheck(line):
    return line.startswith("Wait") or line.startswith("But wait")


def other(line):
    return line[:1].isalpha() and not recheck(line) and not line.startswith(("Hmm", "Actually"))


def split(sampling):
    """The OpenAI-API part of the sampling settings and the part vLLM takes in the request body."""
    sampling = dict(sampling)
    return sampling, {k: sampling.pop(k) for k in ("top_k", "min_p")}


def generate(client, tokenizer, task, caps, bias, model="qwen3.5-4b"):
    """`model`: the served name, or a LoRA adapter's name (6.15)."""
    cap, answer_cap, _, seed, _ = caps
    sampling, extra = split(SAMPLING)
    start = time.perf_counter()
    r = client.chat.completions.create(
        model=model, seed=seed, max_tokens=cap, logit_bias=bias or None,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}, **extra},
        messages=[{"role": "user", "content": task["public"]["prompt"]}], **sampling)
    text, tokens, cut = r.choices[0].message.content or "", r.usage.completion_tokens, r.choices[0].finish_reason == "length"
    forced = bool(answer_cap) and cut and "</think>" not in text
    if forced:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": task["public"]["prompt"]}],
                                               add_generation_prompt=True, enable_thinking=True, tokenize=False)
        r = client.completions.create(model=model, prompt=prompt + text + FORCE, max_tokens=answer_cap,
                                      seed=seed, extra_body=extra, **sampling)
        text, tokens = text + FORCE + r.choices[0].text, tokens + r.usage.completion_tokens
        cut = r.choices[0].finish_reason == "length"
    return {"id": task["id"], "tokens": tokens, "cut": cut, "forced": forced,
            "seconds": round(time.perf_counter() - start, 1), "closed_thinking": "</think>" in text,
            "repeated": round(diag.repeated(text), 3), "text": text}


def run(name, tasks, out, smoke=False):
    from openai import OpenAI
    from transformers import AutoTokenizer
    caps = RUNS[name]
    if smoke:
        caps = (256, 128 if caps[1] else None, *caps[2:])
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    bias = {}
    if caps[4]:
        for word in BANNED:
            ids = tokenizer.encode(word, add_special_tokens=False)
            assert len(ids) == 1, (word, ids)
            bias[str(ids[0])] = -100
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", timeout=1800)
    served = client.models.list().data[0].root
    start = time.time()
    with ThreadPoolExecutor(max_workers=caps[2]) as pool:
        rows = list(pool.map(lambda t: generate(client, tokenizer, t, caps, bias), tasks))
    # The judge after all calls and on fewer threads, as in T2 (rive/runtime/workflow.py), so tests keep their time.
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda pair: diag.judge(pair[0], pair[1], True), zip(tasks, rows)))
    (out / f"{name}.json").write_text(json.dumps(rows))
    s = summary(rows)
    print(f"{time.strftime('%F %T')} run {name} ({served}): {time.time() - start:.0f} s, {json.dumps(s)}", flush=True)


def summary(rows):
    cut = [r for r in rows if r["cut"]]
    lines = sum(sum(map(recheck, r["text"].split("</think>")[0].split("\n"))) for r in rows)
    return {"tasks": len(rows), "solved": sum(r["correct"] for r in rows), "cut": len(cut),
            "cut_inside_thinking": sum(not r["closed_thinking"] for r in cut), "forced": sum(r["forced"] for r in rows),
            "mean_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
            "recheck_lines_per_1k": round(1000 * lines / sum(r["tokens"] for r in rows), 2)}


def positions(tokenizer, task, text):
    """Token ids of prompt plus thinking (truncated) and the positions just before a re-check line and before
    another line: the state from which the model chooses how the next line starts."""
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": task["public"]["prompt"]}],
                                           add_generation_prompt=True, enable_thinking=True, tokenize=False)
    thinking = text.split("</think>")[0]
    enc = tokenizer(prompt + thinking, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"][:MAX_TOKENS], enc["offset_mapping"][:MAX_TOKENS]
    starts = {s: i for i, (s, _) in enumerate(offsets)}
    found = {"recheck": [], "other": []}
    at = len(prompt)
    for line in thinking.split("\n"):
        i = starts.get(at)
        if i is not None and i > 0 and at > len(prompt):
            kind = "recheck" if recheck(line) else "other" if other(line) else None
            if kind:
                found[kind].append(i - 1)
        at += len(line) + 1
    # Evenly spread over the output, at most PER_CLASS of each.
    return ids, {k: v[::max(1, len(v) // PER_CLASS)][:PER_CLASS] for k, v in found.items()}


def auc(pos, neg):
    """Probability that a re-check state projects higher than another state (Mann-Whitney)."""
    import torch
    ranks = torch.cat([pos, neg]).argsort().argsort().float() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def direction(tasks, out, smoke=False):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    by_id = {t["id"]: t for t in tasks}
    half_a = set(sorted(by_id)[::2])  # directions from half A, layers chosen by AUC on half B
    states = {h: {k: [] for k in ("recheck", "other")} for h in ("a", "b")}
    for source in SOURCES[:1] if smoke else SOURCES:
        rows = json.loads((DIAG / f"{source}.json").read_text())
        for row in rows[:8] if smoke else rows:
            ids, pos = positions(tokenizer, by_id[row["id"]], row["text"])
            if not pos["recheck"] or not pos["other"]:
                continue
            with torch.inference_mode():
                hidden = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True,
                               logits_to_keep=1).hidden_states
            h = "a" if row["id"] in half_a else "b"
            for kind, p in pos.items():
                # hidden[l + 1] is the residual stream after layer l
                states[h][kind].append(torch.stack([hidden[l + 1][0, p].float().cpu() for l in LAYERS], 1))
            del hidden
    st = {h: {k: torch.cat(v) for k, v in d.items()} for h, d in states.items()}  # (positions, layers, hidden)

    def mean_diff(sets, j):
        d = torch.cat([s["recheck"][:, j] for s in sets]).mean(0) - torch.cat([s["other"][:, j] for s in sets]).mean(0)
        return d / d.norm()
    aucs = {}
    for j, layer in enumerate(LAYERS):
        d = mean_diff([st["a"]], j)
        aucs[layer] = round(auc(st["b"]["recheck"][:, j] @ d, st["b"]["other"][:, j] @ d), 4)
    info = {"aucs": aucs, "positions": {h: {k: len(v) for k, v in s.items()} for h, s in st.items()}, "edits": {}}
    for name, band in EDITS.items():
        layer = max(band, key=lambda l: (aucs[l], -l))
        torch.save(mean_diff([st["a"], st["b"]], LAYERS.index(layer)), out / f"direction_{name}.pt")
        info["edits"][name] = {"layer": layer, "auc": aucs[layer]}
    (out / "direction.json").write_text(json.dumps(info, indent=1))
    print(f"{time.strftime('%F %T')} direction: {json.dumps(info)}", flush=True)


def edit(name, out, model=MODEL, layers=None, suffix=None, alpha=1.0):
    """A copy of the model whose layers no longer write the direction into the residual stream:
    W <- W - r r^T W for every matrix in WRITERS. The embedding is left as it is (tied to the output layer) and so
    is the multi-token-prediction head (vLLM does not use it here). For the 9B (sections 6.23, 6.24): `model`,
    `layers` (the layers edited; all if None), `suffix` (the copy's name, else `name`) and `alpha` (the share of
    the component taken out: W <- W - alpha r r^T W)."""
    import torch
    from safetensors.torch import load_file, save_file
    r = torch.load(out / f"direction_{name}.pt").to("cuda")
    target = model.parent / f"{model.name}-nowait-{suffix or name}"
    target.mkdir(exist_ok=True)
    edited = 0
    for f in model.iterdir():
        if f.is_dir():  # .cache from the download
            continue
        if f.suffix != ".safetensors":
            shutil.copy2(f, target / f.name)
            continue
        tensors = load_file(f)
        for key, w in tensors.items():
            if key.startswith("model.language_model.layers.") and key.endswith(WRITERS) and (
                    layers is None or int(key.split(".")[3]) in layers):
                w32 = w.to("cuda", torch.float32)
                tensors[key] = (w32 - alpha * torch.outer(r, r @ w32)).to(w.dtype).cpu()
                edited += 1
        save_file(tensors, target / f.name, metadata={"format": "pt"})
    print(f"{time.strftime('%F %T')} edit {name}: {edited} matrices, {target}", flush=True)


def best_edit(s):
    edits = [n for n in EDITS if n in s]
    return max(edits, key=lambda n: (s[n]["solved"], -s[n]["cut"])) if edits else None


def report(out):
    rows = {n: json.loads((out / f"{n}.json").read_text()) for n in RUNS if (out / f"{n}.json").exists()}
    s = {n: summary(r) for n, r in rows.items()}
    res = {"runs": s}
    if (out / "direction.json").exists():
        res["direction"] = json.loads((out / "direction.json").read_text())
    if "t0" in s:
        t0 = s["t0"]
        e = best_edit(s)
        if e:
            fewer_cuts, more = s[e]["cut"] <= t0["cut"] / 2, s[e]["solved"] >= t0["solved"] + BETTER
            res["edit"] = {"best": e, "reading": "works" if fewer_cuts and more
                           else "only stops it earlier" if fewer_cuts else "no use"}
            for control in ("tb", "tw"):
                if control in s:
                    res["edit"][f"better_than_{control}"] = s[e]["solved"] >= s[control]["solved"] + BETTER
            if "e_seed1" in s and "t0_seed1" in s:
                res["edit"]["confirmed"] = s["e_seed1"]["solved"] >= s["t0_seed1"]["solved"] + CONFIRM
        if "t16" in s:
            res["longer_cap_helps"] = s["t16"]["solved"] >= t0["solved"] + BETTER
        arms = [n for n in THINKING_ARMS if n in s]
        best = max(arms, key=lambda n: s[n]["solved"])
        res["thinking_for_t2"] = {"best": best, "solved": s[best]["solved"], "non_thinking_fp8": NON_THINKING_FP8,
                                  "worth_it": s[best]["solved"] >= NON_THINKING_FP8 + BETTER}
    if out.name != "smoke":
        (ROOT / "results/rive_think_fix.json").write_text(json.dumps(res, indent=1) + "\n")
    print("report", json.dumps(res), flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("run", "direction", "edit", "report", "smoke", "best-edit"))
    p.add_argument("names", nargs="*")
    p.add_argument("--out-dir", default=str(Path.home() / "alter/rive/thinkfix"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    tasks = json.loads(diag.PACKS.read_text())["dev"]
    out = Path(args.out_dir) / ("smoke" if args.step == "smoke" or args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    if args.step == "best-edit":  # for the chain: the edit to confirm, if the reading is "works"
        s = {n: summary(json.loads((out / f"{n}.json").read_text())) for n in ("t0", *EDITS)}
        e = best_edit(s)
        works = s[e]["cut"] <= s["t0"]["cut"] / 2 and s[e]["solved"] >= s["t0"]["solved"] + BETTER
        print(e if works else "")
        return
    if args.step == "run":
        for name in args.names:
            if not (out / f"{name}.json").exists():
                run(name, tasks, out)
    elif args.step == "smoke":
        for name in args.names or ("t0", "tb", "tw"):
            run(name, tasks[:2], out, smoke=True)
    elif args.step == "direction":
        direction(tasks, out, smoke=out.name == "smoke")
    elif args.step == "edit":
        for name in args.names:
            edit(name, out)
    report(out)


if __name__ == "__main__":
    main()
