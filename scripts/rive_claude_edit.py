"""Claude edits the 4B model's weights for the non-thinking mode (docs/rive-design.md section 6.16).

Claude (Opus 5.5, through rive/infra/claude_gateway.py) works with a fixed set of tools on T2's development pack. It
reads the original model's outputs, defines two kinds of positions in them and gets the difference-of-means
direction of the residual stream between the two (as in 6.13, from states stored beforehand). It builds edits from
directions (W <- W - alpha r r^T W on chosen layers and matrices, as a LoRA adapter, 6.15) and evaluates them: 74
tasks, two seeds, no thinking. It never sees the selection or the test pack, the hidden tests or the judge, and it
runs no code of its own. At the end it names at most two edits, which are checked on the selection pack against
the original model.
  python scripts/rive_claude_edit.py positions    # with the vLLM server stopped (needs the GPU)
  python scripts/rive_claude_edit.py session      # against the LoRA server (scripts/rive_claude_edit_chain.sh)
  python scripts/rive_claude_edit.py verify
  python scripts/rive_claude_edit.py report
  python scripts/rive_claude_edit.py smoke        # 2 Claude turns, no evaluation, in <out-dir>/smoke
Writes <out-dir> (default ~/alter/rive/claude_edit) and results/rive_claude_edit.json."""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import rive_cutoff_diag as diag  # noqa: E402
import rive_think_fix as fix  # noqa: E402
from rive.infra.claude_gateway import BudgetExceeded, ClaudeGateway  # noqa: E402

HOME = Path.home()
# The original model's non-thinking outputs on the development pack (6.12, its FP8 check, 6.15).
SOURCES = {"bf16_greedy": HOME / "alter/rive/diag/a0_greedy.json", "bf16_card": HOME / "alter/rive/diag/a0_card.json",
           "fp8_greedy": HOME / "alter/rive/diag_fp8/a0_greedy.json",
           "fp8_card": HOME / "alter/rive/diag_fp8/a0_card.json",
           "fp8_card_2": HOME / "alter/rive/loracheck/lora_plain.json"}
LAYERS = list(range(4, 29))   # residual stream after these layers is stored
PER_OUTPUT = 80               # line positions kept per output at most, evenly spread
MAX_TOKENS = 6144
PLAIN_CAP, SEEDS = 4096, (0, 1)
MAX_EVALS, MAX_TURNS, SESSION_USD, MAX_RANK = 10, 80, 8.0, 8
WORKERS = 12
BETTER, CONFIRM = 8, 5        # on the selection pack: seed 0, then seed 1 for the best one
KINDS = {"linear_attn.out_proj.weight": "linear_attn", "self_attn.o_proj.weight": "attn", "mlp.down_proj.weight": "mlp"}
NAME = re.compile(r"^[a-z0-9_]{1,32}$")
URL = "http://127.0.0.1:8000"


def rows_of(path):
    d = json.loads(Path(path).read_text())
    return d["rows"] if isinstance(d, dict) else d


def outcome(row):
    return "solved" if row["correct"] else "cut" if row["cut"] else row["failure"] or "failed"


def packs():
    return json.loads(diag.PACKS.read_text())


# ---------------------------------------------------------------- stored states (GPU, server stopped)

def positions(out):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(fix.MODEL)
    tasks = {t["id"]: t for t in packs()["dev"]}
    plan, meta = [], []
    for source, path in SOURCES.items():
        for row in rows_of(path):
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": tasks[row["id"]]["public"]["prompt"]}],
                                                   add_generation_prompt=True, enable_thinking=False, tokenize=False)
            enc = tokenizer(prompt + row["text"], return_offsets_mapping=True, add_special_tokens=False)
            ids, offsets = enc["input_ids"][:MAX_TOKENS], enc["offset_mapping"][:MAX_TOKENS]
            starts = {s: i for i, (s, _) in enumerate(offsets)}
            if len(prompt) not in starts:
                continue
            found, at = [], len(prompt)
            for line in row["text"].split("\n"):
                i = starts.get(at)
                if i is not None and line.strip():
                    found.append((i - 1, line[:80], (at - len(prompt)) / max(1, len(row["text"]))))
                at += len(line) + 1
            found = found[::max(1, len(found) // PER_OUTPUT)][:PER_OUTPUT]
            base = {"source": source, "task": row["id"], "outcome": outcome(row)}
            pos = [starts[len(prompt)] - 1] + [p for p, _, _ in found]
            meta += [{**base, "kind": "prompt_end", "line": "", "frac": 0.0}]
            meta += [{**base, "kind": "line", "line": text, "frac": round(frac, 3)} for _, text, frac in found]
            plan.append((ids, pos))
    n = len(meta)
    arrays = {l: np.lib.format.open_memmap(out / f"layer_{l}.npy", mode="w+", dtype=np.float16, shape=(n, 2560))
              for l in LAYERS}
    model = AutoModelForImageTextToText.from_pretrained(fix.MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    row = 0
    for ids, pos in plan:
        with torch.inference_mode():
            hidden = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True,
                           logits_to_keep=1).hidden_states
        for l in LAYERS:  # hidden[l + 1] is the residual stream after layer l
            arrays[l][row:row + len(pos)] = hidden[l + 1][0, pos].to(torch.float16).cpu().numpy()
        row += len(pos)
        del hidden
    for a in arrays.values():
        a.flush()
    (out / "positions.json").write_text(json.dumps(meta))
    print(f"{time.strftime('%F %T')} positions: {n} from {len(plan)} outputs, layers {LAYERS[0]}-{LAYERS[-1]}",
          flush=True)


# ---------------------------------------------------------------- the tools

class Lab:
    def __init__(self, out, evaluate=True):
        self.out, self.can_evaluate = out, evaluate
        for d in ("directions", "edits", "evals"):
            (out / d).mkdir(parents=True, exist_ok=True)
        self.meta = json.loads((out / "positions.json").read_text())
        self.layers = {l: np.load(out / f"layer_{l}.npy", mmap_mode="r") for l in LAYERS}
        self.dev = {t["id"]: t for t in packs()["dev"]}
        ids = sorted(self.dev)
        self.half_a = set(ids[::2])
        self.runs = {s: {r["id"]: r for r in rows_of(p)} for s, p in SOURCES.items()}
        for f in (out / "evals").glob("*.json"):
            if not f.stem.startswith("select_"):  # the held-out check's outputs stay out of Claude's reach
                self.runs[f.stem.replace("__", "/")] = {r["id"]: r for r in json.loads(f.read_text())}
        self.evals = json.loads((out / "evals.json").read_text()) if (out / "evals.json").exists() else {}
        self.writers = None

    # -- reading
    def read_output(self, run, task_id, start=0, chars=2500):
        r = self.runs[run][task_id]
        chars = max(1, min(int(chars), 3000))
        return {"run": run, "task": task_id, "outcome": outcome(r), "tokens": r["tokens"], "length": len(r["text"]),
                "start": start, "text": r["text"][start:start + chars]}

    def read_task(self, task_id, chars=2500):
        return {"task": task_id, "statement": self.dev[task_id]["public"]["prompt"][:max(1, min(int(chars), 3000))]}

    def find_outputs(self, run, outcome_is="any", contains=None):
        rx = re.compile(contains) if contains else None
        hits = [{"task": i, "outcome": outcome(r), "tokens": r["tokens"]} for i, r in sorted(self.runs[run].items())
                if (outcome_is == "any" or outcome(r) == outcome_is or (outcome_is == "failed" and not r["correct"]))
                and (rx is None or rx.search(r["text"]))]
        return {"run": run, "count": len(hits), "outputs": hits[:40]}

    def line_stats(self, runs=None, outcome_is="any", words=1, top=30):
        counts = {}
        for run in runs or list(SOURCES):
            for r in self.runs[run].values():
                if outcome_is != "any" and outcome(r) != outcome_is and not (outcome_is == "failed" and not r["correct"]):
                    continue
                for line in r["text"].split("\n"):
                    key = " ".join(line.strip().split(" ")[:max(1, min(int(words), 3))])
                    if key:
                        counts[key] = counts.get(key, 0) + 1
        return {"outcome": outcome_is, "top": sorted(counts.items(), key=lambda kv: -kv[1])[:max(1, min(int(top), 60))]}

    # -- directions
    def select(self, f):
        kind, runs = f.get("kind", "line"), set(f.get("runs") or SOURCES)
        rx = re.compile(f["line_regex"]) if f.get("line_regex") else None
        want, lo, hi = f.get("outcome", "any"), f.get("min_frac", 0.0), f.get("max_frac", 1.0)
        return [i for i, m in enumerate(self.meta)
                if m["kind"] == kind and m["source"] in runs and lo <= m["frac"] <= hi
                and (want == "any" or m["outcome"] == want or (want == "failed" and m["outcome"] != "solved"))
                and (rx is None or rx.match(m["line"].lstrip()))]

    def make_direction(self, name, a, b):
        assert NAME.match(name), "name: lower-case letters, digits and _, at most 32"
        ia, ib = self.select(a), self.select(b)
        split = {k: ([i for i in idx if self.meta[i]["task"] in self.half_a],
                     [i for i in idx if self.meta[i]["task"] not in self.half_a]) for k, idx in (("a", ia), ("b", ib))}
        if min(len(x) for pair in split.values() for x in pair) < 20:
            return {"error": "each kind needs at least 20 positions in each half of the tasks",
                    "counts": {k: [len(x) for x in pair] for k, pair in split.items()}}
        aucs, dirs = {}, []
        for l in LAYERS:
            x = self.layers[l]
            mean = lambda idx: np.asarray(x[sorted(idx)], dtype=np.float32).mean(0)  # noqa: E731
            d = mean(split["a"][0]) - mean(split["b"][0])
            d /= np.linalg.norm(d)
            pa = np.asarray(x[sorted(split["a"][1])], dtype=np.float32) @ d
            pb = np.asarray(x[sorted(split["b"][1])], dtype=np.float32) @ d
            aucs[l] = round(auc(pa, pb), 4)
            full = mean(ia) - mean(ib)
            dirs.append(full / np.linalg.norm(full))
        np.save(self.out / "directions" / f"{name}.npy", np.stack(dirs))
        info = {"name": name, "a": a, "b": b, "counts": {"a": len(ia), "b": len(ib)}, "auc_on_held_out_half": aucs}
        (self.out / "directions" / f"{name}.json").write_text(json.dumps(info))
        return info

    # -- edits
    def load_writers(self):
        from safetensors.torch import load_file
        self.writers = {}
        for f in sorted(fix.MODEL.glob("*.safetensors")):
            for key, w in load_file(f).items():
                if key.startswith("model.language_model.layers.") and key.endswith(tuple(KINDS)):
                    self.writers[key] = w

    def make_edit(self, name, parts):
        import torch
        from safetensors.torch import save_file
        assert NAME.match(name), "name: lower-case letters, digits and _, at most 32"
        assert 1 <= len(parts) <= MAX_RANK, f"1 to {MAX_RANK} parts"
        if self.writers is None:
            self.load_writers()
        specs = []
        for p in parts:
            dirs = np.load(self.out / "directions" / f"{p['direction']}.npy")
            assert p["layer"] in LAYERS, f"layer must be one of {LAYERS[0]}..{LAYERS[-1]} (where states are stored)"
            alpha = float(p["alpha"])
            assert -1.0 <= alpha <= 1.0 and alpha != 0, "alpha in [-1, 1], not 0"
            lo, hi = int(p.get("from_layer", 0)), int(p.get("to_layer", 31))
            assert 0 <= lo <= hi <= 31, "0 <= from_layer <= to_layer <= 31"
            kinds = set(p.get("writers") or KINDS.values())
            assert kinds <= set(KINDS.values()), f"writers from {sorted(KINDS.values())}"
            specs.append((torch.tensor(dirs[LAYERS.index(p["layer"])], dtype=torch.float32), alpha, lo, hi, kinds))
        rank, tensors, touched = len(specs), {}, 0
        for key, w in self.writers.items():
            layer = int(key.split(".")[3])
            kind = next(v for k, v in KINDS.items() if key.endswith(k))
            use = [j for j, s in enumerate(specs) if s[2] <= layer <= s[3] and kind in s[4]]
            if not use:
                continue
            w32 = w.to(torch.float32)
            a = torch.zeros(rank, w.shape[1])
            b = torch.zeros(w.shape[0], rank)
            for j in use:
                r, alpha = specs[j][0], specs[j][1]
                a[j], b[:, j] = r @ w32, -alpha * r
            module = key.removesuffix(".weight")
            tensors[f"base_model.model.{module}.lora_A.weight"] = a.to(w.dtype).contiguous()
            tensors[f"base_model.model.{module}.lora_B.weight"] = b.to(w.dtype).contiguous()
            touched += 1
        target = self.out / "edits" / name
        target.mkdir(exist_ok=True)
        save_file(tensors, target / "adapter_model.safetensors", metadata={"format": "pt"})
        (target / "adapter_config.json").write_text(json.dumps(
            {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": rank, "lora_dropout": 0.0,
             "bias": "none", "fan_in_fan_out": False, "base_model_name_or_path": str(fix.MODEL),
             "target_modules": ["down_proj", "o_proj", "out_proj"]}))
        (target / "parts.json").write_text(json.dumps(parts))
        return {"name": name, "rank": rank, "matrices": touched}

    # -- evaluation
    def evaluate(self, name, pack="dev", seeds=SEEDS):
        """`name` an edit, or "base" for the original model. Returns the summary; rows go to <out>/evals."""
        import httpx
        model = "qwen3.5-4b" if name == "base" else name
        if name != "base":
            r = httpx.post(f"{URL}/v1/load_lora_adapter", json={"lora_name": name,
                                                               "lora_path": str(self.out / "edits" / name)}, timeout=300)
            assert r.status_code == 200, r.text
        try:
            tasks = packs()[pack]
            res = {}
            for seed in seeds:
                start = time.time()
                rows = run_plain(tasks, model, seed)
                tag = f"{name}/s{seed}" if pack == "dev" else f"{pack}_{name}/s{seed}"
                (self.out / "evals" / f"{tag.replace('/', '__')}.json").write_text(json.dumps(rows))
                if pack == "dev":
                    self.runs[tag] = {x["id"]: x for x in rows}
                res[f"s{seed}"] = {**plain_summary(rows), "seconds": round(time.time() - start)}
                ref = self.runs.get(f"base/s{seed}") if pack == "dev" else None
                if ref and name != "base":
                    got = {x["id"] for x in rows if x["correct"]}
                    was = {i for i, x in ref.items() if x["correct"]}
                    res[f"s{seed}"]["vs_base"] = {"gained": sorted(got - was), "lost": sorted(was - got)}
        finally:
            if name != "base":
                httpx.post(f"{URL}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=300)
        if pack == "dev":
            self.evals[name] = res
            (self.out / "evals.json").write_text(json.dumps(self.evals, indent=1))
        return {"name": name, "pack": pack, **res}


def auc(pos, neg):
    ranks = np.concatenate([pos, neg]).argsort().argsort().astype(np.float64) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run_plain(tasks, model, seed):
    from openai import OpenAI
    client = OpenAI(base_url=f"{URL}/v1", api_key="EMPTY", timeout=1800)
    sampling, extra = fix.split(diag.CARD_INSTRUCT)

    def one(task):
        r = client.chat.completions.create(model=model, seed=seed, max_tokens=PLAIN_CAP, **sampling,
                                           extra_body={"chat_template_kwargs": {"enable_thinking": False}, **extra},
                                           messages=[{"role": "user", "content": task["public"]["prompt"]}])
        text = r.choices[0].message.content or ""
        return {"id": task["id"], "tokens": r.usage.completion_tokens, "cut": r.choices[0].finish_reason == "length",
                "forced": False, "closed_thinking": False, "repeated": round(diag.repeated(text), 3), "text": text}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(one, tasks))
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(lambda p: diag.judge(p[0], p[1], False), zip(tasks, rows)))


def plain_summary(rows):
    fails = {}
    for r in rows:
        if not r["correct"]:
            fails[outcome(r)] = fails.get(outcome(r), 0) + 1
    return {"tasks": len(rows), "solved": sum(r["correct"] for r in rows), "cut": sum(r["cut"] for r in rows),
            "mean_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
            "starts_with_code": sum(r["text"].lstrip().startswith("```") for r in rows), "failures": fails}


# ---------------------------------------------------------------- Claude

FILTER = {"type": "object", "description": "Which stored positions: all conditions must hold.", "properties": {
    "kind": {"type": "string", "enum": ["line", "prompt_end"],
             "description": "line: just before a line's first token; prompt_end: just before the answer begins"},
    "line_regex": {"type": "string", "description": "Python regex matched at the start of the line (leading spaces "
                                                    "removed); only for kind line"},
    "outcome": {"type": "string", "description": "the output's outcome: any, solved, failed, cut, wrong_answer, "
                                                 "runtime_error, load_error, time_limit"},
    "runs": {"type": "array", "items": {"type": "string", "enum": list(SOURCES)}},
    "min_frac": {"type": "number", "description": "position in the output as a fraction of its length, from"},
    "max_frac": {"type": "number", "description": "... to"}}}
TOOLS = [
    {"name": "read_output", "description": "Read part of one output (a stored run, or base/s0, <edit>/s1 ...).",
     "input_schema": {"type": "object", "properties": {"run": {"type": "string"}, "task_id": {"type": "string"},
                                                      "start": {"type": "integer"}, "chars": {"type": "integer"}},
                      "required": ["run", "task_id"]}},
    {"name": "read_task", "description": "Read a development task's statement (what the model was asked).",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}, "chars": {"type": "integer"}},
                      "required": ["task_id"]}},
    {"name": "find_outputs", "description": "List outputs of a run by outcome and/or a regex searched in the text.",
     "input_schema": {"type": "object", "properties": {"run": {"type": "string"}, "outcome": {"type": "string"},
                                                      "contains": {"type": "string"}}, "required": ["run"]}},
    {"name": "line_stats", "description": "Most frequent line beginnings (first 1-3 words) in outputs of a given "
                                          "outcome, over stored runs.",
     "input_schema": {"type": "object", "properties": {"runs": {"type": "array", "items": {"type": "string"}},
                                                      "outcome": {"type": "string"}, "words": {"type": "integer"},
                                                      "top": {"type": "integer"}}}},
    {"name": "make_direction", "description": "Difference of means of the residual stream between positions of kind "
                                              "a and of kind b, at every stored layer (4-28). The direction is found "
                                              "on half of the tasks and its AUC (how well the projection separates a "
                                              "from b) measured on the other half; the saved direction uses all.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "a": FILTER, "b": FILTER},
                      "required": ["name", "a", "b"]}},
    {"name": "make_edit", "description": "Build an edit: for each part, every chosen output matrix W in layers "
                                         "from_layer..to_layer becomes W - alpha r r^T W, r the part's direction at "
                                         "`layer`. writers: linear_attn (out_proj, 24 layers), attn (o_proj, the 8 "
                                         "full-attention layers 3, 7, ..., 31), mlp (down_proj, all 32); default all.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "parts": {"type": "array", "items": {
         "type": "object", "properties": {"direction": {"type": "string"}, "layer": {"type": "integer"},
                                          "alpha": {"type": "number"}, "from_layer": {"type": "integer"},
                                          "to_layer": {"type": "integer"},
                                          "writers": {"type": "array", "items": {"type": "string",
                                                                                 "enum": ["linear_attn", "attn",
                                                                                          "mlp"]}}},
         "required": ["direction", "layer", "alpha"]}}}, "required": ["name", "parts"]}},
    {"name": "evaluate", "description": "Run the 74 development tasks without thinking, two seeds, with the edit "
                                        "(about 7 minutes). Counts against the evaluation budget.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "finish", "description": "End the session, naming at most two edits for the held-out check, best "
                                      "first, with a short account of what you found.",
     "input_schema": {"type": "object", "properties": {"edits": {"type": "array", "items": {"type": "string"}},
                                                      "notes": {"type": "string"}}, "required": ["edits", "notes"]}}]

SYSTEM = """You are a research engineer editing the weights of a small language model, Qwen3.5-4B (32 layers, \
residual stream of 2560 dimensions; 24 layers are linear attention, every fourth layer 3, 7, ..., 31 is full \
attention). The goal: more programming tasks solved when the model answers WITHOUT thinking.

How it is measured. 74 development tasks (LiveCodeBench-style, Python), the model answers directly (no thinking \
block), sampling temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5, at most 4096 tokens per answer; hidden \
tests judge the code. Each evaluation runs two seeds. Two runs of the same model differ by about 5 solved tasks \
per seed by chance, so a difference below about 4 in the two-seed mean is noise. At the end you name at most two \
edits; they are then checked on 249 other tasks that you cannot see, against the unedited model, and count only if \
they solve at least 8 more there. Chasing noise on the 74 tasks will not pass that check.

What is known. In THINKING mode, removing the direction "the next line starts with Wait" (difference of means of \
the residual stream after layer 20, between the positions before 'Wait' lines and before other lines; removed from \
every output matrix of every layer) cut endless re-checking and lifted solved tasks a lot. In NON-THINKING mode the \
same edit lost 24 of 249 tasks: the answers started directly with code instead of first explaining the approach \
(96% of answers began with a code block, against 73% unedited). So directions can carry more than one behaviour.

What you can do. Tools let you read outputs and task statements, count line beginnings, compute directions from \
stored residual-stream states (at the start of lines and at the end of the prompt, in 370 earlier non-thinking \
outputs of the unedited model on the same 74 tasks, layers 4-28), build edits from directions (remove, weaken or \
strengthen a direction in what chosen layers write into the residual stream) and evaluate them. You cannot run \
your own code. A status line after each tool result shows the budget left: evaluations, turns and USD. Plan for it \
and call finish before it runs out; if it runs out, the two edits with the best two-seed mean that beat the \
unedited model are taken.

Work like a careful scientist: look at outputs first, form a hypothesis about what makes answers fail (cut-offs, \
wrong answers, crashes), check it with directions (an AUC near 0.5 means the two kinds are not separated), then \
edit and evaluate. Prefer few, well-motivated evaluations to many blind ones, and say honestly what you find."""


def overview(lab):
    base = lab.evals.get("base", {})
    lines = ["Session start. Unedited model on the 74 development tasks (no thinking):"]
    for seed, s in base.items():
        lines.append(f"  base/{seed}: {json.dumps(s)}")
    for source in SOURCES:
        rows = lab.runs[source].values()
        lines.append(f"  stored run {source}: solved {sum(r['correct'] for r in rows)}, cut "
                     f"{sum(r['cut'] for r in rows)} of {len(rows)}")
    lines.append(f"Stored positions: {len(lab.meta)} ({sum(m['kind'] == 'prompt_end' for m in lab.meta)} prompt ends).")
    lines.append(f"Budget: {MAX_EVALS} evaluations, {MAX_TURNS} turns, {SESSION_USD} USD.")
    lines.append("Task ids look like " + ", ".join(sorted(lab.dev)[:3]) + ".")
    return "\n".join(lines)


def call_tool(lab, name, args, state):
    if name == "read_output":
        return lab.read_output(args["run"], args["task_id"], int(args.get("start", 0)), args.get("chars", 2500))
    if name == "read_task":
        return lab.read_task(args["task_id"], args.get("chars", 2500))
    if name == "find_outputs":
        return lab.find_outputs(args["run"], args.get("outcome", "any"), args.get("contains"))
    if name == "line_stats":
        return lab.line_stats(args.get("runs"), args.get("outcome", "any"), args.get("words", 1), args.get("top", 30))
    if name == "make_direction":
        return lab.make_direction(args["name"], args["a"], args["b"])
    if name == "make_edit":
        return lab.make_edit(args["name"], args["parts"])
    if name == "evaluate":
        if state["evals_left"] <= 0:
            return {"error": "no evaluations left; call finish"}
        if not (lab.out / "edits" / args["name"]).exists():
            return {"error": f"no edit called {args['name']}"}
        if not lab.can_evaluate:
            return {"error": "evaluation is off in this run (smoke test)"}
        state["evals_left"] -= 1
        return lab.evaluate(args["name"])
    if name == "finish":
        state["finish"] = {"edits": [e for e in args["edits"] if e in lab.evals][:2], "notes": args["notes"]}
        return {"ok": True}
    return {"error": f"unknown tool {name}"}


def session(out, turns=MAX_TURNS, evaluate=True, usd=SESSION_USD):
    lab = Lab(out, evaluate)
    if evaluate and "base" not in lab.evals:
        print(f"{time.strftime('%F %T')} base: {json.dumps(lab.evaluate('base'))}", flush=True)
    gw = ClaudeGateway(HOME / "alter/rive/claude", session_usd=usd)
    state = {"evals_left": MAX_EVALS - sum(1 for k in lab.evals if k != "base"), "finish": None}
    messages = [{"role": "user", "content": overview(lab)}]
    transcript = (out / "transcript.jsonl").open("a")
    nudges = 0
    for turn in range(turns):
        try:
            msg = gw.create("claude_edit", system=SYSTEM, messages=messages, max_tokens=16000, tools=TOOLS,
                            thinking={"type": "adaptive", "display": "summarized"}, output_config={"effort": "high"},
                            cache_control={"type": "ephemeral", "ttl": "1h"})
        except BudgetExceeded as e:
            print(f"{time.strftime('%F %T')} budget: {e}", flush=True)
            break
        messages.append({"role": "assistant", "content": msg.content})
        transcript.write(json.dumps({"turn": turn, "role": "assistant", "stop_reason": msg.stop_reason,
                                     "content": [b.model_dump() for b in msg.content]}) + "\n")
        transcript.flush()
        uses = [b for b in msg.content if b.type == "tool_use"]
        if msg.stop_reason == "refusal":
            print(f"{time.strftime('%F %T')} refusal at turn {turn}", flush=True)
            break
        if not uses:
            if nudges >= 2:
                break
            nudges += 1
            messages.append({"role": "user", "content": "Continue with the tools; call finish when you are done."})
            continue
        results = []
        for b in uses:
            try:
                res, err = call_tool(lab, b.name, b.input, state), False
            except Exception as e:  # the tool's input was not usable: tell Claude, as a tool error
                res, err = {"error": f"{type(e).__name__}: {e}"}, True
            text = json.dumps(res)[:6000]
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": text, "is_error": err})
            transcript.write(json.dumps({"turn": turn, "tool": b.name, "input": b.input, "result": text[:2000]}) + "\n")
            print(f"{time.strftime('%F %T')} turn {turn} {b.name} {json.dumps(b.input)[:200]} -> {text[:300]}",
                  flush=True)
        status = (f"[budget left: {state['evals_left']} evaluations, {turns - turn - 1} turns, "
                  f"{gw.remaining():.2f} USD]")
        messages.append({"role": "user", "content": results + [{"type": "text", "text": status}]})
        if state["finish"] is not None:
            break
    chosen = state["finish"] or auto_choice(lab)
    (out / "chosen.json").write_text(json.dumps({**chosen, "spent_usd": round(gw.session_spent, 4),
                                                  "evaluations": len([k for k in lab.evals if k != "base"])}, indent=1))
    print(f"{time.strftime('%F %T')} session end: {json.dumps(chosen)[:600]}, spent {gw.session_spent:.2f} USD",
          flush=True)


def auto_choice(lab):
    mean = lambda r: sum(s["solved"] for s in r.values()) / len(r)  # noqa: E731
    base = mean(lab.evals["base"]) if "base" in lab.evals else 0
    ranked = sorted((k for k in lab.evals if k != "base" and mean(lab.evals[k]) > base),
                    key=lambda k: (-mean(lab.evals[k]), sum(s["mean_tokens"] for s in lab.evals[k].values())))
    return {"edits": ranked[:2], "notes": "taken by the rule: the session ended without finish"}


def verify(out):
    lab = Lab(out)
    chosen = json.loads((out / "chosen.json").read_text())
    res = {"chosen": chosen, "select": {}}
    done = out / "verify.json"
    if done.exists():
        res = json.loads(done.read_text())
    for name in ["base"] + chosen["edits"]:
        if name not in res["select"]:
            res["select"][name] = lab.evaluate(name, pack="select", seeds=(0,))["s0"]
            done.write_text(json.dumps(res, indent=1))
            print(f"{time.strftime('%F %T')} select {name} s0: {json.dumps(res['select'][name])}", flush=True)
    base = res["select"]["base"]["solved"]
    passing = [e for e in chosen["edits"] if res["select"][e]["solved"] >= base + BETTER]
    if passing:
        best = passing[0]
        for name in ("base", best):
            key = f"{name}_s1"
            if key not in res["select"]:
                res["select"][key] = lab.evaluate(name, pack="select", seeds=(1,))["s1"]
                done.write_text(json.dumps(res, indent=1))
                print(f"{time.strftime('%F %T')} select {name} s1: {json.dumps(res['select'][key])}", flush=True)
        res["confirmed"] = res["select"][f"{best}_s1"]["solved"] >= res["select"]["base_s1"]["solved"] + CONFIRM
        res["best"] = best
    res["works"] = bool(passing)
    done.write_text(json.dumps(res, indent=1))
    report(out)


def report(out):
    res = {"chosen": json.loads((out / "chosen.json").read_text()) if (out / "chosen.json").exists() else None,
           "dev_evals": json.loads((out / "evals.json").read_text()) if (out / "evals.json").exists() else {},
           "verify": json.loads((out / "verify.json").read_text()) if (out / "verify.json").exists() else None}
    if out.name != "smoke":
        (ROOT / "results/rive_claude_edit.json").write_text(json.dumps(res, indent=1) + "\n")
    print("report", json.dumps(res)[:3000], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("positions", "session", "verify", "report", "smoke"))
    p.add_argument("--out-dir", default=str(HOME / "alter/rive/claude_edit"))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.step == "positions":
        positions(out)
    elif args.step == "session":
        session(out)
    elif args.step == "verify":
        verify(out)
    elif args.step == "report":
        report(out)
    elif args.step == "smoke":
        smoke = out / "smoke"
        smoke.mkdir(exist_ok=True)
        for f in out.glob("layer_*.npy"):  # the stored states, shared
            if not (smoke / f.name).exists():
                (smoke / f.name).symlink_to(f)
        if not (smoke / "positions.json").exists():
            (smoke / "positions.json").symlink_to(out / "positions.json")
        session(smoke, turns=2, evaluate=False, usd=1.0)


if __name__ == "__main__":
    main()
