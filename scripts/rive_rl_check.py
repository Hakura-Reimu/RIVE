"""The check section 6.25 asks for after the halt (docs/rive-design.md): the training pass rate stayed below half of
iteration 0's for 10 iterations, so is the collapse a pipeline error? Two tests.

  1. The served policy is the trained one: the log-probabilities the server (FP8 base, the exported adapter) gives
     the tokens of sampled answers, against the trainer's (nf4 base, the PEFT adapter), for the base model and two
     adapters. A broken export or loading shows as a much larger gap for the adapters than for the base model.
  2. The update goes the intended way: under the adapters before and after one iteration's training, the answers
     with a positive weight should gain likelihood and those with a negative weight lose it.

  python scripts/rive_rl_check.py vllm      # the server (started if down) scores the answers
  python scripts/rive_rl_check.py trainer   # the trainer scores them (the server is put to sleep first)
  python scripts/rive_rl_check.py report    # results/rive_rl_check.json
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rive_rl as rl  # noqa: E402

CHECK = rl.OUT / "check"
ROWS = 8
# test 1: {set: (the iteration whose answers are scored, {policy name: adapter of iteration})}. "own" was added
# after the first set's result: rl-0029 scored on its own answers (iteration 30's), not on rl-0014's
SETS = {"": (15, {rl.BASE: None, "rl-0014": 14, "rl-0029": 29}),
        "own": (30, {rl.BASE: None, "rl-0029": 29})}
p_ = argparse.ArgumentParser(add_help=False)
p_.add_argument("--set", default="")
SET = p_.parse_known_args()[0].set
SAMPLES_OF, POLICIES = SETS[SET]
SUFFIX = f"_{SET}" if SET else ""
BEFORE, AFTER = 29, 30     # test 2: iteration 30 trained adapter 0029 into 0030
MAX_LEN = 3072             # the trainer's cut


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(rl.MODEL)


def batch(it):
    return [json.loads(line) for line in open(rl.OUT / "iters" / f"{it:04d}" / "batch.jsonl")]


def encoded(rows):
    from phase_d_train import encode
    tok = tokenizer()
    out = []
    for r in rows:
        ids, labels = encode(tok, r["prompt"], r["answer"])
        out.append((ids[:MAX_LEN], labels[:MAX_LEN]))
    return out


def test1_rows():
    rows = sorted(batch(SAMPLES_OF), key=lambda r: (r["id"], r["answer"]))
    return random.Random(0).sample(rows, ROWS)


def vllm_step(args):
    rows = test1_rows()
    result = {"ids": [r["id"] for r in rows], "policies": {}}
    for name, it in POLICIES.items():
        rl.use(name, Path(f"{rl.adapter_dir(rl.OUT, it)}-vllm") if it is not None else None)
        per_row = []
        for ids, labels in encoded(rows):
            body = {"model": name, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
            got = json.loads(rl.http("POST", "/v1/completions", body))["choices"][0]["prompt_logprobs"]
            # position p holds the log-probability of token p given the tokens before it
            per_row.append([got[p][str(ids[p])]["logprob"] for p in range(1, len(ids)) if labels[p] != -100])
        result["policies"][name] = per_row
        rl.log(f"check: server scored {len(rows)} answers under {name}")
    CHECK.mkdir(parents=True, exist_ok=True)
    rl.write_json(CHECK / f"vllm{SUFFIX}.json", result)


def token_logprobs(model, ids, labels):
    import torch
    causal = model.get_base_model()
    with torch.no_grad():
        hidden = causal.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).last_hidden_state[0, :-1]
        y = torch.tensor(labels[1:], device="cuda")
        keep = y != -100
        h, y = hidden[keep], y[keep]
        out = []
        for i in range(0, len(y), 512):
            logits = causal.lm_head(h[i:i + 512]).float()
            out += torch.log_softmax(logits, -1).gather(1, y[i:i + 512, None])[:, 0].tolist()
    return out


def trainer_step(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if rl.served() is not None and not rl.sleeping():
        rl.http("POST", "/sleep?level=1")
    model = AutoModelForCausalLM.from_pretrained(rl.MODEL, dtype=torch.bfloat16, device_map={"": 0},
                                                 quantization_config=BitsAndBytesConfig(
                                                     load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                                     bnb_4bit_use_double_quant=True))
    adapters = sorted({it for it in POLICIES.values() if it is not None} | {BEFORE, AFTER})
    model = PeftModel.from_pretrained(model, str(rl.adapter_dir(rl.OUT, adapters[0])), adapter_name=str(adapters[0]))
    for it in adapters[1:]:
        model.load_adapter(str(rl.adapter_dir(rl.OUT, it)), adapter_name=str(it))
    model.eval()

    def score(it, examples):
        if it is None:
            with model.disable_adapter():
                return [token_logprobs(model, ids, labels) for ids, labels in examples]
        model.set_adapter(str(it))
        return [token_logprobs(model, ids, labels) for ids, labels in examples]

    rows = test1_rows()
    first = encoded(rows)
    result = {"ids": [r["id"] for r in rows], "policies": {name: score(it, first) for name, it in POLICIES.items()}}
    if SET:
        rl.write_json(CHECK / f"trainer{SUFFIX}.json", result)
        rl.log("check: trainer scored the answers")
        return
    second_rows = batch(AFTER)
    second = encoded(second_rows)
    result["update"] = {"iteration": AFTER, "weights": [r["weight"] for r in second_rows],
                        "before": [sum(x) for x in score(BEFORE, second)],
                        "after": [sum(x) for x in score(AFTER, second)],
                        "tokens": [sum(t != -100 for t in labels) for _, labels in second]}
    CHECK.mkdir(parents=True, exist_ok=True)
    rl.write_json(CHECK / "trainer.json", result)
    rl.log("check: trainer scored the answers")


def agreement_of(v, t, policies):
    assert v["ids"] == t["ids"]
    agreement = {}
    for name in policies:
        gaps, signed, means = [], [], []
        for a, b in zip(v["policies"][name], t["policies"][name]):
            assert len(a) == len(b)
            gaps.append(statistics.mean(abs(x - y) for x, y in zip(a, b)))
            signed.append(statistics.mean(x - y for x, y in zip(a, b)))
            means.append((statistics.mean(a), statistics.mean(b)))
        agreement[name] = {"mean_abs_gap": round(statistics.mean(gaps), 4), "mean_gap": round(statistics.mean(signed), 4),
                           "server_mean_logprob": round(statistics.mean(m[0] for m in means), 4),
                           "trainer_mean_logprob": round(statistics.mean(m[1] for m in means), 4)}
    return agreement


def report_step(args):
    v, t = (json.loads((CHECK / f).read_text()) for f in ("vllm.json", "trainer.json"))
    agreement = agreement_of(v, t, SETS[""][1])
    u = t["update"]
    moves = [(w, (a - b) / n) for w, b, a, n in zip(u["weights"], u["before"], u["after"], u["tokens"])]
    update = {"iteration": u["iteration"],
              "positive": {"answers": sum(w > 0 for w, _ in moves),
                           "mean_change_per_token": round(statistics.mean(d for w, d in moves if w > 0), 5),
                           "rose": sum(d > 0 for w, d in moves if w > 0)},
              "negative": {"answers": sum(w < 0 for w, _ in moves),
                           "mean_change_per_token": round(statistics.mean(d for w, d in moves if w < 0), 5),
                           "fell": sum(d < 0 for w, d in moves if w < 0)},
              "weighted_change": round(sum(w * (a - b) for w, b, a in zip(u["weights"], u["before"], u["after"])), 3)}
    result = {"answers": len(v["ids"]), "agreement": agreement, "update": update}
    if (CHECK / "trainer_own.json").exists():
        v, t = (json.loads((CHECK / f).read_text()) for f in ("vllm_own.json", "trainer_own.json"))
        result["agreement_own_answers"] = {"iteration": SETS["own"][0], **agreement_of(v, t, SETS["own"][1])}
    rl.write_json(ROOT / "results" / "rive_rl_check.json", result)
    print(json.dumps(result, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["vllm", "trainer", "report"])
    p.add_argument("--set", default="", choices=sorted(SETS), help="test 1's answers and policies (see SETS)")
    args = p.parse_args()
    {"vllm": vllm_step, "trainer": trainer_step, "report": report_step}[args.step](args)


if __name__ == "__main__":
    main()
