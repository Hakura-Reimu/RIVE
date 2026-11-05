"""One training step of the test-reward reinforcement learning (docs/rive-design.md section 6.25): QLoRA on the
answers the policy sampled, each weighted by its advantage (its reward minus the mean reward of its problem's
samples), continuing the adapter and the optimizer state of the previous iteration.

  python scripts/rive_rl_train.py --batch it/batch.jsonl --out adapters/0003 [--init adapters/0002] --steps 2

The loss of a window is sum_i A_i * CE_i / (answer tokens in the window), CE_i being the summed cross-entropy of
answer i: minimizing it raises the likelihood of answers better than their problem's mean and lowers the others.
The model, quantization and LoRA layers are Phase D's (scripts/phase_d_train.py); dropout is off, so that the
likelihood being pushed is the policy's. Answers longer than --max-len tokens (prompt included) are cut there and
carry the loss of their first part only: the card has about 11 GiB free beside the sleeping vLLM server.

--recipe v2 (section 6.26) trains on the sampled token ids (row "ids") at the sampling temperature, with per-token
terms: the loss of a window is -(1/N) sum_i sum_t [w_it min(rho_it A_i, clip(rho_it, 1 - eps_low, 1 + eps_high) A_i)
- beta k3_it], where rho = pi / pi_old (pi_old: this model before the update), w = min(pi_old / mu, cap) with mu the
server's probability (row "logprobs"; 1 where it has none), and k3 = exp(ref - log pi) - (ref - log pi) - 1 with ref
the base model's (the adapter off). pi_old and ref come from a pass over the batch before the first step.
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_d_train import CHUNK, TARGETS, encode, export_for_vllm, loss_sum  # noqa: E402


def encode_ids(tokenizer, prompt, ids, logprobs):
    """v2: the prompt as the server renders it, then the sampled token ids; the end-of-turn token is added (with no
    server probability) when the server did not return it."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False)
    prefix = tokenizer(text, add_special_tokens=False)["input_ids"]
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    body, mu = list(ids), list(logprobs)
    if not body or body[-1] != end:
        body, mu = body + [end], mu + [None]
    return prefix + body, [-100] * len(prefix) + body, mu


def _lp(h, w, y, temperature):
    return torch.log_softmax(torch.nn.functional.linear(h.to(w.dtype), w).float() / temperature, -1).gather(
        1, y[:, None])[:, 0]


def answer_logprobs(model, ids, labels, temperature, grad):
    """The log-probability of every answer token at `temperature`, in pieces of CHUNK tokens."""
    from torch.utils.checkpoint import checkpoint
    causal = model.get_base_model()
    hidden = causal.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
    y = labels[0, 1:]
    keep = y != -100
    h, y, w = hidden[keep], y[keep], causal.lm_head.weight
    if grad:
        parts = [checkpoint(_lp, h[i:i + CHUNK], w, y[i:i + CHUNK], temperature, use_reentrant=False)
                 for i in range(0, len(y), CHUNK)]
    else:
        parts = [_lp(h[i:i + CHUNK], w, y[i:i + CHUNK], temperature) for i in range(0, len(y), CHUNK)]
    return torch.cat(parts)


def v2_terms(lp, old, ref, mu, advantage, args):
    """The per-token loss terms of recipe v2 and what is recorded of them."""
    ratio = torch.exp(lp - old)
    have = ~torch.isnan(mu)
    w = torch.where(have, torch.exp(old - torch.nan_to_num(mu)), torch.ones_like(old))
    capped = w > args.tis_cap
    w = torch.clamp(w, max=args.tis_cap)
    low, high = 1 - args.clip_low, 1 + args.clip_high
    surrogate = torch.minimum(ratio * advantage, torch.clamp(ratio, low, high) * advantage)
    d = ref - lp
    k3 = torch.exp(d) - d - 1
    clipped = (ratio > high) & (advantage > 0) | (ratio < low) & (advantage < 0)
    stats = {"clipped": int(clipped.sum()), "capped": int(capped.sum()), "k3": float(k3.detach().sum()),
             "old_minus_mu": float((old - torch.nan_to_num(mu))[have].abs().sum()), "with_mu": int(have.sum()),
             "logprob": float(lp.detach().sum())}
    return -(w * surrogate - args.kl * k3), stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/alter/models/Qwen3.5-4B"))
    p.add_argument("--batch", required=True, help="jsonl of {prompt, answer, weight}")
    p.add_argument("--out", required=True, help="the adapter written (PEFT), and <out>-vllm for the server")
    p.add_argument("--init", help="the previous iteration's adapter (PEFT); a new one if left out")
    p.add_argument("--steps", type=int, default=2, help="optimizer steps: the batch is split into this many windows")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-len", type=int, default=3072)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recipe", choices=["v1", "v2"], default="v1", help="v1: section 6.25; v2: section 6.26")
    p.add_argument("--temperature", type=float, default=0.7, help="v2: the sampling temperature")
    p.add_argument("--kl", type=float, default=0.04, help="v2: the k3 KL coefficient")
    p.add_argument("--clip-low", type=float, default=0.2)
    p.add_argument("--clip-high", type=float, default=0.28)
    p.add_argument("--tis-cap", type=float, default=2.0)
    args = p.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    out = Path(args.out)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in open(args.batch)]
    rows = [r for r in rows if r["weight"] != 0]
    examples, cut, mus = [], 0, []
    for r in rows:
        if args.recipe == "v2":
            ids, labels, mu = encode_ids(tokenizer, r["prompt"], r["ids"], r["logprobs"])
        else:
            ids, labels = encode(tokenizer, r["prompt"], r["answer"])
            mu = []
        cut += len(ids) > args.max_len
        examples.append((ids[:args.max_len], labels[:args.max_len], r["weight"]))
        mus.append(mu[:sum(t != -100 for t in labels[:args.max_len])])
    order = list(range(len(examples)))
    random.Random(args.seed).shuffle(order)
    windows = [w for w in (order[k::args.steps] for k in range(args.steps)) if w]
    result = {"args": vars(args), "examples": len(examples), "cut_to_max_len": cut,
              "positive": sum(w > 0 for _, _, w in examples), "negative": sum(w < 0 for _, _, w in examples)}

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 quantization_config=BitsAndBytesConfig(
                                                     load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                                     bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.init:
        model = PeftModel.from_pretrained(model, args.init, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                                 target_modules=TARGETS, task_type="CAUSAL_LM"))
    params = [q for q in model.parameters() if q.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    state = Path(args.init) / "optimizer.pt" if args.init else None
    if state and state.exists():
        optim.load_state_dict(torch.load(state, map_location="cuda"))
        for group in optim.param_groups:
            group["lr"] = args.lr

    torch.cuda.reset_peak_memory_stats()
    start, tokens, steps = time.perf_counter(), 0, []
    old, ref = {}, {}
    if args.recipe == "v2":  # pi_old and the base model's log-probabilities, before any step
        model.eval()
        with torch.no_grad():
            for i, (ids, labels, _) in enumerate(examples):
                ids, labels = (torch.tensor([x], device="cuda") for x in (ids, labels))
                old[i] = answer_logprobs(model, ids, labels, args.temperature, grad=False)
                with model.disable_adapter():
                    ref[i] = answer_logprobs(model, ids, labels, args.temperature, grad=False)
    model.train()
    for window in windows:
        n_answer = sum(sum(t != -100 for t in examples[i][1]) for i in window)
        objective, grad_tokens, recorded = 0.0, 0, {}
        for i in window:
            ids, labels, weight = examples[i]
            ids, labels = (torch.tensor([x], device="cuda") for x in (ids, labels))
            if args.recipe == "v2":
                lp = answer_logprobs(model, ids, labels, args.temperature, grad=True)
                mu = torch.tensor([float("nan") if m is None else m for m in mus[i]], device="cuda")
                terms, stats = v2_terms(lp, old[i], ref[i], mu, weight, args)
                loss = terms.sum() / n_answer
                n = len(lp)
                for k, v in stats.items():
                    recorded[k] = recorded.get(k, 0) + v
            else:
                total, n = loss_sum(model, ids, labels)
                loss = weight * total / n_answer
            loss.backward()
            objective += loss.item()
            grad_tokens += n
            tokens += ids.shape[1]
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        optim.step()
        optim.zero_grad(set_to_none=True)
        steps.append({"examples": len(window), "answer_tokens": grad_tokens, "objective": round(objective, 5),
                      "grad_norm": round(norm, 4)})
        if recorded:  # per answer token, except the counts of tokens
            steps[-1].update(clipped=round(recorded["clipped"] / grad_tokens, 4),
                             capped=round(recorded["capped"] / max(recorded["with_mu"], 1), 4),
                             kl=round(recorded["k3"] / grad_tokens, 5),
                             old_minus_mu=round(recorded["old_minus_mu"] / max(recorded["with_mu"], 1), 4),
                             logprob=round(recorded["logprob"] / grad_tokens, 4))
        print(json.dumps(steps[-1]), flush=True)
    seconds = time.perf_counter() - start
    model.save_pretrained(out)
    torch.save(optim.state_dict(), out / "optimizer.pt")
    result.update(steps=steps, seconds=round(seconds), tokens_per_s=round(tokens / max(seconds, 1e-9), 1),
                  peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                  vllm_tensors=export_for_vllm(out, Path(f"{out}-vllm")))
    (out / "train.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
