"""One training step of the test-reward reinforcement learning (docs/rive-design.md section 6.25): QLoRA on the
answers the policy sampled, each weighted by its advantage (its reward minus the mean reward of its problem's
samples), continuing the adapter and the optimizer state of the previous iteration.

  python scripts/rive_rl_train.py --batch it/batch.jsonl --out adapters/0003 [--init adapters/0002] --steps 2

The loss of a window is sum_i A_i * CE_i / (answer tokens in the window), CE_i being the summed cross-entropy of
answer i: minimizing it raises the likelihood of answers better than their problem's mean and lowers the others.
The model, quantization and LoRA layers are Phase D's (scripts/phase_d_train.py); dropout is off, so that the
likelihood being pushed is the policy's. Answers longer than --max-len tokens (prompt included) are cut there and
carry the loss of their first part only: the card has about 11 GiB free beside the sleeping vLLM server.
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
from phase_d_train import TARGETS, encode, export_for_vllm, loss_sum  # noqa: E402


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
    args = p.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    out = Path(args.out)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in open(args.batch)]
    rows = [r for r in rows if r["weight"] != 0]
    examples, cut = [], 0
    for r in rows:
        ids, labels = encode(tokenizer, r["prompt"], r["answer"])
        cut += len(ids) > args.max_len
        examples.append((ids[:args.max_len], labels[:args.max_len], r["weight"]))
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

    model.train()
    torch.cuda.reset_peak_memory_stats()
    start, tokens, steps = time.perf_counter(), 0, []
    for window in windows:
        n_answer = sum(sum(t != -100 for t in examples[i][1]) for i in window)
        objective, grad_tokens = 0.0, 0
        for i in window:
            ids, labels, weight = examples[i]
            ids, labels = (torch.tensor([x], device="cuda") for x in (ids, labels))
            total, n = loss_sum(model, ids, labels)
            (weight * total / n_answer).backward()
            objective += weight * total.item() / n_answer
            grad_tokens += n
            tokens += ids.shape[1]
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        optim.step()
        optim.zero_grad(set_to_none=True)
        steps.append({"examples": len(window), "answer_tokens": grad_tokens, "objective": round(objective, 5),
                      "grad_norm": round(norm, 4)})
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
