"""Phase D training (docs/phase-d-design.md section 4): QLoRA on the correct answers V0 wrote, then a copy of the
adapter renamed so vLLM applies it.

  python scripts/phase_d_train.py --seed 1        # ~/alter/evo/runs/phase-d/train.jsonl -> adapters/seed1(-vllm)
  python scripts/phase_d_train.py --probe 3       # D0: 3 optimizer steps on examples of the full length

Settings from phase0/REPORT.md section 3: nf4 with double quantization, rank 16 on every linear layer of the text
model, expandable CUDA segments (without them WSL silently pages VRAM to host memory). The loss is computed in
chunks under checkpointing: with Qwen3.5's 248,320-token vocabulary, the logits of one 4.5K-token answer alone
would take several GiB.
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
           "gate_proj", "up_proj", "down_proj"]  # every linear layer of the text model but lm_head (Phase 0)
CHUNK = 1024  # answer tokens per piece of the loss
# vLLM loads Qwen3.5 as Qwen3_5ForConditionalGeneration and silently skips adapter weights saved under the
# causal-LM names (phase0/export_adapter_for_vllm.py).
SRC_PREFIX, DST_PREFIX = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."


def encode(tokenizer, prompt, answer):
    """Token ids and labels: the prompt as the server renders it for generation (one user turn, non-thinking),
    then the answer and the end-of-turn token, which alone carry the loss."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False)
    prefix = tokenizer(text, add_special_tokens=False)["input_ids"]
    body = tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
    return prefix + body, [-100] * len(prefix) + body


def export_for_vllm(src, dst):
    from safetensors.torch import load_file, save_file
    renamed = {}
    for key, value in load_file(src / "adapter_model.safetensors").items():
        if key.startswith(SRC_PREFIX):
            key = DST_PREFIX + key[len(SRC_PREFIX):]
        elif not key.startswith(DST_PREFIX):
            raise ValueError(f"unexpected adapter tensor name: {key}")
        renamed[key] = value
    dst.mkdir(parents=True, exist_ok=True)
    save_file(renamed, dst / "adapter_model.safetensors")
    shutil.copy2(src / "adapter_config.json", dst / "adapter_config.json")
    return len(renamed)


def _ce(h, w, y):
    return F.cross_entropy(F.linear(h.to(w.dtype), w).float(), y, reduction="sum")


def loss_sum(model, ids, labels):
    """Summed cross-entropy over the answer tokens, and their number."""
    causal = model.get_base_model()
    hidden = causal.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
    y = labels[0, 1:]
    keep = y != -100
    h, y, w = hidden[keep], y[keep], causal.lm_head.weight
    total = sum(checkpoint(_ce, h[i:i + CHUNK], w, y[i:i + CHUNK], use_reentrant=False)
                for i in range(0, len(y), CHUNK))
    return total, len(y)


def probe_rows(n, max_len, tokenizer):
    """Examples of about max_len tokens for the D0 probe: a long LiveCodeBench statement and MATH solutions joined
    into one long answer. Only their length matters."""
    data = Path(os.path.expanduser("~/alter/data"))
    statements = sorted((json.loads(line)["content"] for line in open(data / "lcb_train.jsonl")), key=len)[-n:]
    fillers = [json.loads(line)["problem"] for line in open(data / "math_train.jsonl")][:400]
    rows = []
    for s in statements:
        answer, i = "", 0
        while len(tokenizer(s + answer, add_special_tokens=False)["input_ids"]) < max_len - 200:  # template room
            answer += fillers[i % len(fillers)] + "\n"
            i += 1
        rows.append({"prompt": s, "answer": answer})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/alter/models/Qwen3.5-4B"))
    p.add_argument("--run-dir", default=os.path.expanduser("~/alter/evo/runs/phase-d"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8, help="examples per optimizer step (one at a time)")
    p.add_argument("--max-len", type=int, default=4608, help="longer examples are left out")
    p.add_argument("--probe", type=int, default=0, help="D0: this many optimizer steps on full-length examples")
    p.add_argument("--no-upcast", action="store_true",
                   help="keep embeddings and norms in bf16 (prepare_model_for_kbit_training moves them to fp32)")
    args = p.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import bitsandbytes as bnb

    run_dir = Path(args.run_dir)
    tag = f"probe-{args.max_len}" if args.probe else f"seed{args.seed}"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = (probe_rows(args.probe * args.grad_accum, args.max_len, tokenizer) if args.probe else
            [json.loads(line) for line in open(run_dir / "train.jsonl")])
    examples = [encode(tokenizer, r["prompt"], r["answer"]) for r in rows]
    skipped = sum(len(ids) > args.max_len for ids, _ in examples)
    examples = [e for e in examples if len(e[0]) <= args.max_len]
    rng = random.Random(args.seed)
    order = []
    for epoch in range(math.ceil(args.epochs)):
        idx = list(range(len(examples)))
        rng.shuffle(idx)
        order += idx[:round(len(idx) * min(1.0, args.epochs - epoch))]
    windows = [order[i:i + args.grad_accum] for i in range(0, len(order), args.grad_accum)]
    if args.probe:
        windows = windows[:args.probe]
    result = {"tag": tag, "args": vars(args), "examples": len(examples), "skipped_too_long": skipped,
              "steps": len(windows), "answer_tokens": sum(sum(t != -100 for t in lab) for _, lab in examples)}
    print(json.dumps(result), flush=True)

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 quantization_config=BitsAndBytesConfig(
                                                     load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                                     bnb_4bit_use_double_quant=True))
    if args.no_upcast:  # the tied 248,320 x 2560 embedding alone is 2.5 GiB in fp32
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    params = [q for q in model.parameters() if q.requires_grad]
    optim = bnb.optim.PagedAdamW8bit(params, lr=args.lr, weight_decay=0.0)
    warmup = max(1, round(0.05 * len(windows)))

    def lr_at(step):  # linear warm-up, then cosine down to a tenth
        if step < warmup:
            return args.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, len(windows) - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    model.train()
    torch.cuda.reset_peak_memory_stats()
    losses, start, tokens = [], time.perf_counter(), 0
    log = open(run_dir / f"train_{tag}.log", "a")
    for step, window in enumerate(windows):
        n_answer = sum(sum(t != -100 for t in examples[i][1]) for i in window)
        step_loss = 0.0
        for i in window:
            ids, labels = (torch.tensor([x], device="cuda") for x in examples[i])
            total, _ = loss_sum(model, ids, labels)
            (total / n_answer).backward()
            step_loss += total.item() / n_answer
            tokens += len(examples[i][0])
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for group in optim.param_groups:
            group["lr"] = lr_at(step)
        optim.step()
        optim.zero_grad(set_to_none=True)
        losses.append(round(step_loss, 4))
        line = (f"step {step + 1}/{len(windows)} loss {step_loss:.4f} lr {lr_at(step):.2e} "
                f"{tokens / (time.perf_counter() - start):.0f} tok/s peak "
                f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    seconds = time.perf_counter() - start
    total_vram = torch.cuda.mem_get_info()[1] / 2**30
    result.update(seconds=round(seconds), tokens_per_s=round(tokens / seconds, 1), loss_first=losses[:5],
                  loss_last=losses[-5:], peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                  peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                  exceeded_physical_vram=torch.cuda.max_memory_reserved() / 2**30 > total_vram)
    adapter = run_dir / "adapters" / tag
    model.save_pretrained(adapter)
    result["vllm_tensors"] = export_for_vllm(adapter, run_dir / "adapters" / f"{tag}-vllm")
    (run_dir / f"train_{tag}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
