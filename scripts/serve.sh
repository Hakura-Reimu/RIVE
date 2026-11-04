#!/usr/bin/env bash
# Long-running vLLM server for Phase A (research loop + evaluation share it). Run as the login user in WSL.
#   bash scripts/serve.sh [model directory] [served name] [more vllm options]    (default: Qwen3.5-4B as qwen3.5-4b)
#   Phase D adds the adapters: ... qwen3.5-4b --enable-lora --max-lora-rank 16 --lora-modules seed1=<dir> ...
# Settings from Phase 0: max_num_seqs <= 24 (Qwen3.5 linear-attention state blocks), text-only.
# 16384 left only 0.4 GiB for KV cache on the 12 GB card; 12288 with 4096-token prefill chunks fits the
# longest Phase A prompts (tool pool K=32: ~5.7k tokens) plus outputs.
# disable_any_whitespace: with JSON-schema decoding the 4B model otherwise emits whitespace until max_tokens.
# RIVE from 2026-11-03 (user's choice, docs/rive-design.md 6.12): weights quantized to FP8 on loading. Weights take
# 4.71 instead of 7.94 GiB and the KV cache holds ~104k instead of ~46k tokens, which allows 20480-token requests.
# Results before this date are bf16 (--dtype bfloat16 without the quantization option and --max-model-len 12288).
set -euo pipefail
exec ~/alter/.venv/bin/vllm serve "${1:-$HOME/alter/models/Qwen3.5-4B}" \
  --served-model-name "${2:-qwen3.5-4b}" \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --quantization fp8_per_tensor --max-model-len 20480 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.86 --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  --seed 0 "${@:3}"
