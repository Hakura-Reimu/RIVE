#!/usr/bin/env bash
# vLLM server for Qwen3.5-9B with a 64k context on the 12 GB RTX 4070 (docs/rive-design.md section 6.23).
#   bash scripts/serve9b.sh [more vllm options]    e.g. --enable-lora --max-lora-rank 64 --max-loras 1
#   MODEL=~/alter/models/Qwen3.5-9B-nowait-e1l8 bash scripts/serve9b.sh ...   another copy of the weights
# FP8 weights would take about 10.4 GiB (the 248,320 x 4096 embedding and output matrices stay bf16, about 4 GB), so
# the linear layers are quantized to MXFP4 while loading (Marlin, 4-bit weights, 16-bit activations; the vision
# tower is neither quantized nor used) and the KV cache is FP8: 8 of the 32 layers keep a KV cache, 16 KiB per
# token. The embedding table stays in host memory (a lookup reads one row per token). Measured 2026-11-04: weights
# 7.26 GiB; KV cache 79,782 tokens without LoRA and 69,335 with --enable-lora --max-lora-rank 16 --max-loras 1; a
# 60,822-token prompt is read in 22 s. Windows keeps about 1.2 GiB of the card, so 0.90 is the highest share.
set -euo pipefail
exec ~/alter/.venv/bin/vllm serve "${MODEL:-$HOME/alter/models/Qwen3.5-9B}" \
  --served-model-name qwen3.5-9b \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --quantization online --quantization-config '{"linear": "mxfp4", "ignore": ["*visual*"]}' \
  --kv-cache-dtype fp8 --max-model-len 65536 --max-num-batched-tokens 2048 \
  --cpu-offload-gb 2.1 --cpu-offload-params embed_tokens \
  --gpu-memory-utilization 0.90 --max-num-seqs 8 \
  --language-model-only \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  --seed 0 "$@"
