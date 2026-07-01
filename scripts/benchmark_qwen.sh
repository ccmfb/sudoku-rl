#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/eval_qwen.py \
  --data data/eval/missing_5_100.jsonl \
  --model Qwen/Qwen3-8B \
  --batch-size 16

uv run python scripts/eval_qwen.py \
  --data data/eval/missing_10_100.jsonl \
  --model Qwen/Qwen3-8B \
  --batch-size 16

uv run python scripts/eval_qwen.py \
  --data data/eval/missing_20_100.jsonl \
  --model Qwen/Qwen3-8B \
  --batch-size 16

uv run python scripts/eval_qwen.py \
  --data data/eval/missing_40_100.jsonl \
  --model Qwen/Qwen3-8B \
  --batch-size 16