# Scripts

Run these commands from the repository root.

## Qwen Baseline Eval

Evaluate the base Qwen policy before training:

```bash
python scripts/eval_qwen.py \
  --data data/eval/radcliffe_100.jsonl \
  --model Qwen/Qwen3-8B \
  --limit 100
```

## Qwen SFT LoRA

Install the GPU and training dependencies:

```bash
uv sync --group gpu
```

Train a first real LoRA adapter on the local Radcliffe split:

```bash
python scripts/train_qwen_sft.py \
  --train data/train/radcliffe_1000.jsonl \
  --model Qwen/Qwen3-8B \
  --output-dir runs/qwen3-8b-sft-lora-radcliffe-1000-500steps \
  --max-steps 500
```

Evaluate the trained adapter:

```bash
python scripts/eval_qwen.py \
  --data data/eval/radcliffe_100.jsonl \
  --model Qwen/Qwen3-8B \
  --adapter runs/qwen3-8b-sft-lora-radcliffe-1000-500steps \
  --limit 100
```

Inspect a few generated grids:

```bash
python scripts/eval_qwen.py \
  --data data/eval/radcliffe_100.jsonl \
  --model Qwen/Qwen3-8B \
  --adapter runs/qwen3-8b-sft-lora-radcliffe-1000-500steps \
  --limit 10 \
  --verbose
```

Adapters and checkpoints should be written under `runs/`, which is ignored by git.
