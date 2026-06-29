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
  --train data/train/all.jsonl \
  --model Qwen/Qwen3-8B \
  --output-dir runs/qwen3-8b-sft-lora-all-500steps \
  --max-steps 500
```

Evaluate the trained adapter:

```bash
python scripts/eval_qwen.py \
  --data data/eval/missing_1_100.jsonl \
  --model Qwen/Qwen3-8B \
  --adapter runs/qwen3-8b-sft-lora-all-500steps
```

Adapters and checkpoints should be written under `runs/`, which is ignored by git.
