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

To send TRL training metrics to W&B, log in once and pass `--wandb`.
Runs use the `sudoku-rl` W&B project and the output directory name as the run name.

```bash
uv run wandb login
uv run python scripts/train_qwen_sft.py \
  --train data/train/all.jsonl \
  --model Qwen/Qwen3-8B \
  --output-dir runs/qwen3-8b-sft-lora-all-500steps \
  --max-steps 500 \
  --wandb
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

## Qwen GRPO LoRA

Train a fresh LoRA adapter from the base model with verifier rewards:

```bash
python scripts/train_qwen_grpo.py \
  --train data/train/all.jsonl \
  --model Qwen/Qwen3-8B \
  --output-dir runs/qwen3-8b-grpo-lora-all-500steps \
  --max-steps 500
```

The same `--wandb` flag works for GRPO.

For a quick smoke run, add `--limit 1000` so TRL only materializes the first 1,000 rows.

Adapters and checkpoints should be written under `runs/`, which is ignored by git.
