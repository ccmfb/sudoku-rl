import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen Sudoku policy with GRPO LoRA.")
    parser.add_argument("--train", required=True, type=Path, help="Path to JSONL rows with sudoku and solution fields.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Qwen model name or checkpoint path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for LoRA adapter checkpoints.")
    parser.add_argument("--limit", default=None, type=int, help="Maximum number of training rows to load.")
    parser.add_argument("--max-steps", default=500, type=int, help="Maximum optimizer steps.")
    parser.add_argument("--batch-size", default=8, type=int, help="Per-device train batch size.")
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", default=1e-6, type=float, help="LoRA learning rate.")
    parser.add_argument("--max-completion-length", default=128, type=int, help="Maximum generated tokens per rollout.")
    parser.add_argument("--num-generations", default=8, type=int, help="Number of completions sampled per prompt group.")
    parser.add_argument("--temperature", default=1.0, type=float, help="Sampling temperature.")
    parser.add_argument("--top-p", default=1.0, type=float, help="Nucleus sampling probability.")
    parser.add_argument("--top-k", default=0, type=int, help="Top-k sampling cutoff; 0 disables top-k filtering.")
    parser.add_argument("--beta", default=0.0, type=float, help="KL coefficient for the reference model.")
    parser.add_argument("--loss-type", choices=["grpo", "bnpo", "dr_grpo", "dapo"], default="dapo", help="TRL GRPO loss variant.")
    parser.add_argument("--scale-rewards", choices=["group", "batch", "none"], default="group", help="Reward normalization strategy.")
    parser.add_argument("--lora-r", default=16, type=int, help="LoRA rank.")
    parser.add_argument("--lora-alpha", default=32, type=int, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", default=0.05, type=float, help="LoRA dropout.")

    return parser.parse_args()


def main() -> None:
    """Train a Qwen Sudoku policy with GRPO LoRA."""
    from sudoku_rl.methods.grpo import train_grpo
    from sudoku_rl.models.peft import make_lora_config

    args = parse_args()
    peft_config = make_lora_config(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    train_grpo(
        args.train,
        model=args.model,
        output_dir=args.output_dir,
        limit=args.limit,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        beta=args.beta,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        peft_config=peft_config,
    )


if __name__ == "__main__":
    main()
