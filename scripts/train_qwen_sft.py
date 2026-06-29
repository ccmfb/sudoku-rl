import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen Sudoku policy with SFT LoRA.")
    parser.add_argument("--train", required=True, type=Path, help="Path to JSONL rows with sudoku and solution fields.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Qwen model name or checkpoint path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for LoRA adapter checkpoints.")
    parser.add_argument("--target", choices=["no-thinking", "thinking"], default="no-thinking", help="SFT example shape.")
    parser.add_argument("--max-steps", default=500, type=int, help="Maximum optimizer steps.")
    parser.add_argument("--batch-size", default=8, type=int, help="Per-device train batch size.")
    parser.add_argument("--gradient-accumulation-steps", default=8, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", default=2e-4, type=float, help="LoRA learning rate.")
    parser.add_argument("--max-seq-length", default=512, type=int, help="Maximum tokenized sequence length.")
    parser.add_argument("--lora-r", default=16, type=int, help="LoRA rank.")
    parser.add_argument("--lora-alpha", default=32, type=int, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", default=0.05, type=float, help="LoRA dropout.")

    return parser.parse_args()


def main() -> None:
    """Train a Qwen Sudoku policy with SFT LoRA."""
    from sudoku_rl.methods.sft import EXAMPLE_BUILDERS, train_sft
    from sudoku_rl.models.peft import make_lora_config
    from sudoku_rl.models.qwen import QWEN_CHAT_EOS_TOKEN

    args = parse_args()
    peft_config = make_lora_config(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    train_sft(
        args.train,
        model=args.model,
        output_dir=args.output_dir,
        example_builder=EXAMPLE_BUILDERS[args.target],
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        peft_config=peft_config,
        eos_token=QWEN_CHAT_EOS_TOKEN,
    )


if __name__ == "__main__":
    main()
