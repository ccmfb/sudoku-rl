import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

QWEN_CHAT_EOS_TOKEN = "<|im_end|>"


def format_training_example(row: dict[str, str]) -> dict[str, object]:
    """Format a row for Qwen supervised fine-tuning."""
    from sudoku_rl.utils import format_prompt

    return {
        "prompt": [{"role": "user", "content": format_prompt(row["sudoku"])}],
        "completion": [{"role": "assistant", "content": f"<answer>{row['solution']}</answer>"}],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen Sudoku policy with SFT LoRA.")
    parser.add_argument("--train", required=True, type=Path, help="Path to JSONL rows with sudoku and solution fields.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Qwen model name or checkpoint path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for LoRA adapter checkpoints.")
    parser.add_argument("--max-steps", default=500, type=int, help="Maximum optimizer steps.")
    parser.add_argument("--batch-size", default=1, type=int, help="Per-device train batch size.")
    parser.add_argument("--gradient-accumulation-steps", default=8, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", default=2e-4, type=float, help="LoRA learning rate.")
    parser.add_argument("--max-seq-length", default=512, type=int, help="Maximum tokenized sequence length.")
    parser.add_argument("--lora-r", default=16, type=int, help="LoRA rank.")
    parser.add_argument("--lora-alpha", default=32, type=int, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", default=0.05, type=float, help="LoRA dropout.")

    return parser.parse_args()


def main() -> None:
    """Train a Qwen Sudoku policy with SFT LoRA."""
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from data.utils import load_jsonl

    rows = list(load_jsonl(args.train))
    train_dataset = Dataset.from_list([format_training_example(row) for row in rows])
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    trainer = SFTTrainer(
        model=args.model,
        args=SFTConfig(
            output_dir=str(args.output_dir),
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_seq_length,
            bf16=True,
            gradient_checkpointing=True,
            save_steps=100,
            logging_steps=10,
            report_to="none",
            model_init_kwargs={"dtype": torch.bfloat16},
            eos_token=QWEN_CHAT_EOS_TOKEN,
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
