from pathlib import Path

from sudoku_rl.models.qwen import QWEN_CHAT_EOS_TOKEN
from sudoku_rl.tasks.sudoku import format_prompt


def format_training_example(row: dict[str, str]) -> dict[str, object]:
    """Format a row for Qwen supervised fine-tuning."""
    return {
        "prompt": [{"role": "user", "content": format_prompt(row["sudoku"])}],
        "completion": [{"role": "assistant", "content": row["solution"]}],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def train_qwen_sft(train_path: Path, model: str, output_dir: Path, max_steps: int = 500, batch_size: int = 1, gradient_accumulation_steps: int = 8, learning_rate: float = 2e-4, max_seq_length: int = 512, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05) -> None:
    """Train a Qwen Sudoku policy with SFT LoRA."""
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from data.utils import load_jsonl
    from sudoku_rl.models.peft import make_lora_config

    rows = list(load_jsonl(train_path))
    train_dataset = Dataset.from_list([format_training_example(row) for row in rows])
    tokenizer = AutoTokenizer.from_pretrained(model)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(output_dir),
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_length=max_seq_length,
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
        peft_config=make_lora_config(r=lora_r, alpha=lora_alpha, dropout=lora_dropout),
    )

    trainer.train()
    trainer.save_model(str(output_dir))
