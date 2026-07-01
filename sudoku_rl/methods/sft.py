from pathlib import Path
from typing import Callable

from sudoku_rl.tasks.sudoku import format_prompt


ExampleBuilder = Callable[[dict[str, str]], dict[str, object]]


def format_no_thinking_example(row: dict[str, str]) -> dict[str, object]:
    """Build no-thinking SFT data."""
    return {
        "prompt": [{"role": "user", "content": format_prompt(row["sudoku"])}],
        "completion": [{"role": "assistant", "content": row["solution"]}],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def format_thinking_example(row: dict[str, str]) -> dict[str, object]:
    """Build thinking SFT data."""
    prompt = (
        f"Solve this Sudoku:\n{row['sudoku']}\n"
        "You may reason through the puzzle first.\n"
        "Put the final completed 81-character solution between <answer> and </answer> tags."
        "\nDo not put anything except the final grid inside <answer>."
    )

    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": f"<answer>{row['solution']}</answer>"}],
        "chat_template_kwargs": {"enable_thinking": True},
    }


EXAMPLE_BUILDERS: dict[str, ExampleBuilder] = {
    "no-thinking": format_no_thinking_example,
    "thinking": format_thinking_example,
}


def _iter_sft_examples(train_path: str | Path, example_builder: ExampleBuilder):
    """Stream formatted SFT data from JSONL."""
    from data.utils import load_jsonl

    for row in load_jsonl(train_path):
        yield example_builder(row)


def train_sft(train_path: str | Path, model: str, output_dir: Path, example_builder: ExampleBuilder = format_no_thinking_example, max_steps: int = 500, batch_size: int = 8, gradient_accumulation_steps: int = 8, learning_rate: float = 2e-4, max_seq_length: int = 512, peft_config=None, eos_token: str | None = None, wandb: bool = False) -> None:
    """Train a causal language model with TRL SFT."""
    import os

    import torch
    from datasets import IterableDataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train_dataset = IterableDataset.from_generator(_iter_sft_examples, gen_kwargs={"train_path": train_path, "example_builder": example_builder})
    tokenizer = AutoTokenizer.from_pretrained(model)
    config_kwargs = {
        "output_dir": str(output_dir),
        "max_steps": max_steps,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "max_length": max_seq_length,
        "bf16": True,
        "gradient_checkpointing": True,
        "save_steps": 100,
        "logging_steps": 10,
        "report_to": "wandb" if wandb else "none",
        "completion_only_loss": True,
        "model_init_kwargs": {"dtype": torch.bfloat16},
    }
    if eos_token is not None: config_kwargs["eos_token"] = eos_token
    if wandb:
        os.environ.setdefault("WANDB_PROJECT", "sudoku-rl")
        config_kwargs["run_name"] = Path(output_dir).name

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**config_kwargs),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
