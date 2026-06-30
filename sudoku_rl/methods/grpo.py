from pathlib import Path
from typing import Any

from sudoku_rl.tasks.rewards import score_attempt
from sudoku_rl.tasks.sudoku import extract_answer, format_prompt


def format_grpo_example(row: dict[str, str]) -> dict[str, object]:
    """Build no-thinking GRPO data."""
    return {
        "prompt": [{"role": "user", "content": format_prompt(row["sudoku"])}],
        "sudoku": row["sudoku"],
        "solution": row["solution"],
    }


def completion_text(completion: Any) -> str:
    """Extract text from a TRL completion."""
    if isinstance(completion, str): return completion
    if isinstance(completion, dict): return str(completion.get("content", ""))
    if isinstance(completion, list): return "".join(completion_text(item) for item in completion)

    return str(completion)


def sudoku_reward(prompts, completions, sudoku, solution, **kwargs) -> list[float]:
    """Score GRPO completions with the Sudoku verifier."""
    rewards = []

    for completion, puzzle, answer in zip(completions, sudoku, solution, strict=True):
        text = completion_text(completion)
        attempt = extract_answer(text) or text.strip()
        rewards.append(score_attempt(attempt, puzzle, answer))

    return rewards


def _iter_grpo_examples(train_path: str | Path, limit: int | None = None):
    """Stream formatted GRPO data from JSONL."""
    from data.utils import load_jsonl

    for row in load_jsonl(train_path, limit=limit):
        yield format_grpo_example(row)


def train_grpo(train_path: str | Path, model: str, output_dir: Path, limit: int | None = None, max_steps: int = 500, batch_size: int = 8, gradient_accumulation_steps: int = 1, learning_rate: float = 1e-6, max_completion_length: int = 128, num_generations: int = 8, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, beta: float = 0.0, loss_type: str = "dapo", scale_rewards: str = "group", peft_config=None) -> None:
    """Train a causal language model with TRL GRPO."""
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    train_dataset = Dataset.from_generator(_iter_grpo_examples, gen_kwargs={"train_path": train_path, "limit": limit})
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=sudoku_reward,
        args=GRPOConfig(
            output_dir=str(output_dir),
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_completion_length=max_completion_length,
            num_generations=num_generations,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            beta=beta,
            loss_type=loss_type,
            scale_rewards=scale_rewards,
            bf16=True,
            gradient_checkpointing=True,
            save_steps=100,
            logging_steps=10,
            report_to="none",
            remove_unused_columns=False,
            chat_template_kwargs={"enable_thinking": False},
            model_init_kwargs={"dtype": torch.bfloat16},
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
