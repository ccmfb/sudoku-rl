import json
import sys
import types

from sudoku_rl.methods.sft import EXAMPLE_BUILDERS, format_no_thinking_example, format_thinking_example, train_sft
from sudoku_rl.tasks.sudoku import format_prompt


SUDOKU = "1..5.37..6.3..8.9......98...1.......8761..........6...........7.8.9.76.47...6.312"
SOLUTION = "198543726643278591527619843914735268876192435235486179462351987381927654759864312"
ROW = {"sudoku": SUDOKU, "solution": SOLUTION}


def test_format_no_thinking_example_uses_prompt_completion_chat_shape() -> None:
    example = format_no_thinking_example(ROW)

    assert example == {
        "prompt": [{"role": "user", "content": format_prompt(SUDOKU)}],
        "completion": [{"role": "assistant", "content": SOLUTION}],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_format_thinking_example_uses_answer_tags_and_thinking_template() -> None:
    example = format_thinking_example(ROW)

    assert example == {
        "prompt": [
            {
                "role": "user",
                "content": (
                    f"Solve this Sudoku:\n{SUDOKU}\n"
                    "You may reason through the puzzle first.\n"
                    "Put the final completed 81-character solution between <answer> and </answer> tags."
                    "\nDo not put anything except the final grid inside <answer>."
                ),
            }
        ],
        "completion": [{"role": "assistant", "content": f"<answer>{SOLUTION}</answer>"}],
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_example_builders_expose_script_targets() -> None:
    assert EXAMPLE_BUILDERS["no-thinking"] is format_no_thinking_example
    assert EXAMPLE_BUILDERS["thinking"] is format_thinking_example


def test_train_qwen_sft_defaults_to_no_thinking_and_batch_eight(monkeypatch, tmp_path) -> None:
    from scripts.train_qwen_sft import parse_args

    monkeypatch.setattr(sys, "argv", ["train_qwen_sft.py", "--train", str(tmp_path / "train.jsonl"), "--output-dir", str(tmp_path / "run")])

    args = parse_args()

    assert args.target == "no-thinking"
    assert args.batch_size == 8


def test_train_sft_streams_jsonl_into_trl_trainer(monkeypatch, tmp_path) -> None:
    calls = {}
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps(ROW) + "\n")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model):
            calls["tokenizer_model"] = model
            return "tokenizer"

    class SFTConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["config"] = self

    class SFTTrainer:
        def __init__(self, model, args, train_dataset, processing_class, peft_config):
            calls["trainer"] = {
                "model": model,
                "args": args,
                "train_dataset": train_dataset,
                "processing_class": processing_class,
                "peft_config": peft_config,
            }

        def train(self):
            calls["trained"] = True

        def save_model(self, path):
            calls["saved_model"] = path

    class IterableDataset:
        @staticmethod
        def from_generator(generator, gen_kwargs):
            calls["dataset_items"] = list(generator(**gen_kwargs))
            return "streaming-dataset"

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(IterableDataset=IterableDataset))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer))
    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=SFTConfig, SFTTrainer=SFTTrainer))

    train_sft(
        train_path,
        model="base-model",
        output_dir=tmp_path,
        example_builder=lambda row: {"solution": row["solution"]},
        max_steps=3,
        batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_seq_length=128,
        peft_config="lora-config",
        eos_token="<eos>",
    )

    assert calls["dataset_items"] == [{"solution": SOLUTION}]
    assert calls["tokenizer_model"] == "base-model"
    assert calls["config"].kwargs == {
        "output_dir": str(tmp_path),
        "max_steps": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "max_length": 128,
        "bf16": True,
        "gradient_checkpointing": True,
        "save_steps": 100,
        "logging_steps": 10,
        "report_to": "none",
        "completion_only_loss": True,
        "model_init_kwargs": {"dtype": "bf16"},
        "eos_token": "<eos>",
    }
    assert calls["trainer"] == {
        "model": "base-model",
        "args": calls["config"],
        "train_dataset": "streaming-dataset",
        "processing_class": "tokenizer",
        "peft_config": "lora-config",
    }
    assert calls["trained"] is True
    assert calls["saved_model"] == str(tmp_path)
