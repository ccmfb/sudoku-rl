import json
import sys
import types

from sudoku_rl.methods.grpo import completion_text, format_grpo_example, sudoku_reward, train_grpo
from sudoku_rl.tasks.sudoku import format_prompt


SUDOKU = "1..5.37..6.3..8.9......98...1.......8761..........6...........7.8.9.76.47...6.312"
SOLUTION = "198543726643278591527619843914735268876192435235486179462351987381927654759864312"
ROW = {"sudoku": SUDOKU, "solution": SOLUTION}


def test_format_grpo_example_keeps_reward_columns() -> None:
    example = format_grpo_example(ROW)

    assert example == {
        "prompt": [{"role": "user", "content": format_prompt(SUDOKU)}],
        "sudoku": SUDOKU,
        "solution": SOLUTION,
    }


def test_completion_text_extracts_trl_conversational_completion() -> None:
    completion = [{"role": "assistant", "content": "first "}, {"role": "assistant", "content": "second"}]

    assert completion_text(completion) == "first second"


def test_sudoku_reward_scores_raw_and_tagged_completions() -> None:
    completions = [
        f" {SOLUTION}\n",
        [{"role": "assistant", "content": f"<answer>{SOLUTION}</answer>"}],
        "123",
    ]

    rewards = sudoku_reward(
        prompts=["first", "second", "third"],
        completions=completions,
        sudoku=[SUDOKU, SUDOKU, SUDOKU],
        solution=[SOLUTION, SOLUTION, SOLUTION],
    )

    assert rewards == [1.0, 1.0, 0.0]


def test_train_qwen_grpo_defaults_to_one_prompt_group(monkeypatch, tmp_path) -> None:
    from scripts.train_qwen_grpo import parse_args

    monkeypatch.setattr(sys, "argv", ["train_qwen_grpo.py", "--train", str(tmp_path / "train.jsonl"), "--output-dir", str(tmp_path / "run")])

    args = parse_args()

    assert args.batch_size == 8
    assert args.gradient_accumulation_steps == 1
    assert args.num_generations == 8
    assert args.learning_rate == 1e-6
    assert args.limit is None
    assert args.adapter is None
    assert args.wandb is False


def test_train_grpo_streams_jsonl_into_trl_trainer(monkeypatch, tmp_path) -> None:
    calls = {}
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps(ROW) + "\n")

    class Tokenizer:
        def __init__(self):
            self.padding_side = None
            self.pad_token = None
            self.eos_token = "<eos>"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model):
            calls["tokenizer_model"] = model
            return Tokenizer()

    class GRPOConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["config"] = self

    class GRPOTrainer:
        def __init__(self, model, reward_funcs, args, train_dataset, processing_class, peft_config):
            calls["trainer"] = {
                "model": model,
                "reward_funcs": reward_funcs,
                "args": args,
                "train_dataset": train_dataset,
                "processing_class": processing_class,
                "peft_config": peft_config,
            }

        def train(self):
            calls["trained"] = True

        def save_model(self, path):
            calls["saved_model"] = path

    class Dataset:
        @staticmethod
        def from_generator(generator, gen_kwargs):
            calls["dataset_kwargs"] = gen_kwargs
            calls["dataset_items"] = list(generator(**gen_kwargs))
            return "map-style-dataset"

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=Dataset))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer))
    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(GRPOConfig=GRPOConfig, GRPOTrainer=GRPOTrainer))
    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    train_grpo(
        train_path,
        model="base-model",
        output_dir=tmp_path,
        limit=1,
        max_steps=3,
        batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-7,
        max_completion_length=64,
        num_generations=4,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        beta=0.01,
        loss_type="dr_grpo",
        scale_rewards="none",
        peft_config="lora-config",
        wandb=True,
    )

    tokenizer = calls["trainer"]["processing_class"]

    assert calls["dataset_kwargs"] == {"train_path": train_path, "limit": 1}
    assert calls["dataset_items"] == [format_grpo_example(ROW)]
    assert calls["tokenizer_model"] == "base-model"
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<eos>"
    assert calls["config"].kwargs == {
        "output_dir": str(tmp_path),
        "max_steps": 3,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-7,
        "max_completion_length": 64,
        "num_generations": 4,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "beta": 0.01,
        "loss_type": "dr_grpo",
        "scale_rewards": "none",
        "bf16": True,
        "gradient_checkpointing": True,
        "save_steps": 100,
        "logging_steps": 10,
        "report_to": "wandb",
        "remove_unused_columns": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "model_init_kwargs": {"dtype": "bf16"},
        "run_name": tmp_path.name,
    }
    assert calls["trainer"] == {
        "model": "base-model",
        "reward_funcs": sudoku_reward,
        "args": calls["config"],
        "train_dataset": "map-style-dataset",
        "processing_class": tokenizer,
        "peft_config": "lora-config",
    }
    assert calls["trained"] is True
    assert calls["saved_model"] == str(tmp_path)


def test_train_grpo_continues_from_trainable_adapter(monkeypatch, tmp_path) -> None:
    calls = {}
    train_path = tmp_path / "train.jsonl"
    adapter_path = tmp_path / "adapter"
    train_path.write_text(json.dumps(ROW) + "\n")

    class Tokenizer:
        def __init__(self):
            self.padding_side = None
            self.pad_token = None
            self.eos_token = "<eos>"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model):
            calls["tokenizer_model"] = model
            return Tokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model, torch_dtype, device_map):
            calls["base_model"] = {"model": model, "torch_dtype": torch_dtype, "device_map": device_map}
            return "base-model-object"

    class PeftModel:
        @staticmethod
        def from_pretrained(model, adapter, is_trainable):
            calls["adapter_model"] = {"model": model, "adapter": adapter, "is_trainable": is_trainable}
            return "trainable-adapter-model"

    class GRPOConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["config"] = self

    class GRPOTrainer:
        def __init__(self, model, reward_funcs, args, train_dataset, processing_class, peft_config):
            calls["trainer"] = {
                "model": model,
                "reward_funcs": reward_funcs,
                "args": args,
                "train_dataset": train_dataset,
                "processing_class": processing_class,
                "peft_config": peft_config,
            }

        def train(self):
            calls["trained"] = True

        def save_model(self, path):
            calls["saved_model"] = path

    class Dataset:
        @staticmethod
        def from_generator(generator, gen_kwargs):
            calls["dataset_items"] = list(generator(**gen_kwargs))
            return "map-style-dataset"

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=Dataset))
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=PeftModel))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModelForCausalLM))
    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(GRPOConfig=GRPOConfig, GRPOTrainer=GRPOTrainer))

    train_grpo(
        train_path,
        model="base-model",
        output_dir=tmp_path / "out",
        adapter_path=adapter_path,
        peft_config="unused-lora-config",
        max_steps=2,
    )

    assert calls["base_model"] == {"model": "base-model", "torch_dtype": "bf16", "device_map": "auto"}
    assert calls["adapter_model"] == {"model": "base-model-object", "adapter": adapter_path, "is_trainable": True}
    assert calls["dataset_items"] == [format_grpo_example(ROW)]
    assert calls["config"].kwargs["output_dir"] == str(tmp_path / "out")
    assert calls["config"].kwargs["max_steps"] == 2
    assert "model_init_kwargs" not in calls["config"].kwargs
    assert calls["trainer"]["model"] == "trainable-adapter-model"
    assert calls["trainer"]["peft_config"] is None
    assert calls["trained"] is True
    assert calls["saved_model"] == str(tmp_path / "out")
