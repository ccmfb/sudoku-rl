import argparse
import sys
import types

import pytest

from scripts import train_qwen_grpo_curriculum as curriculum


def test_log_wandb_round_logs_curriculum_metrics(monkeypatch) -> None:
    calls = {}

    def log(metrics, step):
        calls["metrics"] = metrics
        calls["step"] = step

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(log=log))

    args = argparse.Namespace(target_low=0.10, target_high=0.80, steps_per_round=100)

    curriculum.log_wandb_round(3, {1: 0.99, 10: 0.55}, 10, args)

    assert calls == {
        "metrics": {
            "curriculum/round": 3,
            "curriculum/selected_missing_count": 10,
            "curriculum/target_low": 0.10,
            "curriculum/target_high": 0.80,
            "train/missing_count": 10,
            "train/steps_per_round": 100,
            "eval/missing_1": 0.99,
            "eval/missing_10": 0.55,
        },
        "step": 3,
    }


def test_run_curriculum_starts_from_base_when_adapter_is_missing(monkeypatch, tmp_path) -> None:
    from sudoku_rl.methods import grpo
    from sudoku_rl.models import peft

    calls = {"train": []}
    output_dir = tmp_path / "runs" / "curriculum"
    data_dir = tmp_path / "data" / "train"
    data_dir.mkdir(parents=True)
    (data_dir / "missing_10_7.jsonl").write_text("")

    monkeypatch.setattr(curriculum, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(curriculum, "evaluate_levels", lambda *args: {1: 0.99, 10: 0.55})
    monkeypatch.setattr(grpo, "train_grpo", lambda train_path, **kwargs: calls["train"].append({"train_path": train_path, **kwargs}))
    monkeypatch.setattr(peft, "make_lora_config", lambda **kwargs: {"lora": kwargs})

    args = argparse.Namespace(
        model="base-model",
        adapter=None,
        output_dir=output_dir,
        rounds=1,
        steps_per_round=100,
        train_set_size=7,
        train_limit=None,
        eval_set_size=11,
        eval_limit=None,
        missing_counts=[1, 10],
        target_low=0.10,
        target_high=0.80,
        batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        max_completion_length=128,
        num_generations=8,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        loss_type="dr_grpo",
        scale_rewards="none",
        eval_batch_size=8,
        max_new_tokens=256,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        wandb=False,
    )

    curriculum.run_curriculum(args)

    assert calls["train"][0]["train_path"] == tmp_path / "data" / "train" / "missing_10_7.jsonl"
    assert calls["train"][0]["adapter_path"] is None
    assert calls["train"][0]["peft_config"] == {"lora": {"r": 16, "alpha": 32, "dropout": 0.05}}


@pytest.mark.parametrize(
    ("scores", "selected_missing_count"),
    [
        ({1: 0.97, 2: 0.90, 5: 0.03, 10: 0.0}, 5),
        ({1: 0.99, 2: 0.96, 5: 0.91}, 5),
    ],
)
def test_run_curriculum_selects_fallback_level(monkeypatch, tmp_path, scores, selected_missing_count) -> None:
    from sudoku_rl.methods import grpo
    from sudoku_rl.models import peft

    calls = {"train": []}
    output_dir = tmp_path / "runs" / "curriculum"
    data_dir = tmp_path / "data" / "train"
    data_dir.mkdir(parents=True)
    (data_dir / f"missing_{selected_missing_count}_7.jsonl").write_text("")

    monkeypatch.setattr(curriculum, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(curriculum, "evaluate_levels", lambda *args: scores)
    monkeypatch.setattr(grpo, "train_grpo", lambda train_path, **kwargs: calls["train"].append({"train_path": train_path, **kwargs}))
    monkeypatch.setattr(peft, "make_lora_config", lambda **kwargs: "lora-config")

    args = argparse.Namespace(
        model="base-model",
        adapter=None,
        output_dir=output_dir,
        rounds=1,
        steps_per_round=100,
        train_set_size=7,
        train_limit=None,
        eval_set_size=11,
        eval_limit=None,
        missing_counts=list(scores),
        target_low=0.10,
        target_high=0.80,
        batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        max_completion_length=128,
        num_generations=8,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        loss_type="dr_grpo",
        scale_rewards="none",
        eval_batch_size=8,
        max_new_tokens=256,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        wandb=False,
    )

    curriculum.run_curriculum(args)

    assert calls["train"][0]["train_path"] == tmp_path / "data" / "train" / f"missing_{selected_missing_count}_7.jsonl"


def test_run_curriculum_evaluates_trains_and_hands_off_adapters(monkeypatch, tmp_path) -> None:
    from sudoku_rl.methods import grpo
    from sudoku_rl.models import peft

    calls = {"eval": [], "train": []}
    output_dir = tmp_path / "runs" / "curriculum"
    start_adapter = tmp_path / "runs" / "sft"
    data_dir = tmp_path / "data" / "train"
    data_dir.mkdir(parents=True)
    (data_dir / "missing_20_7.jsonl").write_text("")
    (data_dir / "missing_40_7.jsonl").write_text("")
    score_rounds = [
        {1: 0.99, 10: 0.55, 20: 0.14, 40: 0.01},
        {1: 0.99, 10: 0.90, 20: 0.85, 40: 0.25},
    ]

    def evaluate_levels(model, adapter_path, missing_counts, eval_set_size, eval_limit, max_new_tokens, batch_size):
        calls["eval"].append(
            {
                "model": model,
                "adapter_path": adapter_path,
                "missing_counts": missing_counts,
                "eval_set_size": eval_set_size,
                "eval_limit": eval_limit,
                "max_new_tokens": max_new_tokens,
                "batch_size": batch_size,
            }
        )
        return score_rounds.pop(0)

    def train_grpo(train_path, **kwargs):
        calls["train"].append({"train_path": train_path, **kwargs})

    monkeypatch.setattr(curriculum, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(curriculum, "evaluate_levels", evaluate_levels)
    monkeypatch.setattr(grpo, "train_grpo", train_grpo)
    monkeypatch.setattr(peft, "make_lora_config", lambda **kwargs: "unused-lora-config")

    args = argparse.Namespace(
        model="base-model",
        adapter=start_adapter,
        output_dir=output_dir,
        rounds=2,
        steps_per_round=100,
        train_set_size=7,
        train_limit=3,
        eval_set_size=11,
        eval_limit=5,
        missing_counts=[1, 10, 20, 40],
        target_low=0.10,
        target_high=0.80,
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
        eval_batch_size=4,
        max_new_tokens=128,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        wandb=False,
    )

    curriculum.run_curriculum(args)

    first_round_dir = output_dir / "round-000-missing-20"
    second_round_dir = output_dir / "round-001-missing-40"
    assert calls["eval"][0]["adapter_path"] == start_adapter
    assert calls["eval"][1]["adapter_path"] == first_round_dir
    assert calls["train"][0]["train_path"] == tmp_path / "data" / "train" / "missing_20_7.jsonl"
    assert calls["train"][0]["adapter_path"] == start_adapter
    assert calls["train"][0]["peft_config"] is None
    assert calls["train"][0]["max_steps"] == 100
    assert calls["train"][1]["train_path"] == tmp_path / "data" / "train" / "missing_40_7.jsonl"
    assert calls["train"][1]["adapter_path"] == first_round_dir
    assert calls["train"][1]["output_dir"] == second_round_dir
