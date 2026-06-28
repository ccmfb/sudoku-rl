import sys
import types

import pytest

from sudoku_rl.tasks.sudoku import extract_answer

pytestmark = pytest.mark.gpu

qwen = pytest.importorskip("sudoku_rl.models.qwen")
QwenPolicy = qwen.QwenPolicy


ATTEMPT = "123456789" * 9


def test_extract_answer_returns_last_tagged_grid() -> None:
    text = f"<answer>123</answer> thinking <answer>{ATTEMPT}</answer>"

    assert extract_answer(text) == ATTEMPT


def test_extract_answer_strips_grid_formatting_inside_answer_tags() -> None:
    text = "<answer>" + "\n".join(ATTEMPT[index : index + 9] for index in range(0, 81, 9)) + "</answer>"

    assert extract_answer(text) == ATTEMPT


def test_extract_answer_returns_empty_string_for_missing_or_malformed_answer() -> None:
    assert extract_answer(ATTEMPT) == ""
    assert extract_answer("<answer>123</answer>") == ""


def test_qwen_policy_attempt_extracts_answer_when_thinking() -> None:
    policy = QwenPolicy.__new__(QwenPolicy)
    prompts = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return f"I will reason first. <answer>{ATTEMPT}</answer>"

    policy.complete = complete
    policy.thinking = True

    assert policy.attempt("solve this") == ATTEMPT
    assert prompts == ["solve this"]


def test_qwen_policy_attempt_extracts_batched_answers_when_thinking() -> None:
    policy = QwenPolicy.__new__(QwenPolicy)

    def complete(prompts: list[str]) -> list[str]:
        assert prompts == ["first", "second"]
        return [f"<answer>{ATTEMPT}</answer>", "<answer>123</answer>"]

    policy.complete = complete
    policy.thinking = True

    assert policy.attempt(["first", "second"]) == [ATTEMPT, ""]


def test_qwen_policy_attempt_preserves_raw_response_without_tagged_answer() -> None:
    policy = QwenPolicy.__new__(QwenPolicy)
    response = f"Here is the answer:\n{ATTEMPT}"

    def complete(prompt: str) -> str:
        return response

    policy.complete = complete
    policy.thinking = False

    assert policy.attempt("solve this") == response


def test_qwen_policy_attempt_returns_empty_answer_when_thinking_without_tagged_answer() -> None:
    policy = QwenPolicy.__new__(QwenPolicy)

    def complete(prompt: str) -> str:
        return f"<think>reasoning</think> {ATTEMPT}"

    policy.complete = complete
    policy.thinking = True

    assert policy.attempt("solve this") == ""


@pytest.mark.parametrize(
    ("thinking", "expected_generation_kwargs"),
    [
        (False, {"max_new_tokens": 7, "do_sample": False}),
        (True, {"max_new_tokens": 7, "do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}),
    ],
)
def test_qwen_policy_complete_uses_generation_settings(monkeypatch, thinking: bool, expected_generation_kwargs: dict[str, object]) -> None:
    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return None

    class InputIds:
        shape = (1, 2)

    class Inputs(dict):
        def to(self, device):
            return self

    class Outputs:
        def __getitem__(self, key):
            return ["generated"]

    class Tokenizer:
        def __init__(self):
            self.messages = None
            self.thinking = None
            self.texts = None
            self.padding = None
            self.padding_side = None
            self.pad_token = None
            self.eos_token = "<eos>"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            self.messages = messages
            self.thinking = enable_thinking
            return "chat text"

        def __call__(self, texts, padding, return_tensors):
            self.texts = texts
            self.padding = padding
            return Inputs(input_ids=InputIds())

        def batch_decode(self, generated, skip_special_tokens):
            return ["decoded"]

    class Model:
        device = "cpu"

        def __init__(self):
            self.generation_kwargs = None

        def generate(self, **kwargs):
            self.generation_kwargs = kwargs
            return Outputs()

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(inference_mode=lambda: InferenceMode()))

    policy = QwenPolicy.__new__(QwenPolicy)
    policy.tokenizer = Tokenizer()
    policy.model = Model()
    policy.max_new_tokens = 7
    policy.thinking = thinking

    assert policy.complete("solve this") == "decoded"
    assert policy.tokenizer.thinking is thinking
    assert policy.tokenizer.texts == ["chat text"]
    assert policy.tokenizer.padding is True
    assert policy.tokenizer.padding_side == "left"
    assert policy.tokenizer.pad_token == "<eos>"

    prompt = policy.tokenizer.messages[0]["content"]
    if thinking:
        assert prompt.startswith("solve this\n")
        assert "<answer>" in prompt
    else:
        assert prompt == "solve this"

    generation_kwargs = dict(policy.model.generation_kwargs)
    generation_kwargs.pop("input_ids")
    assert generation_kwargs == expected_generation_kwargs
