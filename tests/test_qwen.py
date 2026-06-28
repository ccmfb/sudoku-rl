import pytest

pytestmark = pytest.mark.gpu

qwen = pytest.importorskip("sudoku_rl.qwen")
QwenPolicy = qwen.QwenPolicy
extract_answer = qwen.extract_answer


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


def test_qwen_policy_attempt_returns_extracted_answer() -> None:
    policy = QwenPolicy.__new__(QwenPolicy)
    prompts = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return f"I will reason first. <answer>{ATTEMPT}</answer>"

    policy.complete = complete

    assert policy.attempt("solve this") == ATTEMPT
    assert prompts == ["solve this"]
