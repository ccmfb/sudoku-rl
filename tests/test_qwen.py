from sudoku_rl.qwen import QwenPolicy


ATTEMPT = "123456789" * 9


def test_qwen_policy_attempt_returns_generated_response() -> None:
    policy = QwenPolicy.__new__(QwenPolicy)
    prompts = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return f"Here is the answer:\n{ATTEMPT}"

    policy.complete = complete

    assert policy.attempt("solve this") == f"Here is the answer:\n{ATTEMPT}"
    assert prompts == ["solve this"]
