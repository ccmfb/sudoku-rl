from sudoku_rl.qwen import extract_attempt


ATTEMPT = "123456789" * 9


def test_extract_attempt_returns_81_digit_solution() -> None:
    assert extract_attempt(f"Here is the answer:\n{ATTEMPT}") == ATTEMPT


def test_extract_attempt_ignores_whitespace() -> None:
    spaced_attempt = " ".join(ATTEMPT[index : index + 9] for index in range(0, 81, 9))

    assert extract_attempt(spaced_attempt) == ATTEMPT


def test_extract_attempt_rejects_zero_or_dots() -> None:
    assert extract_attempt("." * 81) == ""
    assert extract_attempt("0" * 81) == ""


def test_extract_attempt_returns_empty_string_without_attempt() -> None:
    assert extract_attempt("I cannot solve this one.") == ""
