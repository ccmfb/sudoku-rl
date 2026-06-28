import pytest

from sudoku_rl.utils import evaluate_attempts, format_prompt, print_sudoku, score_attempt


SUDOKU = "1..5.37..6.3..8.9......98...1.......8761..........6...........7.8.9.76.47...6.312"
SOLUTION = "198543726643278591527619843914735268876192435235486179462351987381927654759864312"


def test_print_sudoku_formats_grid(capsys: pytest.CaptureFixture[str]) -> None:
    print_sudoku(SUDOKU)

    assert capsys.readouterr().out == (
        "+-------+-------+-------+\n"
        "| 1 . . | 5 . 3 | 7 . . |\n"
        "| 6 . 3 | . . 8 | . 9 . |\n"
        "| . . . | . . 9 | 8 . . |\n"
        "+-------+-------+-------+\n"
        "| . 1 . | . . . | . . . |\n"
        "| 8 7 6 | 1 . . | . . . |\n"
        "| . . . | . . 6 | . . . |\n"
        "+-------+-------+-------+\n"
        "| . . . | . . . | . . 7 |\n"
        "| . 8 . | 9 . 7 | 6 . 4 |\n"
        "| 7 . . | . 6 . | 3 1 2 |\n"
        "+-------+-------+-------+\n"
    )


def test_print_sudoku_prints_invalid_characters(capsys: pytest.CaptureFixture[str]) -> None:
    sudoku = "0" + "." * 80

    print_sudoku(sudoku)

    assert capsys.readouterr().out == f"INVALID: {sudoku}\n"


def test_print_sudoku_prints_wrong_length(capsys: pytest.CaptureFixture[str]) -> None:
    sudoku = "." * 80

    print_sudoku(sudoku)

    assert capsys.readouterr().out == f"INVALID: {sudoku}\n"


def test_score_attempt_returns_one_for_solution() -> None:
    assert score_attempt(SOLUTION, SUDOKU, SOLUTION) == 1.0


def test_score_attempt_scores_correct_missing_cells() -> None:
    missing_indices = [index for index, value in enumerate(SUDOKU) if value == "."]
    correct_indices = missing_indices[: len(missing_indices) // 2]
    wrong_indices = missing_indices[len(missing_indices) // 2 :]
    attempt = list(SOLUTION)

    for index in wrong_indices:
        attempt[index] = "1" if SOLUTION[index] != "1" else "2"

    expected = len(correct_indices) / len(missing_indices)

    assert score_attempt("".join(attempt), SUDOKU, SOLUTION) == expected


def test_score_attempt_returns_zero_when_clues_change() -> None:
    attempt = "2" + SOLUTION[1:]

    assert score_attempt(attempt, SUDOKU, SOLUTION) == 0.0


def test_score_attempt_returns_zero_for_invalid_attempt() -> None:
    assert score_attempt("." * 81, SUDOKU, SOLUTION) == 0.0


def test_format_prompt() -> None:
    assert format_prompt(SUDOKU) == (
        "Solve this Sudoku:\n"
        f"{SUDOKU}\n"
        "You may reason through the puzzle first.\n"
        "Put the final completed 81-character solution between <answer> and </answer> tags.\n"
        "Do not put anything except the final grid inside <answer>."
    )


def test_evaluate_attempts_scores_generated_attempts() -> None:
    class FixedPolicy:
        def __init__(self) -> None:
            self.prompts = []

        def attempt(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return SOLUTION

    policy = FixedPolicy()

    score = evaluate_attempts([{"sudoku": SUDOKU, "solution": SOLUTION}], policy)

    assert score == 1.0
    assert policy.prompts == [format_prompt(SUDOKU)]


def test_evaluate_attempts_verbose_prints_invalid_attempt(capsys: pytest.CaptureFixture[str]) -> None:
    class FixedPolicy:
        def attempt(self, prompt: str) -> str:
            return "123"

    score = evaluate_attempts([{"sudoku": SUDOKU, "solution": SOLUTION}], FixedPolicy(), verbose=True)
    output = capsys.readouterr().out

    assert score == 0.0
    assert "ATTEMPT:" in output
    assert "INVALID: 123" in output
    assert "SOLUTION:" in output
    assert "SCORE: 0.0" in output


def test_evaluate_attempts_returns_zero_for_empty_rows() -> None:
    class FixedPolicy:
        def attempt(self, prompt: str) -> str:
            return SOLUTION

    assert evaluate_attempts([], FixedPolicy()) == 0.0
