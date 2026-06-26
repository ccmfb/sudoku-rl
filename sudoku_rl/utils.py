from collections.abc import Callable, Iterable


def evaluate_attempts(rows: Iterable[dict[str, str]], generate_attempt: Callable[[str, str], str]) -> float:
    """Score generated attempts for sudoku rows."""
    total_score = 0.0
    count = 0

    for row in rows:
        sudoku = row["sudoku"]
        solution = row["solution"]
        prompt = format_prompt(sudoku)
        attempt = generate_attempt(sudoku, prompt)

        total_score += score_attempt(attempt, sudoku, solution)
        count += 1

    if count == 0: return 0.0

    return total_score / count

def format_prompt(sudoku: str) -> str:
    """Format a sudoku prompt for a model."""
    return f"Solve this Sudoku:\n{sudoku}\nReturn only the completed 81-character solution."

def score_attempt(attempt: str, sudoku: str, solution: str) -> float:
    """Computes score of an attempted solution."""
    if not valid_solution(attempt): return 0.0

    for index, value in enumerate(sudoku):
        if value == ".": continue
        if value != attempt[index]: return 0.0

    missing_count = sudoku.count(".")
    correct_count = 0

    for index, value in enumerate(sudoku):
        if value != ".": continue
        if attempt[index] == solution[index]: correct_count += 1

    return correct_count / missing_count

def valid_sudoku(sudoku: str) -> bool:
    """Checks whether a sudoku string is valid."""
    invalid = set(sudoku) - set(".123456789")

    if invalid: return False
    if len(sudoku) != 81: return False

    return True

def valid_solution(solution: str) -> bool:
    """Checks whether a solution string to a sudoku is valid."""
    invalid = set(solution) - set("123456789")

    if invalid: return False
    if len(solution) != 81: return False

    return True

def print_sudoku(sudoku: str) -> None:
    """Print a sudoku string in the terminal using ASCII."""
    if not valid_sudoku(sudoku): raise ValueError(f"Invalid sudoku: {sudoku}")

    border = "+-------+-------+-------+"

    print(border)
    for row_index in range(9):
        row = sudoku[row_index * 9 : (row_index + 1) * 9]

        chunks = [
            " ".join(row[0:3]),
            " ".join(row[3:6]),
            " ".join(row[6:9]),
        ]

        print("| " + " | ".join(chunks) + " |")

        if row_index in {2, 5, 8}:
            print(border)
