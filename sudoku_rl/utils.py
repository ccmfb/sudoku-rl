from collections.abc import Iterable


def evaluate_attempts(rows: Iterable[dict[str, str]], policy, verbose: bool = False, batch_size: int = 1) -> float:
    """Score generated attempts for sudoku rows."""
    total_score = 0.0
    count = 0
    rows = list(rows)

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start : batch_start + batch_size]
        prompts = [format_prompt(row["sudoku"]) for row in batch]
        attempts = [policy.attempt(prompts[0])] if batch_size == 1 else policy.attempt(prompts)

        for offset, row in enumerate(batch):
            row_index = batch_start + offset
            sudoku = row["sudoku"]
            solution = row["solution"]
            attempt = attempts[offset]
            score = score_attempt(attempt, sudoku, solution)
            total_score += score
            count += 1

            if not verbose: continue

            print()
            print(f"ATTEMPT {row_index}:")
            print_sudoku(attempt)
            print()
            print(f"SOLUTION {row_index}:")
            print_sudoku(solution)
            print()
            print(f"SCORE: {score}")
            print("=*" * 50)

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
    if not valid_sudoku(sudoku):
        print(f"INVALID: {sudoku}")
        return


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
