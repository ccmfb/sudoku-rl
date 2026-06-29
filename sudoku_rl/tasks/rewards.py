from sudoku_rl.tasks.sudoku import valid_solution


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
