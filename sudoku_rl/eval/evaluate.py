from collections.abc import Iterable

from sudoku_rl.tasks.rewards import score_attempt
from sudoku_rl.tasks.sudoku import format_prompt, print_sudoku


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
