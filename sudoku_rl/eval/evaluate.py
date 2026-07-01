from collections.abc import Iterable

from sudoku_rl.tasks.rewards import score_attempt
from sudoku_rl.tasks.sudoku import format_prompt, print_sudoku


def evaluate_attempt_rows(rows: Iterable[dict[str, str]], policy, verbose: bool = False, batch_size: int = 1) -> list[dict[str, object]]:
    """Score generated attempts and return one result row per sudoku."""
    results = []
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
            results.append({"index": row_index, "sudoku": sudoku, "solution": solution, "attempt": attempt, "score": score})

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

    return results


def evaluate_attempts(rows: Iterable[dict[str, str]], policy, verbose: bool = False, batch_size: int = 1) -> float:
    """Score generated attempts for sudoku rows."""
    results = evaluate_attempt_rows(rows, policy, verbose=verbose, batch_size=batch_size)
    if not results: return 0.0

    return sum(float(row["score"]) for row in results) / len(results)
