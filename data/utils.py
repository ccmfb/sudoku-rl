import csv
import json
from collections.abc import Iterator
from pathlib import Path

from sudoku_rl.utils import valid_sudoku, valid_solution


def load_jsonl(path: str | Path, limit: int | None = None) -> Iterator[dict[str, str]]:
    """Load sudoku/solution rows from JSONL."""
    with Path(path).open() as file:
        for row_index, line in enumerate(file):
            if limit is not None and row_index >= limit:
                break

            row = json.loads(line)
            sudoku = row["sudoku"]
            solution = row["solution"]

            if not valid_sudoku(sudoku): raise ValueError(f"Invalid sudoku: {sudoku}")
            if not valid_solution(solution): raise ValueError(f"Invalid solution: {solution}")

            yield {"sudoku": sudoku, "solution": solution}


def load_radcliffe_csv(path: str | Path, limit: int | None = None) -> Iterator[dict[str, str]]:
    """Load Radcliffe CSV rows as sudoku/solution dictionaries."""
    with Path(path).open(newline="") as file:
        for row_index, row in enumerate(csv.DictReader(file)):
            if limit is not None and row_index >= limit:
                break

            sudoku = row["puzzle"].strip()
            solution = row["solution"].strip()

            if not valid_sudoku(sudoku): raise ValueError(f"Invalid sudoku: {sudoku}")
            if not valid_solution(solution): raise ValueError(f"Invalid solution: {solution}")

            yield {"sudoku": sudoku, "solution": solution}


def load_rohanrao_csv(path: str | Path, limit: int | None = None) -> Iterator[dict[str, str]]:
    """Load Rohan Rao CSV rows as sudoku/solution dictionaries."""
    with Path(path).open(newline="") as file:
        for row_index, row in enumerate(csv.DictReader(file)):
            if limit is not None and row_index >= limit:
                break

            sudoku = row["puzzle"].strip().replace("0", ".")
            solution = row["solution"].strip()

            if not valid_sudoku(sudoku): raise ValueError(f"Invalid sudoku: {sudoku}")
            if not valid_solution(solution): raise ValueError(f"Invalid solution: {solution}")

            yield {"sudoku": sudoku, "solution": solution}


def load_bryanpark_csv(path: str | Path, limit: int | None = None) -> Iterator[dict[str, str]]:
    """Load Bryan Park CSV rows as sudoku/solution dictionaries."""
    with Path(path).open(newline="") as file:
        for row_index, row in enumerate(csv.DictReader(file)):
            if limit is not None and row_index >= limit:
                break

            sudoku = row["quizzes"].strip().replace("0", ".")
            solution = row["solutions"].strip()

            if not valid_sudoku(sudoku): raise ValueError(f"Invalid sudoku: {sudoku}")
            if not valid_solution(solution): raise ValueError(f"Invalid solution: {solution}")

            yield {"sudoku": sudoku, "solution": solution}
