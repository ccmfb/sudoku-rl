import csv
import json
from collections.abc import Iterator
from pathlib import Path

from sudoku_rl.utils import valid_sudoku, valid_solution


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

def extract_splits(rows: Iterator[dict[str, str]], train_path: str | Path, eval_path: str | Path, train_count: int, eval_count: int) -> None:
    """Write train and eval JSONL splits from sudoku/solution rows."""
    train_path = Path(train_path)
    eval_path = Path(eval_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("w") as train_file, eval_path.open("w") as eval_file:
        for row_index, row in enumerate(rows):
            if row_index < train_count:
                output_file = train_file
            elif row_index < train_count + eval_count:
                output_file = eval_file
            else:
                break

            output_file.write(json.dumps({"sudoku": row["sudoku"], "solution": row["solution"]}) + "\n")
