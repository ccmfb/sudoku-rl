import hashlib
import json
import random
from pathlib import Path

from data.utils import load_bryanpark_csv, load_radcliffe_csv, load_rohanrao_csv

ROOT = Path(__file__).resolve().parents[1]
EVAL_PERCENT = 5
SHUFFLE_SEED = "sudoku-rl-v1"
TRAIN_PATH = ROOT / "data" / "train" / "all.jsonl"
EVAL_PATH = ROOT / "data" / "eval" / "all.jsonl"
SOURCES = [
    (ROOT / "data" / "radcliffe-3-million-sudoku-puzzles-with-ratings" / "sudoku-3m.csv", load_radcliffe_csv),
    (ROOT / "data" / "rohanrao-sudoku" / "sudoku.csv", load_rohanrao_csv),
    (ROOT / "data" / "bryanpark-sudoku" / "sudoku.csv", load_bryanpark_csv),
]


def is_eval_solution(solution: str) -> bool:
    """Return whether a solved grid belongs in eval."""
    digest = hashlib.blake2b(solution.encode(), digest_size=8).hexdigest()
    return int(digest, 16) % 100 < EVAL_PERCENT


def canonical_json_row(row: dict[str, str]) -> str:
    """Return a canonical JSONL payload."""
    return json.dumps({"sudoku": row["sudoku"], "solution": row["solution"]})


def write_shuffled_rows(path: Path, rows: list[str], split_name: str) -> None:
    """Write rows in deterministic shuffled order."""
    random.Random(f"{SHUFFLE_SEED}:{split_name}").shuffle(rows)
    with path.open("w") as file:
        for row in rows:
            file.write(row + "\n")


def build_splits() -> dict[str, int]:
    """Build local train/eval splits from available raw datasets."""
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "eval": 0, "eval_duplicate_solutions": 0}
    seen_eval_solutions = set()
    found_source = False
    train_rows = []
    eval_rows = []

    for path, load_rows in SOURCES:
        if not path.exists():
            continue

        found_source = True
        for row in load_rows(path):
            solution = row["solution"]
            if not is_eval_solution(solution):
                train_rows.append(canonical_json_row(row))
                counts["train"] += 1
                continue

            if solution in seen_eval_solutions:
                counts["eval_duplicate_solutions"] += 1
                continue

            seen_eval_solutions.add(solution)
            eval_rows.append(canonical_json_row(row))
            counts["eval"] += 1

    if not found_source: raise FileNotFoundError("No raw dataset files found.")

    write_shuffled_rows(TRAIN_PATH, train_rows, "train")
    write_shuffled_rows(EVAL_PATH, eval_rows, "eval")
    return counts


if __name__ == "__main__":
    print(json.dumps(build_splits(), indent=2))
