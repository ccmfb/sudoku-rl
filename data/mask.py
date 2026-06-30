import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path

from data.utils import load_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = "eval"
MISSING_COUNTS = [1, 2, 5, 10, 20, 40]
DEFAULT_SET_SIZE = 100


def mask_positions(solution: str, missing_count: int) -> list[int]:
    """Choose deterministic missing positions for a solution."""
    scores = []

    for position in range(81):
        text = f"{solution}:{missing_count}:{position}"
        digest = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
        scores.append((int(digest, 16), position))

    return [position for _, position in sorted(scores)[:missing_count]]


def masked_sudoku(solution: str, missing_count: int) -> str:
    """Mask a solved grid with a fixed number of missing cells."""
    sudoku = list(solution)

    for position in mask_positions(solution, missing_count):
        sudoku[position] = "."

    return "".join(sudoku)


def write_masked_rows(rows: list[dict[str, str]], split_name: str, missing_count: int, set_size: int) -> None:
    """Write one masked set."""
    path = ROOT / "data" / split_name / f"missing_{missing_count}_{set_size}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        for row in rows:
            solution = row["solution"]
            file.write(json.dumps({"sudoku": masked_sudoku(solution, missing_count), "solution": solution}) + "\n")


def build_masked_sets(split_name: str = DEFAULT_SPLIT, set_size: int = DEFAULT_SET_SIZE) -> dict[str, int | str]:
    """Build masked sets from disjoint split windows."""
    if set_size <= 0: raise ValueError("set_size must be positive")

    rows = load_jsonl(ROOT / "data" / split_name / "all.jsonl")
    input_count = 0

    for missing_count in MISSING_COUNTS:
        selected_rows = list(islice(rows, set_size))
        input_count += len(selected_rows)
        if len(selected_rows) < set_size:
            raise ValueError(f"Need {set_size} rows for missing_{missing_count}, got {len(selected_rows)}")

        write_masked_rows(selected_rows, split_name, missing_count, set_size)

    return {"split": split_name, "input": input_count, "sets": len(MISSING_COUNTS)}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["eval", "train"], default=DEFAULT_SPLIT)
    parser.add_argument("--set-size", type=int, default=DEFAULT_SET_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build_masked_sets(args.split, args.set_size), indent=2))
