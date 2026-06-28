import hashlib
import json
from pathlib import Path

from data.utils import load_jsonl

ROOT = Path(__file__).resolve().parents[1]
MISSING_COUNTS = [1, 2, 5, 10, 20, 40]
INPUT_PATH = ROOT / "data" / "eval" / "sample_100.jsonl"
OUTPUT_DIR = ROOT / "data" / "eval"


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


def write_masked_rows(rows: list[dict[str, str]], missing_count: int) -> None:
    """Write one masked eval set."""
    path = OUTPUT_DIR / f"missing_{missing_count}_100.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        for row in rows:
            solution = row["solution"]
            file.write(json.dumps({"sudoku": masked_sudoku(solution, missing_count), "solution": solution}) + "\n")


def build_masked_sets() -> dict[str, int]:
    """Build masked eval sets from the 100-row eval sample."""
    rows = list(load_jsonl(INPUT_PATH))

    for missing_count in MISSING_COUNTS:
        write_masked_rows(rows, missing_count)

    return {"input": len(rows), "sets": len(MISSING_COUNTS)}


if __name__ == "__main__":
    print(json.dumps(build_masked_sets(), indent=2))
