import hashlib
import json
from pathlib import Path

from data.utils import load_jsonl

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_COUNT = 100
INPUT_PATH = ROOT / "data" / "eval" / "all.jsonl"
SAMPLE_PATH = ROOT / "data" / "eval" / "sample_100.jsonl"


def sample_score(row: dict[str, str]) -> int:
    """Return a deterministic sample ordering score."""
    text = row["solution"] + ":sample_100"
    digest = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
    return int(digest, 16)


def write_rows(rows: list[dict[str, str]]) -> None:
    """Write the sampled eval rows."""
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SAMPLE_PATH.open("w") as file:
        for row in rows:
            file.write(json.dumps({"sudoku": row["sudoku"], "solution": row["solution"]}) + "\n")


def build_sample() -> dict[str, int]:
    """Build a deterministic 100-row eval sample."""
    rows = list(load_jsonl(INPUT_PATH))
    if len(rows) < SAMPLE_COUNT: raise ValueError(f"Need {SAMPLE_COUNT} rows, got {len(rows)}")

    sample = sorted(rows, key=sample_score)[:SAMPLE_COUNT]
    write_rows(sample)
    return {"input": len(rows), "sample": len(sample)}


if __name__ == "__main__":
    print(json.dumps(build_sample(), indent=2))
