import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a Qwen policy on Sudoku JSONL rows.")
    parser.add_argument("--data", required=True, type=Path, help="Path to JSONL rows with sudoku and solution fields.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Qwen model name or fine-tuned checkpoint path.")
    parser.add_argument("--adapter", default=None, help="Optional adapter checkpoint path.")
    parser.add_argument("--limit", default=None, type=int, help="Maximum number of rows to evaluate.")
    parser.add_argument("--max-new-tokens", default=256, type=int, help="Maximum number of generated tokens per prompt.")
    parser.add_argument("--batch-size", default=1, type=int, help="Number of rows to generate at once.")
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen thinking mode in the chat template.")
    parser.add_argument("--verbose", action="store_true", help="Print each attempt, solution, and row score.")

    return parser.parse_args()


def make_result_path(created_at: datetime | None = None) -> Path:
    """Build the timestamped JSONL result path."""
    created_at = created_at or datetime.now(timezone.utc)
    filename = created_at.strftime("%Y%m%dT%H%M%S.%fZ.jsonl")
    return RESULTS_DIR / filename


def write_results(path: Path, args: argparse.Namespace, result_rows: list[dict[str, object]], score: float, created_at: datetime) -> None:
    """Write one eval run as summary and question JSONL rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "type": "summary",
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "model": args.model,
        "adapter": args.adapter,
        "data": str(args.data),
        "limit": args.limit,
        "rows": len(result_rows),
        "thinking": args.thinking,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "score": score,
    }

    with path.open("w") as file:
        file.write(json.dumps(summary) + "\n")
        for result_row in result_rows:
            file.write(json.dumps({"type": "question", **result_row}) + "\n")


def main() -> None:
    """Evaluate Qwen on a Sudoku JSONL file."""
    from data.utils import load_jsonl
    from sudoku_rl.eval.evaluate import evaluate_attempt_rows
    from sudoku_rl.models.qwen import QwenPolicy

    args = parse_args()
    rows = list(load_jsonl(args.data, limit=args.limit))
    policy = QwenPolicy(args.model, adapter_path=args.adapter, max_new_tokens=args.max_new_tokens, thinking=args.thinking)
    result_rows = evaluate_attempt_rows(rows, policy, verbose=args.verbose, batch_size=args.batch_size)
    score = sum(float(row["score"]) for row in result_rows) / len(result_rows) if result_rows else 0.0
    created_at = datetime.now(timezone.utc)
    result_path = make_result_path(created_at)
    write_results(result_path, args, result_rows, score, created_at)

    print(f"model: {args.model}")
    print(f"adapter: {args.adapter or ''}")
    print(f"data: {args.data}")
    print(f"limit: {args.limit or ''}")
    print(f"thinking: {args.thinking}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"batch_size: {args.batch_size}")
    print(f"score: {score:.6f}")
    print(f"results: {result_path}")


if __name__ == "__main__":
    main()
