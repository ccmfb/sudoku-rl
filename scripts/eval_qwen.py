import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a Qwen policy on Sudoku JSONL rows.")
    parser.add_argument("--data", required=True, type=Path, help="Path to JSONL rows with sudoku and solution fields.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Qwen model name or fine-tuned checkpoint path.")
    parser.add_argument("--adapter", default=None, help="Optional adapter checkpoint path.")
    parser.add_argument("--limit", default=None, type=int, help="Maximum number of rows to evaluate.")
    parser.add_argument("--max-new-tokens", default=256, type=int, help="Maximum number of generated tokens per prompt.")
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen thinking mode in the chat template.")

    return parser.parse_args()


def main() -> None:
    """Evaluate Qwen on a Sudoku JSONL file."""
    from data.utils import load_jsonl
    from sudoku_rl.qwen import QwenPolicy
    from sudoku_rl.utils import evaluate_attempts

    args = parse_args()
    rows = load_jsonl(args.data, limit=args.limit)
    policy = QwenPolicy(args.model, adapter_path=args.adapter, max_new_tokens=args.max_new_tokens, thinking=args.thinking)
    score = evaluate_attempts(rows, policy)

    print(f"model: {args.model}")
    print(f"adapter: {args.adapter or ''}")
    print(f"data: {args.data}")
    print(f"limit: {args.limit or ''}")
    print(f"score: {score:.6f}")


if __name__ == "__main__":
    main()
