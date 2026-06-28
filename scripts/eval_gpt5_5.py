import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODEL = "gpt-5.5"
REASONING = {"effort": "none"}


def load_project_env() -> None:
    """Load repo .env values into the process environment."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")


class GPT55Policy:
    """Generate Sudoku attempts with GPT-5.5 and no reasoning budget."""

    def __init__(self) -> None:
        load_project_env()

        try:
            from openai import OpenAI
        except ImportError as error:
            message = "Install the OpenAI SDK or run with: uv run --with openai python scripts/eval_gpt5_5.py --data data/eval/missing_1_100.jsonl"
            raise SystemExit(message) from error

        self.client = OpenAI()

    def attempt(self, prompt: str) -> str:
        """Generate a Sudoku attempt."""
        response = self.client.responses.create(
            model=MODEL,
            input=prompt,
            reasoning=REASONING,
            text={"verbosity": "low"},
            store=False,
        )

        return response.output_text.strip()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate GPT-5.5 with no reasoning budget on Sudoku JSONL rows.")
    parser.add_argument("--data", required=True, type=Path, help="Path to JSONL rows with sudoku and solution fields.")
    parser.add_argument("--limit", default=None, type=int, help="Maximum number of rows to evaluate.")
    parser.add_argument("--verbose", action="store_true", help="Print each attempt, solution, and row score.")

    return parser.parse_args()


def main() -> None:
    """Evaluate GPT-5.5 on a Sudoku JSONL file."""
    from data.utils import load_jsonl
    from sudoku_rl.utils import evaluate_attempts

    args = parse_args()
    rows = load_jsonl(args.data, limit=args.limit)
    policy = GPT55Policy()
    score = evaluate_attempts(rows, policy, verbose=args.verbose)

    print(f"model: {MODEL}")
    print(f"reasoning: {REASONING}")
    print(f"data: {args.data}")
    print(f"limit: {args.limit or ''}")
    print(f"score: {score:.6f}")


if __name__ == "__main__":
    main()
