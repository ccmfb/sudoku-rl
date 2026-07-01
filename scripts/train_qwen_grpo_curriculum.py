import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MISSING_COUNTS = [1, 2, 5, 10, 20, 40]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen Sudoku policy with a dynamic GRPO curriculum.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Qwen model name or checkpoint path.")
    parser.add_argument("--adapter", default=None, type=Path, help="Optional LoRA adapter checkpoint to continue training.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for curriculum round checkpoints.")
    parser.add_argument("--rounds", default=20, type=int, help="Number of curriculum rounds.")
    parser.add_argument("--steps-per-round", default=100, type=int, help="GRPO optimizer steps per curriculum round.")
    parser.add_argument("--train-set-size", default=10000, type=int, help="Suffix size for data/train/missing_X_SIZE.jsonl files.")
    parser.add_argument("--train-limit", default=None, type=int, help="Maximum number of rows to load from each train file.")
    parser.add_argument("--eval-set-size", default=100, type=int, help="Suffix size for data/eval/missing_X_SIZE.jsonl files.")
    parser.add_argument("--eval-limit", default=None, type=int, help="Maximum number of eval rows per missing-count level.")
    parser.add_argument("--missing-counts", nargs="+", default=MISSING_COUNTS, type=int, help="Missing-count levels to evaluate and choose from.")
    parser.add_argument("--target-low", default=0.20, type=float, help="Lowest score treated as learnable.")
    parser.add_argument("--target-high", default=0.80, type=float, help="Highest score treated as learnable.")
    parser.add_argument("--batch-size", default=8, type=int, help="Per-device train batch size.")
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", default=1e-6, type=float, help="LoRA learning rate.")
    parser.add_argument("--max-completion-length", default=128, type=int, help="Maximum generated tokens per rollout.")
    parser.add_argument("--num-generations", default=8, type=int, help="Number of completions sampled per prompt group.")
    parser.add_argument("--temperature", default=1.0, type=float, help="Sampling temperature.")
    parser.add_argument("--top-p", default=1.0, type=float, help="Nucleus sampling probability.")
    parser.add_argument("--top-k", default=0, type=int, help="Top-k sampling cutoff; 0 disables top-k filtering.")
    parser.add_argument("--beta", default=0.0, type=float, help="KL coefficient for the reference model.")
    parser.add_argument("--loss-type", choices=["grpo", "bnpo", "dr_grpo", "dapo"], default="dr_grpo", help="TRL GRPO loss variant.")
    parser.add_argument("--scale-rewards", choices=["group", "batch", "none"], default="none", help="Reward normalization strategy.")
    parser.add_argument("--eval-batch-size", default=8, type=int, help="Number of eval rows to generate at once.")
    parser.add_argument("--max-new-tokens", default=256, type=int, help="Maximum generated tokens per eval prompt.")
    parser.add_argument("--lora-r", default=16, type=int, help="LoRA rank.")
    parser.add_argument("--lora-alpha", default=32, type=int, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", default=0.05, type=float, help="LoRA dropout.")
    parser.add_argument("--wandb", action="store_true", help="Log TRL training metrics to W&B.")

    return parser.parse_args()


def evaluate_levels(model: str, adapter_path: str | Path | None, missing_counts: list[int], eval_set_size: int, eval_limit: int | None, max_new_tokens: int, batch_size: int) -> dict[int, float]:
    """Evaluate one adapter on every curriculum level."""
    from data.utils import load_jsonl
    from sudoku_rl.eval.evaluate import evaluate_attempts
    from sudoku_rl.models.qwen import QwenPolicy

    adapter_text = str(adapter_path) if adapter_path is not None else None
    policy = QwenPolicy(model, adapter_path=adapter_text, max_new_tokens=max_new_tokens, thinking=False)
    scores = {}

    for missing_count in missing_counts:
        eval_path = PROJECT_ROOT / "data" / "eval" / f"missing_{missing_count}_{eval_set_size}.jsonl"
        rows = list(load_jsonl(eval_path, limit=eval_limit))
        scores[missing_count] = evaluate_attempts(rows, policy, batch_size=batch_size)

    return scores


def log_wandb_round(round_index: int, scores: dict[int, float], missing_count: int, args: argparse.Namespace) -> None:
    """Log one curriculum round to W&B."""
    import wandb

    metrics = {
        "curriculum/round": round_index,
        "curriculum/selected_missing_count": missing_count,
        "curriculum/target_low": args.target_low,
        "curriculum/target_high": args.target_high,
        "train/missing_count": missing_count,
        "train/steps_per_round": args.steps_per_round,
    }
    metrics.update({f"eval/missing_{count}": score for count, score in scores.items()})
    wandb.log(metrics, step=round_index)


def run_curriculum(args: argparse.Namespace) -> None:
    """Run dynamic chunked GRPO curriculum training."""
    from sudoku_rl.methods.grpo import train_grpo
    from sudoku_rl.models.peft import make_lora_config

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = args.adapter
    peft_config = None if adapter_path is not None else make_lora_config(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    for round_index in range(args.rounds):
        scores = evaluate_levels(args.model, adapter_path, args.missing_counts, args.eval_set_size, args.eval_limit, args.max_new_tokens, args.eval_batch_size)
        learnable = [missing_count for missing_count, score in scores.items() if args.target_low <= score <= args.target_high]
        too_hard = [missing_count for missing_count, score in scores.items() if score < args.target_low]
        missing_count = max(learnable) if learnable else min(too_hard) if too_hard else max(scores)
        train_path = PROJECT_ROOT / "data" / "train" / f"missing_{missing_count}_{args.train_set_size}.jsonl"
        if not train_path.exists(): raise FileNotFoundError(f"Missing train curriculum file: {train_path}")

        round_dir = args.output_dir / f"round-{round_index:03d}-missing-{missing_count}"
        train_grpo(
            train_path,
            model=args.model,
            output_dir=round_dir,
            limit=args.train_limit,
            max_steps=args.steps_per_round,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_completion_length=args.max_completion_length,
            num_generations=args.num_generations,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            beta=args.beta,
            loss_type=args.loss_type,
            scale_rewards=args.scale_rewards,
            adapter_path=adapter_path,
            peft_config=peft_config,
            wandb=args.wandb,
        )

        if args.wandb: log_wandb_round(round_index, scores, missing_count, args)
        score_text = " ".join(f"missing_{count}={scores[count]:.4f}" for count in args.missing_counts)
        print(f"round {round_index}: selected missing_{missing_count} {score_text}")

        adapter_path = round_dir
        peft_config = None


def main() -> None:
    """Train Qwen with a dynamic GRPO curriculum."""
    run_curriculum(parse_args())


if __name__ == "__main__":
    main()
