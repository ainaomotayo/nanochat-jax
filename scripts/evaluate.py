#!/usr/bin/env python3
"""Evaluation entry point for nanochat-jax.

Runs lm-eval-harness benchmarks against a nanochat-jax checkpoint, or
a randomly initialised model for sanity-checking.

Usage examples::

    # Quick sanity check with random weights
    python scripts/evaluate.py --checkpoint RANDOM --suite quick --limit 10

    # Evaluate a real checkpoint on standard tasks
    python scripts/evaluate.py --checkpoint checkpoints/step_010000 --suite standard

    # Custom task list
    python scripts/evaluate.py --checkpoint checkpoints/step_010000 \\
        --tasks hellaswag,piqa --num_fewshot 5 --batch_size 16
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import structlog

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a nanochat-jax model with lm-eval-harness"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "tpu"],
        help="Device backend (default: cpu)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help='Path to checkpoint directory, or "RANDOM" for random-weight testing',
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of lm-eval task names",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        choices=["quick", "standard", "full"],
        help="Predefined task suite (overridden by --tasks)",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit examples per task (for quick testing)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Directory to write results.json and summary.md",
    )
    return parser.parse_args()


def _resolve_tasks(args: argparse.Namespace) -> list[str]:
    """Determine the task list from --tasks or --suite flags."""
    from nanochat.evaluation.suite import QUICK_TASKS, STANDARD_TASKS

    if args.tasks is not None:
        return [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.suite == "quick":
        return list(QUICK_TASKS)
    elif args.suite == "standard":
        return list(STANDARD_TASKS)
    elif args.suite == "full":
        # Full = standard tasks; extend as the project grows
        return list(STANDARD_TASKS)

    # Default to quick
    return list(QUICK_TASKS)


def _build_random_model(device: str, batch_size: int):
    """Create a small randomly-initialised model wrapped in the adapter."""
    from flax import nnx

    from nanochat.config import ModelConfig
    from nanochat.evaluation.lm_eval_adapter import NanoChatJAXModel
    from nanochat.model.transformer import TransformerLM
    from nanochat.tokenizer.bpe import BPETokenizer

    cfg = ModelConfig.for_scale("nano")
    tokenizer = BPETokenizer()
    # Override vocab_size to match tokenizer
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        max_seq_len=cfg.max_seq_len,
    )
    model = TransformerLM(cfg, rngs=nnx.Rngs(params=42, dropout=43))

    logger.info(
        "random_model_created",
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        vocab_size=cfg.vocab_size,
    )

    return NanoChatJAXModel(model, tokenizer, cfg, batch_size=batch_size)


def _print_summary_table(task_results: dict[str, dict]) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  {'Task':<25} {'Metric':<15} {'Value':>10}")
    print("-" * 60)

    for task_name in sorted(task_results):
        metrics = task_results[task_name]
        first = True
        for metric_name in sorted(metrics):
            value = metrics[metric_name]
            display_task = task_name if first else ""
            if isinstance(value, float):
                print(f"  {display_task:<25} {metric_name:<15} {value:>10.4f}")
            else:
                print(f"  {display_task:<25} {metric_name:<15} {str(value):>10}")
            first = False

    print("=" * 60 + "\n")


def main() -> None:
    args = parse_args()

    # Setup device FIRST, before any JAX operations
    from nanochat.core.device import setup_device

    setup_device(args.device)

    from nanochat.evaluation.suite import run_eval_suite

    tasks = _resolve_tasks(args)
    logger.info(
        "evaluate_start",
        checkpoint=args.checkpoint,
        device=args.device,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    # Build model
    model_obj = None
    if args.checkpoint.upper() == "RANDOM":
        model_obj = _build_random_model(args.device, args.batch_size)

    task_results = run_eval_suite(
        checkpoint_path=args.checkpoint if args.checkpoint.upper() != "RANDOM" else ".",
        device=args.device,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        output_path=args.output_path,
        model_obj=model_obj,
    )

    _print_summary_table(task_results)


if __name__ == "__main__":
    main()
