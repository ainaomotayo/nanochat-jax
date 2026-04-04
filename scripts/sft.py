#!/usr/bin/env python3
"""SFT (Supervised Fine-Tuning) entry point for nanochat-jax.

Usage:
    # Full fine-tuning with synthetic data
    python scripts/sft.py --device cpu --base_checkpoint RANDOM --dataset smoltalk --steps 100

    # LoRA fine-tuning
    python scripts/sft.py --device cpu --base_checkpoint RANDOM --dataset smoltalk --lora_rank 8

    # Fine-tune from a checkpoint
    python scripts/sft.py --device gpu --base_checkpoint checkpoints/step_010000 --lora_rank 16
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import structlog

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SFT."""
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for nanochat-jax",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "gpu", "tpu"],
        help="Device to train on (default: cpu).",
    )
    parser.add_argument(
        "--base_checkpoint", type=str, default="RANDOM",
        help="Path to base checkpoint directory, or RANDOM for random init.",
    )
    parser.add_argument(
        "--dataset", type=str, default="smoltalk",
        help="Dataset name. 'smoltalk' uses built-in synthetic conversations.",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=0,
        help="LoRA rank. 0 = full fine-tuning, >0 = LoRA fine-tuning.",
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=None,
        help="LoRA alpha. Defaults to lora_rank.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5,
        help="Peak learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size.",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=128,
        help="Maximum sequence length for SFT examples.",
    )
    parser.add_argument(
        "--model_size", type=str, default="nano",
        choices=["nano", "small", "medium", "large", "xlarge"],
        help="Model size preset (used when base_checkpoint=RANDOM).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints/sft/",
        help="Directory for saving SFT checkpoints.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    """Main SFT entry point."""
    args = parse_args()

    # Setup device first (before any JAX operations)
    from nanochat.core.device import setup_device
    setup_device(args.device)

    import jax
    from flax import nnx

    from nanochat.config import ModelConfig, TrainingConfig
    from nanochat.model.transformer import TransformerLM
    from nanochat.training.sft_trainer import SFTTrainer, SFTDataset, SimpleTokenizer
    from nanochat.training.checkpoint import CheckpointManager

    # Build model config
    model_cfg = ModelConfig.for_scale(args.model_size)
    if args.max_seq_len:
        model_cfg = ModelConfig.for_scale(
            args.model_size, max_seq_len=args.max_seq_len,
        )

    # Build model
    rngs = nnx.Rngs(params=args.seed, dropout=args.seed + 1)
    model = TransformerLM(model_cfg, rngs=rngs)

    # Load base checkpoint if specified
    if args.base_checkpoint != "RANDOM":
        ckpt_path = Path(args.base_checkpoint)
        if not ckpt_path.exists():
            logger.error("checkpoint_not_found", path=str(ckpt_path))
            sys.exit(1)
        ckpt_mgr = CheckpointManager(ckpt_path.parent)
        step = ckpt_mgr.load(ckpt_path, model)
        logger.info("base_checkpoint_loaded", step=step, path=str(ckpt_path))
    else:
        logger.info("using_random_init", model_size=args.model_size)

    # Apply LoRA if requested
    use_lora = args.lora_rank > 0
    if use_lora:
        from nanochat.training.lora import apply_lora, count_lora_params, count_base_params

        lora_rngs = nnx.Rngs(params=args.seed + 100)
        apply_lora(
            model,
            rank=args.lora_rank,
            rngs=lora_rngs,
            alpha=args.lora_alpha,
        )
        n_lora = count_lora_params(model)
        n_base = count_base_params(model)
        logger.info(
            "lora_config",
            rank=args.lora_rank,
            alpha=args.lora_alpha or args.lora_rank,
            lora_params=n_lora,
            base_params=n_base,
            reduction=f"{n_lora / max(n_base, 1) * 100:.2f}%",
        )

    # Build tokenizer (simple byte-level for testing)
    tokenizer = SimpleTokenizer(vocab_size=model_cfg.vocab_size)

    # Build dataset
    conversations = None  # Will use synthetic data
    if args.dataset != "smoltalk":
        logger.warning(
            "unknown_dataset_using_synthetic",
            dataset=args.dataset,
        )

    dataset = SFTDataset(
        conversations=conversations,
        tokenizer=tokenizer,
        max_seq_len=model_cfg.max_seq_len,
    )

    # Build training config
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        optimizer="adamw",
        learning_rate=args.learning_rate,
        warmup_steps=min(50, args.steps // 10),
        total_steps=args.steps,
        weight_decay=0.01,
        dtype="float32",
        param_dtype="float32",
        checkpoint_dir=args.output_dir,
        save_every_steps=max(100, args.steps // 5),
        eval_every_steps=args.steps + 1,  # no eval during SFT
    )

    logger.info(
        "sft_config",
        model_size=args.model_size,
        dataset=args.dataset,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        steps=args.steps,
        batch_size=args.batch_size,
        max_seq_len=model_cfg.max_seq_len,
    )

    # Build trainer and train
    trainer = SFTTrainer(
        model=model,
        dataset=dataset,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        use_lora=use_lora,
    )

    results = trainer.train()

    # Merge LoRA weights if applicable
    if use_lora:
        from nanochat.training.lora import merge_lora
        merge_lora(model)
        logger.info("lora_weights_merged")

        # Save merged model
        merged_dir = Path(args.output_dir) / "merged"
        merged_ckpt = CheckpointManager(str(merged_dir))
        merged_ckpt.save(args.steps, model, results)
        logger.info("merged_model_saved", path=str(merged_dir))

    logger.info("sft_complete", **results)


if __name__ == "__main__":
    main()
