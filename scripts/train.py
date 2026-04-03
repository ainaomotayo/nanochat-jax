#!/usr/bin/env python3
"""Training entry point for nanochat-jax.

Usage:
    python scripts/train.py
    python scripts/train.py model=nano training=overfit_single
    python scripts/train.py model=small training=chinchilla
    python scripts/train.py training.learning_rate=1e-3 training.total_steps=5000
"""
from __future__ import annotations

import os
import sys
import structlog

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
from flax import nnx

from nanochat.config import ModelConfig, TrainingConfig, DataConfig
from nanochat.model.transformer import TransformerLM
from nanochat.model.param_count import count_params, format_param_count
from nanochat.training.trainer import Trainer
from nanochat.data.loader import build_dataloader

logger = structlog.get_logger()


def main() -> None:
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train nanochat-jax model")
    parser.add_argument("--model-size", type=str, default="nano",
                       choices=["nano", "small", "medium", "large", "xlarge"],
                       help="Model size preset")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--use-synthetic", action="store_true", default=True,
                       help="Use synthetic data (for testing)")
    parser.add_argument("--data-path", type=str, default=None, help="Path to HDF5 token dataset")
    args = parser.parse_args()

    # Build configs
    model_cfg = ModelConfig.for_scale(args.model_size)
    train_cfg = TrainingConfig(
        batch_size=args.batch_size or (2 if args.model_size == "nano" else 8),
        learning_rate=args.learning_rate or 3e-4,
        total_steps=args.total_steps or (200 if args.model_size == "nano" else 10000),
        warmup_steps=args.warmup_steps or (0 if args.model_size == "nano" else 100),
        checkpoint_dir=args.checkpoint_dir,
        eval_every_steps=args.eval_every or 100,
        save_every_steps=args.save_every or 500,
        dtype=args.dtype,
        param_dtype="float32",
        weight_decay=0.1 if args.model_size != "nano" else 0.0,
    )

    logger.info("config", model_size=args.model_size, model=model_cfg.model_dump(),
                training=train_cfg.model_dump())

    # Build model
    rngs = nnx.Rngs(params=args.seed, dropout=args.seed + 1)
    model = TransformerLM(model_cfg, rngs=rngs)

    param_counts = count_params(model)
    logger.info("model_built", **{k: f"{v:,}" for k, v in param_counts.items()})

    # Build data loaders
    if args.use_synthetic or args.data_path is None:
        # Synthetic data loader for testing
        def synthetic_loader():
            rng = jax.random.PRNGKey(args.seed + 100)
            while True:
                rng, batch_rng = jax.random.split(rng)
                ids = jax.random.randint(batch_rng, (train_cfg.batch_size, model_cfg.max_seq_len),
                                         0, model_cfg.vocab_size)
                yield {
                    "input_ids": ids,
                    "labels": ids,
                    "attention_mask": jnp.ones_like(ids),
                }
        train_loader = synthetic_loader()
        val_loader = synthetic_loader()
        logger.info("using_synthetic_data")
    else:
        from nanochat.data.dataset import TokenDataset
        train_ds = TokenDataset(args.data_path, model_cfg.max_seq_len, split="train")
        val_ds = TokenDataset(args.data_path, model_cfg.max_seq_len, split="val")
        train_loader = build_dataloader(train_ds, train_cfg.batch_size, shuffle=True, seed=args.seed)
        val_loader = build_dataloader(val_ds, train_cfg.batch_size, shuffle=False, seed=args.seed)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )

    results = trainer.train()
    logger.info("training_finished", **results)


if __name__ == "__main__":
    main()
