#!/usr/bin/env python3
"""Training entry point for nanochat-jax.

Usage:
    python scripts/train.py --model-size nano --use-synthetic --total-steps 50
    python scripts/train.py --model-size nano --data-path data/shakespeare_char.h5
    python scripts/train.py --model-size small --device gpu --dtype bfloat16
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
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="muon",
                       choices=["muon", "adamw"],
                       help="Optimizer: muon (default) or adamw")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "gpu", "tpu"],
                       help="Device backend (default: cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--use-synthetic", action="store_true", default=False,
                       help="Use synthetic data (for testing, no dataset needed)")
    parser.add_argument("--data-path", type=str, default=None, help="Path to HDF5 token dataset")
    args = parser.parse_args()

    # ── Device setup (must happen before any JAX computation) ──────────
    from nanochat.core.device import setup_device
    setup_device(args.device)

    # Build configs
    model_cfg = ModelConfig.for_scale(args.model_size)
    train_cfg = TrainingConfig(
        batch_size=args.batch_size or (2 if args.model_size == "nano" else 8),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate or 3e-4,
        total_steps=args.total_steps or (200 if args.model_size == "nano" else 10000),
        warmup_steps=args.warmup_steps or (0 if args.model_size == "nano" else 100),
        optimizer=args.optimizer,
        checkpoint_dir=args.checkpoint_dir,
        eval_every_steps=args.eval_every or 100,
        save_every_steps=args.save_every or 500,
        dtype=args.dtype,
        param_dtype="float32",
        weight_decay=0.1 if args.model_size != "nano" else 0.0,
    )

    logger.info("config", model_size=args.model_size, device=args.device,
                optimizer=args.optimizer, dtype=args.dtype,
                total_steps=train_cfg.total_steps, batch_size=train_cfg.batch_size)

    # Build model
    rngs = nnx.Rngs(params=args.seed, dropout=args.seed + 1)
    model = TransformerLM(model_cfg, rngs=rngs)

    param_counts = count_params(model)
    logger.info("model_built", **{k: f"{v:,}" for k, v in param_counts.items()})

    # Build data loaders
    if args.use_synthetic or args.data_path is None:
        # Synthetic data loader — generate S+1 tokens so labels can be shifted by 1.
        # labels[t] = token[t+1], matching TokenDataset's pre-shifted convention.
        def synthetic_loader():
            rng = jax.random.PRNGKey(args.seed + 100)
            S = model_cfg.max_seq_len
            while True:
                rng, batch_rng = jax.random.split(rng)
                ids = jax.random.randint(batch_rng, (train_cfg.batch_size, S + 1),
                                         0, model_cfg.vocab_size)
                yield {
                    "input_ids": ids[:, :-1],          # [B, S]
                    "labels":    ids[:, 1:],            # [B, S] pre-shifted
                    "attention_mask": jnp.ones((train_cfg.batch_size, S), jnp.int32),
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
