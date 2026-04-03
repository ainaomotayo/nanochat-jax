"""Optimizer construction with weight decay masking."""
from __future__ import annotations
import jax
import optax
import structlog
from nanochat.config.training_config import TrainingConfig
from nanochat.training.scheduler import build_lr_schedule

logger = structlog.get_logger()


def _weight_decay_mask(path: tuple, _: jax.Array) -> bool:
    """Return True if parameter should have weight decay applied.

    Exclude: norms (gamma/beta), biases, embedding tables.
    Include: all weight matrices.
    """
    path_str = "/".join(str(p) for p in path)
    # Exclude norms, biases, embeddings
    exclude_patterns = ("norm", "bias", "embed", "gamma", "beta")
    for pattern in exclude_patterns:
        if pattern in path_str.lower():
            return False
    return True


def build_optimizer(
    cfg: TrainingConfig,
) -> optax.GradientTransformation:
    """Build AdamW optimizer with gradient clipping and LR schedule.

    Weight decay is applied only to weight matrices (not norms, biases, embeddings).
    Uses optax.multi_transform for per-parameter-group treatment.

    Args:
        cfg: Training configuration

    Returns:
        optax.GradientTransformation ready to wrap in nnx.Optimizer
    """
    schedule = build_lr_schedule(
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.total_steps,
        min_lr_ratio=cfg.min_lr_ratio,
        lr_decay_steps=cfg.lr_decay_steps,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=cfg.beta1,
            b2=cfg.beta2,
            eps=cfg.epsilon,
            weight_decay=cfg.weight_decay,
            mask=lambda params: jax.tree_util.tree_map_with_path(
                _weight_decay_mask, params
            ),
        ),
    )

    logger.info("optimizer_built", type=cfg.optimizer, lr=cfg.learning_rate,
                wd=cfg.weight_decay, clip=cfg.grad_clip_norm)
    return optimizer
