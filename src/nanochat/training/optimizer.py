"""Optimizer construction supporting AdamW and Muon.

Dispatcher that builds either AdamW (for baselines/ablations) or Muon
(nanochat production default) from TrainingConfig.
"""

from __future__ import annotations

import jax
import optax
import structlog

from nanochat.config.training_config import TrainingConfig
from nanochat.training.scheduler import build_lr_schedule
from nanochat.training.muon import build_muon_optimizer

logger = structlog.get_logger()


def _weight_decay_mask(path: tuple, _: jax.Array) -> bool:
    """Return True if parameter should have weight decay applied.

    Excludes: all normalization params (none exist since RMSNorm is
    parameterless, but keep for safety), biases, embedding tables.
    Includes: all 2D weight matrices (projections).
    """
    path_str = "/".join(str(p) for p in path)
    exclude = ("norm", "bias", "embed", "gamma", "beta", "table",
               "alpha_attn", "alpha_ffn", "raw_alpha", "raw_beta")
    for pattern in exclude:
        if pattern in path_str.lower():
            return False
    return True


def build_optimizer(
    cfg: TrainingConfig,
) -> optax.GradientTransformation:
    """Build an optimizer from TrainingConfig.

    Routes to:
    - Muon: ``cfg.optimizer == "muon"`` (nanochat default)
    - AdamW: ``cfg.optimizer == "adamw"``

    Both variants include:
    - Global gradient norm clipping
    - LR schedule (linear warmup + cosine decay)
    - Decoupled weight decay (matrices only)

    Args:
        cfg: Training configuration.

    Returns:
        optax.GradientTransformation ready for nnx.Optimizer.
    """
    schedule = build_lr_schedule(
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.total_steps,
        min_lr_ratio=cfg.min_lr_ratio,
        lr_decay_steps=cfg.lr_decay_steps,
    )

    if cfg.optimizer == "muon":
        optimizer = build_muon_optimizer(
            learning_rate=schedule,
            momentum=cfg.muon_momentum,
            nesterov=cfg.muon_nesterov,
            ns_steps=cfg.muon_ns_steps,
            weight_decay=cfg.muon_weight_decay,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        logger.info(
            "optimizer_built",
            type="muon",
            lr=cfg.learning_rate,
            momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps,
            wd=cfg.muon_weight_decay,
            clip=cfg.grad_clip_norm,
        )

    elif cfg.optimizer == "adamw":
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
        logger.info(
            "optimizer_built",
            type="adamw",
            lr=cfg.learning_rate,
            wd=cfg.weight_decay,
            clip=cfg.grad_clip_norm,
        )

    else:
        raise ValueError(
            f"Unknown optimizer '{cfg.optimizer}'. "
            "Supported: 'muon', 'adamw'."
        )

    return optimizer
