"""Auto-scaling hyperparameters from a single --depth integer.

Maps depth levels 1-10 to complete model and training configurations,
enabling quick experimentation across a wide range of model sizes
(~1M to ~6B parameters) with a single CLI argument.
"""
from __future__ import annotations

import math

import structlog

from nanochat.config.model_config import ModelConfig
from nanochat.config.training_config import TrainingConfig

logger = structlog.get_logger()

# ── Depth-to-architecture mapping ────────────────────────────────
# Each entry: (d_model, n_layers, n_heads, n_kv_heads, max_seq_len, approx_params_str)
_DEPTH_TABLE: dict[int, dict] = {
    1: dict(d_model=128,  n_layers=4,  n_heads=4,  n_kv_heads=4, max_seq_len=256),    # ~1M
    2: dict(d_model=256,  n_layers=6,  n_heads=4,  n_kv_heads=4, max_seq_len=512),    # ~10M
    3: dict(d_model=512,  n_layers=6,  n_heads=8,  n_kv_heads=8, max_seq_len=1024),   # ~50M
    4: dict(d_model=768,  n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=1024),   # ~130M
    5: dict(d_model=1024, n_layers=12, n_heads=16, n_kv_heads=8, max_seq_len=2048),   # ~240M
    6: dict(d_model=1536, n_layers=18, n_heads=16, n_kv_heads=8, max_seq_len=2048),   # ~500M
    7: dict(d_model=2048, n_layers=24, n_heads=32, n_kv_heads=8, max_seq_len=4096),   # ~1.3B
    8: dict(d_model=2560, n_layers=28, n_heads=32, n_kv_heads=8, max_seq_len=4096),   # ~2.5B
    9: dict(d_model=3584, n_layers=30, n_heads=32, n_kv_heads=8, max_seq_len=4096),   # ~4.5B
    10: dict(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8, max_seq_len=4096),  # ~6B
}

# ── Training HP scaling rules ────────────────────────────────────
# (learning_rate, batch_size, warmup_steps, total_steps)
_TRAINING_TABLE: dict[int, dict] = {
    1:  dict(learning_rate=1e-3,  batch_size=64,  warmup_steps=200,   total_steps=5_000),
    2:  dict(learning_rate=6e-4,  batch_size=64,  warmup_steps=500,   total_steps=10_000),
    3:  dict(learning_rate=3e-4,  batch_size=32,  warmup_steps=1_000, total_steps=20_000),
    4:  dict(learning_rate=3e-4,  batch_size=32,  warmup_steps=2_000, total_steps=50_000),
    5:  dict(learning_rate=2e-4,  batch_size=16,  warmup_steps=2_000, total_steps=100_000),
    6:  dict(learning_rate=1.5e-4, batch_size=16, warmup_steps=3_000, total_steps=150_000),
    7:  dict(learning_rate=1e-4,  batch_size=8,   warmup_steps=3_000, total_steps=200_000),
    8:  dict(learning_rate=8e-5,  batch_size=8,   warmup_steps=4_000, total_steps=300_000),
    9:  dict(learning_rate=6e-5,  batch_size=4,   warmup_steps=4_000, total_steps=400_000),
    10: dict(learning_rate=5e-5,  batch_size=4,   warmup_steps=5_000, total_steps=500_000),
}


def _validate_depth(depth: int) -> None:
    """Raise ValueError if depth is out of the supported 1-10 range."""
    if not isinstance(depth, int) or depth < 1 or depth > 10:
        raise ValueError(
            f"depth must be an integer between 1 and 10 (inclusive), got {depth!r}"
        )


def model_config_from_depth(depth: int, **overrides) -> ModelConfig:
    """Build a ModelConfig from a depth level (1-10).

    Depth 1 yields a ~1M param model suitable for quick tests;
    depth 10 yields a ~6B param model for large-scale experiments.

    Args:
        depth: Integer from 1 to 10 selecting the model scale.
        **overrides: Additional keyword arguments forwarded to ModelConfig.

    Returns:
        A fully configured ModelConfig instance.

    Raises:
        ValueError: If depth is outside [1, 10].
    """
    _validate_depth(depth)
    params = {**_DEPTH_TABLE[depth], **overrides}
    cfg = ModelConfig(**params)
    logger.info(
        "auto_scale.model",
        depth=depth,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        max_seq_len=cfg.max_seq_len,
    )
    return cfg


def training_config_from_depth(depth: int, **overrides) -> TrainingConfig:
    """Build a TrainingConfig from a depth level (1-10).

    Automatically scales learning rate, batch size, warmup, and total
    training steps to match the model size implied by depth.

    Args:
        depth: Integer from 1 to 10 selecting the training scale.
        **overrides: Additional keyword arguments forwarded to TrainingConfig.

    Returns:
        A fully configured TrainingConfig instance.

    Raises:
        ValueError: If depth is outside [1, 10].
    """
    _validate_depth(depth)
    params = {**_TRAINING_TABLE[depth], **overrides}
    cfg = TrainingConfig(**params)
    logger.info(
        "auto_scale.training",
        depth=depth,
        lr=cfg.learning_rate,
        batch_size=cfg.batch_size,
        warmup=cfg.warmup_steps,
        total_steps=cfg.total_steps,
    )
    return cfg
