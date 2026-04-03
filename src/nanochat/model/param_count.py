"""Parameter counting and FLOP estimation utilities for nanochat-jax.

Provides three main functions:

- :func:`count_params`: counts trainable parameters grouped by component
  (embedding, attention, FFN, norms, LM head).
- :func:`estimate_flops_per_token`: estimates the floating-point
  operations for a single forward pass per token.
- :func:`compute_mfu`: computes Model FLOP Utilization (MFU) given
  measured throughput and hardware peak FLOPS.

These utilities are essential for training monitoring, cost estimation,
and hardware utilization analysis.

References:
    - Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
    - Hoffmann et al., "Training Compute-Optimal Large Language Models"
      (Chinchilla, 2022)
    - PaLM: Chowdhery et al., "PaLM: Scaling Language Modeling with
      Pathways" (2022) --- MFU definition
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import structlog
from flax import nnx

from nanochat.config.model_config import ModelConfig

log = structlog.get_logger(__name__)


# ======================================================================
# Parameter counting
# ======================================================================


def count_params(model: Any) -> dict[str, int]:
    """Count trainable parameters grouped by component.

    Traverses the model's NNX parameter state and classifies each
    parameter array into one of the following groups based on its
    path in the parameter tree:

    - **embedding**: token embedding weights.
    - **attention**: all attention-related parameters (Q/K/V projections,
      output projection).
    - **ffn**: feed-forward network parameters (gate, up, down
      projections).
    - **norms**: normalization layer parameters (RMSNorm gamma, etc.).
    - **lm_head**: output projection (only when embeddings are untied).
    - **other**: any parameters that don't match the above categories.
    - **total**: sum of all parameters.

    Args:
        model: A :class:`TransformerLM` instance (or any ``nnx.Module``).

    Returns:
        A dictionary mapping group names to parameter counts.

    Example:
        >>> from nanochat.config.model_config import ModelConfig
        >>> from nanochat.model.transformer import TransformerLM
        >>> cfg = ModelConfig.for_scale("nano")
        >>> rngs = nnx.Rngs(params=0)
        >>> model = TransformerLM(cfg, rngs=rngs)
        >>> counts = count_params(model)
        >>> counts["total"] > 0
        True
    """
    # Extract all parameters from the model
    state = nnx.state(model, nnx.Param)

    groups: dict[str, int] = {
        "embedding": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "lm_head": 0,
        "other": 0,
        "total": 0,
    }

    # Flatten the state tree to get (path, value) pairs
    flat_state = jax.tree_util.tree_leaves_with_path(state)

    for key_path, leaf in flat_state:
        # Build a dot-separated path string for classification
        path_str = _keypath_to_str(key_path)
        param_size = leaf.size if hasattr(leaf, "size") else 0

        # Classify the parameter by its path
        group = _classify_param(path_str)
        groups[group] += param_size
        groups["total"] += param_size

    log.debug(
        "param_count.summary",
        embedding=groups["embedding"],
        attention=groups["attention"],
        ffn=groups["ffn"],
        norms=groups["norms"],
        lm_head=groups["lm_head"],
        other=groups["other"],
        total=groups["total"],
    )

    return groups


def _keypath_to_str(key_path: tuple) -> str:
    """Convert a JAX key path tuple to a dot-separated string.

    Args:
        key_path: Tuple of JAX path keys (e.g., ``DictKey``,
            ``SequenceKey``, ``GetAttrKey``, ``FlattenedIndexKey``).

    Returns:
        A human-readable dot-separated path string.
    """
    parts: list[str] = []
    for key in key_path:
        # Handle different JAX key types
        if hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return ".".join(parts)


def _classify_param(path: str) -> str:
    """Classify a parameter into a component group based on its path.

    Args:
        path: Dot-separated parameter path string.

    Returns:
        One of ``"embedding"``, ``"attention"``, ``"ffn"``, ``"norms"``,
        ``"lm_head"``, or ``"other"``.
    """
    path_lower = path.lower()

    # LM head (must check before "embed" to avoid misclassification)
    if "lm_head" in path_lower:
        return "lm_head"

    # Embeddings
    if "embed" in path_lower:
        return "embedding"

    # Attention components
    attention_keywords = (
        "attention", "attn", "q_proj", "k_proj", "v_proj",
        "out_proj", "qkv_proj",
    )
    if any(kw in path_lower for kw in attention_keywords):
        return "attention"

    # Feed-forward components
    ffn_keywords = (
        "ffn", "feedforward", "gate_proj", "up_proj", "down_proj",
        "mlp", "fc1", "fc2",
    )
    if any(kw in path_lower for kw in ffn_keywords):
        return "ffn"

    # Normalization layers
    norm_keywords = ("norm", "gamma", "layernorm", "rmsnorm")
    if any(kw in path_lower for kw in norm_keywords):
        return "norms"

    return "other"


# ======================================================================
# FLOP estimation
# ======================================================================


def estimate_flops_per_token(cfg: ModelConfig) -> float:
    """Estimate floating-point operations for one forward pass per token.

    Uses the standard approximation for transformer FLOPs, accounting
    for the model's specific attention layout (MHA vs. GQA/MQA) and
    feed-forward type (standard MLP vs. gated variants like SwiGLU).

    Per-layer breakdown (all counts are multiply-accumulate pairs,
    hence the factor of 2 per matmul):

    **Attention**:
        - Q projection: ``2 * d_model * (n_heads * d_head)``
        - K projection: ``2 * d_model * (n_kv_heads * d_head)``
        - V projection: ``2 * d_model * (n_kv_heads * d_head)``
        - Attention scores: ``2 * n_heads * d_head * seq_len``
        - Attention context: ``2 * n_heads * d_head * seq_len``
        - Output projection: ``2 * (n_heads * d_head) * d_model``

    **FFN** (gated variant, e.g. SwiGLU):
        - Gate projection: ``2 * d_model * d_ff``
        - Up projection: ``2 * d_model * d_ff``
        - Down projection: ``2 * d_ff * d_model``
        Total: ``3 * 2 * d_model * d_ff``

    **FFN** (standard MLP):
        - Up projection: ``2 * d_model * d_ff``
        - Down projection: ``2 * d_ff * d_model``
        Total: ``2 * 2 * d_model * d_ff``

    **Embedding + LM head**: ``2 * d_model * vocab_size``

    Args:
        cfg: Model architecture configuration.

    Returns:
        Estimated FLOPs per token for a single forward pass (float).

    Example:
        >>> from nanochat.config.model_config import ModelConfig
        >>> cfg = ModelConfig.for_scale("small")
        >>> flops = estimate_flops_per_token(cfg)
        >>> flops > 0
        True
    """
    d_model = cfg.d_model
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    n_kv_heads = cfg.n_kv_heads
    d_head = cfg.d_head
    d_ff = cfg.d_ff
    vocab_size = cfg.vocab_size
    seq_len = cfg.max_seq_len

    # -- Per-layer attention FLOPs --
    # Q projection: d_model -> n_heads * d_head
    flops_q_proj = 2 * d_model * (n_heads * d_head)
    # K projection: d_model -> n_kv_heads * d_head
    flops_k_proj = 2 * d_model * (n_kv_heads * d_head)
    # V projection: d_model -> n_kv_heads * d_head
    flops_v_proj = 2 * d_model * (n_kv_heads * d_head)
    # Attention score computation: each head computes q @ k^T
    # per token: dot product with all seq_len keys
    flops_attn_scores = 2 * n_heads * d_head * seq_len
    # Attention context: weighted sum of values
    flops_attn_context = 2 * n_heads * d_head * seq_len
    # Output projection: n_heads * d_head -> d_model
    flops_out_proj = 2 * (n_heads * d_head) * d_model

    flops_attn_per_layer = (
        flops_q_proj
        + flops_k_proj
        + flops_v_proj
        + flops_attn_scores
        + flops_attn_context
        + flops_out_proj
    )

    # -- Per-layer FFN FLOPs --
    # Gated variants (SwiGLU, GeGLU) have 3 projections; standard MLP has 2
    is_gated = cfg.ffn_type in ("swiglu", "geglu")
    n_ffn_projections = 3 if is_gated else 2
    flops_ffn_per_layer = n_ffn_projections * 2 * d_model * d_ff

    # -- Total per-layer FLOPs --
    flops_per_layer = flops_attn_per_layer + flops_ffn_per_layer

    # -- All layers --
    flops_all_layers = n_layers * flops_per_layer

    # -- Embedding + LM head (shared or separate) --
    # Forward pass through embedding lookup is essentially free (gather),
    # but the LM head matmul costs 2 * d_model * vocab_size per token
    flops_lm_head = 2 * d_model * vocab_size

    total_flops = float(flops_all_layers + flops_lm_head)

    log.debug(
        "flops_estimate",
        flops_attn_per_layer=flops_attn_per_layer,
        flops_ffn_per_layer=flops_ffn_per_layer,
        flops_per_layer=flops_per_layer,
        flops_all_layers=flops_all_layers,
        flops_lm_head=flops_lm_head,
        total_flops_per_token=total_flops,
    )

    return total_flops


def estimate_training_flops(
    cfg: ModelConfig,
    n_tokens: int,
) -> float:
    """Estimate total training FLOPs using the 3x forward-pass rule.

    For standard backpropagation, the total compute is approximately
    ``3 * forward_flops * n_tokens`` (1x forward + 2x backward).

    Args:
        cfg: Model architecture configuration.
        n_tokens: Total number of training tokens.

    Returns:
        Estimated total training FLOPs (float).
    """
    fwd_flops = estimate_flops_per_token(cfg)
    total = 3.0 * fwd_flops * n_tokens
    log.debug(
        "training_flops_estimate",
        fwd_flops_per_token=fwd_flops,
        n_tokens=n_tokens,
        total_training_flops=total,
    )
    return total


# ======================================================================
# Model FLOP Utilization (MFU)
# ======================================================================


def compute_mfu(
    throughput_tps: float,
    cfg: ModelConfig,
    peak_flops: float,
) -> float:
    """Compute Model FLOP Utilization (MFU).

    MFU measures how efficiently the hardware is utilized relative to
    its theoretical peak performance.  It is defined as:

    .. math::

        \\text{MFU} = \\frac{\\text{actual FLOPS}}{\\text{peak FLOPS}}

    where actual FLOPS is computed from the measured training throughput
    (tokens per second) and the estimated FLOPs per token (including
    the 3x factor for forward + backward).

    Args:
        throughput_tps: Measured training throughput in tokens per
            second.
        cfg: Model architecture configuration (used to estimate
            per-token FLOPs).
        peak_flops: Theoretical peak FLOPS of the hardware (e.g.,
            ``312e12`` for an A100 GPU in bfloat16).

    Returns:
        MFU as a fraction in ``[0, 1]``.  Values above ``0.5`` are
        considered excellent for language model training.

    Raises:
        ValueError: If *peak_flops* is not positive.

    Example:
        >>> from nanochat.config.model_config import ModelConfig
        >>> cfg = ModelConfig.for_scale("small")
        >>> mfu = compute_mfu(
        ...     throughput_tps=50_000.0,
        ...     cfg=cfg,
        ...     peak_flops=312e12,
        ... )
        >>> 0.0 <= mfu <= 1.0
        True
    """
    if peak_flops <= 0:
        raise ValueError(
            f"peak_flops must be positive, got {peak_flops}"
        )
    if throughput_tps < 0:
        raise ValueError(
            f"throughput_tps must be non-negative, got {throughput_tps}"
        )

    # FLOPs per token for training (3x forward: 1x fwd + 2x bwd)
    flops_per_token = 3.0 * estimate_flops_per_token(cfg)

    # Actual FLOPS achieved
    actual_flops = throughput_tps * flops_per_token

    mfu = actual_flops / peak_flops

    log.debug(
        "mfu.compute",
        throughput_tps=throughput_tps,
        flops_per_token=flops_per_token,
        actual_flops=actual_flops,
        peak_flops=peak_flops,
        mfu=mfu,
    )

    return mfu


# ======================================================================
# Formatting helpers
# ======================================================================


def format_param_count(counts: dict[str, int]) -> str:
    """Format parameter counts into a human-readable summary table.

    Args:
        counts: Dictionary returned by :func:`count_params`.

    Returns:
        A multi-line string with aligned component names and counts.

    Example:
        >>> counts = {"embedding": 1_000_000, "attention": 2_000_000,
        ...           "ffn": 3_000_000, "norms": 1_000, "lm_head": 0,
        ...           "other": 0, "total": 6_001_000}
        >>> print(format_param_count(counts))  # doctest: +SKIP
    """
    lines: list[str] = ["Parameter Count Summary", "=" * 40]

    # Display order (total last)
    display_order = [
        "embedding", "attention", "ffn", "norms", "lm_head", "other",
    ]

    for group in display_order:
        count = counts.get(group, 0)
        if count > 0:
            lines.append(f"  {group:<15s} {count:>15,d}  ({_human_readable(count)})")

    lines.append("-" * 40)
    total = counts.get("total", 0)
    lines.append(f"  {'total':<15s} {total:>15,d}  ({_human_readable(total)})")

    return "\n".join(lines)


def _human_readable(n: int) -> str:
    """Convert a parameter count to a human-readable string.

    Args:
        n: Number of parameters.

    Returns:
        Abbreviated string like ``"1.2M"`` or ``"350K"``.
    """
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
