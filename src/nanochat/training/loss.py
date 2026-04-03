"""Loss functions for language model training."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import optax
import structlog

logger = structlog.get_logger()

IGNORE_INDEX: int = -100


def cross_entropy_loss(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array | None = None,
    label_smoothing: float = 0.0,
    z_loss_weight: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute masked cross-entropy loss with optional label smoothing and z-loss.

    Args:
        logits: Raw logits [batch, seq, vocab_size]
        labels: Target token IDs [batch, seq]. Use -100 for ignored positions.
        mask: Optional boolean mask [batch, seq]. True = compute loss. If None, uses labels != IGNORE_INDEX.
        label_smoothing: Smoothing factor in [0, 0.2]. 0 = standard CE.
        z_loss_weight: Weight for z-loss regularizer. 0 = disabled.

    Returns:
        loss: Scalar mean loss
        metrics: Dict with ce_loss, z_loss, n_tokens
    """
    if mask is None:
        mask = (labels != IGNORE_INDEX).astype(jnp.float32)  # [B, S]
    else:
        mask = mask.astype(jnp.float32)

    # Replace ignored labels with 0 (valid index) to avoid indexing errors
    safe_labels = jnp.where(labels == IGNORE_INDEX, 0, labels)  # [B, S]

    vocab_size = logits.shape[-1]

    # Standard cross-entropy
    if label_smoothing == 0.0:
        ce = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits.reshape(-1, vocab_size),
            labels=safe_labels.reshape(-1),
        ).reshape(logits.shape[:2])  # [B, S]
    else:
        # One-hot with smoothing
        one_hot = jax.nn.one_hot(safe_labels, vocab_size)  # [B, S, V]
        smooth = one_hot * (1.0 - label_smoothing) + label_smoothing / vocab_size
        ce = -jnp.sum(smooth * jax.nn.log_softmax(logits, axis=-1), axis=-1)  # [B, S]

    # Apply mask
    ce_masked = ce * mask  # [B, S]
    n_tokens = jnp.maximum(mask.sum(), 1.0)
    ce_loss = ce_masked.sum() / n_tokens

    # Z-loss (PaLM-style logit regularization)
    z_loss = jnp.float32(0.0)
    if z_loss_weight > 0.0:
        log_z = jax.nn.logsumexp(logits, axis=-1)  # [B, S]
        z_loss = z_loss_weight * (log_z ** 2 * mask).sum() / n_tokens

    total_loss = ce_loss + z_loss

    metrics = {
        "ce_loss": ce_loss,
        "z_loss": z_loss,
        "n_tokens": n_tokens,
    }

    return total_loss, metrics
