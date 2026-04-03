"""Evaluation metrics for language models."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import math


def perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss. PPL = exp(loss). Clips loss at 20 to avoid inf."""
    return math.exp(min(loss, 20.0))


def bits_per_byte(loss: float, avg_chars_per_token: float = 4.0) -> float:
    """Compute bits per byte from loss. BPB = loss / log(2) / avg_chars_per_token."""
    return loss / math.log(2) / avg_chars_per_token


def token_accuracy(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array | None = None,
) -> float:
    """Fraction of positions where argmax(logits) == label.
    logits: [batch, seq, vocab], labels: [batch, seq], mask: [batch, seq]
    """
    preds = jnp.argmax(logits, axis=-1)  # [batch, seq]
    correct = (preds == labels)  # [batch, seq]
    if mask is not None:
        correct = correct & mask.astype(jnp.bool_)
        n_total = jnp.maximum(mask.sum(), 1.0)
    else:
        n_total = jnp.float32(correct.size)
    return float(correct.sum() / n_total)


def top_k_accuracy(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array | None = None,
    k: int = 5,
) -> float:
    """Fraction of positions where label is in top-k predictions."""
    top_k_preds = jax.lax.top_k(logits, k)[1]  # [batch, seq, k]
    labels_expanded = labels[..., None]  # [batch, seq, 1]
    in_top_k = jnp.any(top_k_preds == labels_expanded, axis=-1)  # [batch, seq]
    if mask is not None:
        in_top_k = in_top_k & mask.astype(jnp.bool_)
        n_total = jnp.maximum(mask.sum(), 1.0)
    else:
        n_total = jnp.float32(in_top_k.size)
    return float(in_top_k.sum() / n_total)
