"""Embedding layers for the nanochat-jax transformer.

This module provides:
- **TokenEmbedding**: Wraps ``nnx.Embed`` with weight-tying support via
  ``as_logits`` for projecting hidden states back to vocabulary logits.
- **RotaryEmbedding**: Implements Rotary Position Embeddings (RoPE) as a
  pure-JAX utility class (not an ``nnx.Module``) that precomputes sin/cos
  frequency tables and applies them to query/key tensors.

References:
    - RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position
      Embedding" (2021)
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import structlog
from flax import nnx

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Token Embedding
# ---------------------------------------------------------------------------


class TokenEmbedding(nnx.Module):
    """Token embedding with optional weight-tied output projection.

    Wraps ``nnx.Embed`` and provides a convenience ``as_logits`` method
    that projects hidden states to vocabulary logits by multiplying with
    the transposed embedding matrix (weight tying).

    Attributes:
        vocab_size: Size of the token vocabulary.
        d_model: Dimensionality of the embedding vectors.
        embed: Underlying ``nnx.Embed`` layer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        init_std: float = 0.02,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize TokenEmbedding.

        Args:
            vocab_size: Number of unique tokens in the vocabulary.
            d_model: Dimensionality of each embedding vector.
            init_std: Standard deviation for the truncated-normal
                initializer. Defaults to ``0.02``.
            rngs: Flax NNX RNG container used by ``nnx.Embed``.

        Raises:
            ValueError: If *vocab_size* or *d_model* is not positive.
        """
        if vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, got {vocab_size}"
            )
        if d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {d_model}"
            )

        self.vocab_size = vocab_size
        self.d_model = d_model

        # nnx.Embed stores weights in self.embed.embedding (nnx.Param)
        self.embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=d_model,
            rngs=rngs,
        )

        # Re-initialize embedding table with custom std if requested
        if init_std != 0.02:
            key = rngs.params()
            self.embed.embedding = nnx.Param(
                jax.random.truncated_normal(
                    key, -2.0, 2.0, shape=(vocab_size, d_model)
                )
                * init_std
            )  # [vocab_size, d_model]

        log.debug(
            "token_embedding.init",
            vocab_size=vocab_size,
            d_model=d_model,
            init_std=init_std,
        )

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        """Look up embeddings for the given token IDs.

        Args:
            token_ids: Integer token IDs of shape ``(batch, seq_len)``.

        Returns:
            Embedding vectors of shape ``(batch, seq_len, d_model)``.
        """
        # token_ids: [batch, seq_len]
        embeddings = self.embed(token_ids)  # [batch, seq_len, d_model]
        return embeddings

    def as_logits(self, hidden: jax.Array) -> jax.Array:
        """Project hidden states to vocabulary logits via weight tying.

        Computes ``hidden @ embedding_matrix.T`` to produce un-normalized
        logits over the vocabulary.

        Args:
            hidden: Hidden-state tensor of shape
                ``(batch, seq_len, d_model)``.

        Returns:
            Logit tensor of shape ``(batch, seq_len, vocab_size)``.
        """
        # hidden: [batch, seq_len, d_model]
        # self.embed.embedding.value: [vocab_size, d_model]
        weight = self.embed.embedding.value  # [vocab_size, d_model]
        logits = hidden @ weight.T  # [batch, seq_len, vocab_size]
        return logits

    def __repr__(self) -> str:
        return (
            f"TokenEmbedding(vocab_size={self.vocab_size}, "
            f"d_model={self.d_model})"
        )


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------


class RotaryEmbedding:
    """Rotary Position Embedding (RoPE).

    This is a **pure-JAX utility class** (not an ``nnx.Module``) that
    precomputes sinusoidal frequency tables and provides a static method
    to apply rotary embeddings to query/key tensors.

    The rotation is applied by splitting each head vector into two halves,
    then combining them with the precomputed cos/sin tables:

    .. code-block:: text

        x1, x2 = x[..., :d//2], x[..., d//2:]
        rotated = concat([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1)

    Attributes:
        d_head: Dimensionality of each attention head.
        max_seq_len: Maximum sequence length for precomputed tables.
        base: Base frequency for the sinusoidal encoding.
        cos_table: Precomputed cosine table of shape
            ``(max_seq_len, d_head // 2)``.
        sin_table: Precomputed sine table of shape
            ``(max_seq_len, d_head // 2)``.
    """

    def __init__(
        self,
        d_head: int,
        max_seq_len: int,
        base: float = 10_000.0,
    ) -> None:
        """Initialize RotaryEmbedding and precompute frequency tables.

        Args:
            d_head: Dimensionality of each attention head. Must be even.
            max_seq_len: Maximum sequence length to support.
            base: Base frequency for the sinusoidal encoding.
                Defaults to ``10000.0``.

        Raises:
            ValueError: If *d_head* is not a positive even integer or
                *max_seq_len* is not positive.
        """
        if d_head <= 0 or d_head % 2 != 0:
            raise ValueError(
                f"d_head must be a positive even integer, got {d_head}"
            )
        if max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be positive, got {max_seq_len}"
            )

        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute and store cos/sin tables
        self.cos_table, self.sin_table = self.precompute_freqs(
            d_head, max_seq_len, base
        )

        log.debug(
            "rotary_embedding.init",
            d_head=d_head,
            max_seq_len=max_seq_len,
            base=base,
            cos_shape=self.cos_table.shape,
            sin_shape=self.sin_table.shape,
        )

    @staticmethod
    def precompute_freqs(
        d_head: int,
        max_seq_len: int,
        base: float = 10_000.0,
    ) -> Tuple[jax.Array, jax.Array]:
        """Precompute cosine and sine frequency tables for RoPE.

        Args:
            d_head: Head dimensionality (must be even).
            max_seq_len: Maximum sequence length.
            base: Base frequency. Defaults to ``10000.0``.

        Returns:
            A tuple ``(cos_table, sin_table)`` each of shape
            ``(max_seq_len, d_head // 2)``.
        """
        half_d = d_head // 2

        # Compute inverse frequencies: theta_i = 1 / (base^(2i / d_head))
        # for i in [0, 1, ..., half_d - 1]
        freq_indices = jnp.arange(0, half_d, dtype=jnp.float32)  # [half_d]
        inv_freq = 1.0 / (
            base ** (2.0 * freq_indices / d_head)
        )  # [half_d]

        # Position indices
        positions = jnp.arange(0, max_seq_len, dtype=jnp.float32)  # [max_seq_len]

        # Outer product: each position multiplied by each frequency
        # Result shape: [max_seq_len, half_d]
        angles = jnp.outer(positions, inv_freq)  # [max_seq_len, half_d]

        # Compute cos and sin tables
        cos_table = jnp.cos(angles)  # [max_seq_len, half_d]
        sin_table = jnp.sin(angles)  # [max_seq_len, half_d]

        return cos_table, sin_table

    def get_freqs(
        self,
        seq_len: int,
        offset: int = 0,
    ) -> Tuple[jax.Array, jax.Array]:
        """Retrieve cos/sin slices for a given sequence length and offset.

        This enables incremental decoding by allowing the caller to fetch
        frequency tables starting from a position offset.

        Args:
            seq_len: Number of positions to retrieve.
            offset: Starting position index. Defaults to ``0``.

        Returns:
            A tuple ``(cos_slice, sin_slice)`` each of shape
            ``(seq_len, d_head // 2)``.

        Raises:
            ValueError: If ``offset + seq_len`` exceeds *max_seq_len*.
        """
        end = offset + seq_len
        if end > self.max_seq_len:
            raise ValueError(
                f"Requested positions [{offset}, {end}) exceed "
                f"max_seq_len={self.max_seq_len}. Consider increasing "
                f"max_seq_len or using a shorter sequence."
            )

        cos_slice = self.cos_table[offset:end, :]  # [seq_len, half_d]
        sin_slice = self.sin_table[offset:end, :]  # [seq_len, half_d]

        return cos_slice, sin_slice

    @staticmethod
    def apply(
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        offset: int = 0,
    ) -> jax.Array:
        """Apply rotary position embedding to a query or key tensor.

        Uses the split-and-rotate approach:
        - Split the head dimension into two halves ``(x1, x2)``.
        - Rotate: ``concat([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1)``.

        Args:
            x: Input tensor of shape ``(batch, n_heads, seq_len, d_head)``.
            cos: Cosine table of shape ``(seq_len, d_head // 2)`` (broadcast
                over batch and heads).
            sin: Sine table of shape ``(seq_len, d_head // 2)`` (broadcast
                over batch and heads).
            offset: Positional offset (unused here since cos/sin should
                already be sliced; kept for API compatibility).

        Returns:
            Rotated tensor of the same shape as *x*:
            ``(batch, n_heads, seq_len, d_head)``.
        """
        # x: [batch, n_heads, seq_len, d_head]
        d_head = x.shape[-1]
        half_d = d_head // 2

        # Split head dimension into two halves
        x1 = x[..., :half_d]  # [batch, n_heads, seq_len, half_d]
        x2 = x[..., half_d:]  # [batch, n_heads, seq_len, half_d]

        # Reshape cos/sin for broadcasting: [1, 1, seq_len, half_d]
        cos = cos[None, None, :, :]  # [1, 1, seq_len, half_d]
        sin = sin[None, None, :, :]  # [1, 1, seq_len, half_d]

        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin  # [batch, n_heads, seq_len, half_d]
        rotated_x2 = x1 * sin + x2 * cos  # [batch, n_heads, seq_len, half_d]

        # Concatenate halves back together
        rotated = jnp.concatenate(
            [rotated_x1, rotated_x2], axis=-1
        )  # [batch, n_heads, seq_len, d_head]

        return rotated

    def __repr__(self) -> str:
        return (
            f"RotaryEmbedding(d_head={self.d_head}, "
            f"max_seq_len={self.max_seq_len}, base={self.base})"
        )
