"""Per-token value embeddings for nanochat-faithful implementation.

Value embeddings provide each token with a learned residual vector that is
independent of context. This is distinct from input embeddings (which are
consumed at the bottom of the stack) — value embeddings inject a
token-specific bias directly into the attention output at each layer.

Mechanism::

    # During attention output in each TransformerBlock:
    v_embed = value_embedding_table[token_ids]   # [B, S, d_model]
    attn_output = attn_output + v_embed          # residual injection

Design rationale:
- Attention already computes value vectors (v_proj) that are contextually
  mixed. But value embeddings bypass the attention mechanism entirely and
  add a per-token bias that is always present regardless of context.
- This is analogous to "position-free static knowledge" per token.
- Initialization: near-zero (uniform[-eps, eps]) so that at the start of
  training the model behaves identically to one without value embeddings.

References:
    - nanochat architecture (conditional memory / value bias)
    - Similar to "value residual learning" in MLA-style architectures
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import structlog
from flax import nnx

log = structlog.get_logger(__name__)


class ValueEmbedding(nnx.Module):
    """Per-token value embedding table.

    Maps each token ID to a learned vector that is added to the attention
    output at each transformer block. Shared across all layers by default
    (the same table is instantiated once and passed to all blocks).

    Attributes:
        vocab_size: Size of the token vocabulary.
        d_model: Dimension of the value vectors (matches d_model).
        table: Embedding parameter of shape ``(vocab_size, d_model)``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        init_scale: float = 1e-4,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the value embedding table.

        Args:
            vocab_size: Vocabulary size.
            d_model: Embedding / hidden dimension.
            init_scale: Std for initialization. Very small (1e-4) so that
                value embeddings start as a near-zero perturbation.
            rngs: Flax NNX RNG container.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Initialize near zero — value embeddings start as no-ops and
        # grow as needed during training.
        key = rngs.params()
        self.table = nnx.Param(
            jax.random.normal(key, (vocab_size, d_model)) * init_scale
        )  # [vocab_size, d_model]

        log.debug(
            "value_embedding.init",
            vocab_size=vocab_size,
            d_model=d_model,
            init_scale=init_scale,
        )

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        """Look up value embeddings for a batch of token IDs.

        Args:
            token_ids: Integer token IDs of shape ``(batch, seq_len)``.

        Returns:
            Value vectors of shape ``(batch, seq_len, d_model)``.
        """
        # Simple embedding lookup
        return self.table.get_value()[token_ids]  # [B, S, d_model]

    def __repr__(self) -> str:
        return f"ValueEmbedding(vocab_size={self.vocab_size}, d_model={self.d_model})"
