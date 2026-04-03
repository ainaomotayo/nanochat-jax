"""Grouped-Query Attention for the nanochat-jax transformer.

This module implements :class:`GroupedQueryAttention`, a unified attention
mechanism that supports Multi-Head Attention (MHA), Multi-Query Attention
(MQA), and Grouped-Query Attention (GQA) through a single parameterization
controlled by the ratio ``n_heads / n_kv_heads``.

**Attention Variants and Memory Comparison**

+----------+-----------+------------------+-----------------------------+
| Variant  | n_kv_heads| KV cache / layer | Example                     |
+==========+===========+==================+=============================+
| MHA      | = n_heads | 2 * B*S*d_model  | GPT-2, BERT                 |
+----------+-----------+------------------+-----------------------------+
| GQA      | < n_heads | 2 * B*S*n_kv*d_h | LLaMA-2 70B (n_kv=8)       |
|          | > 1       |                  | Reduces KV cache by n_groups|
+----------+-----------+------------------+-----------------------------+
| MQA      | = 1       | 2 * B*S*d_head   | PaLM, Falcon                |
|          |           |                  | Minimal KV cache            |
+----------+-----------+------------------+-----------------------------+

For a model with ``d_model=4096, n_heads=32, d_head=128``:
- MHA (n_kv=32): KV cache = 2 * 32 * 128 = 8192 floats/token/layer
- GQA (n_kv=8):  KV cache = 2 *  8 * 128 = 2048 floats/token/layer (4x smaller)
- MQA (n_kv=1):  KV cache = 2 *  1 * 128 =  256 floats/token/layer (32x smaller)

GQA provides a good balance: near-MHA quality with significantly reduced
memory during inference, enabling larger batch sizes and longer contexts.

References:
    - MHA: Vaswani et al., "Attention Is All You Need" (2017)
    - MQA: Shazeer, "Fast Transformer Decoding: One Write-Head is All
      You Need" (2019)
    - GQA: Ainslie et al., "GQA: Training Generalized Multi-Query
      Transformer Models from Multi-Head Checkpoints" (2023)
    - RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary
      Position Embedding" (2021)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import structlog
from flax import nnx

from nanochat.config.model_config import ModelConfig
from nanochat.model.embeddings import RotaryEmbedding

log = structlog.get_logger(__name__)


class GroupedQueryAttention(nnx.Module):
    """Grouped-Query Attention supporting MHA, GQA, and MQA.

    This module computes scaled dot-product attention with support for:
    - Rotary Position Embeddings (RoPE) applied to queries and keys.
    - KV caching for efficient autoregressive decoding.
    - Grouped-Query Attention where multiple query heads share a single
      key/value head, reducing memory and computation.

    The attention variant is determined by the relationship between
    ``n_heads`` (query heads) and ``n_kv_heads`` (key/value heads):

    - ``n_kv_heads == n_heads``: Standard Multi-Head Attention (MHA)
    - ``1 < n_kv_heads < n_heads``: Grouped-Query Attention (GQA)
    - ``n_kv_heads == 1``: Multi-Query Attention (MQA)

    Attributes:
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value attention heads.
        d_head: Dimensionality of each attention head.
        n_groups: Number of query heads per key/value head
            (``n_heads // n_kv_heads``).
        q_proj: Linear projection for queries
            (``d_model -> n_heads * d_head``).
        k_proj: Linear projection for keys
            (``d_model -> n_kv_heads * d_head``).
        v_proj: Linear projection for values
            (``d_model -> n_kv_heads * d_head``).
        out_proj: Linear projection for output
            (``n_heads * d_head -> d_model``).
        dropout: Attention dropout layer.
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize GroupedQueryAttention.

        Creates the four linear projections (Q, K, V, output) and the
        attention dropout layer. Projection dimensions are derived from
        the model configuration.

        Args:
            cfg: Model configuration specifying architecture hyperparameters.
                Required fields: ``d_model``, ``n_heads``, ``n_kv_heads``,
                ``d_head``, ``use_bias``, ``dropout_rate``.
            rngs: Flax NNX RNG container used for parameter initialization
                and dropout randomness.

        Raises:
            ValueError: If ``n_heads`` is not divisible by ``n_kv_heads``.
        """
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.n_groups = cfg.n_heads // cfg.n_kv_heads

        # Query projection: d_model -> n_heads * d_head
        self.q_proj = nnx.Linear(
            cfg.d_model,
            cfg.n_heads * cfg.d_head,
            use_bias=cfg.use_bias,
            rngs=rngs,
        )

        # Key projection: d_model -> n_kv_heads * d_head
        self.k_proj = nnx.Linear(
            cfg.d_model,
            cfg.n_kv_heads * cfg.d_head,
            use_bias=cfg.use_bias,
            rngs=rngs,
        )

        # Value projection: d_model -> n_kv_heads * d_head
        self.v_proj = nnx.Linear(
            cfg.d_model,
            cfg.n_kv_heads * cfg.d_head,
            use_bias=cfg.use_bias,
            rngs=rngs,
        )

        # Output projection: n_heads * d_head -> d_model
        self.out_proj = nnx.Linear(
            cfg.n_heads * cfg.d_head,
            cfg.d_model,
            use_bias=cfg.use_bias,
            rngs=rngs,
        )

        # Attention dropout
        self.dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

        log.debug(
            "grouped_query_attention.init",
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            d_head=self.d_head,
            n_groups=self.n_groups,
            d_model=cfg.d_model,
            use_bias=cfg.use_bias,
            dropout_rate=cfg.dropout_rate,
        )

    def _expand_kv(self, x: jax.Array) -> jax.Array:
        """Expand key/value heads to match the number of query heads.

        For standard MHA (``n_groups == 1``), this is a no-op. For GQA/MQA,
        each KV head is repeated ``n_groups`` times so that every query head
        has a corresponding key/value head for the attention computation.

        Args:
            x: Key or value tensor of shape
                ``(batch, n_kv_heads, seq_len, d_head)``.

        Returns:
            Expanded tensor of shape ``(batch, n_heads, seq_len, d_head)``
            where ``n_heads = n_kv_heads * n_groups``.
        """
        if self.n_groups == 1:
            return x  # MHA: no expansion needed

        # x: [B, n_kv_heads, S, d_head]
        # Repeat each KV head n_groups times along the head dimension
        # Result: [B, n_kv_heads * n_groups, S, d_head] = [B, n_heads, S, d_head]
        return jnp.repeat(x, repeats=self.n_groups, axis=1)

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: jax.Array,
        kv_cache: Optional[Tuple[jax.Array, jax.Array]] = None,
        deterministic: bool = True,
    ) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Compute grouped-query attention with RoPE and optional KV cache.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            cos: Cosine frequencies for RoPE of shape
                ``(seq_len, d_head // 2)``. Obtained from
                :meth:`RotaryEmbedding.get_freqs`.
            sin: Sine frequencies for RoPE of shape
                ``(seq_len, d_head // 2)``. Obtained from
                :meth:`RotaryEmbedding.get_freqs`.
            mask: Boolean attention mask of shape
                ``(batch_or_1, 1, seq_len, seq_len_total)`` where ``True``
                indicates positions that **can** be attended to and ``False``
                indicates positions that should be masked out. For causal
                decoding, ``seq_len_total = past_len + seq_len``.
            kv_cache: Optional tuple ``(k_cache, v_cache)`` from previous
                decoding steps. Each tensor has shape
                ``(batch, n_kv_heads, past_len, d_head)``. When provided,
                new keys and values are concatenated with the cache along
                the sequence dimension.
            deterministic: If ``True``, disables dropout (inference mode).
                If ``False``, applies dropout (training mode).

        Returns:
            A tuple ``(output, new_kv_cache)`` where:
            - ``output``: Attention output of shape
              ``(batch, seq_len, d_model)``.
            - ``new_kv_cache``: Updated KV cache tuple
              ``(k, v)`` each of shape
              ``(batch, n_kv_heads, total_seq_len, d_head)`` where
              ``total_seq_len = past_len + seq_len``. These are the
              **un-expanded** KV tensors (before GQA head replication).

        Shape flow::

            Input:  x [B, S, D]
                        |
                Q,K,V projections
                        |
            Q: [B, S, n_heads * d_head]   -> reshape -> [B, S, n_heads, d_head]    -> transpose -> [B, n_heads, S, d_head]
            K: [B, S, n_kv_heads * d_head] -> reshape -> [B, S, n_kv_heads, d_head] -> transpose -> [B, n_kv_heads, S, d_head]
            V: [B, S, n_kv_heads * d_head] -> reshape -> [B, S, n_kv_heads, d_head] -> transpose -> [B, n_kv_heads, S, d_head]
                        |
                Apply RoPE to Q, K
                        |
                Concatenate with KV cache (if present)
                K: [B, n_kv_heads, S_total, d_head]
                V: [B, n_kv_heads, S_total, d_head]
                        |
                Expand KV for GQA: repeat n_kv_heads -> n_heads
                K_expanded: [B, n_heads, S_total, d_head]
                V_expanded: [B, n_heads, S_total, d_head]
                        |
                Attention scores: Q @ K^T / sqrt(d_head)
                scores: [B, n_heads, S, S_total]
                        |
                Apply mask + softmax + dropout
                        |
                Weighted sum: weights @ V
                context: [B, n_heads, S, d_head]
                        |
                Transpose + reshape: [B, S, n_heads * d_head]
                        |
                Output projection: [B, S, D]
        """
        batch_size, seq_len, _ = x.shape

        # -----------------------------------------------------------------
        # Step 1: Project queries, keys, and values
        # -----------------------------------------------------------------
        q = self.q_proj(x)  # [B, S, n_heads * d_head]
        k = self.k_proj(x)  # [B, S, n_kv_heads * d_head]
        v = self.v_proj(x)  # [B, S, n_kv_heads * d_head]

        # -----------------------------------------------------------------
        # Step 2: Reshape to multi-head format and transpose
        # -----------------------------------------------------------------
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_head)
        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, n_heads, S, d_head]

        k = k.reshape(batch_size, seq_len, self.n_kv_heads, self.d_head)
        k = jnp.transpose(k, (0, 2, 1, 3))  # [B, n_kv_heads, S, d_head]

        v = v.reshape(batch_size, seq_len, self.n_kv_heads, self.d_head)
        v = jnp.transpose(v, (0, 2, 1, 3))  # [B, n_kv_heads, S, d_head]

        # -----------------------------------------------------------------
        # Step 3: Apply Rotary Position Embeddings to Q and K
        # -----------------------------------------------------------------
        # cos, sin: [S, d_head // 2]
        # RotaryEmbedding.apply expects [B, n_heads, S, d_head]
        q = RotaryEmbedding.apply(q, cos, sin)  # [B, n_heads, S, d_head]
        k = RotaryEmbedding.apply(k, cos, sin)  # [B, n_kv_heads, S, d_head]

        # -----------------------------------------------------------------
        # Step 4: Concatenate with KV cache (for autoregressive decoding)
        # -----------------------------------------------------------------
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # k_cache: [B, n_kv_heads, past_len, d_head]
            # v_cache: [B, n_kv_heads, past_len, d_head]
            k = jnp.concatenate(
                [k_cache, k], axis=2
            )  # [B, n_kv_heads, past_len + S, d_head]
            v = jnp.concatenate(
                [v_cache, v], axis=2
            )  # [B, n_kv_heads, past_len + S, d_head]

        # Save un-expanded KV for the cache return
        new_kv_cache = (k, v)

        # -----------------------------------------------------------------
        # Step 5: Expand KV heads for GQA
        # -----------------------------------------------------------------
        # Repeat each KV head n_groups times to match query head count
        k_expanded = self._expand_kv(k)  # [B, n_heads, S_total, d_head]
        v_expanded = self._expand_kv(v)  # [B, n_heads, S_total, d_head]

        # -----------------------------------------------------------------
        # Step 6: Compute scaled dot-product attention scores
        # -----------------------------------------------------------------
        scale = math.sqrt(self.d_head)
        # Q @ K^T: [B, n_heads, S, d_head] @ [B, n_heads, d_head, S_total]
        #        -> [B, n_heads, S, S_total]
        scores = jnp.matmul(q, jnp.transpose(k_expanded, (0, 1, 3, 2))) / scale

        # -----------------------------------------------------------------
        # Step 7: Apply attention mask
        # -----------------------------------------------------------------
        # mask: [B_or_1, 1, S, S_total], True = attend, False = mask out
        # Use a large negative value instead of -inf to avoid NaN in softmax
        # when an entire row is masked (e.g., padding tokens).
        scores = jnp.where(mask, scores, jnp.float32(-1e9))

        # -----------------------------------------------------------------
        # Step 8: Softmax and dropout
        # -----------------------------------------------------------------
        weights = jax.nn.softmax(scores, axis=-1)  # [B, n_heads, S, S_total]
        weights = self.dropout(weights, deterministic=deterministic)

        # -----------------------------------------------------------------
        # Step 9: Compute weighted sum of values
        # -----------------------------------------------------------------
        # weights @ V: [B, n_heads, S, S_total] @ [B, n_heads, S_total, d_head]
        #            -> [B, n_heads, S, d_head]
        context = jnp.matmul(weights, v_expanded)  # [B, n_heads, S, d_head]

        # -----------------------------------------------------------------
        # Step 10: Reshape back to [B, S, d_model] and project
        # -----------------------------------------------------------------
        # Transpose: [B, n_heads, S, d_head] -> [B, S, n_heads, d_head]
        context = jnp.transpose(context, (0, 2, 1, 3))
        # Reshape: [B, S, n_heads, d_head] -> [B, S, n_heads * d_head]
        context = context.reshape(batch_size, seq_len, self.n_heads * self.d_head)

        # Output projection: [B, S, n_heads * d_head] -> [B, S, d_model]
        output = self.out_proj(context)  # [B, S, d_model]

        return output, new_kv_cache

    def __repr__(self) -> str:
        variant = "MHA"
        if self.n_groups > 1 and self.n_kv_heads > 1:
            variant = "GQA"
        elif self.n_kv_heads == 1:
            variant = "MQA"
        return (
            f"GroupedQueryAttention("
            f"variant={variant}, "
            f"n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, "
            f"d_head={self.d_head}, "
            f"n_groups={self.n_groups})"
        )
