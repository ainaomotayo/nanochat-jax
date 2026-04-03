"""Grouped-Query Attention with nanochat-faithful extensions.

Faithful nanochat port adds:

1. **QK normalization**: L2-normalize Q and K per head position before
   computing attention scores. This bounds dot-product magnitudes and
   stabilizes training at depth.

2. **QK scale**: Use ``qk_scale_factor / sqrt(d_head)`` instead of the
   standard ``1 / sqrt(d_head)``. nanochat default: 1.2.

3. **Logit softcap**: Apply ``cap * tanh(logits / cap)`` BEFORE softmax
   to prevent extreme logit spikes. nanochat default: cap=30.0.

4. **Sliding window attention**: Optionally restrict each query to attend
   only to the last ``window_size`` keys (local attention), with optional
   leading global tokens that are always attendable.

Attention variants (unchanged from original):
- MHA:  n_kv_heads == n_heads
- GQA:  1 < n_kv_heads < n_heads
- MQA:  n_kv_heads == 1

References:
    - GQA: Ainslie et al. (2023)
    - RoPE: Su et al. (2021)
    - QK norm: Wortsman et al. (2023), Gemma-2 tech report (2024)
    - Logit softcap: Gemma-2 tech report (2024)
    - Sliding window: Beltagy et al. (2020), Mistral (2023)
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


def _l2_normalize(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    """L2-normalize along the last axis.

    Args:
        x: Input array. Normalization is applied per last-dim vector.
        eps: Small constant to avoid division by zero.

    Returns:
        Unit-norm array of the same shape as *x*.
    """
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)


def _build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    n_global_tokens: int = 1,
) -> jax.Array:
    """Build a causal sliding-window attention mask.

    Each query at position q can attend to:
    - All key positions within the window: max(0, q - window_size + 1) ≤ k ≤ q
    - Global key positions: k < n_global_tokens (always attended regardless of window)

    The result is combined with the standard causal mask so that future
    positions are never attended to.

    Args:
        seq_len: Sequence length (both query and key dimension).
        window_size: Local attention window size in tokens.
        n_global_tokens: Number of leading tokens treated as global
            (always-attendable keys).

    Returns:
        Boolean mask of shape ``(1, 1, seq_len, seq_len)`` where ``True``
        means "this (q, k) pair is allowed to attend."
    """
    q_idx = jnp.arange(seq_len)[:, None]   # [S, 1]
    k_idx = jnp.arange(seq_len)[None, :]   # [1, S]

    # Standard causal: k ≤ q
    causal = k_idx <= q_idx

    # Local window: q - window_size < k  (within window)
    in_window = k_idx > (q_idx - window_size)

    # Global tokens: k < n_global_tokens (always visible)
    is_global = k_idx < n_global_tokens

    # Combined: (causal AND in_window) OR is_global
    mask = (causal & in_window) | is_global

    return mask[None, None, :, :]  # [1, 1, S, S]


class GroupedQueryAttention(nnx.Module):
    """Grouped-Query Attention with nanochat extensions.

    Supports MHA / GQA / MQA through the n_heads / n_kv_heads ratio.

    nanochat extensions (all opt-in via ModelConfig):
    - QK L2 normalization before RoPE
    - Attention scale = qk_scale_factor / sqrt(d_head)
    - Logit softcap: cap * tanh(logits / cap)
    - Sliding window mask with global tokens

    Attributes:
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.
        d_head: Head dimension.
        n_groups: n_heads // n_kv_heads.
        use_qk_norm: Whether to L2-normalize Q and K.
        attn_scale: Scalar attention scale (qk_scale_factor / sqrt(d_head)).
        logit_softcap: Softcap value or None.
        sliding_window_size: Window size or None for full attention.
        n_global_tokens: Always-visible global token count.
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize GroupedQueryAttention.

        Args:
            cfg: Model configuration.
            rngs: Flax NNX RNG container.
        """
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.n_groups = cfg.n_heads // cfg.n_kv_heads

        # nanochat extensions
        self.use_qk_norm = cfg.use_qk_norm
        self.attn_scale = cfg.qk_scale_factor / math.sqrt(cfg.d_head)
        self.logit_softcap = cfg.logit_softcap
        self.sliding_window_size = cfg.sliding_window_size
        self.n_global_tokens = cfg.n_global_tokens

        # Projections
        self.q_proj = nnx.Linear(
            cfg.d_model, cfg.n_heads * cfg.d_head,
            use_bias=cfg.use_bias, rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            cfg.d_model, cfg.n_kv_heads * cfg.d_head,
            use_bias=cfg.use_bias, rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            cfg.d_model, cfg.n_kv_heads * cfg.d_head,
            use_bias=cfg.use_bias, rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            cfg.n_heads * cfg.d_head, cfg.d_model,
            use_bias=cfg.use_bias, rngs=rngs,
        )

        self.dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

        log.debug(
            "grouped_query_attention.init",
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            d_head=self.d_head,
            use_qk_norm=self.use_qk_norm,
            attn_scale=self.attn_scale,
            logit_softcap=self.logit_softcap,
            sliding_window_size=self.sliding_window_size,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand_kv(self, x: jax.Array) -> jax.Array:
        """Repeat KV heads to match query head count for GQA."""
        if self.n_groups == 1:
            return x
        return jnp.repeat(x, repeats=self.n_groups, axis=1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: jax.Array,
        kv_cache: Optional[Tuple[jax.Array, jax.Array]] = None,
        deterministic: bool = True,
    ) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Compute attention with RoPE, optional QK norm, softcap, and window mask.

        Args:
            x: Hidden states ``(batch, seq_len, d_model)``.
            cos: RoPE cosines ``(seq_len, d_head // 2)``.
            sin: RoPE sines ``(seq_len, d_head // 2)``.
            mask: Boolean causal mask ``(1_or_B, 1, seq_len, seq_total)``,
                True = attend. Pre-built by TransformerLM.
            kv_cache: Optional ``(k_cache, v_cache)`` for incremental decoding.
            deterministic: If True, disables attention dropout.

        Returns:
            Tuple ``(output, new_kv_cache)`` where output has shape
            ``(batch, seq_len, d_model)`` and new_kv_cache contains
            updated key/value tensors for all attended positions.
        """
        B, S, _ = x.shape

        # ----------------------------------------------------------------
        # 1. Linear projections
        # ----------------------------------------------------------------
        q = self.q_proj(x)  # [B, S, n_heads * d_head]
        k = self.k_proj(x)  # [B, S, n_kv_heads * d_head]
        v = self.v_proj(x)  # [B, S, n_kv_heads * d_head]

        # ----------------------------------------------------------------
        # 2. Reshape to multi-head format
        # ----------------------------------------------------------------
        q = q.reshape(B, S, self.n_heads, self.d_head)
        q = jnp.transpose(q, (0, 2, 1, 3))          # [B, n_heads, S, d_head]

        k = k.reshape(B, S, self.n_kv_heads, self.d_head)
        k = jnp.transpose(k, (0, 2, 1, 3))          # [B, n_kv_heads, S, d_head]

        v = v.reshape(B, S, self.n_kv_heads, self.d_head)
        v = jnp.transpose(v, (0, 2, 1, 3))          # [B, n_kv_heads, S, d_head]

        # ----------------------------------------------------------------
        # 3. QK L2 normalization (nanochat feature)
        #    Applied BEFORE RoPE so rotation operates on unit vectors.
        # ----------------------------------------------------------------
        if self.use_qk_norm:
            q = _l2_normalize(q.astype(jnp.float32)).astype(q.dtype)
            k = _l2_normalize(k.astype(jnp.float32)).astype(k.dtype)

        # ----------------------------------------------------------------
        # 4. Apply RoPE
        # ----------------------------------------------------------------
        q = RotaryEmbedding.apply(q, cos, sin)  # [B, n_heads, S, d_head]
        k = RotaryEmbedding.apply(k, cos, sin)  # [B, n_kv_heads, S, d_head]

        # ----------------------------------------------------------------
        # 5. Concatenate KV cache (incremental decoding)
        # ----------------------------------------------------------------
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = jnp.concatenate([k_cache, k], axis=2)
            v = jnp.concatenate([v_cache, v], axis=2)

        new_kv_cache = (k, v)   # store pre-expansion KV

        # ----------------------------------------------------------------
        # 6. Expand KV heads for GQA
        # ----------------------------------------------------------------
        k_exp = self._expand_kv(k)  # [B, n_heads, S_total, d_head]
        v_exp = self._expand_kv(v)  # [B, n_heads, S_total, d_head]

        # ----------------------------------------------------------------
        # 7. Attention scores with nanochat scale
        #    scale = qk_scale_factor / sqrt(d_head)  (not 1/sqrt(d_head))
        # ----------------------------------------------------------------
        # [B, n_heads, S, d_head] @ [B, n_heads, d_head, S_total]
        scores = jnp.matmul(
            q.astype(jnp.float32),
            jnp.transpose(k_exp.astype(jnp.float32), (0, 1, 3, 2))
        ) * self.attn_scale   # [B, n_heads, S, S_total]

        # ----------------------------------------------------------------
        # 8. Logit softcap (nanochat feature)
        #    cap * tanh(logits / cap) — bounds logits to [-cap, cap]
        #    Applied before masking so masked positions aren't used.
        # ----------------------------------------------------------------
        if self.logit_softcap is not None:
            cap = float(self.logit_softcap)
            scores = cap * jnp.tanh(scores / cap)

        # ----------------------------------------------------------------
        # 9. Apply attention mask (causal ± sliding window)
        #    mask: True = valid, False = masked out
        # ----------------------------------------------------------------
        # Apply sliding window on top of the pre-built causal mask if needed.
        if self.sliding_window_size is not None:
            S_total = k_exp.shape[2]
            window_mask = _build_sliding_window_mask(
                S_total, self.sliding_window_size, self.n_global_tokens
            )  # [1, 1, S_total, S_total]
            # Slice to [1, 1, S, S_total] to match the current query length
            window_mask = window_mask[:, :, S_total - S:, :]
            combined_mask = mask & window_mask
        else:
            combined_mask = mask

        # Replace masked positions with large negative value (not -inf to
        # avoid NaN when entire row is masked).
        scores = jnp.where(combined_mask, scores, jnp.float32(-1e9))

        # ----------------------------------------------------------------
        # 10. Softmax + dropout
        # ----------------------------------------------------------------
        weights = jax.nn.softmax(scores, axis=-1)  # [B, n_heads, S, S_total]
        weights = self.dropout(weights, deterministic=deterministic)

        # ----------------------------------------------------------------
        # 11. Weighted sum of values
        # ----------------------------------------------------------------
        context = jnp.matmul(weights, v_exp.astype(jnp.float32))  # [B, n_heads, S, d_head]

        # ----------------------------------------------------------------
        # 12. Reshape + output projection
        # ----------------------------------------------------------------
        context = jnp.transpose(context, (0, 2, 1, 3))  # [B, S, n_heads, d_head]
        context = context.reshape(B, S, self.n_heads * self.d_head).astype(x.dtype)

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
            f"variant={variant}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, d_head={self.d_head}, "
            f"qk_norm={self.use_qk_norm}, softcap={self.logit_softcap}, "
            f"window={self.sliding_window_size})"
        )
