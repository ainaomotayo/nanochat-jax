"""Pre-norm transformer block for the nanochat-jax decoder.

This module implements :class:`TransformerBlock`, a single decoder layer
that follows the pre-norm residual pattern used in LLaMA, Gemma, and
other modern open-weight language models:

.. code-block:: text

    x = x + Attention(Norm(x))
    x = x + FFN(Norm(x))

The block supports:

- **Grouped-query attention** via :class:`GroupedQueryAttention`.
- **Configurable feed-forward networks**: SwiGLU, GeGLU, or standard MLP,
  selected at construction time through :attr:`ModelConfig.ffn_type`.
- **Gradient checkpointing** (``jax.checkpoint`` / ``jax.remat``):
  enabled by setting the class attribute ``use_remat = True`` on the
  block *before* the forward pass, which trades compute for memory
  during backpropagation.

References:
    - Pre-norm residual: Xiong et al., "On Layer Normalization in the
      Transformer Architecture" (2020)
    - Gradient checkpointing: Chen et al., "Training Deep Nets with
      Sublinear Memory Cost" (2016)
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import structlog
from flax import nnx

from nanochat.config.model_config import ModelConfig
from nanochat.model.norms import RMSNorm
from nanochat.model.attention import GroupedQueryAttention
from nanochat.model.feedforward import SwiGLUFFN, GeGLUFFN, StandardMLP

log = structlog.get_logger(__name__)


# Mapping from config string to feed-forward class
_FFN_REGISTRY: dict[str, type] = {
    "swiglu": SwiGLUFFN,
    "geglu": GeGLUFFN,
    "mlp": StandardMLP,
}


class TransformerBlock(nnx.Module):
    """Pre-norm transformer block: ``x = x + Attn(Norm(x))``, ``x = x + FFN(Norm(x))``.

    Each block contains two sub-layers with independent RMSNorm
    normalization and residual connections.  The attention sub-layer
    uses grouped-query attention with rotary position embeddings, while
    the feed-forward sub-layer is selected from the ``_FFN_REGISTRY``
    based on the :attr:`ModelConfig.ffn_type` string.

    Attributes:
        layer_idx: Zero-based index of this block within the stack.
        use_remat: Class-level flag; when ``True``, the forward
            computation is wrapped in ``jax.checkpoint`` to reduce
            activation memory at the cost of recomputation.
        attn_norm: Pre-attention RMSNorm.
        ffn_norm: Pre-FFN RMSNorm.
        attention: Grouped-query attention sub-layer.
        ffn: Feed-forward sub-layer (SwiGLU, GeGLU, or MLP).
    """

    # Set to True on the *class* (or per-instance) to enable gradient
    # checkpointing.  This is toggled externally (e.g. by the training
    # harness) rather than being a constructor argument so that the
    # parameter pytree is identical with and without remat.
    use_remat: bool = False

    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize a single transformer block.

        Args:
            cfg: Model architecture configuration.
            layer_idx: Zero-based layer index (used for logging and
                potential per-layer hyperparameter scheduling).
            rngs: Flax NNX RNG container.

        Raises:
            ValueError: If ``cfg.ffn_type`` is not in the FFN registry.
        """
        if cfg.ffn_type not in _FFN_REGISTRY:
            available = ", ".join(sorted(_FFN_REGISTRY.keys()))
            raise ValueError(
                f"Unknown ffn_type '{cfg.ffn_type}'. "
                f"Available types: {available}"
            )

        self.layer_idx = layer_idx

        # -- Pre-norm layers ------------------------------------------------
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps, rngs=rngs)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps, rngs=rngs)

        # -- Attention sub-layer -------------------------------------------
        self.attention = GroupedQueryAttention(cfg, rngs=rngs)

        # -- Feed-forward sub-layer ----------------------------------------
        ffn_cls = _FFN_REGISTRY[cfg.ffn_type]
        self.ffn = ffn_cls(cfg, rngs=rngs)

        log.debug(
            "transformer_block.init",
            layer_idx=layer_idx,
            ffn_type=cfg.ffn_type,
            use_remat=self.use_remat,
        )

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array],
        kv_cache: Optional[Tuple[jax.Array, jax.Array]] = None,
        deterministic: bool = True,
    ) -> Tuple[jax.Array, Optional[Tuple[jax.Array, jax.Array]]]:
        """Forward pass through the transformer block.

        Args:
            x: Input hidden states of shape ``(batch, seq_len, d_model)``.
            cos: RoPE cosine frequencies of shape
                ``(seq_len, d_head // 2)``.
            sin: RoPE sine frequencies of shape
                ``(seq_len, d_head // 2)``.
            mask: Attention mask of shape ``(1, 1, seq_len, seq_total)``
                or ``(batch, 1, seq_len, seq_total)``.  ``True``/``1``
                positions are attended to; ``False``/``0`` are masked out.
            kv_cache: Optional tuple ``(cached_k, cached_v)`` from a
                previous forward pass for incremental decoding.
            deterministic: When ``True``, dropout is disabled.

        Returns:
            A tuple ``(output, new_kv_cache)`` where *output* has shape
            ``(batch, seq_len, d_model)`` and *new_kv_cache* is the
            updated key/value cache (or ``None`` if caching is disabled
            in the attention layer).
        """
        if self.use_remat:
            return self._forward_with_remat(
                x, cos, sin, mask, kv_cache, deterministic
            )
        return self._forward(x, cos, sin, mask, kv_cache, deterministic)

    def _forward(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array],
        kv_cache: Optional[Tuple[jax.Array, jax.Array]],
        deterministic: bool,
    ) -> Tuple[jax.Array, Optional[Tuple[jax.Array, jax.Array]]]:
        """Core forward logic without gradient checkpointing.

        This is factored out so that ``_forward_with_remat`` can wrap it
        with ``jax.checkpoint``.
        """
        # --- Pre-norm attention with residual ---
        residual = x  # [B, S, d_model]
        x_normed = self.attn_norm(x)  # [B, S, d_model]
        attn_out, new_kv_cache = self.attention(
            x_normed,
            cos,
            sin,
            mask,
            kv_cache=kv_cache,
            deterministic=deterministic,
        )  # attn_out: [B, S, d_model]
        x = residual + attn_out  # [B, S, d_model]

        # --- Pre-norm FFN with residual ---
        residual = x  # [B, S, d_model]
        x_normed = self.ffn_norm(x)  # [B, S, d_model]
        ffn_out = self.ffn(x_normed, deterministic=deterministic)  # [B, S, d_model]
        x = residual + ffn_out  # [B, S, d_model]

        return x, new_kv_cache

    def _forward_with_remat(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array],
        kv_cache: Optional[Tuple[jax.Array, jax.Array]],
        deterministic: bool,
    ) -> Tuple[jax.Array, Optional[Tuple[jax.Array, jax.Array]]]:
        """Forward pass wrapped in ``jax.checkpoint`` for gradient checkpointing.

        During the backward pass, intermediate activations will be
        recomputed rather than stored, reducing peak memory at the cost
        of ~33% additional compute.
        """

        @jax.checkpoint
        def _checkpointed_forward(
            x_in: jax.Array,
            cos_in: jax.Array,
            sin_in: jax.Array,
        ) -> jax.Array:
            """Checkpointed portion: attention + FFN (no cache for remat)."""
            # Pre-norm attention
            x_normed = self.attn_norm(x_in)  # [B, S, d_model]
            attn_out, _ = self.attention(
                x_normed,
                cos_in,
                sin_in,
                mask,
                kv_cache=None,
                deterministic=deterministic,
            )  # [B, S, d_model]
            h = x_in + attn_out  # [B, S, d_model]

            # Pre-norm FFN
            h_normed = self.ffn_norm(h)  # [B, S, d_model]
            ffn_out = self.ffn(
                h_normed, deterministic=deterministic
            )  # [B, S, d_model]
            h = h + ffn_out  # [B, S, d_model]
            return h

        # When using remat during training we ignore the KV cache
        # (caching is only used during inference, where remat is off).
        output = _checkpointed_forward(x, cos, sin)

        # Still compute the KV cache outside the checkpoint boundary
        # so it can be used for inference if the caller requests it.
        # During training with remat, kv_cache is typically None.
        if kv_cache is not None:
            x_normed = self.attn_norm(x)
            _, new_kv_cache = self.attention(
                x_normed,
                cos,
                sin,
                mask,
                kv_cache=kv_cache,
                deterministic=deterministic,
            )
        else:
            new_kv_cache = None

        return output, new_kv_cache

    def __repr__(self) -> str:
        return (
            f"TransformerBlock(layer_idx={self.layer_idx}, "
            f"use_remat={self.use_remat})"
        )
