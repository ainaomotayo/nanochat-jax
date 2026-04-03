"""Decoder-only transformer language model for nanochat-jax.

This module implements :class:`TransformerLM`, the top-level model that
composes token embeddings, rotary position encodings, a stack of
:class:`TransformerBlock` layers, and a language-modelling head into a
complete autoregressive language model.

Architecture overview:

.. code-block:: text

    input_ids -> TokenEmbedding -> [TransformerBlock x N] -> RMSNorm -> LM Head -> logits

Key features:

- **Weight tying**: when ``cfg.tie_embeddings`` is ``True``, the output
  projection reuses the token embedding matrix (no separate ``lm_head``).
- **GPT-NeoX-style initialization**: output projection weights in
  attention and FFN sub-layers are scaled by ``1 / sqrt(2 * n_layers)``
  to stabilize training at depth.
- **KV caching**: each layer returns an updated cache for efficient
  autoregressive generation.
- **Gradient checkpointing**: can be enabled per-layer by setting
  ``TransformerBlock.use_remat = True``.

References:
    - GPT-NeoX: Black et al., "GPT-NeoX-20B: An Open-Source Autoregressive
      Language Model" (2022)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import structlog
from flax import nnx

from nanochat.config.model_config import ModelConfig
from nanochat.model.embeddings import TokenEmbedding, RotaryEmbedding
from nanochat.model.norms import RMSNorm
from nanochat.model.block import TransformerBlock

log = structlog.get_logger(__name__)


class TransformerLM(nnx.Module):
    """Decoder-only transformer language model.

    Combines token embeddings, rotary position encodings, a stack of
    pre-norm transformer blocks, and a (possibly weight-tied) language
    modelling head.

    Attributes:
        cfg: Frozen model configuration.
        embed: Token embedding layer with optional weight-tied logit
            projection.
        rope: Rotary position embedding utility (not an nnx.Module).
        layers: List of :class:`TransformerBlock` decoder layers.
        final_norm: Final RMSNorm applied before the LM head.
        lm_head: Output linear projection (only present when
            ``cfg.tie_embeddings`` is ``False``).
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize the full transformer language model.

        Args:
            cfg: Model architecture configuration specifying dimensions,
                layer count, attention layout, FFN type, and more.
            rngs: Flax NNX RNG container used for all sub-module
                parameter initialization.

        Raises:
            ValueError: Propagated from sub-modules if the configuration
                contains invalid combinations.
        """
        self.cfg = cfg

        # -- Token embedding ------------------------------------------------
        self.embed = TokenEmbedding(
            cfg.vocab_size,
            cfg.d_model,
            cfg.init_std,
            rngs=rngs,
        )

        # -- Rotary position embedding (pure-JAX utility, no parameters) ----
        self.rope = RotaryEmbedding(
            cfg.d_head,
            cfg.max_seq_len,
            cfg.rope_base,
        )

        # -- Transformer block stack ----------------------------------------
        # nnx.List is required for Flax NNX pytree traversal of module lists
        self.layers = nnx.List([
            TransformerBlock(cfg, layer_idx=i, rngs=rngs)
            for i in range(cfg.n_layers)
        ])

        # -- Final normalization --------------------------------------------
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps, rngs=rngs)

        # -- LM head (only when embeddings are not tied) --------------------
        if not cfg.tie_embeddings:
            self.lm_head = nnx.Linear(
                cfg.d_model,
                cfg.vocab_size,
                use_bias=False,
                rngs=rngs,
            )

        # -- Apply GPT-NeoX-style weight initialization ---------------------
        self._init_weights()

        log.info(
            "transformer_lm.init",
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            d_ff=cfg.d_ff,
            tie_embeddings=cfg.tie_embeddings,
            ffn_type=cfg.ffn_type,
            max_seq_len=cfg.max_seq_len,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids: jax.Array,
        kv_caches: Optional[List[Tuple[jax.Array, jax.Array]]] = None,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> Tuple[jax.Array, List[Optional[Tuple[jax.Array, jax.Array]]]]:
        """Run the full forward pass of the language model.

        Args:
            input_ids: Integer token IDs of shape ``(batch, seq_len)``.
            kv_caches: Optional list of per-layer KV caches for
                incremental decoding.  Each element is a tuple
                ``(cached_keys, cached_values)`` or ``None``.
            attention_mask: Optional boolean/int mask of shape
                ``(batch, seq_len)`` where ``True``/``1`` marks valid
                positions and ``False``/``0`` marks padding.
            deterministic: When ``True``, disables dropout.

        Returns:
            A tuple ``(logits, new_kv_caches)`` where:

            - *logits* has shape ``(batch, seq_len, vocab_size)``
            - *new_kv_caches* is a list of per-layer cache tuples
        """
        B, S = input_ids.shape

        # 1. Token embedding
        x = self.embed(input_ids)  # [B, S, d_model]

        # 2. Get RoPE frequencies for this sequence length
        cos, sin = self.rope.get_freqs(S)  # [S, d_head // 2] each

        # 3. Build causal attention mask [1, 1, S, S_total]
        #    For standard (non-cached) forward: S_total == S
        #    For cached decoding: S_total would be larger, but we handle
        #    that at the attention layer level via kv_cache concatenation.
        causal = jnp.tril(
            jnp.ones((S, S), dtype=jnp.bool_)
        )  # [S, S]
        mask = causal[None, None, :, :]  # [1, 1, S, S]

        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask: [B, S] -> [B, 1, 1, S]
            pad_mask = attention_mask[:, None, None, :]  # [B, 1, 1, S]
            mask = mask & pad_mask  # [B, 1, S, S] via broadcast

        # 4. Forward through transformer block stack
        new_kv_caches: list[Optional[Tuple[jax.Array, jax.Array]]] = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(
                x, cos, sin, mask, cache, deterministic
            )  # x: [B, S, d_model]
            new_kv_caches.append(new_cache)

        # 5. Final layer normalization
        x = self.final_norm(x)  # [B, S, d_model]

        # 6. Project to vocabulary logits
        if self.cfg.tie_embeddings:
            logits = self.embed.as_logits(x)  # [B, S, vocab_size]
        else:
            logits = self.lm_head(x)  # [B, S, vocab_size]

        # Optional output logit scaling (e.g. Gemma-style)
        if self.cfg.output_logits_scale is not None:
            logits = logits * self.cfg.output_logits_scale

        return logits, new_kv_caches

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply GPT-NeoX-style output-projection scaling.

        Scales the weights of output projection layers (attention
        ``out_proj`` and FFN ``down_proj``) by ``1 / sqrt(2 * n_layers)``
        to prevent the residual stream variance from growing with depth.

        This is called once at the end of ``__init__`` after all
        sub-modules have been constructed with their default
        initialization.
        """
        n_layers = self.cfg.n_layers
        if n_layers == 0:
            return

        output_scale = 1.0 / math.sqrt(2.0 * n_layers)

        for layer in self.layers:
            # Scale attention output projection
            if hasattr(layer.attention, "out_proj"):
                _scale_param(layer.attention.out_proj, output_scale)

            # Scale FFN down projection (output projection)
            if hasattr(layer.ffn, "down_proj"):
                _scale_param(layer.ffn.down_proj, output_scale)

        log.debug(
            "transformer_lm.init_weights",
            n_layers=n_layers,
            output_scale=output_scale,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on all transformer blocks.

        This sets ``use_remat = True`` on every :class:`TransformerBlock`
        in the layer stack, causing their forward passes to be wrapped in
        ``jax.checkpoint`` during backpropagation.
        """
        for layer in self.layers:
            layer.use_remat = True
        log.info(
            "transformer_lm.gradient_checkpointing",
            enabled=True,
            n_layers=len(self.layers),
        )

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing on all transformer blocks."""
        for layer in self.layers:
            layer.use_remat = False
        log.info(
            "transformer_lm.gradient_checkpointing",
            enabled=False,
            n_layers=len(self.layers),
        )

    def __repr__(self) -> str:
        return (
            f"TransformerLM(\n"
            f"  vocab_size={self.cfg.vocab_size},\n"
            f"  d_model={self.cfg.d_model},\n"
            f"  n_layers={self.cfg.n_layers},\n"
            f"  n_heads={self.cfg.n_heads},\n"
            f"  n_kv_heads={self.cfg.n_kv_heads},\n"
            f"  d_ff={self.cfg.d_ff},\n"
            f"  ffn_type={self.cfg.ffn_type!r},\n"
            f"  tie_embeddings={self.cfg.tie_embeddings},\n"
            f"  max_seq_len={self.cfg.max_seq_len},\n"
            f")"
        )


# ======================================================================
# Private helpers
# ======================================================================


def _scale_param(linear: nnx.Linear, scale: float) -> None:
    """Scale the kernel of an ``nnx.Linear`` layer in-place.

    Args:
        linear: An ``nnx.Linear`` module whose ``kernel`` parameter
            will be multiplied by *scale*.
        scale: Multiplicative scaling factor.
    """
    if hasattr(linear, "kernel"):
        linear.kernel = nnx.Param(linear.kernel.value * scale)
