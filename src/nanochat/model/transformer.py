"""Decoder-only transformer language model — nanochat-faithful port.

Architecture::

    input_ids -> TokenEmbedding -> [TransformerBlock x N] -> RMSNorm -> LM Head -> logits

nanochat-faithful implementation:
- Parameterless RMSNorm (no gamma)
- relu² MLP activation
- QK L2 normalization + 1.2x scale factor
- Logit softcap (30.0) in attention
- Per-layer alpha scalars on attention and FFN outputs
- Smear/Backout causal token mixing
- Shared value embedding table injected at each block
- Depth-aware weight initialization (from_depth style)
- rope_base = 100000
- Untied input/output embeddings (default)

Weight initialization (nanochat from_depth style):
    At layer l (0-indexed), the residual projections (out_proj, down_proj)
    are scaled by 1 / sqrt(2 * (l + 1)). This prevents the residual
    stream variance from growing with depth, regardless of total n_layers.

    This is more principled than GPT-NeoX's global 1/sqrt(2*n_layers)
    because it accounts for the actual depth at each layer.
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
from nanochat.model.value_embeddings import ValueEmbedding

log = structlog.get_logger(__name__)


class TransformerLM(nnx.Module):
    """Nanochat-faithful decoder-only transformer language model.

    Attributes:
        cfg: Frozen model configuration.
        embed: Token embedding layer.
        rope: Rotary position embedding utility.
        value_embed: Optional shared value embedding table.
        layers: Stack of TransformerBlock decoder layers.
        final_norm: Parameterless RMSNorm before LM head.
        lm_head: Output linear projection (when tie_embeddings=False).
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize TransformerLM.

        Args:
            cfg: Model architecture configuration.
            rngs: Flax NNX RNG container.
        """
        self.cfg = cfg

        # -- Token embedding ------------------------------------------------
        self.embed = TokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.init_std, rngs=rngs,
        )

        # -- Rotary position embedding (no parameters) ----------------------
        self.rope = RotaryEmbedding(cfg.d_head, cfg.max_seq_len, cfg.rope_base)

        # -- Shared value embedding table -----------------------------------
        # Single table shared across all layers to keep parameter count bounded.
        # Blocks receive a reference to this table (not a copy).
        if cfg.use_value_embeddings:
            self.value_embed = ValueEmbedding(
                cfg.vocab_size, cfg.d_model, rngs=rngs,
            )
        else:
            self.value_embed = None  # type: ignore[assignment]

        # -- Transformer block stack ----------------------------------------
        self.layers = nnx.List([
            TransformerBlock(
                cfg,
                layer_idx=i,
                value_embed=self.value_embed,
                rngs=rngs,
            )
            for i in range(cfg.n_layers)
        ])

        # -- Final RMSNorm --------------------------------------------------
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps, rngs=rngs)

        # -- LM head --------------------------------------------------------
        if not cfg.tie_embeddings:
            self.lm_head = nnx.Linear(
                cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs,
            )
        else:
            self.lm_head = None  # type: ignore[assignment]

        # -- Depth-aware weight initialization ------------------------------
        self._init_weights_from_depth()

        log.info(
            "transformer_lm.init",
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            d_ff=cfg.d_ff,
            ffn_type=cfg.ffn_type,
            tie_embeddings=cfg.tie_embeddings,
            use_qk_norm=cfg.use_qk_norm,
            logit_softcap=cfg.logit_softcap,
            use_value_embeddings=cfg.use_value_embeddings,
            use_per_layer_scalars=cfg.use_per_layer_scalars,
            use_smear=cfg.use_smear,
            sliding_window_size=cfg.sliding_window_size,
            rope_base=cfg.rope_base,
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
        """Run the full forward pass.

        Args:
            input_ids: Token IDs ``(batch, seq_len)``.
            kv_caches: Optional per-layer KV caches for incremental decoding.
                Each element is ``(k, v)`` or ``None``.
            attention_mask: Optional boolean/int mask ``(batch, seq_len)``
                where True/1 marks valid positions.
            deterministic: When True, disables dropout.

        Returns:
            Tuple ``(logits, new_kv_caches)`` where logits has shape
            ``(batch, seq_len, vocab_size)``.
        """
        B, S = input_ids.shape

        # 1. Token embedding
        x = self.embed(input_ids)  # [B, S, d_model]

        # 2. RoPE frequencies for this sequence length
        cos, sin = self.rope.get_freqs(S)  # [S, d_head//2]

        # 3. Build causal mask
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        mask = causal[None, None, :, :]  # [1, 1, S, S]

        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :]  # [B, 1, 1, S]
            mask = mask & pad_mask  # [B, 1, S, S]

        # 4. Transformer block stack
        new_kv_caches: list[Optional[Tuple[jax.Array, jax.Array]]] = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(
                x, cos, sin, mask, cache, deterministic,
                token_ids=input_ids,   # for value embeddings
            )
            new_kv_caches.append(new_cache)

        # 5. Final RMSNorm
        x = self.final_norm(x)  # [B, S, d_model]

        # 6. LM head
        if not self.cfg.tie_embeddings:
            logits = self.lm_head(x)  # [B, S, vocab_size]
        else:
            logits = self.embed.as_logits(x)  # [B, S, vocab_size]

        # 7. Optional output logit scaling
        if self.cfg.output_logits_scale is not None:
            logits = logits * self.cfg.output_logits_scale

        return logits, new_kv_caches

    # ------------------------------------------------------------------
    # Weight initialization — nanochat from_depth style
    # ------------------------------------------------------------------

    def _init_weights_from_depth(self) -> None:
        """Apply nanochat from_depth weight initialization.

        Scales residual output projections (attention out_proj and FFN
        down_proj) at each layer by 1 / sqrt(2 * (layer_idx + 1)).

        Rationale:
            After l layers, the residual stream has accumulated 2*(l+1)
            additions (one from attention, one from FFN per layer). To
            keep the variance of the residual stream O(1) at any depth,
            each addition should contribute O(1/sqrt(2*(l+1))).

            This is strictly more principled than GPT-NeoX's global
            1/sqrt(2*n_layers) because it adapts to actual depth rather
            than total depth.

        Effect:
            Early layers have larger-magnitude residual projections
            (since the stream is still small). Deep layers have smaller
            projections to avoid variance blow-up.
        """
        for layer_idx, layer in enumerate(self.layers):
            # Depth-aware scale: 1/sqrt(2*(l+1))
            depth_scale = 1.0 / math.sqrt(2.0 * (layer_idx + 1))

            if hasattr(layer.attention, "out_proj"):
                _scale_linear(layer.attention.out_proj, depth_scale)

            if hasattr(layer.ffn, "down_proj"):
                _scale_linear(layer.ffn.down_proj, depth_scale)

        log.debug(
            "transformer_lm.init_weights_from_depth",
            n_layers=self.cfg.n_layers,
            style="from_depth",
        )

    # ------------------------------------------------------------------
    # Gradient checkpointing helpers
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on all transformer blocks."""
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
            f"  use_qk_norm={self.cfg.use_qk_norm},\n"
            f"  logit_softcap={self.cfg.logit_softcap},\n"
            f"  rope_base={self.cfg.rope_base},\n"
            f")"
        )


# ======================================================================
# Private helpers
# ======================================================================


def _scale_linear(linear: nnx.Linear, scale: float) -> None:
    """Scale the kernel of an nnx.Linear layer in-place.

    Args:
        linear: Linear module whose ``kernel`` parameter is multiplied.
        scale: Multiplicative factor.
    """
    if hasattr(linear, "kernel"):
        linear.kernel = nnx.Param(linear.kernel.get_value() * scale)
