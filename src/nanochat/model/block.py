"""Pre-norm transformer block with nanochat-faithful extensions.

Faithful nanochat port extends the standard pre-norm block with:

1. **Per-layer scalars**: Learnable scalar weights (alpha_attn, alpha_ffn)
   applied to the attention and FFN outputs before the residual addition.
   Initialized to 1.0. These allow the model to learn the optimal
   contribution of each sub-layer independently.

2. **Smear / Backout token mixing**: Optional causal token-mixing applied
   before attention (smear blends each token with its predecessor) and
   corrective backout applied to the attention output.

3. **Value embeddings**: Optional per-token value lookup added to the
   attention output. Requires token_ids to be passed through.

The full forward pass with all nanochat features enabled::

    # Smear: blend x with predecessor
    x_smear, x_prev = smear(x)

    # Pre-norm attention
    attn_out, kv_cache = attention(norm(x_smear), ...)

    # Backout: remove smear residual from attn output
    attn_out = backout(attn_out, x_prev)

    # Value embedding injection
    attn_out = attn_out + value_embed(token_ids)

    # Residual with learnable scalar
    x = x + alpha_attn * attn_out

    # Pre-norm FFN
    ffn_out = ffn(norm(x))
    x = x + alpha_ffn * ffn_out

References:
    - Pre-norm residual: Xiong et al. (2020)
    - Per-layer scalars: nanochat, MAGNETO (Wang et al., 2022)
    - Smear/Backout: nanochat architecture
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
from nanochat.model.feedforward import (
    ReLUSquaredMLP, SwiGLUFFN, GeGLUFFN, StandardMLP,
)
from nanochat.model.value_embeddings import ValueEmbedding
from nanochat.model.token_mixing import Smear, Backout

log = structlog.get_logger(__name__)

_FFN_REGISTRY: dict[str, type] = {
    "relu2": ReLUSquaredMLP,
    "swiglu": SwiGLUFFN,
    "geglu": GeGLUFFN,
    "gelu": StandardMLP,
}


class TransformerBlock(nnx.Module):
    """Pre-norm transformer block with nanochat extensions.

    Attributes:
        layer_idx: Zero-based layer index.
        use_remat: Enable gradient checkpointing via jax.checkpoint.
        attn_norm: Pre-attention RMSNorm (parameterless).
        ffn_norm: Pre-FFN RMSNorm (parameterless).
        attention: GroupedQueryAttention with QK norm, softcap, window.
        ffn: Feed-forward network (relu² by default).
        alpha_attn: Learnable scalar on attention output.
        alpha_ffn: Learnable scalar on FFN output.
        smear: Optional causal token-mixing (before attention).
        backout: Optional corrective mixing (after attention).
        value_embed: Optional per-token value embedding.
    """

    use_remat: bool = False

    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int,
        *,
        value_embed: Optional[ValueEmbedding] = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize TransformerBlock.

        Args:
            cfg: Model configuration.
            layer_idx: Zero-based index of this block.
            value_embed: Shared ValueEmbedding module (created once in
                TransformerLM and passed to all blocks). None if
                cfg.use_value_embeddings is False.
            rngs: Flax NNX RNG container.

        Raises:
            ValueError: If cfg.ffn_type is not in the FFN registry.
        """
        if cfg.ffn_type not in _FFN_REGISTRY:
            available = ", ".join(sorted(_FFN_REGISTRY.keys()))
            raise ValueError(
                f"Unknown ffn_type '{cfg.ffn_type}'. Available: {available}"
            )

        self.layer_idx = layer_idx
        self.cfg_use_smear = cfg.use_smear
        self.cfg_use_value_embed = cfg.use_value_embeddings
        self.cfg_use_scalars = cfg.use_per_layer_scalars

        # -- Normalization (parameterless RMSNorm) -------------------------
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps, rngs=rngs)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps, rngs=rngs)

        # -- Attention -------------------------------------------------------
        self.attention = GroupedQueryAttention(cfg, rngs=rngs)

        # -- FFN -------------------------------------------------------------
        ffn_cls = _FFN_REGISTRY[cfg.ffn_type]
        self.ffn = ffn_cls(cfg, rngs=rngs)

        # -- Per-layer scalars -----------------------------------------------
        # Initialize to 1.0 (identity behavior at init).
        # These become effective after a few gradient steps.
        if cfg.use_per_layer_scalars:
            self.alpha_attn = nnx.Param(jnp.ones(()))  # scalar
            self.alpha_ffn = nnx.Param(jnp.ones(()))   # scalar
        else:
            self.alpha_attn = None  # type: ignore[assignment]
            self.alpha_ffn = None   # type: ignore[assignment]

        # -- Smear / Backout -------------------------------------------------
        if cfg.use_smear:
            self.smear = Smear(cfg.d_model, rngs=rngs)
            self.backout = Backout(cfg.d_model, rngs=rngs)
        else:
            self.smear = None   # type: ignore[assignment]
            self.backout = None  # type: ignore[assignment]

        # -- Value embedding (shared, passed in from TransformerLM) ----------
        # Not owned by this block — just a reference for forward-pass lookup.
        self._value_embed = value_embed

        log.debug(
            "transformer_block.init",
            layer_idx=layer_idx,
            ffn_type=cfg.ffn_type,
            use_smear=cfg.use_smear,
            use_scalars=cfg.use_per_layer_scalars,
            use_value_embed=cfg.use_value_embeddings and value_embed is not None,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array],
        kv_cache: Optional[Tuple[jax.Array, jax.Array]] = None,
        deterministic: bool = True,
        token_ids: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Optional[Tuple[jax.Array, jax.Array]]]:
        """Forward pass through the transformer block.

        Args:
            x: Hidden states ``(batch, seq_len, d_model)``.
            cos: RoPE cosines ``(seq_len, d_head // 2)``.
            sin: RoPE sines ``(seq_len, d_head // 2)``.
            mask: Attention mask ``(1_or_B, 1, seq_len, seq_total)``.
            kv_cache: Optional KV cache for incremental decoding.
            deterministic: When True, disables dropout.
            token_ids: Integer token IDs ``(batch, seq_len)`` required for
                value embeddings. Can be None if use_value_embeddings=False.

        Returns:
            ``(output, new_kv_cache)`` where output has the same shape as x.
        """
        if self.use_remat:
            return self._forward_with_remat(
                x, cos, sin, mask, kv_cache, deterministic, token_ids
            )
        return self._forward(x, cos, sin, mask, kv_cache, deterministic, token_ids)

    def _forward(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array],
        kv_cache: Optional[Tuple[jax.Array, jax.Array]],
        deterministic: bool,
        token_ids: Optional[jax.Array],
    ) -> Tuple[jax.Array, Optional[Tuple[jax.Array, jax.Array]]]:
        """Core forward logic (no gradient checkpointing)."""

        # ----------------------------------------------------------------
        # Attention sub-layer
        # ----------------------------------------------------------------
        residual = x  # [B, S, d_model]

        # Smear: blend x with causal predecessor before normalization
        x_prev = None
        x_for_attn = x
        if self.smear is not None:
            x_for_attn, x_prev = self.smear(x)

        # Pre-norm
        x_normed = self.attn_norm(x_for_attn)  # [B, S, d_model]

        # Attention
        attn_out, new_kv_cache = self.attention(
            x_normed, cos, sin, mask,
            kv_cache=kv_cache, deterministic=deterministic,
        )  # [B, S, d_model]

        # Backout: remove smear residual from attention output
        if self.backout is not None and x_prev is not None:
            attn_out = self.backout(attn_out, x_prev)

        # Value embedding injection (token-specific static residual)
        if self._value_embed is not None and token_ids is not None:
            v_embed = self._value_embed(token_ids)  # [B, S, d_model]
            attn_out = attn_out + v_embed

        # Residual with per-layer scalar
        if self.alpha_attn is not None:
            x = residual + self.alpha_attn.get_value() * attn_out
        else:
            x = residual + attn_out

        # ----------------------------------------------------------------
        # FFN sub-layer
        # ----------------------------------------------------------------
        residual = x

        # Pre-norm
        x_normed = self.ffn_norm(x)  # [B, S, d_model]

        # FFN
        ffn_out = self.ffn(x_normed, deterministic=deterministic)  # [B, S, d_model]

        # Residual with per-layer scalar
        if self.alpha_ffn is not None:
            x = residual + self.alpha_ffn.get_value() * ffn_out
        else:
            x = residual + ffn_out

        return x, new_kv_cache

    def _forward_with_remat(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array],
        kv_cache: Optional[Tuple[jax.Array, jax.Array]],
        deterministic: bool,
        token_ids: Optional[jax.Array],
    ) -> Tuple[jax.Array, Optional[Tuple[jax.Array, jax.Array]]]:
        """Forward pass with jax.checkpoint for gradient checkpointing.

        During backpropagation, intermediate activations are recomputed
        rather than stored, reducing peak memory by ~33%.

        Note: KV cache is computed outside the checkpoint boundary so it
        is accessible during inference without re-triggering remat.
        """

        @jax.checkpoint
        def _checkpointed(
            x_in: jax.Array,
            cos_in: jax.Array,
            sin_in: jax.Array,
            token_ids_in: Optional[jax.Array],
        ) -> jax.Array:
            x_out, _ = self._forward(
                x_in, cos_in, sin_in, mask, None, deterministic, token_ids_in
            )
            return x_out

        output = _checkpointed(x, cos, sin, token_ids)

        # Compute KV cache outside the checkpoint boundary (inference only).
        if kv_cache is not None:
            x_prev = None
            x_for_attn = x
            if self.smear is not None:
                x_for_attn, x_prev = self.smear(x)
            x_normed = self.attn_norm(x_for_attn)
            _, new_kv_cache = self.attention(
                x_normed, cos, sin, mask,
                kv_cache=kv_cache, deterministic=deterministic,
            )
        else:
            new_kv_cache = None

        return output, new_kv_cache

    def __repr__(self) -> str:
        return (
            f"TransformerBlock(layer_idx={self.layer_idx}, "
            f"ffn_type={type(self.ffn).__name__}, "
            f"smear={self.smear is not None}, "
            f"scalars={self.alpha_attn is not None})"
        )
