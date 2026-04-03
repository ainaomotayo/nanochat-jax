"""Feed-forward network variants for the nanochat-jax transformer.

Faithful nanochat port default: ReLUSquaredMLP using relu²(x) = x * relu(x).

Activation comparison:
- **relu²**: x * relu(x) = relu(x)² for x≥0, 0 for x<0. Smooth, non-saturating,
  efficient (no gating branch needed). nanochat default.
- **SwiGLU**: silu(gate) * up. Requires 3 projections. Used by LLaMA/Mistral.
- **GeGLU**: gelu(gate) * up. Similar to SwiGLU.
- **StandardMLP**: GELU on single projection. Classic GPT-2 style.

All classes share identical ``__init__``/``__call__`` signatures for
drop-in replacement.

References:
    - relu²: So et al., "Primer: Searching for Efficient Transformers" (2021)
    - SwiGLU: Shazeer, "GLU Variants Improve Transformer" (2020)
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import structlog
from flax import nnx

from nanochat.config.model_config import ModelConfig

log = structlog.get_logger(__name__)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


# ---------------------------------------------------------------------------
# relu² MLP — nanochat default
# ---------------------------------------------------------------------------


class ReLUSquaredMLP(nnx.Module):
    """Two-layer MLP with relu²(x) = x * relu(x) activation.

    This is the nanochat-faithful FFN variant. Unlike gated FFNs (SwiGLU,
    GeGLU), it uses only two linear projections, making it computationally
    equivalent to a standard MLP with the same d_ff.

    Architecture::

        x ---> fc1 ---> relu²(·) ---> fc2 ---> dropout ---> out

    where relu²(x) = x * relu(x) = max(0, x)² (for x > 0), 0 (for x ≤ 0).

    Attributes:
        d_ff: Intermediate dimension (default: 4 * d_model).
        fc1: Expansion projection (d_model → d_ff).
        down_proj: Contraction projection (d_ff → d_model). Alias: fc2.
        dropout: Output dropout.
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize ReLUSquaredMLP.

        Args:
            cfg: Model config. Uses d_model, d_ff (auto: 4*d_model), dropout_rate.
            rngs: Flax NNX RNG container.
        """
        self.d_ff = cfg.d_ff if cfg.d_ff is not None else 4 * cfg.d_model

        # Expansion: d_model → d_ff
        self.gate_proj = nnx.Linear(
            cfg.d_model, self.d_ff, use_bias=cfg.use_bias, rngs=rngs
        )

        # No up_proj (unlike SwiGLU) — relu² needs only one projection
        self.up_proj = None  # type: ignore[assignment]

        # Contraction: d_ff → d_model
        self.down_proj = nnx.Linear(
            self.d_ff, cfg.d_model, use_bias=cfg.use_bias, rngs=rngs
        )

        self.dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

        log.debug(
            "relu2_mlp.init",
            d_model=cfg.d_model,
            d_ff=self.d_ff,
            dropout_rate=cfg.dropout_rate,
        )

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        """Apply relu² MLP.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            deterministic: If True, disables dropout.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Expand and apply relu²: x * relu(x)
        h = self.gate_proj(x)          # [B, S, d_ff]
        h = h * jax.nn.relu(h)         # relu²: x * max(0, x) — bf16-safe, no saturation

        # Contract
        out = self.down_proj(h)        # [B, S, d_model]
        return self.dropout(out, deterministic=deterministic)

    def __repr__(self) -> str:
        return f"ReLUSquaredMLP(d_ff={self.d_ff})"


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------


class SwiGLUFFN(nnx.Module):
    """Gated FFN with SiLU (Swish) activation.

    Architecture::

        gate = silu(gate_proj(x))
        up   = up_proj(x)
        out  = down_proj(gate * up)

    Used by LLaMA, Mistral, and most modern decoder-only LLMs.
    d_ff defaults to round_up(2/3 * 4 * d_model, 256) to equalize
    parameter count with a standard MLP using 4 * d_model.
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs) -> None:
        if cfg.d_ff is not None:
            self.d_ff = cfg.d_ff
        else:
            raw = int(2.0 / 3.0 * 4 * cfg.d_model)
            self.d_ff = _round_up_to_multiple(raw, 256)

        self.gate_proj = nnx.Linear(cfg.d_model, self.d_ff, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(cfg.d_model, self.d_ff, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(self.d_ff, cfg.d_model, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

        log.debug("swiglu_ffn.init", d_model=cfg.d_model, d_ff=self.d_ff)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        gate = jax.nn.silu(self.gate_proj(x))  # [B, S, d_ff]
        up = self.up_proj(x)                    # [B, S, d_ff]
        out = self.down_proj(gate * up)         # [B, S, d_model]
        return self.dropout(out, deterministic=deterministic)

    def __repr__(self) -> str:
        return f"SwiGLUFFN(d_ff={self.d_ff})"


# ---------------------------------------------------------------------------
# GeGLU FFN
# ---------------------------------------------------------------------------


class GeGLUFFN(nnx.Module):
    """Gated FFN with GELU activation.

    Identical to SwiGLUFFN but with GELU instead of SiLU gating.
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs) -> None:
        if cfg.d_ff is not None:
            self.d_ff = cfg.d_ff
        else:
            raw = int(2.0 / 3.0 * 4 * cfg.d_model)
            self.d_ff = _round_up_to_multiple(raw, 256)

        self.gate_proj = nnx.Linear(cfg.d_model, self.d_ff, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(cfg.d_model, self.d_ff, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(self.d_ff, cfg.d_model, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

        log.debug("geglu_ffn.init", d_model=cfg.d_model, d_ff=self.d_ff)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        gate = jax.nn.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        return self.dropout(out, deterministic=deterministic)

    def __repr__(self) -> str:
        return f"GeGLUFFN(d_ff={self.d_ff})"


# ---------------------------------------------------------------------------
# Standard GELU MLP
# ---------------------------------------------------------------------------


class StandardMLP(nnx.Module):
    """Classical two-layer MLP with GELU activation.

    Architecture::

        out = fc2(gelu(fc1(x)))
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs) -> None:
        self.d_ff = cfg.d_ff if cfg.d_ff is not None else 4 * cfg.d_model

        self.gate_proj = nnx.Linear(
            cfg.d_model, self.d_ff, use_bias=cfg.use_bias, rngs=rngs
        )
        self.up_proj = None  # type: ignore[assignment]
        self.down_proj = nnx.Linear(
            self.d_ff, cfg.d_model, use_bias=cfg.use_bias, rngs=rngs
        )
        self.dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

        log.debug("standard_mlp.init", d_model=cfg.d_model, d_ff=self.d_ff)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        hidden = jax.nn.gelu(self.gate_proj(x))  # [B, S, d_ff]
        out = self.down_proj(hidden)              # [B, S, d_model]
        return self.dropout(out, deterministic=deterministic)

    def __repr__(self) -> str:
        return f"StandardMLP(d_ff={self.d_ff})"
