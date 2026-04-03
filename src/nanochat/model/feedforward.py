"""Feed-forward network variants for the nanochat-jax transformer.

This module provides three drop-in-replaceable FFN implementations:

- :class:`SwiGLUFFN`: Gated Linear Unit with SiLU (Swish) activation.
  Used by LLaMA, Mistral, and most modern LLMs. The default for nanochat.
- :class:`GeGLUFFN`: Gated Linear Unit with GELU activation.
  Used by PaLM and some T5 variants.
- :class:`StandardMLP`: Classical two-layer MLP with GELU activation.
  Used by GPT-2, BERT, and earlier transformers.

All three classes share identical ``__init__`` and ``__call__`` signatures
so they can be swapped without code changes.

**Gated FFN Architecture (SwiGLU / GeGLU)**::

    x ----+-----> gate_proj ----> activation ---+
          |                                     |---> element-wise multiply ---> down_proj ---> dropout ---> out
          +-----> up_proj ----------------------+

**Standard MLP Architecture**::

    x ----> fc1 ----> GELU ----> dropout ----> fc2 ----> dropout ----> out

**Parameter Count Comparison** (for d_model=D, d_ff=F):

+-------------+----------------+---------------------------------------+
| Variant     | Parameters     | Notes                                 |
+=============+================+=======================================+
| SwiGLU/GeGLU| 3 * D * F      | gate_proj + up_proj + down_proj       |
+-------------+----------------+---------------------------------------+
| Standard MLP| 2 * D * F      | fc1 + fc2 (plus biases if enabled)    |
+-------------+----------------+---------------------------------------+

To compensate, gated variants typically use d_ff = round(2/3 * 4D) instead
of d_ff = 4D, making total parameters roughly equivalent while improving
quality.

References:
    - SwiGLU: Shazeer, "GLU Variants Improve Transformer" (2020)
    - GELU: Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016)
    - GeGLU: Shazeer, "GLU Variants Improve Transformer" (2020)
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
    """Round *value* up to the nearest multiple of *multiple*.

    Parameters
    ----------
    value:
        The integer value to round.
    multiple:
        The alignment boundary (must be positive).

    Returns
    -------
    int
        The smallest integer >= *value* that is divisible by *multiple*.
    """
    return math.ceil(value / multiple) * multiple


def _compute_gated_d_ff(d_model: int, d_ff: Optional[int]) -> int:
    """Compute the intermediate dimension for gated FFN variants.

    If ``d_ff`` is already specified, returns it directly. Otherwise
    computes the standard SwiGLU/GeGLU intermediate size:
    ``round_up(2/3 * 4 * d_model, 256)``.

    Args:
        d_model: Model hidden dimension.
        d_ff: Explicit intermediate dimension, or ``None`` for auto-compute.

    Returns:
        The intermediate feed-forward dimension.
    """
    if d_ff is not None:
        return d_ff
    raw = int(2.0 / 3.0 * 4 * d_model)
    return _round_up_to_multiple(raw, 256)


def _compute_standard_d_ff(d_model: int, d_ff: Optional[int]) -> int:
    """Compute the intermediate dimension for standard MLP.

    If ``d_ff`` is already specified, returns it directly. Otherwise
    uses the classical ``4 * d_model``.

    Args:
        d_model: Model hidden dimension.
        d_ff: Explicit intermediate dimension, or ``None`` for auto-compute.

    Returns:
        The intermediate feed-forward dimension.
    """
    if d_ff is not None:
        return d_ff
    return 4 * d_model


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------


class SwiGLUFFN(nnx.Module):
    """Gated feed-forward network with SiLU (Swish) activation.

    Implements the SwiGLU variant from Shazeer (2020), which is the
    default FFN in LLaMA, Mistral, and most modern decoder-only LLMs.

    The forward pass computes::

        gate = silu(gate_proj(x))       # [B, S, d_ff]
        up   = up_proj(x)               # [B, S, d_ff]
        out  = down_proj(gate * up)     # [B, S, d_model]
        out  = dropout(out)

    The element-wise gating (``gate * up``) allows the network to learn
    which features to amplify or suppress, providing better gradient flow
    than standard ReLU/GELU MLPs.

    Attributes:
        d_ff: Intermediate (hidden) dimension of the feed-forward network.
        gate_proj: Linear projection for the gating branch
            (``d_model -> d_ff``).
        up_proj: Linear projection for the value branch
            (``d_model -> d_ff``).
        down_proj: Linear projection back to model dimension
            (``d_ff -> d_model``).
        dropout: Dropout layer applied to the output.
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize SwiGLUFFN.

        Args:
            cfg: Model configuration. Uses ``d_model``, ``d_ff``, and
                ``dropout_rate``. If ``cfg.d_ff`` is ``None``, it is
                auto-computed as ``round_up(2/3 * 4 * d_model, 256)``.
            rngs: Flax NNX RNG container for parameter initialization
                and dropout randomness.
        """
        self.d_ff = _compute_gated_d_ff(cfg.d_model, cfg.d_ff)

        # Gate projection: applies SiLU activation to control information flow
        # d_model -> d_ff, no bias (standard for gated FFNs)
        self.gate_proj = nnx.Linear(
            cfg.d_model, self.d_ff, use_bias=False, rngs=rngs
        )

        # Up projection: produces the value signal to be gated
        # d_model -> d_ff, no bias
        self.up_proj = nnx.Linear(
            cfg.d_model, self.d_ff, use_bias=False, rngs=rngs
        )

        # Down projection: maps back to model dimension
        # d_ff -> d_model, no bias
        self.down_proj = nnx.Linear(
            self.d_ff, cfg.d_model, use_bias=False, rngs=rngs
        )

        # Output dropout
        self.dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

        log.debug(
            "swiglu_ffn.init",
            d_model=cfg.d_model,
            d_ff=self.d_ff,
            dropout_rate=cfg.dropout_rate,
        )

    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
    ) -> jax.Array:
        """Apply SwiGLU feed-forward transformation.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            deterministic: If ``True``, disables dropout (inference mode).
                If ``False``, applies dropout (training mode).

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.

        Shape flow::

            x: [B, S, d_model]
                |
            gate_proj -> silu: [B, S, d_ff]
            up_proj:           [B, S, d_ff]
                |
            gate * up:         [B, S, d_ff]
                |
            down_proj:         [B, S, d_model]
                |
            dropout:           [B, S, d_model]
        """
        # Gating branch with SiLU (Swish) activation
        gate = jax.nn.silu(self.gate_proj(x))  # [B, S, d_ff]

        # Value branch (no activation)
        up = self.up_proj(x)  # [B, S, d_ff]

        # Element-wise gating and down-projection
        out = self.down_proj(gate * up)  # [B, S, d_model]

        # Apply dropout
        return self.dropout(out, deterministic=deterministic)  # [B, S, d_model]

    def __repr__(self) -> str:
        return f"SwiGLUFFN(d_ff={self.d_ff})"


# ---------------------------------------------------------------------------
# GeGLU Feed-Forward Network
# ---------------------------------------------------------------------------


class GeGLUFFN(nnx.Module):
    """Gated feed-forward network with GELU activation.

    Implements the GeGLU variant from Shazeer (2020). Identical to
    :class:`SwiGLUFFN` except the gating branch uses GELU instead of SiLU.

    The forward pass computes::

        gate = gelu(gate_proj(x))       # [B, S, d_ff]
        up   = up_proj(x)               # [B, S, d_ff]
        out  = down_proj(gate * up)     # [B, S, d_model]
        out  = dropout(out)

    Attributes:
        d_ff: Intermediate (hidden) dimension of the feed-forward network.
        gate_proj: Linear projection for the gating branch
            (``d_model -> d_ff``).
        up_proj: Linear projection for the value branch
            (``d_model -> d_ff``).
        down_proj: Linear projection back to model dimension
            (``d_ff -> d_model``).
        dropout: Dropout layer applied to the output.
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize GeGLUFFN.

        Args:
            cfg: Model configuration. Uses ``d_model``, ``d_ff``, and
                ``dropout_rate``. If ``cfg.d_ff`` is ``None``, it is
                auto-computed as ``round_up(2/3 * 4 * d_model, 256)``.
            rngs: Flax NNX RNG container for parameter initialization
                and dropout randomness.
        """
        self.d_ff = _compute_gated_d_ff(cfg.d_model, cfg.d_ff)

        # Gate projection: applies GELU activation to control information flow
        # d_model -> d_ff, no bias
        self.gate_proj = nnx.Linear(
            cfg.d_model, self.d_ff, use_bias=False, rngs=rngs
        )

        # Up projection: produces the value signal to be gated
        # d_model -> d_ff, no bias
        self.up_proj = nnx.Linear(
            cfg.d_model, self.d_ff, use_bias=False, rngs=rngs
        )

        # Down projection: maps back to model dimension
        # d_ff -> d_model, no bias
        self.down_proj = nnx.Linear(
            self.d_ff, cfg.d_model, use_bias=False, rngs=rngs
        )

        # Output dropout
        self.dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

        log.debug(
            "geglu_ffn.init",
            d_model=cfg.d_model,
            d_ff=self.d_ff,
            dropout_rate=cfg.dropout_rate,
        )

    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
    ) -> jax.Array:
        """Apply GeGLU feed-forward transformation.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            deterministic: If ``True``, disables dropout (inference mode).
                If ``False``, applies dropout (training mode).

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.

        Shape flow::

            x: [B, S, d_model]
                |
            gate_proj -> gelu: [B, S, d_ff]
            up_proj:           [B, S, d_ff]
                |
            gate * up:         [B, S, d_ff]
                |
            down_proj:         [B, S, d_model]
                |
            dropout:           [B, S, d_model]
        """
        # Gating branch with GELU activation
        gate = jax.nn.gelu(self.gate_proj(x))  # [B, S, d_ff]

        # Value branch (no activation)
        up = self.up_proj(x)  # [B, S, d_ff]

        # Element-wise gating and down-projection
        out = self.down_proj(gate * up)  # [B, S, d_model]

        # Apply dropout
        return self.dropout(out, deterministic=deterministic)  # [B, S, d_model]

    def __repr__(self) -> str:
        return f"GeGLUFFN(d_ff={self.d_ff})"


# ---------------------------------------------------------------------------
# Standard MLP Feed-Forward Network
# ---------------------------------------------------------------------------


class StandardMLP(nnx.Module):
    """Classical two-layer MLP with GELU activation.

    Implements the original transformer feed-forward network used in
    GPT-2, BERT, and other early transformers.

    The forward pass computes::

        out = fc2(gelu(fc1(x)))         # [B, S, d_model]
        out = dropout(out)

    Unlike the gated variants (:class:`SwiGLUFFN`, :class:`GeGLUFFN`),
    this MLP has no gating mechanism and uses only two linear projections
    instead of three, resulting in fewer parameters for the same ``d_ff``.

    Attributes:
        d_ff: Intermediate (hidden) dimension of the feed-forward network.
        fc1: First linear projection (``d_model -> d_ff``).
        fc2: Second linear projection (``d_ff -> d_model``).
        dropout: Dropout layer applied to the output.
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize StandardMLP.

        Args:
            cfg: Model configuration. Uses ``d_model``, ``d_ff``, and
                ``dropout_rate``. If ``cfg.d_ff`` is ``None``, it defaults
                to ``4 * d_model``.
            rngs: Flax NNX RNG container for parameter initialization
                and dropout randomness.
        """
        self.d_ff = _compute_standard_d_ff(cfg.d_model, cfg.d_ff)

        # First projection: expand to intermediate dimension
        # d_model -> d_ff, uses bias following GPT-2 convention
        self.gate_proj = nnx.Linear(
            cfg.d_model, self.d_ff, use_bias=cfg.use_bias, rngs=rngs
        )

        # Unused but kept for interface compatibility with gated variants.
        # StandardMLP does not have a separate up_proj or gating mechanism.
        # We alias fc1/fc2 as gate_proj/down_proj below for compatibility,
        # but provide fc1/fc2 properties for clarity.
        self.up_proj = None  # type: ignore[assignment]

        # Second projection: contract back to model dimension
        # d_ff -> d_model
        self.down_proj = nnx.Linear(
            self.d_ff, cfg.d_model, use_bias=cfg.use_bias, rngs=rngs
        )

        # Output dropout
        self.dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

        log.debug(
            "standard_mlp.init",
            d_model=cfg.d_model,
            d_ff=self.d_ff,
            dropout_rate=cfg.dropout_rate,
        )

    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
    ) -> jax.Array:
        """Apply standard MLP feed-forward transformation.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            deterministic: If ``True``, disables dropout (inference mode).
                If ``False``, applies dropout (training mode).

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.

        Shape flow::

            x: [B, S, d_model]
                |
            fc1 (gate_proj) -> gelu:  [B, S, d_ff]
                |
            fc2 (down_proj):          [B, S, d_model]
                |
            dropout:                  [B, S, d_model]
        """
        # Expand to intermediate dimension with GELU activation
        hidden = jax.nn.gelu(self.gate_proj(x))  # [B, S, d_ff]

        # Contract back to model dimension
        out = self.down_proj(hidden)  # [B, S, d_model]

        # Apply dropout
        return self.dropout(out, deterministic=deterministic)  # [B, S, d_model]

    def __repr__(self) -> str:
        return f"StandardMLP(d_ff={self.d_ff})"
