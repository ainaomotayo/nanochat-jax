"""Normalization layers for the nanochat-jax transformer.

Faithful nanochat port: RMSNorm is PARAMETERLESS — no learned scale (gamma).
This matches nanochat's norm() function exactly:

    y = x / sqrt(mean(x²) + eps)

Design rationale:
- Removing gamma reduces parameters and eliminates a potential source of
  training instability at depth.
- The residual stream magnitude is controlled by weight initialization
  (from_depth scaling) rather than learned norms.
- This is a deliberate nanochat departure from LLaMA/Gemma which use
  learned gamma.

References:
    - RMSNorm: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
    - nanochat: parameterless variant (no affine transform)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import structlog
from flax import nnx

log = structlog.get_logger(__name__)


class RMSNorm(nnx.Module):
    """Parameterless Root Mean Square Layer Normalization.

    Normalizes the input by its RMS value WITHOUT any learnable scale
    or shift parameters. This is the nanochat-faithful variant.

    Formula::

        output = x / sqrt(mean(x², axis=-1, keepdims=True) + eps)

    The computation is promoted to float32 for numerical stability
    regardless of input dtype, then cast back.

    Attributes:
        d_model: Dimensionality of the input features (stored for repr).
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize parameterless RMSNorm.

        Args:
            d_model: Input feature dimensionality (informational; not used
                in computation since there are no parameters to shape).
            eps: Stability epsilon added inside the square root.
            rngs: Flax NNX RNG container (required by convention; unused
                since this module has no parameters).
        """
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.d_model = d_model
        self.eps = eps
        # No gamma parameter — this is intentional for nanochat fidelity.
        # Do NOT add self.gamma here.

        log.debug("rmsnorm.init", d_model=d_model, eps=eps, parameterless=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply parameterless RMSNorm.

        Args:
            x: Input tensor of shape ``(..., d_model)``.

        Returns:
            RMS-normalized tensor of the same shape and dtype as *x*.
        """
        input_dtype = x.dtype

        # Promote to float32 for numerically stable variance computation.
        x_f32 = x.astype(jnp.float32)  # [..., d_model]

        # mean(x²) along the feature axis
        mean_sq = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)  # [..., 1]

        # Reciprocal RMS
        rms_inv = jax.lax.rsqrt(mean_sq + self.eps)  # [..., 1]

        # Normalize — no gamma scaling
        normed = x_f32 * rms_inv  # [..., d_model]

        return normed.astype(input_dtype)

    def __repr__(self) -> str:
        return f"RMSNorm(d_model={self.d_model}, eps={self.eps}, parameterless=True)"


class LayerNorm(nnx.Module):
    """Standard Layer Normalization with learnable affine parameters.

    Retained for completeness and ablation experiments. Not used in the
    nanochat-faithful default configuration.

    Formula::

        output = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.d_model = d_model
        self.eps = eps
        self.gamma = nnx.Param(jnp.ones((d_model,)))
        self.beta = nnx.Param(jnp.zeros((d_model,)))

        log.debug("layernorm.init", d_model=d_model, eps=eps)

    def __call__(self, x: jax.Array) -> jax.Array:
        input_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x_f32 - mean), axis=-1, keepdims=True)
        normed = (x_f32 - mean) * jax.lax.rsqrt(var + self.eps)
        output = normed * self.gamma.get_value().astype(jnp.float32) + self.beta.get_value().astype(jnp.float32)
        return output.astype(input_dtype)

    def __repr__(self) -> str:
        return f"LayerNorm(d_model={self.d_model}, eps={self.eps})"
