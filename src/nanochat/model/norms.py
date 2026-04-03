"""Normalization layers for the nanochat-jax transformer.

This module provides RMSNorm and LayerNorm implementations as Flax NNX modules.
RMSNorm is the preferred normalization for modern LLMs (LLaMA, Gemma, etc.)
due to its simplicity and comparable performance to LayerNorm.

References:
    - RMSNorm: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
    - LayerNorm: Ba, Kiros & Hinton, "Layer Normalization" (2016)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import structlog
from flax import nnx

log = structlog.get_logger(__name__)


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input by its RMS value and scales by a learnable parameter.
    Unlike LayerNorm, RMSNorm does not re-center (no mean subtraction or bias),
    which reduces computation while maintaining effectiveness.

    Formula::

        output = (x / sqrt(mean(x^2, axis=-1, keepdims=True) + eps)) * gamma

    Attributes:
        d_model: Dimensionality of the input features.
        eps: Small constant for numerical stability in the denominator.
        gamma: Learnable scale parameter of shape ``(d_model,)``.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize RMSNorm.

        Args:
            d_model: Dimensionality of the input features.
            eps: Small constant added inside the square root for numerical
                stability. Defaults to ``1e-6``.
            rngs: Flax NNX RNG container (required by NNX convention, not
                consumed by this module).

        Raises:
            ValueError: If *d_model* is not a positive integer.
        """
        if d_model <= 0:
            raise ValueError(
                f"d_model must be a positive integer, got {d_model}"
            )

        self.d_model = d_model
        self.eps = eps
        self.gamma = nnx.Param(jnp.ones((d_model,)))  # [d_model]

        log.debug(
            "rmsnorm.init",
            d_model=d_model,
            eps=eps,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply RMSNorm to the input tensor.

        The computation is performed in float32 for numerical stability,
        regardless of the input dtype. The output is cast back to the
        original input dtype.

        Args:
            x: Input tensor of shape ``(..., d_model)`` where the last
                dimension must equal *d_model*.

        Returns:
            Normalized tensor of the same shape and dtype as *x*.

        Raises:
            ValueError: If the last dimension of *x* does not match *d_model*.
        """
        input_dtype = x.dtype  # preserve original dtype for output cast

        # Upcast to float32 for numerical stability during norm computation
        x_f32 = x.astype(jnp.float32)  # [..., d_model]

        # Compute mean of squared values along feature dimension
        mean_sq = jnp.mean(
            jnp.square(x_f32), axis=-1, keepdims=True
        )  # [..., 1]

        # Compute the reciprocal of the RMS (inverse root mean square)
        rms_inv = jax.lax.rsqrt(mean_sq + self.eps)  # [..., 1]

        # Normalize and scale by learnable gamma
        normed = x_f32 * rms_inv  # [..., d_model]
        scaled = normed * self.gamma.value.astype(jnp.float32)  # [..., d_model]

        # Cast back to input dtype (e.g. bfloat16)
        return scaled.astype(input_dtype)  # [..., d_model]

    def __repr__(self) -> str:
        return f"RMSNorm(d_model={self.d_model}, eps={self.eps})"


class LayerNorm(nnx.Module):
    """Standard Layer Normalization with learnable affine parameters.

    Normalizes the input by subtracting the mean and dividing by the
    standard deviation, then applies a learnable scale (*gamma*) and
    shift (*beta*).

    Formula::

        output = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

    Attributes:
        d_model: Dimensionality of the input features.
        eps: Small constant for numerical stability.
        gamma: Learnable scale parameter of shape ``(d_model,)``.
        beta: Learnable shift parameter of shape ``(d_model,)``.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize LayerNorm.

        Args:
            d_model: Dimensionality of the input features.
            eps: Small constant added inside the square root for numerical
                stability. Defaults to ``1e-5``.
            rngs: Flax NNX RNG container (required by NNX convention, not
                consumed by this module).

        Raises:
            ValueError: If *d_model* is not a positive integer.
        """
        if d_model <= 0:
            raise ValueError(
                f"d_model must be a positive integer, got {d_model}"
            )

        self.d_model = d_model
        self.eps = eps
        self.gamma = nnx.Param(jnp.ones((d_model,)))  # [d_model]
        self.beta = nnx.Param(jnp.zeros((d_model,)))  # [d_model]

        log.debug(
            "layernorm.init",
            d_model=d_model,
            eps=eps,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply LayerNorm to the input tensor.

        The computation is performed in float32 for numerical stability,
        regardless of the input dtype. The output is cast back to the
        original input dtype.

        Args:
            x: Input tensor of shape ``(..., d_model)`` where the last
                dimension must equal *d_model*.

        Returns:
            Normalized tensor of the same shape and dtype as *x*.
        """
        input_dtype = x.dtype  # preserve original dtype for output cast

        # Upcast to float32 for numerical stability
        x_f32 = x.astype(jnp.float32)  # [..., d_model]

        # Compute mean along feature dimension
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)  # [..., 1]

        # Compute variance along feature dimension
        var = jnp.mean(
            jnp.square(x_f32 - mean), axis=-1, keepdims=True
        )  # [..., 1]

        # Normalize
        normed = (x_f32 - mean) * jax.lax.rsqrt(var + self.eps)  # [..., d_model]

        # Apply learnable affine transform
        gamma_f32 = self.gamma.value.astype(jnp.float32)  # [d_model]
        beta_f32 = self.beta.value.astype(jnp.float32)  # [d_model]
        output = normed * gamma_f32 + beta_f32  # [..., d_model]

        # Cast back to input dtype
        return output.astype(input_dtype)  # [..., d_model]

    def __repr__(self) -> str:
        return f"LayerNorm(d_model={self.d_model}, eps={self.eps})"
