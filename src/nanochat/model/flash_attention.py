"""Flash Attention with automatic fallback.

Provides three attention backends:

1. **naive_attention** -- Standard O(N^2) scaled dot-product attention.
   Always available; supports GQA via broadcasting.

2. **jax_native_attention** -- Uses ``jax.nn.dot_product_attention``
   (available since JAX >= 0.4.28).  Leverages XLA Flash Attention on
   TPU and cuDNN Flash Attention on supported GPUs.  Falls back to
   naive if the JAX API is missing.

3. **get_attention_fn** -- Factory that returns the best available
   backend at import time.
"""

from __future__ import annotations

import warnings
from typing import Callable

import jax
import jax.numpy as jnp
import structlog

log = structlog.get_logger(__name__)

# Type alias for attention functions.
AttentionFn = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array | None, float | None],
    jax.Array,
]


def naive_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array | None = None,
    scale: float | None = None,
) -> jax.Array:
    """Standard O(N^2) scaled dot-product attention.

    Supports GQA: when ``k`` and ``v`` have fewer heads than ``q``,
    they are broadcast-expanded to match the query head count.

    Args:
        q: Query tensor ``(B, H, S, D)``.
        k: Key tensor ``(B, Kh, S, D)`` where ``Kh`` divides ``H``.
        v: Value tensor ``(B, Kh, S, D)``.
        mask: Boolean mask ``(B, 1, S, S)`` or ``None``.
            ``True`` = attend, ``False`` = masked out.
        scale: Attention scale. Defaults to ``1 / sqrt(D)``.

    Returns:
        Output tensor ``(B, H, S, D)``.
    """
    B, H, S, D = q.shape
    Kh = k.shape[1]

    if scale is None:
        scale = D ** -0.5

    # GQA: expand KV heads to match query heads via repeat.
    if Kh != H:
        n_groups = H // Kh
        k = jnp.repeat(k, repeats=n_groups, axis=1)
        v = jnp.repeat(v, repeats=n_groups, axis=1)

    # Compute attention scores.
    scores = jnp.matmul(
        q.astype(jnp.float32),
        jnp.transpose(k.astype(jnp.float32), (0, 1, 3, 2)),
    ) * scale  # (B, H, S, S)

    if mask is not None:
        scores = jnp.where(mask, scores, jnp.float32(-1e9))

    weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(weights, v.astype(jnp.float32))  # (B, H, S, D)
    return output.astype(q.dtype)


# Detect whether jax.nn.dot_product_attention is available.
_HAS_JAX_SDPA: bool = hasattr(jax.nn, "dot_product_attention")


def jax_native_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array | None = None,
    scale: float | None = None,
) -> jax.Array:
    """Attention via ``jax.nn.dot_product_attention`` (JAX >= 0.4.28).

    Falls back to :func:`naive_attention` if the JAX API is unavailable.

    The JAX native path may dispatch to XLA Flash Attention on TPU or
    cuDNN Flash Attention on supported NVIDIA GPUs, yielding O(N) memory.

    Args:
        q: Query tensor ``(B, H, S, D)``.
        k: Key tensor ``(B, Kh, S, D)`` where ``Kh`` divides ``H``.
        v: Value tensor ``(B, Kh, S, D)``.
        mask: Boolean mask ``(B, 1, S, S)`` or ``None``.
        scale: Attention scale. Defaults to ``1 / sqrt(D)``.

    Returns:
        Output tensor ``(B, H, S, D)``.
    """
    if not _HAS_JAX_SDPA:
        return naive_attention(q, k, v, mask, scale)

    B, H, S, D = q.shape
    Kh = k.shape[1]

    if scale is None:
        scale = D ** -0.5

    # jax.nn.dot_product_attention expects (B, T, N, H) layout.
    # Our inputs are (B, H, S, D) so transpose axes 1 and 2.
    q_t = jnp.transpose(q.astype(jnp.float32), (0, 2, 1, 3))  # (B, S, H, D)
    k_t = jnp.transpose(k.astype(jnp.float32), (0, 2, 1, 3))  # (B, S, Kh, D)
    v_t = jnp.transpose(v.astype(jnp.float32), (0, 2, 1, 3))  # (B, S, Kh, D)

    # Convert boolean mask (B, 1, S, S) to additive bias (B, H, S, S)
    # for the native API.  The API expects bias shape (B, N, T, S).
    if mask is not None:
        bias = jnp.where(mask, jnp.float32(0.0), jnp.float32(-1e9))
        # Broadcast (B, 1, S, S) -> (B, H, S, S) explicitly.
        bias = jnp.broadcast_to(bias, (B, H, S, S))
    else:
        bias = None

    # The native API handles GQA natively when K < N.
    out_t = jax.nn.dot_product_attention(
        q_t, k_t, v_t,
        bias=bias,
        scale=scale,
    )  # (B, S, H, D)

    # Transpose back to (B, H, S, D).
    output = jnp.transpose(out_t, (0, 2, 1, 3))
    return output.astype(q.dtype)


def get_attention_fn(use_flash: bool = True) -> AttentionFn:
    """Return the best available attention function.

    Selection order:
    1. If *use_flash* is ``False``: always return :func:`naive_attention`.
    2. If ``jax.nn.dot_product_attention`` exists: return
       :func:`jax_native_attention`.
    3. Otherwise: return :func:`naive_attention` with a warning.

    Args:
        use_flash: Whether to attempt hardware-accelerated attention.

    Returns:
        A callable with signature
        ``(q, k, v, mask, scale) -> output``.
    """
    if not use_flash:
        log.info("flash_attention.disabled", backend="naive")
        return naive_attention

    if _HAS_JAX_SDPA:
        log.info("flash_attention.enabled", backend="jax_native")
        return jax_native_attention

    warnings.warn(
        "jax.nn.dot_product_attention not found (requires JAX >= 0.4.28). "
        "Falling back to naive O(N^2) attention.",
        stacklevel=2,
    )
    log.warning("flash_attention.fallback", backend="naive", reason="jax_sdpa_missing")
    return naive_attention
