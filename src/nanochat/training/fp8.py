"""FP8 training support with graceful fallback.

Provides FP8 (8-bit floating point) matrix multiplication when hardware
and JAX support it.  On unsupported platforms the functions silently
fall back to the current dtype, adding zero overhead.

FP8 format used: ``float8_e4m3fn`` (4 exponent, 3 mantissa bits,
no infinities) -- the standard training format on Hopper/Blackwell GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import structlog

log = structlog.get_logger(__name__)


def is_fp8_available() -> bool:
    """Check whether FP8 computation is available.

    Returns ``True`` when both conditions hold:

    1. ``jnp.float8_e4m3fn`` exists as a dtype (JAX >= 0.4.14).
    2. The default device is a GPU (FP8 on CPU is pointless).

    Returns:
        ``True`` if FP8 is usable, ``False`` otherwise.
    """
    if not hasattr(jnp, "float8_e4m3fn"):
        return False

    try:
        devices = jax.devices()
        if not devices:
            return False
        platform = devices[0].platform
        if platform not in ("gpu", "cuda"):
            return False
    except Exception:
        return False

    return True


def fp8_matmul(
    a: jax.Array,
    b: jax.Array,
    scale_a: float = 1.0,
    scale_b: float = 1.0,
) -> jax.Array:
    """Matrix multiplication with optional FP8 quantization.

    When FP8 is available the operands are:
    1. Scaled by ``1 / scale_*``.
    2. Cast to ``float8_e4m3fn``.
    3. Multiplied.
    4. The result is cast back to ``float32`` and rescaled by
       ``scale_a * scale_b``.

    When FP8 is *not* available, this is just ``a @ b`` in the
    original dtype -- no overhead.

    Args:
        a: Left operand ``(..., M, K)``.
        b: Right operand ``(..., K, N)``.
        scale_a: Amax-derived scale for *a*.
        scale_b: Amax-derived scale for *b*.

    Returns:
        Result ``(..., M, N)``.
    """
    if not is_fp8_available():
        return a @ b

    fp8 = jnp.float8_e4m3fn
    original_dtype = a.dtype

    # Scale down, cast to fp8, matmul, cast back, scale up.
    a_fp8 = (a.astype(jnp.float32) / scale_a).astype(fp8)
    b_fp8 = (b.astype(jnp.float32) / scale_b).astype(fp8)

    # Matmul in float32 after promoting from fp8.
    result = jnp.matmul(a_fp8.astype(jnp.float32), b_fp8.astype(jnp.float32))
    result = result * (scale_a * scale_b)

    return result.astype(original_dtype)


@dataclass
class FP8Config:
    """Configuration for FP8 training.

    Attributes:
        enabled: Whether to use FP8 matmuls where possible.
        compute_dtype: The dtype to accumulate in (always float32 for FP8).
        amax_history_len: Number of past amax values to track for
            dynamic scaling.
    """

    enabled: bool = False
    compute_dtype: jnp.dtype = field(default=jnp.float32)
    amax_history_len: int = 16

    def __post_init__(self) -> None:
        if self.enabled and not is_fp8_available():
            log.warning(
                "fp8.unavailable",
                reason="FP8 requested but not supported on this device. "
                       "Falling back to standard precision.",
            )
            self.enabled = False
        if self.enabled:
            log.info("fp8.enabled", amax_history_len=self.amax_history_len)
