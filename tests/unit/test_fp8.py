"""Tests for FP8 training support."""
import jax
import jax.numpy as jnp
import pytest

from nanochat.training.fp8 import FP8Config, fp8_matmul, is_fp8_available


def test_fp8_availability_check():
    """is_fp8_available should return a bool without raising."""
    result = is_fp8_available()
    assert isinstance(result, bool)


def test_fp8_matmul_shapes():
    """fp8_matmul should produce the correct output shape."""
    M, K, N = 16, 32, 64
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K))
    b = jax.random.normal(k2, (K, N))

    out = fp8_matmul(a, b)
    assert out.shape == (M, N), f"Expected ({M}, {N}), got {out.shape}"


def test_fp8_matmul_approximate_correctness():
    """fp8_matmul should produce results close to standard matmul.

    On CPU (no FP8), results should be exactly equal since fp8_matmul
    falls back to a @ b.  On GPU with FP8, allow some tolerance for
    quantization error.
    """
    M, K, N = 8, 16, 8
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K))
    b = jax.random.normal(k2, (K, N))

    reference = a @ b
    result = fp8_matmul(a, b, scale_a=1.0, scale_b=1.0)

    if is_fp8_available():
        # FP8 quantization introduces some error; allow generous tolerance.
        assert jnp.allclose(reference, result, atol=0.5, rtol=0.1), (
            f"FP8 matmul too far from reference: "
            f"max diff = {float(jnp.max(jnp.abs(reference - result)))}"
        )
    else:
        # Fallback path: should be exactly equal.
        assert jnp.allclose(reference, result, atol=1e-6), (
            "Fallback fp8_matmul should match standard matmul exactly"
        )


def test_fp8_config_defaults():
    """FP8Config should have sensible defaults."""
    cfg = FP8Config()
    assert cfg.enabled is False
    assert cfg.amax_history_len == 16
    assert cfg.compute_dtype == jnp.float32


def test_fp8_config_graceful_degradation():
    """FP8Config(enabled=True) should auto-disable on unsupported hardware."""
    cfg = FP8Config(enabled=True)
    if not is_fp8_available():
        assert cfg.enabled is False, (
            "FP8Config should auto-disable when hardware doesn't support FP8"
        )
