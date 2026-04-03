"""Unit tests for parameterless RMSNorm — nanochat faithful variant.

Regression tests verify:
1. Parameterless (no gamma)
2. Formula: y = x / sqrt(mean(x²) + eps)
3. RMS of output ≈ 1.0
4. Scale equivariance
5. Dtype preservation
6. Numerical stability
"""

import math
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nanochat.model.norms import RMSNorm


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0)


class TestRMSNormParameterless:
    """RMSNorm must have zero trainable parameters."""

    def test_no_gamma_attribute(self, rngs):
        norm = RMSNorm(64, rngs=rngs)
        # gamma must not exist as an nnx.Param attribute
        gamma_val = getattr(norm, "gamma", None)
        assert gamma_val is None or not isinstance(gamma_val, nnx.Param), (
            "Nanochat RMSNorm must be parameterless — found 'gamma' nnx.Param"
        )

    def test_zero_parameters(self, rngs):
        norm = RMSNorm(64, rngs=rngs)
        params = nnx.state(norm, nnx.Param)
        leaves = jax.tree_util.tree_leaves(params)
        n_params = sum(v.size for v in leaves)
        assert n_params == 0, f"Expected 0 params, got {n_params}"


class TestRMSNormFormula:
    """Mathematical correctness tests."""

    def test_formula(self, rngs):
        """y = x / sqrt(mean(x²) + eps)."""
        norm = RMSNorm(4, eps=1e-6, rngs=rngs)
        x = jnp.array([[1.0, 2.0, 3.0, 4.0]])

        out = norm(x)

        rms = math.sqrt((1 + 4 + 9 + 16) / 4 + 1e-6)
        expected = np.array([[1/rms, 2/rms, 3/rms, 4/rms]])
        np.testing.assert_allclose(np.array(out), expected, rtol=1e-5)

    def test_output_rms_near_one(self, rngs):
        """Output RMS should be ≈ 1.0 (no learned scale)."""
        norm = RMSNorm(128, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 32, 128))
        out = norm(x)
        rms_out = jnp.sqrt(jnp.mean(jnp.square(out), axis=-1))
        np.testing.assert_allclose(
            np.array(rms_out), np.ones_like(rms_out), atol=1e-4
        )

    def test_scale_equivariance(self, rngs):
        """norm(c*x) == norm(x) for any scalar c != 0."""
        norm = RMSNorm(64, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 64))
        out1 = norm(x)
        out2 = norm(3.0 * x)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-5)

    def test_gradient_flows(self, rngs):
        """Gradients through parameterless RMSNorm must be finite."""
        norm = RMSNorm(64, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(2), (2, 8, 64))

        def fn(x_in):
            return norm(x_in).sum()

        grad = jax.grad(fn)(x)
        assert jnp.isfinite(grad).all(), "Non-finite gradient through RMSNorm"


class TestRMSNormDtype:
    """Dtype preservation tests."""

    def test_float32_preserved(self, rngs):
        norm = RMSNorm(32, rngs=rngs)
        x = jnp.ones((2, 4, 32), dtype=jnp.float32)
        assert norm(x).dtype == jnp.float32

    def test_bfloat16_preserved(self, rngs):
        norm = RMSNorm(32, rngs=rngs)
        x = jnp.ones((2, 4, 32), dtype=jnp.bfloat16)
        assert norm(x).dtype == jnp.bfloat16

    def test_shape_preserved(self, rngs):
        norm = RMSNorm(64, rngs=rngs)
        for shape in [(1, 10, 64), (4, 32, 64)]:
            x = jax.random.normal(jax.random.PRNGKey(0), shape)
            assert norm(x).shape == shape


class TestRMSNormStability:
    """Numerical stability tests."""

    def test_near_zero_no_nan(self, rngs):
        norm = RMSNorm(16, eps=1e-6, rngs=rngs)
        x = jnp.full((1, 4, 16), 1e-20)
        out = norm(x)
        assert not jnp.any(jnp.isnan(out))
        assert not jnp.any(jnp.isinf(out))

    def test_large_input_no_nan(self, rngs):
        norm = RMSNorm(16, rngs=rngs)
        x = jnp.full((1, 4, 16), 1e6)
        out = norm(x)
        assert not jnp.any(jnp.isnan(out))

    def test_bfloat16_no_nan(self, rngs):
        norm = RMSNorm(64, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(3), (2, 8, 64)).astype(jnp.bfloat16)
        out = norm(x)
        assert jnp.isfinite(out).all()
