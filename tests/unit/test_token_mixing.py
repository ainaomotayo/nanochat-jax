"""Unit tests for Smear and Backout token-mixing operations.

Tests:
1. Smear: causal constraint (no future information)
2. Smear: near-no-op at initialization
3. Smear: output shape
4. Backout: corrective behavior
5. Both: gradient flow
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nanochat.model.token_mixing import Smear, Backout


class TestSmear:
    def test_output_shape(self):
        smear = Smear(64, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 8, 64))
        out, _ = smear(x)
        assert out.shape == (2, 8, 64)

    def test_returns_x_prev(self):
        """Smear should return (x_smeared, x_prev)."""
        smear = Smear(32, rngs=nnx.Rngs(0))
        x = jnp.ones((1, 4, 32))
        result = smear(x)
        assert isinstance(result, tuple) and len(result) == 2

    def test_causal_first_position(self):
        """First position (t=0) should use zero padding as predecessor."""
        smear = Smear(4, rngs=nnx.Rngs(0))
        x = jnp.array([[[1.0, 2.0, 3.0, 4.0],    # pos 0
                         [5.0, 6.0, 7.0, 8.0]]])  # pos 1

        _, x_prev = smear(x)

        # x_prev[0] should be zero-padded
        np.testing.assert_allclose(
            np.array(x_prev[0, 0]), np.zeros(4), atol=1e-8,
            err_msg="First position predecessor should be zero (causal)"
        )
        # x_prev[1] should be x[0]
        np.testing.assert_allclose(
            np.array(x_prev[0, 1]), np.array([1.0, 2.0, 3.0, 4.0]), atol=1e-6,
        )

    def test_near_noop_at_init(self):
        """At initialization (raw_alpha=-10), smear should be nearly identity."""
        smear = Smear(64, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))
        x_smeared, _ = smear(x)

        # alpha ≈ sigmoid(-10) ≈ 4.5e-5, so output ≈ x
        max_diff = float(jnp.max(jnp.abs(x_smeared - x)))
        assert max_diff < 0.01, (
            f"Smear is not near-identity at init: max_diff={max_diff}"
        )

    def test_finite_output(self):
        smear = Smear(64, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 64))
        out, prev = smear(x)
        assert jnp.isfinite(out).all()
        assert jnp.isfinite(prev).all()

    def test_gradient_flows(self):
        smear = Smear(32, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(2), (1, 4, 32))

        grad_x = jax.grad(lambda x_in: smear(x_in)[0].sum())(x)
        assert jnp.isfinite(grad_x).all(), "Non-finite gradient through Smear"


class TestBackout:
    def test_output_shape(self):
        backout = Backout(64, rngs=nnx.Rngs(0))
        attn_out = jnp.ones((2, 8, 64))
        x_prev = jnp.ones((2, 8, 64))
        out = backout(attn_out, x_prev)
        assert out.shape == (2, 8, 64)

    def test_near_noop_at_init(self):
        """At initialization (raw_beta=-10), backout should be nearly identity."""
        backout = Backout(64, rngs=nnx.Rngs(0))
        attn_out = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))
        x_prev = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 64))

        out = backout(attn_out, x_prev)
        max_diff = float(jnp.max(jnp.abs(out - attn_out)))
        assert max_diff < 0.01, f"Backout not near-identity at init: {max_diff}"

    def test_finite_output(self):
        backout = Backout(64, rngs=nnx.Rngs(0))
        attn_out = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))
        x_prev = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 64))
        out = backout(attn_out, x_prev)
        assert jnp.isfinite(out).all()

    def test_gradient_flows(self):
        backout = Backout(32, rngs=nnx.Rngs(0))
        attn_out = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 32))
        x_prev = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 32))

        grad_attn = jax.grad(lambda a: backout(a, x_prev).sum())(attn_out)
        assert jnp.isfinite(grad_attn).all(), "Non-finite gradient through Backout"
