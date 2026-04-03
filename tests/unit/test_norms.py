"""Tests for normalization layers."""
import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from nanochat.model.norms import RMSNorm, LayerNorm


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 8, 64))
        out = norm(x)
        assert out.shape == (2, 8, 64)

    def test_unit_rms(self):
        norm = RMSNorm(64, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))
        out = norm(x)
        rms = jnp.sqrt(jnp.mean(out ** 2, axis=-1))
        # After normalization (with gamma=1), RMS should be ~1
        assert jnp.allclose(rms, 1.0, atol=0.1)

    def test_scale_equivariance(self):
        norm = RMSNorm(64, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))
        out1 = norm(x)
        out2 = norm(3.0 * x)
        assert jnp.allclose(out1, out2, atol=1e-5)

    def test_gradient_flows(self):
        norm = RMSNorm(64, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))
        def fn(x):
            return norm(x).sum()
        grad = jax.grad(fn)(x)
        assert jnp.isfinite(grad).all()

    def test_bf16_no_nan(self):
        norm = RMSNorm(64, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64)).astype(jnp.bfloat16)
        out = norm(x)
        assert jnp.isfinite(out).all()


class TestLayerNorm:
    def test_output_shape(self):
        norm = LayerNorm(64, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 8, 64))
        out = norm(x)
        assert out.shape == (2, 8, 64)

    def test_zero_mean(self):
        norm = LayerNorm(64, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))
        out = norm(x)
        mean = jnp.mean(out, axis=-1)
        assert jnp.allclose(mean, 0.0, atol=1e-5)
