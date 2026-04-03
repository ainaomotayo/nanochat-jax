"""Tests for feedforward modules."""
import jax
import jax.numpy as jnp
from flax import nnx
from nanochat.config import ModelConfig
from nanochat.model.feedforward import SwiGLUFFN, GeGLUFFN, StandardMLP


def test_swiglu_shape(nano_config):
    ffn = SwiGLUFFN(nano_config, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 8, nano_config.d_model))
    out = ffn(x, deterministic=True)
    assert out.shape == (2, 8, nano_config.d_model)


def test_geglu_shape(nano_config):
    ffn = GeGLUFFN(nano_config, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 8, nano_config.d_model))
    out = ffn(x, deterministic=True)
    assert out.shape == (2, 8, nano_config.d_model)


def test_standard_mlp_shape(nano_config):
    ffn = StandardMLP(nano_config, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 8, nano_config.d_model))
    out = ffn(x, deterministic=True)
    assert out.shape == (2, 8, nano_config.d_model)


def test_all_ffn_finite(nano_config):
    for cls in [SwiGLUFFN, GeGLUFFN, StandardMLP]:
        ffn = cls(nano_config, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, nano_config.d_model))
        out = ffn(x, deterministic=True)
        assert jnp.isfinite(out).all(), f"{cls.__name__} produced non-finite output"
