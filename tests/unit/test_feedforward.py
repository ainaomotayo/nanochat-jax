"""Unit tests for feed-forward modules.

Covers:
- ReLUSquaredMLP (nanochat default): activation correctness, no gating branch
- SwiGLUFFN, GeGLUFFN, StandardMLP: shape and finiteness
- relu² activation formula: x * relu(x) = max(0, x)^2
- Regression: all FFN variants produce finite outputs
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nanochat.config import ModelConfig
from nanochat.model.feedforward import (
    ReLUSquaredMLP, SwiGLUFFN, GeGLUFFN, StandardMLP
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mini_cfg(**kwargs) -> ModelConfig:
    base = dict(
        vocab_size=256, d_model=32, n_heads=4, n_kv_heads=4,
        n_layers=1, d_ff=128, max_seq_len=16,
        use_qk_norm=False, logit_softcap=None,
        use_value_embeddings=False, use_per_layer_scalars=False,
        use_smear=False,
    )
    base.update(kwargs)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
# relu² correctness
# ---------------------------------------------------------------------------

class TestReLUSquaredActivation:
    """Verify relu²(x) = x * relu(x) = max(0, x)² mathematically."""

    def test_positive_values(self):
        """For x > 0: relu²(x) = x²."""
        x = jnp.array([0.5, 1.0, 2.0, 3.0])
        result = x * jax.nn.relu(x)
        expected = x ** 2
        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-6)

    def test_negative_values_zero(self):
        """For x < 0: relu²(x) = 0."""
        x = jnp.array([-0.5, -1.0, -2.0, -3.0])
        result = x * jax.nn.relu(x)
        np.testing.assert_allclose(np.array(result), np.zeros(4), atol=1e-8)

    def test_zero_value(self):
        """At x = 0: relu²(0) = 0."""
        assert float(jnp.array(0.0) * jax.nn.relu(jnp.array(0.0))) == 0.0

    def test_bf16_safe(self):
        """relu² should not produce NaN in bfloat16."""
        x = jax.random.normal(jax.random.PRNGKey(0), (16, 128)).astype(jnp.bfloat16)
        result = x * jax.nn.relu(x)
        assert jnp.isfinite(result).all()


class TestReLUSquaredMLP:
    """ReLUSquaredMLP module tests."""

    def test_output_shape(self):
        cfg = _mini_cfg(ffn_type="relu2")
        ffn = ReLUSquaredMLP(cfg, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 8, 32))
        out = ffn(x)
        assert out.shape == (2, 8, 32)

    def test_no_gating_branch(self):
        """ReLUSquaredMLP must not have up_proj (no gating)."""
        cfg = _mini_cfg(ffn_type="relu2")
        ffn = ReLUSquaredMLP(cfg, rngs=nnx.Rngs(0))
        assert ffn.up_proj is None, "relu² MLP should have no up_proj"

    def test_has_gate_and_down_proj(self):
        """Must have gate_proj (fc1) and down_proj (fc2)."""
        cfg = _mini_cfg(ffn_type="relu2")
        ffn = ReLUSquaredMLP(cfg, rngs=nnx.Rngs(0))
        assert hasattr(ffn, "gate_proj")
        assert hasattr(ffn, "down_proj")

    def test_finite_output(self):
        cfg = _mini_cfg(ffn_type="relu2")
        ffn = ReLUSquaredMLP(cfg, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 32))
        assert jnp.isfinite(ffn(x)).all()

    def test_bfloat16_no_nan(self):
        cfg = _mini_cfg(ffn_type="relu2")
        ffn = ReLUSquaredMLP(cfg, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 32)).astype(jnp.bfloat16)
        out = ffn(x)
        assert jnp.isfinite(out).all()

    def test_d_ff_default(self):
        """d_ff should default to 4 * d_model for relu²."""
        cfg = _mini_cfg(ffn_type="relu2", d_ff=None)
        ffn = ReLUSquaredMLP(cfg, rngs=nnx.Rngs(0))
        assert ffn.d_ff == 4 * 32  # 4 * d_model

    def test_gradient_flows(self):
        cfg = _mini_cfg(ffn_type="relu2")
        ffn = ReLUSquaredMLP(cfg, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 32))

        grad_x = jax.grad(lambda x_in: ffn(x_in).sum())(x)
        assert jnp.isfinite(grad_x).all(), "Non-finite gradient through relu² MLP"


# ---------------------------------------------------------------------------
# Other FFN variants — shape and finiteness regression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ffn_cls,ffn_type", [
    (SwiGLUFFN, "swiglu"),
    (GeGLUFFN, "geglu"),
    (StandardMLP, "gelu"),
])
def test_ffn_shape(ffn_cls, ffn_type):
    cfg = _mini_cfg(ffn_type=ffn_type)
    ffn = ffn_cls(cfg, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 8, 32))
    out = ffn(x)
    assert out.shape == (2, 8, 32), f"{ffn_cls.__name__} shape mismatch"


@pytest.mark.parametrize("ffn_cls,ffn_type", [
    (SwiGLUFFN, "swiglu"),
    (GeGLUFFN, "geglu"),
    (StandardMLP, "gelu"),
])
def test_ffn_finite(ffn_cls, ffn_type):
    cfg = _mini_cfg(ffn_type=ffn_type)
    ffn = ffn_cls(cfg, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 32))
    out = ffn(x)
    assert jnp.isfinite(out).all(), f"{ffn_cls.__name__} non-finite output"
