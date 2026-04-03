"""Tests for full transformer model."""
import jax
import jax.numpy as jnp
from flax import nnx
from nanochat.model.transformer import TransformerLM


def test_forward_shape(nano_config):
    model = TransformerLM(nano_config, rngs=nnx.Rngs(params=0, dropout=1))
    ids = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    logits, caches = model(ids, deterministic=True)
    assert logits.shape == (1, 4, nano_config.vocab_size)
    assert len(caches) == nano_config.n_layers


def test_forward_finite(nano_config):
    model = TransformerLM(nano_config, rngs=nnx.Rngs(params=0, dropout=1))
    ids = jax.random.randint(jax.random.PRNGKey(0), (2, 8), 0, nano_config.vocab_size)
    logits, _ = model(ids, deterministic=True)
    assert jnp.isfinite(logits).all()


def test_gradient_flows(nano_config):
    model = TransformerLM(nano_config, rngs=nnx.Rngs(params=0, dropout=1))
    ids = jax.random.randint(jax.random.PRNGKey(0), (1, 8), 0, nano_config.vocab_size)

    def loss_fn(model):
        logits, _ = model(ids, deterministic=True)
        return logits.sum()

    grads = nnx.grad(loss_fn)(model)
    leaves = jax.tree.leaves(grads)
    assert len(leaves) > 0
    for leaf in leaves:
        if hasattr(leaf, 'shape'):
            assert jnp.isfinite(leaf).all()
