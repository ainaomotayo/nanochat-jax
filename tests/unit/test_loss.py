"""Tests for loss functions."""
import jax.numpy as jnp
from nanochat.training.loss import cross_entropy_loss, IGNORE_INDEX


def test_basic_loss():
    logits = jnp.zeros((2, 4, 10))
    labels = jnp.zeros((2, 4), dtype=jnp.int32)
    loss, metrics = cross_entropy_loss(logits, labels)
    assert loss.shape == ()
    assert float(loss) > 0


def test_masked_loss():
    logits = jnp.zeros((1, 4, 10))
    labels = jnp.array([[0, 1, IGNORE_INDEX, IGNORE_INDEX]], dtype=jnp.int32)
    loss, metrics = cross_entropy_loss(logits, labels)
    assert float(metrics["n_tokens"]) == 2.0


def test_z_loss():
    logits = jnp.ones((1, 4, 10)) * 10.0
    labels = jnp.zeros((1, 4), dtype=jnp.int32)
    _, metrics_no_z = cross_entropy_loss(logits, labels, z_loss_weight=0.0)
    loss_z, metrics_z = cross_entropy_loss(logits, labels, z_loss_weight=1e-4)
    assert float(metrics_z["z_loss"]) > 0
