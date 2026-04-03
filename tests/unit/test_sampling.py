"""Tests for sampling strategies."""
import jax
import jax.numpy as jnp
from nanochat.inference.sampling import greedy_sample, temperature_sample, top_k_sample, top_p_sample


def test_greedy():
    logits = jnp.array([[0.0, 0.0, 10.0, 0.0]])
    tokens = greedy_sample(logits)
    assert int(tokens[0]) == 2


def test_temperature_zero_is_greedy():
    logits = jnp.array([[0.0, 0.0, 10.0, 0.0]])
    rng = jax.random.PRNGKey(0)
    tokens = temperature_sample(logits, rng, temperature=1e-6)
    assert int(tokens[0]) == 2


def test_top_k_filters():
    rng = jax.random.PRNGKey(42)
    logits = jnp.array([[10.0, 9.0, 1.0, 0.0, 0.0]])
    tokens = top_k_sample(logits, rng, k=2, temperature=0.1)
    assert int(tokens[0]) in [0, 1]


def test_top_p_filters():
    rng = jax.random.PRNGKey(42)
    logits = jnp.array([[10.0, 9.0, -10.0, -10.0, -10.0]])
    tokens = top_p_sample(logits, rng, p=0.9, temperature=0.1)
    assert int(tokens[0]) in [0, 1]
