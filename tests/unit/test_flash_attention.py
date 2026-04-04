"""Tests for flash attention module."""
import jax
import jax.numpy as jnp
import pytest

from nanochat.model.flash_attention import (
    get_attention_fn,
    naive_attention,
)


@pytest.fixture
def qkv():
    """Standard BHSD tensors for attention tests."""
    B, H, S, D = 2, 4, 16, 32
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (B, H, S, D))
    k = jax.random.normal(k2, (B, H, S, D))
    v = jax.random.normal(k3, (B, H, S, D))
    return q, k, v


@pytest.fixture
def causal_mask():
    """Causal boolean mask (B=1, 1, S, S)."""
    S = 16
    return jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))[None, None, :, :]


def test_naive_attention_shapes(qkv):
    """naive_attention should return the same shape as the query."""
    q, k, v = qkv
    out = naive_attention(q, k, v, mask=None, scale=None)
    assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"


def test_naive_attention_causal_mask(qkv, causal_mask):
    """With a causal mask, changing future keys should not affect past outputs."""
    q, k, v = qkv

    out1 = naive_attention(q, k, v, mask=causal_mask, scale=None)

    # Perturb the last 4 key/value positions.
    k2 = k.at[:, :, 12:, :].set(jax.random.normal(jax.random.PRNGKey(99), k[:, :, 12:, :].shape))
    v2 = v.at[:, :, 12:, :].set(jax.random.normal(jax.random.PRNGKey(100), v[:, :, 12:, :].shape))
    out2 = naive_attention(q, k2, v2, mask=causal_mask, scale=None)

    # Positions 0-11 should be identical (they can't attend to 12-15).
    assert jnp.allclose(out1[:, :, :12, :], out2[:, :, :12, :], atol=1e-5), (
        "Causal masking failed: past positions affected by future key changes"
    )


def test_get_attention_fn_returns_callable():
    """get_attention_fn should return a callable for both use_flash settings."""
    fn_flash = get_attention_fn(use_flash=True)
    fn_naive = get_attention_fn(use_flash=False)
    assert callable(fn_flash)
    assert callable(fn_naive)


def test_attention_output_finite(qkv, causal_mask):
    """All attention backends should produce finite outputs."""
    q, k, v = qkv
    fn = get_attention_fn(use_flash=True)
    out = fn(q, k, v, causal_mask, None)
    assert jnp.all(jnp.isfinite(out)), "Attention output contains non-finite values"


def test_gqa_broadcasting():
    """GQA: n_kv_heads < n_heads should broadcast correctly."""
    B, H, Kh, S, D = 2, 8, 2, 16, 32
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (B, H, S, D))
    k = jax.random.normal(k2, (B, Kh, S, D))
    v = jax.random.normal(k3, (B, Kh, S, D))

    out = naive_attention(q, k, v, mask=None, scale=None)
    assert out.shape == (B, H, S, D), f"GQA output shape mismatch: {out.shape}"
    assert jnp.all(jnp.isfinite(out)), "GQA output contains non-finite values"

    # Also test via get_attention_fn.
    fn = get_attention_fn(use_flash=True)
    out2 = fn(q, k, v, mask=None, scale=None)
    assert out2.shape == (B, H, S, D)
