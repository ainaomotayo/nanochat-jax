"""Tests for attention module."""
import jax
import jax.numpy as jnp
from flax import nnx
from nanochat.config import ModelConfig
from nanochat.model.attention import GroupedQueryAttention
from nanochat.model.embeddings import RotaryEmbedding


def test_attention_output_shape(nano_config):
    rngs = nnx.Rngs(params=0, dropout=1)
    attn = GroupedQueryAttention(nano_config, rngs=rngs)
    B, S, D = 2, 8, nano_config.d_model
    x = jax.random.normal(jax.random.PRNGKey(0), (B, S, D))
    rope = RotaryEmbedding(nano_config.d_head, nano_config.max_seq_len)
    cos, sin = rope.get_freqs(S)
    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))[None, None]
    out, cache = attn(x, cos, sin, mask)
    assert out.shape == (B, S, D)
    assert cache is not None


def test_gqa_attention(gqa_config):
    rngs = nnx.Rngs(params=0, dropout=1)
    attn = GroupedQueryAttention(gqa_config, rngs=rngs)
    B, S, D = 2, 8, gqa_config.d_model
    x = jax.random.normal(jax.random.PRNGKey(0), (B, S, D))
    rope = RotaryEmbedding(gqa_config.d_head, gqa_config.max_seq_len)
    cos, sin = rope.get_freqs(S)
    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))[None, None]
    out, cache = attn(x, cos, sin, mask)
    assert out.shape == (B, S, D)


def test_attention_causal(nano_config):
    """Verify causal masking: changing future tokens shouldn't affect past outputs."""
    rngs = nnx.Rngs(params=0, dropout=1)
    attn = GroupedQueryAttention(nano_config, rngs=rngs)
    B, S, D = 1, 8, nano_config.d_model
    x1 = jax.random.normal(jax.random.PRNGKey(0), (B, S, D))
    x2 = x1.at[:, 4:, :].set(jax.random.normal(jax.random.PRNGKey(1), (B, 4, D)))
    rope = RotaryEmbedding(nano_config.d_head, nano_config.max_seq_len)
    cos, sin = rope.get_freqs(S)
    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))[None, None]
    out1, _ = attn(x1, cos, sin, mask)
    out2, _ = attn(x2, cos, sin, mask)
    # First 4 positions should be identical
    assert jnp.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)
