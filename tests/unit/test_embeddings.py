"""Tests for embedding layers."""
import jax
import jax.numpy as jnp
from flax import nnx
from nanochat.model.embeddings import TokenEmbedding, RotaryEmbedding


class TestTokenEmbedding:
    def test_output_shape(self):
        embed = TokenEmbedding(256, 64, 0.02, rngs=nnx.Rngs(0))
        ids = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        out = embed(ids)
        assert out.shape == (2, 3, 64)

    def test_as_logits_shape(self):
        embed = TokenEmbedding(256, 64, 0.02, rngs=nnx.Rngs(0))
        hidden = jnp.ones((2, 3, 64))
        logits = embed.as_logits(hidden)
        assert logits.shape == (2, 3, 256)


class TestRotaryEmbedding:
    def test_freqs_shape(self):
        rope = RotaryEmbedding(d_head=16, max_seq_len=32)
        cos, sin = rope.get_freqs(16)
        assert cos.shape == (16, 8)
        assert sin.shape == (16, 8)

    def test_apply_preserves_shape(self):
        rope = RotaryEmbedding(d_head=16, max_seq_len=32)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 8, 16))
        cos, sin = rope.get_freqs(8)
        out = RotaryEmbedding.apply(x, cos, sin)
        assert out.shape == x.shape

    def test_rotation_changes_values(self):
        rope = RotaryEmbedding(d_head=16, max_seq_len=32)
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, 4, 16))
        cos, sin = rope.get_freqs(4)
        out = RotaryEmbedding.apply(x, cos, sin)
        assert not jnp.allclose(x, out)
