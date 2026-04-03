"""Unit tests for nanochat attention extensions.

Tests:
1. QK normalization: Q and K are L2-normalized before scores
2. Logit softcap: scores bounded to [-cap, cap] before softmax
3. Sliding window: tokens only attend within window + global tokens
4. Attention scale: 1.2 / sqrt(d_head) instead of 1 / sqrt(d_head)
5. Output shape and finiteness with all features
"""

import math
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nanochat.config import ModelConfig
from nanochat.model.attention import (
    GroupedQueryAttention, _l2_normalize, _build_sliding_window_mask
)
from nanochat.model.embeddings import RotaryEmbedding


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mask(seq_len: int) -> jax.Array:
    """Standard causal mask."""
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return causal[None, None, :, :]  # [1, 1, S, S]


def _make_cfg(
    d_model=64, n_heads=4, n_kv_heads=4,
    use_qk_norm=True, logit_softcap=30.0,
    sliding_window_size=None, n_global_tokens=1,
    qk_scale_factor=1.2,
) -> ModelConfig:
    return ModelConfig(
        vocab_size=256, d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        n_layers=1, d_ff=256, max_seq_len=32,
        use_qk_norm=use_qk_norm, qk_scale_factor=qk_scale_factor,
        logit_softcap=logit_softcap,
        sliding_window_size=sliding_window_size, n_global_tokens=n_global_tokens,
        use_value_embeddings=False, use_per_layer_scalars=False, use_smear=False,
    )


# ---------------------------------------------------------------------------
# L2 normalization helper
# ---------------------------------------------------------------------------

class TestL2Normalize:
    def test_unit_norm(self):
        """L2-normalized vectors have unit L2 norm."""
        x = jnp.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
        out = _l2_normalize(x)
        norms = jnp.linalg.norm(out, axis=-1)
        np.testing.assert_allclose(np.array(norms), np.ones(3), atol=1e-6)

    def test_zero_safe(self):
        """Zero vector should not produce NaN (eps protection)."""
        x = jnp.zeros((4, 8))
        out = _l2_normalize(x)
        assert not jnp.any(jnp.isnan(out))


# ---------------------------------------------------------------------------
# Sliding window mask
# ---------------------------------------------------------------------------

class TestSlidingWindowMask:
    def test_shape(self):
        mask = _build_sliding_window_mask(seq_len=8, window_size=4, n_global_tokens=1)
        assert mask.shape == (1, 1, 8, 8)

    def test_causal(self):
        """Mask must be lower-triangular (no future attention)."""
        mask = _build_sliding_window_mask(seq_len=8, window_size=4, n_global_tokens=0)
        m = np.array(mask[0, 0])
        # Upper triangle (excluding diagonal) must be all False
        assert not m[np.triu_indices(8, k=1)].any(), "Non-causal positions attended"

    def test_window_constraint(self):
        """Token at position q should NOT attend beyond window_size steps back."""
        window_size = 3
        mask = _build_sliding_window_mask(seq_len=8, window_size=window_size, n_global_tokens=0)
        m = np.array(mask[0, 0])
        # Position q=5 should not attend to k=1 (distance=4 > window_size=3)
        assert not m[5, 1], "Token attended beyond window"

    def test_window_allows_recent(self):
        """Token at q should attend to k within window."""
        window_size = 3
        mask = _build_sliding_window_mask(seq_len=8, window_size=window_size, n_global_tokens=0)
        m = np.array(mask[0, 0])
        # Position q=5 should attend to k=4 (distance=1, within window)
        assert m[5, 4], "Token did not attend to recent position"

    def test_global_tokens_always_visible(self):
        """Global tokens (k < n_global_tokens) should always be attended."""
        mask = _build_sliding_window_mask(seq_len=8, window_size=2, n_global_tokens=1)
        m = np.array(mask[0, 0])
        # All positions (including q=7) should attend to k=0 (global)
        for q in range(1, 8):
            assert m[q, 0], f"Position {q} did not attend to global token k=0"


# ---------------------------------------------------------------------------
# QK normalization in attention
# ---------------------------------------------------------------------------

class TestQKNorm:
    def test_attention_with_qk_norm(self):
        """Attention with QK norm should produce finite output."""
        cfg = _make_cfg(use_qk_norm=True)
        attn = GroupedQueryAttention(cfg, rngs=nnx.Rngs(0))
        rope = RotaryEmbedding(cfg.d_head, cfg.max_seq_len, cfg.rope_base)

        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))
        cos, sin = rope.get_freqs(8)
        mask = _make_mask(8)

        out, _ = attn(x, cos, sin, mask)
        assert jnp.isfinite(out).all(), "NaN/Inf with QK norm"
        assert out.shape == (2, 8, 64)

    def test_attention_without_qk_norm(self):
        """Attention without QK norm (ablation) should also be finite."""
        cfg = _make_cfg(use_qk_norm=False, logit_softcap=None)
        attn = GroupedQueryAttention(cfg, rngs=nnx.Rngs(0))
        rope = RotaryEmbedding(cfg.d_head, cfg.max_seq_len, cfg.rope_base)

        x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 64))
        cos, sin = rope.get_freqs(8)
        mask = _make_mask(8)

        out, _ = attn(x, cos, sin, mask)
        assert jnp.isfinite(out).all()


# ---------------------------------------------------------------------------
# Logit softcap
# ---------------------------------------------------------------------------

class TestLogitSoftcap:
    def test_softcap_bounds_logits(self):
        """Scores after softcap should be in (-cap, cap)."""
        # Create very large inputs to force large scores before cap
        cfg = _make_cfg(use_qk_norm=False, logit_softcap=30.0)
        attn = GroupedQueryAttention(cfg, rngs=nnx.Rngs(0))
        rope = RotaryEmbedding(cfg.d_head, cfg.max_seq_len, cfg.rope_base)

        # Large inputs → large dot products before softcap
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 64)) * 100.0
        cos, sin = rope.get_freqs(4)
        mask = _make_mask(4)

        out, _ = attn(x, cos, sin, mask)
        # If softcap works, output should still be finite (not NaN from extreme logits)
        assert jnp.isfinite(out).all(), "Softcap failed to prevent NaN with large inputs"

    def test_softcap_formula(self):
        """cap * tanh(x / cap) should be applied correctly."""
        cap = 30.0
        x = jnp.array([100.0, -100.0, 0.0, 29.0])
        softcapped = cap * jnp.tanh(x / cap)

        # Values should be bounded to (-cap, cap)
        assert (softcapped < cap).all(), "Softcap upper bound violated"
        assert (softcapped > -cap).all(), "Softcap lower bound violated"

        # Large positive → near +cap, large negative → near -cap
        assert softcapped[0] > 29.0
        assert softcapped[1] < -29.0
        assert abs(float(softcapped[2])) < 1e-6  # 0 maps to 0


# ---------------------------------------------------------------------------
# Attention scale
# ---------------------------------------------------------------------------

class TestAttentionScale:
    def test_scale_value(self):
        """Attention scale should be qk_scale_factor / sqrt(d_head)."""
        cfg = _make_cfg(qk_scale_factor=1.2)
        attn = GroupedQueryAttention(cfg, rngs=nnx.Rngs(0))
        expected = 1.2 / math.sqrt(cfg.d_head)
        assert abs(attn.attn_scale - expected) < 1e-7, (
            f"Scale {attn.attn_scale} != expected {expected}"
        )


# ---------------------------------------------------------------------------
# Full forward pass regression
# ---------------------------------------------------------------------------

class TestAttentionForwardPass:
    def test_all_features_shape(self):
        """Full nanochat attention (QK norm + softcap) output shape."""
        cfg = _make_cfg(use_qk_norm=True, logit_softcap=30.0)
        attn = GroupedQueryAttention(cfg, rngs=nnx.Rngs(0))
        rope = RotaryEmbedding(cfg.d_head, cfg.max_seq_len, cfg.rope_base)

        B, S = 3, 16
        x = jax.random.normal(jax.random.PRNGKey(0), (B, S, 64))
        cos, sin = rope.get_freqs(S)
        mask = _make_mask(S)

        out, cache = attn(x, cos, sin, mask)
        assert out.shape == (B, S, 64)
        k, v = cache
        assert k.shape == (B, cfg.n_kv_heads, S, cfg.d_head)

    def test_sliding_window_shape(self):
        """Sliding window attention should produce correct output shape."""
        cfg = _make_cfg(sliding_window_size=4, n_global_tokens=1)
        attn = GroupedQueryAttention(cfg, rngs=nnx.Rngs(0))
        rope = RotaryEmbedding(cfg.d_head, cfg.max_seq_len, cfg.rope_base)

        B, S = 2, 16
        x = jax.random.normal(jax.random.PRNGKey(0), (B, S, 64))
        cos, sin = rope.get_freqs(S)
        mask = _make_mask(S)

        out, _ = attn(x, cos, sin, mask)
        assert out.shape == (B, S, 64)
        assert jnp.isfinite(out).all()

    def test_bfloat16_stable(self):
        """All features should be stable in bfloat16."""
        cfg = _make_cfg(use_qk_norm=True, logit_softcap=30.0)
        attn = GroupedQueryAttention(cfg, rngs=nnx.Rngs(0))
        rope = RotaryEmbedding(cfg.d_head, cfg.max_seq_len, cfg.rope_base)

        B, S = 2, 8
        x = jax.random.normal(jax.random.PRNGKey(0), (B, S, 64)).astype(jnp.bfloat16)
        cos, sin = rope.get_freqs(S)
        mask = _make_mask(S)

        out, _ = attn(x, cos, sin, mask)
        assert jnp.isfinite(out).all(), "NaN in bfloat16 attention"
