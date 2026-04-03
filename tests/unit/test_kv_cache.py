"""Tests for KV cache."""
import jax.numpy as jnp
from nanochat.inference.kv_cache import KVCache, init_kv_caches
from nanochat.config import ModelConfig


def test_kv_cache_init():
    cache = KVCache.init(batch_size=2, n_kv_heads=4, max_len=32, d_head=16)
    assert cache.keys.shape == (2, 4, 32, 16)
    assert cache.position == 0


def test_kv_cache_update():
    cache = KVCache.init(2, 4, 32, 16)
    new_k = jnp.ones((2, 4, 1, 16))
    new_v = jnp.ones((2, 4, 1, 16))
    updated = cache.update(new_k, new_v)
    assert updated.position == 1
    assert jnp.allclose(updated.keys[:, :, 0:1, :], new_k)


def test_init_kv_caches():
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4, max_seq_len=32)
    caches = init_kv_caches(cfg, batch_size=1, max_len=32)
    assert len(caches) == 2
