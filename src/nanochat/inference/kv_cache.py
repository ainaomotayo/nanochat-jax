"""KV Cache for autoregressive inference."""
from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from nanochat.config import ModelConfig


@dataclass
class KVCache:
    """Key-Value cache for a single transformer layer.

    Pre-allocated fixed-size arrays. Position pointer tracks fill level.
    Uses jax.lax.dynamic_update_slice for XLA-compatible updates.
    """
    keys: jax.Array      # [batch, n_kv_heads, max_len, d_head]
    values: jax.Array    # [batch, n_kv_heads, max_len, d_head]
    position: int

    @staticmethod
    def init(batch_size: int, n_kv_heads: int, max_len: int, d_head: int,
             dtype: jnp.dtype = jnp.bfloat16) -> KVCache:
        """Create empty KV cache."""
        return KVCache(
            keys=jnp.zeros((batch_size, n_kv_heads, max_len, d_head), dtype=dtype),
            values=jnp.zeros((batch_size, n_kv_heads, max_len, d_head), dtype=dtype),
            position=0,
        )

    def update(self, new_k: jax.Array, new_v: jax.Array) -> KVCache:
        """Insert new K/V at current position. Returns updated cache.
        new_k, new_v: [batch, n_kv_heads, seq_new, d_head]
        """
        seq_new = new_k.shape[2]
        keys = jax.lax.dynamic_update_slice(
            self.keys, new_k, (0, 0, self.position, 0)
        )
        values = jax.lax.dynamic_update_slice(
            self.values, new_v, (0, 0, self.position, 0)
        )
        return KVCache(keys=keys, values=values, position=self.position + seq_new)

    def get_valid(self) -> tuple[jax.Array, jax.Array]:
        """Return filled portion of cache."""
        return (
            jax.lax.dynamic_slice(self.keys, (0, 0, 0, 0),
                                   (self.keys.shape[0], self.keys.shape[1], self.position, self.keys.shape[3])),
            jax.lax.dynamic_slice(self.values, (0, 0, 0, 0),
                                   (self.values.shape[0], self.values.shape[1], self.position, self.values.shape[3])),
        )


def init_kv_caches(cfg: ModelConfig, batch_size: int, max_len: int,
                   dtype: jnp.dtype = jnp.bfloat16) -> list[KVCache]:
    """Initialize one KVCache per transformer layer."""
    d_head = cfg.d_model // cfg.n_heads
    return [
        KVCache.init(batch_size, cfg.n_kv_heads, max_len, d_head, dtype)
        for _ in range(cfg.n_layers)
    ]
