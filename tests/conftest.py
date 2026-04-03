"""Shared pytest fixtures for nanochat-jax tests."""
from __future__ import annotations
import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from nanochat.config import ModelConfig, TrainingConfig


@pytest.fixture(scope="session")
def nano_config() -> ModelConfig:
    """Tiny model config for fast testing: ~100K params."""
    return ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        d_ff=128,
        max_seq_len=32,
        dropout_rate=0.0,
        norm_eps=1e-6,
        norm_type="rmsnorm",
        ffn_type="swiglu",
        pos_encoding="rope",
        rope_base=10000.0,
        tie_embeddings=True,
        use_bias=False,
        init_std=0.02,
    )


@pytest.fixture(scope="session")
def gqa_config() -> ModelConfig:
    """Config with GQA (fewer KV heads) for testing."""
    return ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=32,
        dropout_rate=0.0,
        tie_embeddings=True,
        use_bias=False,
    )


@pytest.fixture(scope="session")
def train_config() -> TrainingConfig:
    """Small training config for testing."""
    return TrainingConfig(
        batch_size=2,
        learning_rate=1e-3,
        warmup_steps=0,
        total_steps=50,
        weight_decay=0.0,
        dtype="float32",
        param_dtype="float32",
        eval_every_steps=25,
        save_every_steps=100,
    )


@pytest.fixture(scope="session")
def rngs():
    return nnx.Rngs(params=0, dropout=1)


@pytest.fixture
def dummy_batch(nano_config):
    """Random batch matching nano_config dimensions."""
    B, S = 2, nano_config.max_seq_len
    rng = jax.random.PRNGKey(42)
    return {
        "input_ids": jax.random.randint(rng, (B, S), 0, nano_config.vocab_size),
        "labels": jax.random.randint(rng, (B, S), 0, nano_config.vocab_size),
        "attention_mask": jnp.ones((B, S), dtype=jnp.int32),
    }
