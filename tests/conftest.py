"""Shared pytest fixtures for nanochat-jax tests.

Provides model configs and training configs for unit/integration tests.
The ``nano_config`` fixture matches nanochat-faithful defaults (relu²,
parameterless RMSNorm, QK norm, etc.) for correctness testing.
An ``ablation_config`` disables all nanochat extensions for isolated tests.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from nanochat.config import ModelConfig, TrainingConfig


@pytest.fixture(scope="session")
def nano_config() -> ModelConfig:
    """Tiny nanochat-faithful config for fast testing (~100K params)."""
    return ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        d_ff=256,
        max_seq_len=32,
        dropout_rate=0.0,
        norm_eps=1e-6,
        norm_type="rmsnorm",
        ffn_type="relu2",         # nanochat default
        pos_encoding="rope",
        rope_base=100_000.0,      # nanochat default
        tie_embeddings=False,     # nanochat default
        use_bias=False,
        init_std=0.02,
        use_qk_norm=True,
        qk_scale_factor=1.2,
        logit_softcap=30.0,
        use_value_embeddings=True,
        use_per_layer_scalars=True,
        use_smear=True,
        sliding_window_size=None,
    )


@pytest.fixture(scope="session")
def ablation_config() -> ModelConfig:
    """Config with all nanochat extensions disabled for isolated tests."""
    return ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        d_ff=256,
        max_seq_len=32,
        dropout_rate=0.0,
        ffn_type="gelu",
        rope_base=10000.0,
        tie_embeddings=True,
        use_qk_norm=False,
        logit_softcap=None,
        use_value_embeddings=False,
        use_per_layer_scalars=False,
        use_smear=False,
    )


@pytest.fixture(scope="session")
def gqa_config() -> ModelConfig:
    """Config with GQA (fewer KV heads) for testing grouped-query attention."""
    return ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        max_seq_len=32,
        dropout_rate=0.0,
        tie_embeddings=False,
        use_qk_norm=True,
        logit_softcap=30.0,
        use_value_embeddings=False,
        use_per_layer_scalars=False,
        use_smear=False,
    )


@pytest.fixture(scope="session")
def sliding_window_config() -> ModelConfig:
    """Config with sliding window attention for testing."""
    return ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        d_ff=256,
        max_seq_len=32,
        dropout_rate=0.0,
        ffn_type="relu2",
        tie_embeddings=False,
        use_qk_norm=True,
        logit_softcap=30.0,
        use_value_embeddings=False,
        use_per_layer_scalars=False,
        use_smear=False,
        sliding_window_size=8,
        n_global_tokens=1,
    )


@pytest.fixture(scope="session")
def train_config() -> TrainingConfig:
    """Small training config for testing."""
    return TrainingConfig(
        batch_size=2,
        optimizer="adamw",    # Use AdamW for test speed (Muon is slower to JIT)
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
def muon_train_config() -> TrainingConfig:
    """Training config using the Muon optimizer."""
    return TrainingConfig(
        batch_size=2,
        optimizer="muon",
        learning_rate=1e-3,
        warmup_steps=0,
        total_steps=10,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
        muon_weight_decay=0.01,
        dtype="float32",
        param_dtype="float32",
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
