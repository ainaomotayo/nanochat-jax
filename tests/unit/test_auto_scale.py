"""Tests for auto-scaling configuration from depth levels."""
from __future__ import annotations

import pytest

from nanochat.config.auto_scale import model_config_from_depth, training_config_from_depth
from nanochat.config.model_config import ModelConfig
from nanochat.config.training_config import TrainingConfig


class TestModelConfigFromDepth:
    """Test model_config_from_depth produces valid configs at each depth."""

    def test_depth_1_produces_small_model(self):
        cfg = model_config_from_depth(1)
        assert isinstance(cfg, ModelConfig)
        assert cfg.d_model == 128
        assert cfg.n_layers == 4
        assert cfg.n_heads == 4
        assert cfg.n_kv_heads == 4
        assert cfg.max_seq_len == 256

    def test_depth_10_produces_large_model(self):
        cfg = model_config_from_depth(10)
        assert isinstance(cfg, ModelConfig)
        assert cfg.d_model == 4096
        assert cfg.n_layers == 32
        assert cfg.n_heads == 32
        assert cfg.n_kv_heads == 8
        assert cfg.max_seq_len == 4096

    def test_all_depths_valid_configs(self):
        for depth in range(1, 11):
            cfg = model_config_from_depth(depth)
            assert isinstance(cfg, ModelConfig)
            # Verify d_model is divisible by n_heads
            assert cfg.d_model % cfg.n_heads == 0
            # Verify n_heads is divisible by n_kv_heads
            assert cfg.n_heads % cfg.n_kv_heads == 0
            # Verify d_ff was auto-computed
            assert cfg.d_ff is not None
            assert cfg.d_ff > 0
            # Verify d_head computed correctly
            assert cfg.d_head == cfg.d_model // cfg.n_heads

    def test_depth_out_of_range_raises(self):
        with pytest.raises(ValueError, match="depth must be an integer between 1 and 10"):
            model_config_from_depth(0)
        with pytest.raises(ValueError, match="depth must be an integer between 1 and 10"):
            model_config_from_depth(11)
        with pytest.raises(ValueError, match="depth must be an integer between 1 and 10"):
            model_config_from_depth(-1)

    def test_depth_non_integer_raises(self):
        with pytest.raises(ValueError, match="depth must be an integer between 1 and 10"):
            model_config_from_depth(2.5)  # type: ignore

    def test_overrides_applied(self):
        cfg = model_config_from_depth(1, vocab_size=512)
        assert cfg.vocab_size == 512
        assert cfg.d_model == 128  # unchanged from depth=1

    def test_monotonic_d_model(self):
        d_models = [model_config_from_depth(d).d_model for d in range(1, 11)]
        for i in range(1, len(d_models)):
            assert d_models[i] >= d_models[i - 1]


class TestTrainingConfigFromDepth:
    """Test training_config_from_depth produces valid configs."""

    def test_depth_1_training(self):
        cfg = training_config_from_depth(1)
        assert isinstance(cfg, TrainingConfig)
        assert cfg.learning_rate == 1e-3
        assert cfg.batch_size == 64
        assert cfg.warmup_steps == 200
        assert cfg.total_steps == 5_000

    def test_depth_10_training(self):
        cfg = training_config_from_depth(10)
        assert isinstance(cfg, TrainingConfig)
        assert cfg.learning_rate == 5e-5
        assert cfg.batch_size == 4
        assert cfg.total_steps == 500_000

    def test_all_depths_valid_training(self):
        for depth in range(1, 11):
            cfg = training_config_from_depth(depth)
            assert isinstance(cfg, TrainingConfig)
            assert cfg.learning_rate > 0
            assert cfg.batch_size > 0
            assert cfg.warmup_steps > 0
            assert cfg.total_steps > 0
            assert cfg.warmup_steps < cfg.total_steps

    def test_depth_out_of_range_raises(self):
        with pytest.raises(ValueError, match="depth must be an integer between 1 and 10"):
            training_config_from_depth(0)
        with pytest.raises(ValueError, match="depth must be an integer between 1 and 10"):
            training_config_from_depth(11)

    def test_lr_decreases_with_depth(self):
        lrs = [training_config_from_depth(d).learning_rate for d in range(1, 11)]
        # Overall trend: LR should decrease (first > last)
        assert lrs[0] > lrs[-1]

    def test_overrides_applied(self):
        cfg = training_config_from_depth(3, optimizer="adamw")
        assert cfg.optimizer == "adamw"
        assert cfg.batch_size == 32  # unchanged from depth=3
