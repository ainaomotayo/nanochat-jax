"""Tests for optimizer construction."""
from nanochat.config import TrainingConfig
from nanochat.training.optimizer import build_optimizer


def test_build_optimizer():
    cfg = TrainingConfig(learning_rate=1e-3, total_steps=100)
    tx = build_optimizer(cfg)
    assert tx is not None
