"""Tests for LR scheduler."""
from nanochat.training.scheduler import build_lr_schedule


def test_schedule_warmup():
    schedule = build_lr_schedule(learning_rate=1e-3, warmup_steps=10, total_steps=100)
    lr_0 = schedule(0)
    lr_10 = schedule(10)
    assert float(lr_0) < float(lr_10)


def test_schedule_decay():
    schedule = build_lr_schedule(learning_rate=1e-3, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    lr_peak = schedule(10)
    lr_end = schedule(100)
    assert float(lr_end) < float(lr_peak)
