"""Training module for nanochat-jax."""
from nanochat.training.loss import cross_entropy_loss
from nanochat.training.optimizer import build_optimizer
from nanochat.training.scheduler import build_lr_schedule
from nanochat.training.trainer import Trainer
from nanochat.training.checkpoint import CheckpointManager
from nanochat.training.rl_trainer import GRPOConfig, GRPOTrainer, RewardFunction

__all__ = [
    "cross_entropy_loss", "build_optimizer", "build_lr_schedule",
    "Trainer", "CheckpointManager",
    "GRPOConfig", "GRPOTrainer", "RewardFunction",
]
