"""Configuration module for nanochat-jax.

This module provides structured, validated configuration classes for all
components of the nanochat training pipeline, including model architecture,
training hyperparameters, data loading, and scaling law experiments.

All configuration classes use Pydantic v2 for runtime validation, type
checking, and serialization support.
"""

from nanochat.config.model_config import ModelConfig
from nanochat.config.training_config import TrainingConfig
from nanochat.config.data_config import DataConfig
from nanochat.config.scaling_config import ScalingConfig

__all__ = ["ModelConfig", "TrainingConfig", "DataConfig", "ScalingConfig"]
