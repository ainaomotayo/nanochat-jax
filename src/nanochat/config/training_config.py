"""Training hyperparameter configuration for nanochat-jax.

Defines the :class:`TrainingConfig` Pydantic model that captures all
training-loop knobs: optimizer settings, learning-rate schedule,
gradient clipping, checkpointing, evaluation cadence, and numerical
precision.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator, computed_field


class TrainingConfig(BaseModel):
    """Training hyperparameter configuration.

    Attributes
    ----------
    batch_size : int
        Per-device micro-batch size (in sequences).
    gradient_accumulation_steps : int
        Number of micro-batches accumulated before a weight update.
    optimizer : str
        Optimizer name — ``"adamw"`` or ``"sgd"``.
    learning_rate : float
        Peak learning rate after warmup.
    weight_decay : float
        L2 / decoupled weight-decay coefficient.
    beta1 : float
        Adam first-moment exponential decay rate.
    beta2 : float
        Adam second-moment exponential decay rate.
    epsilon : float
        Adam numerical stability constant.
    grad_clip_norm : float
        Maximum global gradient norm for clipping.
    warmup_steps : int
        Number of linear warmup steps.
    total_steps : int
        Total number of training steps.
    lr_decay_steps : int | None
        Steps over which the cosine decay runs.  Defaults to
        *total_steps* when ``None``.
    min_lr_ratio : float
        Ratio of minimum LR to peak LR at end of decay.
    token_budget : int | None
        Optional total token budget for the run.
    checkpoint_dir : str
        Directory for saving checkpoints.
    save_every_steps : int
        Checkpoint saving frequency in steps.
    keep_last_n : int
        Number of most-recent checkpoints to retain.
    resume_from : str | None
        Path to a checkpoint to resume training from.
    eval_every_steps : int
        Evaluation frequency in steps.
    eval_steps : int
        Number of evaluation batches per evaluation run.
    dtype : str
        Compute dtype for forward/backward passes.
    param_dtype : str
        Parameter storage dtype.
    """

    batch_size: int = Field(
        default=32,
        ge=1,
        le=65536,
        description="Per-device micro-batch size (in sequences).",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        le=4096,
        description="Number of micro-batches accumulated before a weight update.",
    )
    optimizer: Literal["adamw", "sgd"] = Field(
        default="adamw",
        description='Optimizer name — "adamw" or "sgd".',
    )
    learning_rate: float = Field(
        default=3e-4,
        gt=0.0,
        le=10.0,
        description="Peak learning rate after warmup.",
    )
    weight_decay: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="L2 / decoupled weight-decay coefficient.",
    )
    beta1: float = Field(
        default=0.9,
        ge=0.0,
        lt=1.0,
        description="Adam first-moment exponential decay rate.",
    )
    beta2: float = Field(
        default=0.95,
        ge=0.0,
        lt=1.0,
        description="Adam second-moment exponential decay rate.",
    )
    epsilon: float = Field(
        default=1e-8,
        gt=0.0,
        le=1e-1,
        description="Adam numerical stability constant.",
    )
    grad_clip_norm: float = Field(
        default=1.0,
        gt=0.0,
        le=1000.0,
        description="Maximum global gradient norm for clipping.",
    )
    warmup_steps: int = Field(
        default=2000,
        ge=0,
        le=10_000_000,
        description="Number of linear warmup steps.",
    )
    total_steps: int = Field(
        default=100_000,
        ge=1,
        le=100_000_000,
        description="Total number of training steps.",
    )
    lr_decay_steps: Optional[int] = Field(
        default=None,
        ge=1,
        le=100_000_000,
        description=(
            "Steps over which the cosine decay runs. "
            "Defaults to total_steps when None."
        ),
    )
    min_lr_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Ratio of minimum LR to peak LR at end of decay.",
    )
    token_budget: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional total token budget for the run.",
    )
    checkpoint_dir: str = Field(
        default="checkpoints/",
        min_length=1,
        description="Directory for saving checkpoints.",
    )
    save_every_steps: int = Field(
        default=5000,
        ge=1,
        le=10_000_000,
        description="Checkpoint saving frequency in steps.",
    )
    keep_last_n: int = Field(
        default=3,
        ge=1,
        le=1000,
        description="Number of most-recent checkpoints to retain.",
    )
    resume_from: Optional[str] = Field(
        default=None,
        description="Path to a checkpoint to resume training from.",
    )
    eval_every_steps: int = Field(
        default=1000,
        ge=1,
        le=10_000_000,
        description="Evaluation frequency in steps.",
    )
    eval_steps: int = Field(
        default=100,
        ge=1,
        le=100_000,
        description="Number of evaluation batches per evaluation run.",
    )
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="bfloat16",
        description="Compute dtype for forward/backward passes.",
    )
    param_dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="float32",
        description="Parameter storage dtype.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _default_lr_decay_steps(self) -> "TrainingConfig":
        """Set *lr_decay_steps* to *total_steps* when not provided."""
        if self.lr_decay_steps is None:
            self.lr_decay_steps = self.total_steps
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def effective_batch_size(self) -> int:
        """Effective batch size accounting for gradient accumulation.

        Returns ``batch_size * gradient_accumulation_steps``.
        """
        return self.batch_size * self.gradient_accumulation_steps

    # ------------------------------------------------------------------
    # Utility class methods
    # ------------------------------------------------------------------

    @classmethod
    def compute_total_steps(
        cls,
        token_budget: int,
        seq_len: int,
        batch_size: int,
    ) -> int:
        """Derive the number of training steps from a token budget.

        Parameters
        ----------
        token_budget:
            Total number of tokens to train on.
        seq_len:
            Sequence length per example.
        batch_size:
            Effective batch size (micro-batch * gradient accumulation).

        Returns
        -------
        int
            The number of training steps required to exhaust the budget,
            rounded up.

        Raises
        ------
        ValueError
            If any argument is non-positive.
        """
        if token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        tokens_per_step = seq_len * batch_size
        return math.ceil(token_budget / tokens_per_step)
