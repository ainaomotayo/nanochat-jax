"""Training hyperparameter configuration for nanochat-jax.

Supports both AdamW (default for ablations) and Muon (nanochat default
for production runs) optimizers.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator, computed_field


class TrainingConfig(BaseModel):
    """Training hyperparameter configuration.

    Two optimizer modes:
    - ``"adamw"``: Standard AdamW. Good for baselines and ablations.
    - ``"muon"``: Muon (Momentum + Newton-Schulz orthogonalization).
      nanochat default. Better scaling behavior on weight matrices.
    """

    # ------------------------------------------------------------------
    # Batch configuration
    # ------------------------------------------------------------------
    batch_size: int = Field(
        default=32, ge=1, le=65536,
        description="Per-device micro-batch size (in sequences).",
    )
    gradient_accumulation_steps: int = Field(
        default=1, ge=1, le=4096,
        description="Number of micro-batches accumulated before a weight update.",
    )

    # ------------------------------------------------------------------
    # Optimizer selection
    # ------------------------------------------------------------------
    optimizer: Literal["adamw", "muon", "sgd"] = Field(
        default="muon",
        description='Optimizer: "muon" (nanochat default), "adamw", or "sgd".',
    )

    # ------------------------------------------------------------------
    # Learning rate and schedule
    # ------------------------------------------------------------------
    learning_rate: float = Field(
        default=3e-4, gt=0.0, le=10.0,
        description="Peak learning rate after warmup.",
    )
    warmup_steps: int = Field(
        default=2000, ge=0, le=10_000_000,
        description="Number of linear warmup steps.",
    )
    total_steps: int = Field(
        default=100_000, ge=1, le=100_000_000,
        description="Total number of training steps.",
    )
    lr_decay_steps: Optional[int] = Field(
        default=None, ge=1, le=100_000_000,
        description="Steps over which cosine decay runs. Defaults to total_steps.",
    )
    min_lr_ratio: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Ratio of minimum LR to peak LR at end of decay.",
    )

    # ------------------------------------------------------------------
    # AdamW hyperparameters (used when optimizer == "adamw")
    # ------------------------------------------------------------------
    weight_decay: float = Field(
        default=0.1, ge=0.0, le=10.0,
        description="Decoupled weight-decay coefficient (AdamW).",
    )
    beta1: float = Field(
        default=0.9, ge=0.0, lt=1.0,
        description="Adam first-moment decay rate.",
    )
    beta2: float = Field(
        default=0.95, ge=0.0, lt=1.0,
        description="Adam second-moment decay rate.",
    )
    epsilon: float = Field(
        default=1e-8, gt=0.0, le=1e-1,
        description="Adam numerical stability constant.",
    )

    # ------------------------------------------------------------------
    # Muon hyperparameters (used when optimizer == "muon")
    # ------------------------------------------------------------------
    muon_momentum: float = Field(
        default=0.95, ge=0.0, lt=1.0,
        description="Muon momentum coefficient. nanochat default: 0.95.",
    )
    muon_nesterov: bool = Field(
        default=True,
        description="Use Nesterov-style gradient in Muon.",
    )
    muon_ns_steps: int = Field(
        default=5, ge=1, le=20,
        description="Newton-Schulz orthogonalization iteration count.",
    )
    muon_weight_decay: float = Field(
        default=0.01, ge=0.0, le=10.0,
        description="Decoupled weight decay for Muon.",
    )

    # ------------------------------------------------------------------
    # Gradient clipping (shared by all optimizers)
    # ------------------------------------------------------------------
    grad_clip_norm: float = Field(
        default=1.0, gt=0.0, le=1000.0,
        description="Maximum global gradient norm for clipping.",
    )

    # ------------------------------------------------------------------
    # Token budget
    # ------------------------------------------------------------------
    token_budget: Optional[int] = Field(
        default=None, ge=1,
        description="Optional total token budget. Overrides total_steps if set.",
    )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    checkpoint_dir: str = Field(
        default="checkpoints/", min_length=1,
        description="Directory for saving checkpoints.",
    )
    save_every_steps: int = Field(
        default=5000, ge=1, le=10_000_000,
        description="Checkpoint saving frequency in steps.",
    )
    keep_last_n: int = Field(
        default=3, ge=1, le=1000,
        description="Number of most-recent checkpoints to retain.",
    )
    resume_from: Optional[str] = Field(
        default=None,
        description="Path to a checkpoint to resume training from.",
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    eval_every_steps: int = Field(
        default=1000, ge=1, le=10_000_000,
        description="Evaluation frequency in steps.",
    )
    eval_steps: int = Field(
        default=100, ge=1, le=100_000,
        description="Number of evaluation batches per evaluation run.",
    )

    # ------------------------------------------------------------------
    # Numerical precision
    # ------------------------------------------------------------------
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
        if self.lr_decay_steps is None:
            self.lr_decay_steps = self.total_steps
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def effective_batch_size(self) -> int:
        """Effective batch size = batch_size * gradient_accumulation_steps."""
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
        """Derive training steps from a token budget."""
        if token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return math.ceil(token_budget / (seq_len * batch_size))

    @classmethod
    def for_scale_experiment(
        cls,
        token_budget: int,
        seq_len: int,
        batch_size: int = 32,
        **overrides,
    ) -> "TrainingConfig":
        """Build a TrainingConfig for a scaling law experiment run.

        Computes total_steps from the token budget and sets reasonable
        warmup (5% of total steps, capped at 2000).

        Args:
            token_budget: Total tokens to train on.
            seq_len: Sequence length per example.
            batch_size: Effective batch size.
            **overrides: Additional field overrides.
        """
        total = cls.compute_total_steps(token_budget, seq_len, batch_size)
        warmup = min(2000, max(100, total // 20))
        return cls(
            total_steps=total,
            warmup_steps=warmup,
            batch_size=batch_size,
            **overrides,
        )
