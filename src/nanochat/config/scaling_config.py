"""Scaling-law experiment configuration for nanochat-jax.

Defines the :class:`ScalingConfig` Pydantic model that drives
multi-run scaling-law sweeps across model sizes, token budgets,
and compute budgets.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ScalingConfig(BaseModel):
    """Scaling-law experiment configuration.

    This configuration controls a sweep of training runs that vary
    model size and/or token budget in order to fit neural scaling laws.

    Attributes
    ----------
    type : str
        Scaling experiment type.  ``"scale_n"`` varies model parameters
        at a fixed token budget, ``"scale_d"`` varies data at a fixed
        model size, and ``"chinchilla"`` co-optimises both according to
        the Chinchilla ratio.
    description : str
        Free-text description of the experiment.
    model_configs : list[str]
        List of model scale names (e.g. ``["nano", "small", "medium"]``)
        to include in the sweep.
    fixed_token_budget : int | None
        A single token budget applied to every run (used with
        ``"scale_n"``).
    token_budgets : list[int] | None
        Per-run token budgets (used with ``"scale_d"``).
    compute_budgets_flops : list[float] | None
        Per-run compute budgets in FLOPs (used with ``"chinchilla"``).
    chinchilla_ratio : float
        Tokens-to-parameters ratio for Chinchilla-optimal allocation.
    seeds : list[int]
        Random seeds for each independent repetition of every run.
    output_dir : str
        Root directory for experiment artefacts and logs.
    """

    type: Literal["scale_n", "scale_d", "chinchilla"] = Field(
        default="scale_n",
        description=(
            "Scaling experiment type: 'scale_n' varies model size, "
            "'scale_d' varies data, 'chinchilla' co-optimises both."
        ),
    )
    description: str = Field(
        default="",
        description="Free-text description of the experiment.",
    )
    model_configs: list[str] = Field(
        default_factory=list,
        description=(
            'List of model scale names (e.g. ["nano", "small"]) '
            "to include in the sweep."
        ),
    )
    fixed_token_budget: Optional[int] = Field(
        default=None,
        ge=1,
        description="A single token budget applied to every run.",
    )
    token_budgets: Optional[list[int]] = Field(
        default=None,
        description="Per-run token budgets.",
    )
    compute_budgets_flops: Optional[list[float]] = Field(
        default=None,
        description="Per-run compute budgets in FLOPs.",
    )
    chinchilla_ratio: float = Field(
        default=20.0,
        gt=0.0,
        description="Tokens-to-parameters ratio for Chinchilla-optimal allocation.",
    )
    seeds: list[int] = Field(
        default_factory=lambda: [42],
        description="Random seeds for each independent repetition.",
    )
    output_dir: str = Field(
        default="outputs/scaling",
        min_length=1,
        description="Root directory for experiment artefacts and logs.",
    )
