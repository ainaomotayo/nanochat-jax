"""Data loading and tokenization configuration for nanochat-jax.

Defines the :class:`DataConfig` Pydantic model that specifies dataset
source, Hugging Face identifiers, tokenizer choice, caching behaviour,
and data-loading parallelism.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data pipeline configuration.

    Attributes
    ----------
    source : str
        Short identifier for the dataset source (used for logging
        and directory naming).
    hf_dataset : str
        Hugging Face dataset identifier
        (e.g. ``"Skylion007/openwebtext"``).
    hf_split : str
        Dataset split to use (e.g. ``"train"``).
    val_fraction : float
        Fraction of training data to reserve for validation.
    tokenizer_name : str
        Name of the tokenizer to use (e.g. ``"cl100k_base"`` for tiktoken
        or a Hugging Face tokenizer identifier).
    cache_dir : str
        Local directory for caching downloaded and tokenized data.
    max_samples : int | None
        Optional cap on the number of samples to use.  ``None`` means
        use the entire dataset.
    num_workers : int
        Number of parallel workers for data preprocessing.
    chunk_size : int
        Number of examples per processing chunk for map operations.
    """

    source: str = Field(
        default="openwebtext",
        min_length=1,
        description="Short identifier for the dataset source.",
    )
    hf_dataset: str = Field(
        default="Skylion007/openwebtext",
        min_length=1,
        description="Hugging Face dataset identifier.",
    )
    hf_split: str = Field(
        default="train",
        min_length=1,
        description='Dataset split to use (e.g. "train").',
    )
    val_fraction: float = Field(
        default=0.005,
        ge=0.0,
        lt=1.0,
        description="Fraction of training data to reserve for validation.",
    )
    tokenizer_name: str = Field(
        default="cl100k_base",
        min_length=1,
        description="Name of the tokenizer to use.",
    )
    cache_dir: str = Field(
        default="data/",
        min_length=1,
        description="Local directory for caching downloaded and tokenized data.",
    )
    max_samples: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Optional cap on the number of samples to use. "
            "None means use the entire dataset."
        ),
    )
    num_workers: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Number of parallel workers for data preprocessing.",
    )
    chunk_size: int = Field(
        default=10000,
        ge=1,
        le=10_000_000,
        description="Number of examples per processing chunk for map operations.",
    )
