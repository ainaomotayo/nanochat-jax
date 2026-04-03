"""Model architecture configuration for nanochat-jax.

Defines the :class:`ModelConfig` Pydantic model that fully specifies
transformer architecture hyperparameters, including attention layout,
feed-forward network type, positional encoding, and weight initialization.

Predefined scale presets (nano through xlarge) are available via
:meth:`ModelConfig.for_scale`.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator, computed_field


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round *value* up to the nearest multiple of *multiple*.

    Parameters
    ----------
    value:
        The integer value to round.
    multiple:
        The alignment boundary.

    Returns
    -------
    int
        The smallest integer >= *value* that is divisible by *multiple*.
    """
    return math.ceil(value / multiple) * multiple


class ModelConfig(BaseModel):
    """Transformer model architecture configuration.

    All fields carry sensible defaults that correspond to a small
    decoder-only transformer.  Use :meth:`for_scale` to obtain
    battle-tested presets for various model sizes.

    Attributes
    ----------
    vocab_size : int
        Size of the token vocabulary.
    d_model : int
        Hidden dimension of the transformer.
    n_layers : int
        Number of transformer decoder layers.
    n_heads : int
        Number of query attention heads.
    n_kv_heads : int
        Number of key/value attention heads (for GQA / MQA).
    d_ff : int | None
        Feed-forward intermediate dimension.  When ``None`` it is
        automatically computed based on *ffn_type*.
    max_seq_len : int
        Maximum sequence length the model can process.
    dropout_rate : float
        Dropout probability applied throughout the model.
    norm_eps : float
        Epsilon for layer normalization variants.
    norm_type : str
        Normalization type — ``"rmsnorm"`` or ``"layernorm"``.
    ffn_type : str
        Feed-forward block type — ``"swiglu"`` or ``"gelu"``.
    pos_encoding : str
        Positional encoding scheme — ``"rope"`` or ``"learned"``.
    rope_base : float
        Base frequency for Rotary Position Embeddings.
    tie_embeddings : bool
        Whether to tie input and output embedding weights.
    use_bias : bool
        Whether to use bias terms in linear projections.
    init_std : float
        Standard deviation for weight initialization.
    output_logits_scale : float | None
        Optional scaling factor applied to output logits before softmax.
    """

    vocab_size: int = Field(
        default=32000,
        ge=1,
        le=1_000_000,
        description="Size of the token vocabulary.",
    )
    d_model: int = Field(
        default=512,
        ge=1,
        le=65536,
        description="Hidden dimension of the transformer.",
    )
    n_layers: int = Field(
        default=6,
        ge=1,
        le=1024,
        description="Number of transformer decoder layers.",
    )
    n_heads: int = Field(
        default=8,
        ge=1,
        le=1024,
        description="Number of query attention heads.",
    )
    n_kv_heads: int = Field(
        default=8,
        ge=1,
        le=1024,
        description="Number of key/value attention heads (for GQA / MQA).",
    )
    d_ff: Optional[int] = Field(
        default=None,
        ge=1,
        le=262144,
        description=(
            "Feed-forward intermediate dimension. When None it is "
            "automatically computed from d_model and ffn_type."
        ),
    )
    max_seq_len: int = Field(
        default=2048,
        ge=1,
        le=1_048_576,
        description="Maximum sequence length the model can process.",
    )
    dropout_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout probability applied throughout the model.",
    )
    norm_eps: float = Field(
        default=1e-6,
        gt=0.0,
        le=1e-1,
        description="Epsilon for layer normalization variants.",
    )
    norm_type: Literal["rmsnorm", "layernorm"] = Field(
        default="rmsnorm",
        description='Normalization type — "rmsnorm" or "layernorm".',
    )
    ffn_type: Literal["swiglu", "gelu"] = Field(
        default="swiglu",
        description='Feed-forward block type — "swiglu" or "gelu".',
    )
    pos_encoding: Literal["rope", "learned"] = Field(
        default="rope",
        description='Positional encoding scheme — "rope" or "learned".',
    )
    rope_base: float = Field(
        default=10000.0,
        gt=0.0,
        description="Base frequency for Rotary Position Embeddings.",
    )
    tie_embeddings: bool = Field(
        default=True,
        description="Whether to tie input and output embedding weights.",
    )
    use_bias: bool = Field(
        default=False,
        description="Whether to use bias terms in linear projections.",
    )
    init_std: float = Field(
        default=0.02,
        gt=0.0,
        le=1.0,
        description="Standard deviation for weight initialization.",
    )
    output_logits_scale: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Optional scaling factor applied to output logits before softmax.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("n_kv_heads")
    @classmethod
    def _n_kv_heads_divides_n_heads(cls, v: int, info) -> int:  # noqa: N805
        """Ensure *n_heads* is divisible by *n_kv_heads*."""
        n_heads = info.data.get("n_heads")
        if n_heads is not None and n_heads % v != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({v})."
            )
        return v

    @field_validator("d_model")
    @classmethod
    def _d_model_divisible_by_n_heads(cls, v: int, info) -> int:  # noqa: N805
        """Ensure *d_model* is divisible by *n_heads*."""
        n_heads = info.data.get("n_heads")
        if n_heads is not None and v % n_heads != 0:
            raise ValueError(
                f"d_model ({v}) must be divisible by n_heads ({n_heads})."
            )
        return v

    @model_validator(mode="after")
    def _auto_compute_d_ff(self) -> "ModelConfig":
        """Auto-compute *d_ff* when it is not explicitly provided.

        For ``swiglu`` the intermediate size is
        ``ceil(2/3 * 4 * d_model)`` rounded up to the nearest multiple
        of 256.  For ``gelu`` it is simply ``4 * d_model``.
        """
        if self.d_ff is None:
            if self.ffn_type == "swiglu":
                raw = int(2.0 / 3.0 * 4 * self.d_model)
                self.d_ff = _round_up_to_multiple(raw, 256)
            else:
                self.d_ff = 4 * self.d_model
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def d_head(self) -> int:
        """Dimension of each attention head (``d_model // n_heads``)."""
        return self.d_model // self.n_heads

    @computed_field  # type: ignore[misc]
    @property
    def n_groups(self) -> int:
        """Number of query groups per key/value head (``n_heads // n_kv_heads``)."""
        return self.n_heads // self.n_kv_heads

    @computed_field  # type: ignore[misc]
    @property
    def is_gqa(self) -> bool:
        """Whether the model uses grouped-query attention (``n_kv_heads < n_heads``)."""
        return self.n_kv_heads < self.n_heads

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def for_scale(cls, name: str, **overrides) -> "ModelConfig":
        """Return a :class:`ModelConfig` for a predefined model scale.

        Parameters
        ----------
        name:
            One of ``"nano"``, ``"small"``, ``"medium"``, ``"large"``,
            or ``"xlarge"``.
        **overrides:
            Any additional keyword arguments are forwarded to the
            constructor and override the preset values.

        Returns
        -------
        ModelConfig
            A fully validated configuration instance.

        Raises
        ------
        ValueError
            If *name* is not a recognized scale preset.
        """
        presets: dict[str, dict] = {
            "nano": dict(
                vocab_size=256,
                d_model=128,
                n_layers=4,
                n_heads=4,
                n_kv_heads=4,
                max_seq_len=64,
            ),
            "small": dict(
                d_model=512,
                n_layers=6,
                n_heads=8,
                n_kv_heads=8,
                max_seq_len=2048,
            ),
            "medium": dict(
                d_model=1024,
                n_layers=12,
                n_heads=16,
                n_kv_heads=8,
                max_seq_len=2048,
            ),
            "large": dict(
                d_model=2048,
                n_layers=24,
                n_heads=32,
                n_kv_heads=8,
                max_seq_len=4096,
            ),
            "xlarge": dict(
                d_model=4096,
                n_layers=32,
                n_heads=32,
                n_kv_heads=8,
                max_seq_len=4096,
            ),
        }
        name = name.lower()
        if name not in presets:
            available = ", ".join(sorted(presets.keys()))
            raise ValueError(
                f"Unknown scale preset '{name}'. Available presets: {available}"
            )
        params = {**presets[name], **overrides}
        return cls(**params)
