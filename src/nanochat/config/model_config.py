"""Model architecture configuration for nanochat-jax.

Faithful nanochat port configuration: defaults match nanochat's architectural
choices exactly (relu², parameterless RMSNorm, rope_base=100000, untied
embeddings, QK norm, logit softcap, value embeddings, per-layer scalars,
sliding window attention, and smear/backout token mixing).

Predefined scale presets are available via :meth:`ModelConfig.for_scale`
and :meth:`ModelConfig.from_depth`.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator, computed_field


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round *value* up to the nearest multiple of *multiple*."""
    return math.ceil(value / multiple) * multiple


class ModelConfig(BaseModel):
    """Transformer model architecture configuration.

    Defaults are set to match nanochat's architectural choices:
    - relu² MLP activation (no gating branch)
    - Parameterless RMSNorm (no learned scale)
    - QK L2 normalization with 1.2x scale factor
    - Logit softcap at 30.0 before softmax
    - Value embeddings per token
    - Per-layer learnable scalars on attention and FFN outputs
    - Smear/backout token mixing
    - rope_base = 100000
    - Untied input/output embeddings
    """

    # ------------------------------------------------------------------
    # Vocabulary and dimensions
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------
    dropout_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout probability applied throughout the model.",
    )

    # ------------------------------------------------------------------
    # Normalization — nanochat: parameterless RMSNorm, no learned scale
    # ------------------------------------------------------------------
    norm_eps: float = Field(
        default=1e-6,
        gt=0.0,
        le=1e-1,
        description="Epsilon for RMSNorm numerical stability.",
    )
    norm_type: Literal["rmsnorm", "layernorm"] = Field(
        default="rmsnorm",
        description='Normalization type. "rmsnorm" is parameterless in nanochat.',
    )

    # ------------------------------------------------------------------
    # FFN — nanochat default: relu² (x * relu(x)), no gating branch
    # ------------------------------------------------------------------
    ffn_type: Literal["relu2", "swiglu", "geglu", "gelu"] = Field(
        default="relu2",
        description=(
            'Feed-forward block type. "relu2" is nanochat default: '
            'x * relu(x) with d_ff = 4 * d_model.'
        ),
    )

    # ------------------------------------------------------------------
    # Positional encoding — nanochat uses rope_base=100000
    # ------------------------------------------------------------------
    pos_encoding: Literal["rope", "learned"] = Field(
        default="rope",
        description='Positional encoding scheme.',
    )
    rope_base: float = Field(
        default=100_000.0,
        gt=0.0,
        description="Base frequency for Rotary Position Embeddings (nanochat: 100000).",
    )

    # ------------------------------------------------------------------
    # Embeddings — nanochat: untied (separate input/output embeddings)
    # ------------------------------------------------------------------
    tie_embeddings: bool = Field(
        default=False,
        description="Whether to tie input and output embedding weights (nanochat: False).",
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

    # ------------------------------------------------------------------
    # QK normalization — nanochat: L2 norm Q and K, then scale by 1.2/sqrt(d_head)
    # ------------------------------------------------------------------
    use_qk_norm: bool = Field(
        default=True,
        description="Apply L2 normalization to Q and K before attention.",
    )
    qk_scale_factor: float = Field(
        default=1.2,
        gt=0.0,
        le=10.0,
        description=(
            "Multiplier on 1/sqrt(d_head) attention scale after QK norm. "
            "nanochat uses 1.2."
        ),
    )

    # ------------------------------------------------------------------
    # Logit softcap — nanochat: 30 * tanh(logits / 30) before softmax
    # ------------------------------------------------------------------
    logit_softcap: Optional[float] = Field(
        default=30.0,
        gt=0.0,
        description="Softcap applied to attention logits before softmax (nanochat: 30.0).",
    )

    # ------------------------------------------------------------------
    # Output logit scaling (Gemma-style, optional)
    # ------------------------------------------------------------------
    output_logits_scale: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Optional scaling factor applied to LM head logits before loss.",
    )

    # ------------------------------------------------------------------
    # Value embeddings — nanochat-specific: per-token learned residual
    # ------------------------------------------------------------------
    use_value_embeddings: bool = Field(
        default=True,
        description=(
            "Add per-token value embeddings to attention output. "
            "nanochat-specific feature."
        ),
    )

    # ------------------------------------------------------------------
    # Per-layer scalars — nanochat: learnable scale on attn and FFN outputs
    # ------------------------------------------------------------------
    use_per_layer_scalars: bool = Field(
        default=True,
        description=(
            "Add learnable scalar weights on attention and FFN outputs. "
            "nanochat-specific feature."
        ),
    )

    # ------------------------------------------------------------------
    # Smear/Backout token mixing — nanochat-specific
    # ------------------------------------------------------------------
    use_smear: bool = Field(
        default=True,
        description=(
            "Apply causal token-mixing (smear) before attention and "
            "backout correction after. nanochat-specific feature."
        ),
    )

    # ------------------------------------------------------------------
    # Sliding window attention — optional, None = full causal attention
    # ------------------------------------------------------------------
    sliding_window_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=1_048_576,
        description=(
            "Local attention window size. None = full causal attention. "
            "When set, tokens only attend to the last sliding_window_size positions."
        ),
    )
    n_global_tokens: int = Field(
        default=1,
        ge=0,
        le=1024,
        description=(
            "Number of leading global tokens (e.g., BOS) that can be "
            "attended to from anywhere regardless of window size."
        ),
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("n_kv_heads")
    @classmethod
    def _n_kv_heads_divides_n_heads(cls, v: int, info) -> int:
        n_heads = info.data.get("n_heads")
        if n_heads is not None and n_heads % v != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({v})."
            )
        return v

    @field_validator("d_model")
    @classmethod
    def _d_model_divisible_by_n_heads(cls, v: int, info) -> int:
        n_heads = info.data.get("n_heads")
        if n_heads is not None and v % n_heads != 0:
            raise ValueError(
                f"d_model ({v}) must be divisible by n_heads ({n_heads})."
            )
        return v

    @model_validator(mode="after")
    def _auto_compute_d_ff(self) -> "ModelConfig":
        """Auto-compute d_ff when not provided.

        - relu2 / gelu / standard MLP: 4 * d_model (no gating overhead)
        - swiglu / geglu: ceil(2/3 * 4 * d_model) rounded to multiple of 256
          (compensates for the three-matrix structure)
        """
        if self.d_ff is None:
            if self.ffn_type in ("swiglu", "geglu"):
                raw = int(2.0 / 3.0 * 4 * self.d_model)
                self.d_ff = _round_up_to_multiple(raw, 256)
            else:
                # relu2 and gelu: standard 4x expansion
                self.d_ff = 4 * self.d_model
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def d_head(self) -> int:
        """Per-head dimension: d_model // n_heads."""
        return self.d_model // self.n_heads

    @computed_field  # type: ignore[misc]
    @property
    def n_groups(self) -> int:
        """Query groups per KV head: n_heads // n_kv_heads."""
        return self.n_heads // self.n_kv_heads

    @computed_field  # type: ignore[misc]
    @property
    def is_gqa(self) -> bool:
        """True when grouped-query attention is active."""
        return self.n_kv_heads < self.n_heads

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def for_scale(cls, name: str, **overrides) -> "ModelConfig":
        """Return a ModelConfig for a predefined model scale.

        All presets use nanochat-faithful defaults (relu², parameterless
        RMSNorm, rope_base=100000, untied embeddings).

        Parameters
        ----------
        name:
            One of ``"nano"``, ``"small"``, ``"medium"``, ``"large"``,
            or ``"xlarge"``.
        **overrides:
            Any additional keyword arguments override preset values.
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

    @classmethod
    def from_depth(
        cls,
        n_layers: int,
        d_model: int,
        *,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        vocab_size: int = 32000,
        max_seq_len: int = 2048,
        **overrides,
    ) -> "ModelConfig":
        """Construct a ModelConfig from depth + width, auto-sizing other dims.

        This factory mirrors nanochat's from_depth() convention where model
        size is specified by (n_layers, d_model) and all derived dimensions
        (d_head, d_ff, n_kv_heads) are computed automatically.

        Parameters
        ----------
        n_layers:
            Number of transformer layers.
        d_model:
            Hidden dimension.
        n_heads:
            Number of query heads.
        n_kv_heads:
            KV heads for GQA. Defaults to n_heads (MHA).
        vocab_size:
            Vocabulary size.
        max_seq_len:
            Maximum sequence length.
        **overrides:
            Additional field overrides.
        """
        if n_kv_heads is None:
            n_kv_heads = n_heads
        params = dict(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        params.update(overrides)
        return cls(**params)
