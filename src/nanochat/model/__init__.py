"""Model components for nanochat-jax transformer.

Nanochat-faithful architecture:
- RMSNorm: parameterless (no gamma)
- ReLUSquaredMLP: default FFN (x * relu(x))
- GroupedQueryAttention: QK norm + softcap + sliding window
- ValueEmbedding: per-token value residual
- Smear / Backout: causal token-mixing
"""

from nanochat.model.norms import RMSNorm, LayerNorm
from nanochat.model.embeddings import TokenEmbedding, RotaryEmbedding
from nanochat.model.attention import GroupedQueryAttention
from nanochat.model.feedforward import (
    ReLUSquaredMLP, SwiGLUFFN, GeGLUFFN, StandardMLP,
)
from nanochat.model.value_embeddings import ValueEmbedding
from nanochat.model.token_mixing import Smear, Backout
from nanochat.model.block import TransformerBlock
from nanochat.model.transformer import TransformerLM
from nanochat.model.param_count import count_params, estimate_flops_per_token, compute_mfu

__all__ = [
    # Norms
    "RMSNorm",
    "LayerNorm",
    # Embeddings
    "TokenEmbedding",
    "RotaryEmbedding",
    "ValueEmbedding",
    # Attention
    "GroupedQueryAttention",
    # FFN variants
    "ReLUSquaredMLP",
    "SwiGLUFFN",
    "GeGLUFFN",
    "StandardMLP",
    # Token mixing
    "Smear",
    "Backout",
    # Blocks
    "TransformerBlock",
    "TransformerLM",
    # Utilities
    "count_params",
    "estimate_flops_per_token",
    "compute_mfu",
]
