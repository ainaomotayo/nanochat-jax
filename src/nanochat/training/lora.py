"""LoRA (Low-Rank Adaptation) for nanochat-jax.

Implements LoRA linear layers that decompose weight updates into low-rank
matrices A and B, so the effective weight is W' = W + (alpha/rank) * A @ B.

B is initialized to zeros and A to small random values, so at initialization
W' = W (the model behaves identically to the base model).

Usage::

    model = TransformerLM(cfg, rngs=rngs)
    apply_lora(model, rank=16, rngs=rngs)          # wrap Q/K/V/out_proj
    trainable = get_trainable_params(model)         # only LoRA params
    # ... train ...
    merge_lora(model)                               # fold into base weights

References:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import structlog
from flax import nnx

logger = structlog.get_logger()


class LoRAParam(nnx.Variable):
    """Variable type tag for LoRA-specific parameters.

    Used by get_trainable_params to filter only LoRA weights from the full
    model state, leaving base model parameters frozen.
    """
    pass


class LoRALinear(nnx.Module):
    """Linear layer with low-rank adaptation.

    Wraps an existing nnx.Linear and adds trainable low-rank matrices
    lora_A (in_features, rank) and lora_B (rank, out_features).

    The forward pass computes::

        y = x @ W^T + x @ (A @ B)^T * (alpha / rank)

    where W is the frozen base weight (kernel).

    At initialization, B is zeros so the output matches the base layer exactly.

    Attributes:
        base_linear: The original nnx.Linear (frozen during LoRA training).
        lora_A: Low-rank input projection (in_features, rank).
        lora_B: Low-rank output projection (rank, out_features).
        scaling: alpha / rank.
    """

    def __init__(
        self,
        base_linear: nnx.Linear,
        rank: int,
        alpha: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize LoRALinear by wrapping an existing nnx.Linear.

        Args:
            base_linear: The original linear layer to adapt.
            rank: LoRA rank (typically 4-64).
            alpha: LoRA scaling factor (typically equal to rank).
            rngs: Flax NNX RNG container.
        """
        self.base_linear = base_linear
        self.rank = rank
        self.scaling = alpha / rank

        # Infer dimensions from the base kernel: shape is (in_features, out_features)
        in_features, out_features = base_linear.kernel[...].shape

        # A: small random init (Kaiming-uniform-like scale)
        a_init = jax.random.normal(
            rngs.params(), (in_features, rank)
        ) * (1.0 / rank)
        self.lora_A = LoRAParam(a_init)

        # B: zero init (so W' = W at initialization)
        self.lora_B = LoRAParam(jnp.zeros((rank, out_features)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: base output + scaled low-rank delta.

        Args:
            x: Input tensor (..., in_features).

        Returns:
            Output tensor (..., out_features).
        """
        # Base linear forward
        base_out = self.base_linear(x)

        # LoRA delta: x @ A @ B * scaling
        lora_out = (x @ self.lora_A[...]) @ self.lora_B[...] * self.scaling

        return base_out + lora_out


def apply_lora(
    model: nnx.Module,
    rank: int,
    rngs: nnx.Rngs,
    alpha: float | None = None,
) -> None:
    """Replace Q/K/V/out_proj in all attention layers with LoRALinear.

    Modifies the model in-place. After this call, only LoRA parameters
    should be trained (use get_trainable_params to filter).

    Args:
        model: TransformerLM model.
        rank: LoRA rank.
        rngs: RNG container for LoRA parameter initialization.
        alpha: LoRA alpha. Defaults to rank (scaling = 1.0).
    """
    if alpha is None:
        alpha = float(rank)

    n_replaced = 0
    for layer in model.layers:
        attn = layer.attention

        for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            base_linear = getattr(attn, proj_name)
            if isinstance(base_linear, LoRALinear):
                continue  # already wrapped
            lora_linear = LoRALinear(base_linear, rank, alpha, rngs=rngs)
            setattr(attn, proj_name, lora_linear)
            n_replaced += 1

    logger.info(
        "lora_applied",
        rank=rank,
        alpha=alpha,
        n_replaced=n_replaced,
    )


def get_lora_params(model: nnx.Module) -> nnx.State:
    """Extract only LoRA parameters from the model.

    Returns an nnx.State containing only LoRAParam variables, suitable
    for passing as the optimizer target.

    Args:
        model: Model with LoRA layers applied.

    Returns:
        nnx.State with only LoRA parameters.
    """
    return nnx.state(model, LoRAParam)


def count_lora_params(model: nnx.Module) -> int:
    """Count the number of trainable LoRA parameters.

    Args:
        model: Model with LoRA layers applied.

    Returns:
        Total number of LoRA parameter elements.
    """
    lora_state = get_lora_params(model)
    leaves = jax.tree.leaves(lora_state)
    return sum(x.size for x in leaves)


def count_base_params(model: nnx.Module) -> int:
    """Count the number of base (non-LoRA) parameters.

    Args:
        model: Model (with or without LoRA).

    Returns:
        Total number of base parameter elements.
    """
    base_state = nnx.state(model, nnx.Param)
    leaves = jax.tree.leaves(base_state)
    return sum(x.size for x in leaves)


def merge_lora(model: nnx.Module) -> None:
    """Fold LoRA weights back into the base linear kernels.

    After merging, the LoRALinear modules are replaced with plain
    nnx.Linear modules that contain the merged weights. The model
    will produce identical outputs but without any LoRA overhead.

    Args:
        model: Model with LoRA layers applied.
    """
    n_merged = 0
    for layer in model.layers:
        attn = layer.attention

        for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            module = getattr(attn, proj_name)
            if not isinstance(module, LoRALinear):
                continue

            base_linear = module.base_linear
            # Merge: W_new = W + scaling * A @ B
            delta = module.lora_A[...] @ module.lora_B[...] * module.scaling
            base_linear.kernel[...] = base_linear.kernel[...] + delta

            # Replace LoRALinear with the merged base linear
            setattr(attn, proj_name, base_linear)
            n_merged += 1

    logger.info("lora_merged", n_merged=n_merged)
