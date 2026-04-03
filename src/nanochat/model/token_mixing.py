"""Smear and Backout token-mixing operations for nanochat.

These are nanochat-specific causal token-mixing operations that provide
cheap local context integration before and after the attention sublayer.

Overview
--------

**Smear** (applied BEFORE attention norm):
    Blends each token's representation with its immediate predecessor
    via a per-feature learnable interpolation coefficient:

        x_smear[t] = (1 - alpha) * x[t] + alpha * x[t-1]
        where alpha = sigmoid(raw_alpha)   (element-wise, per feature)

    - Initialized to alpha ≈ 0 (no-op: raw_alpha = -10 → sigmoid ≈ 0)
    - First position sees zero padding (causal constraint preserved)
    - JIT-compilable: implemented as a masked shift + blend, no Python loops

    Rationale: Provides a cheap "prior" about the preceding token that
    the attention mechanism can then use or override. This offloads simple
    local patterns from attention.

**Backout** (applied AFTER attention output):
    Removes the smear-introduced correlation from the attention output.
    After attention has operated on smeared representations, the output
    contains a residual of the blended context. Backout removes this
    to prevent double-counting when the result is added back to the
    (unsmeared) residual stream:

        out_backed[t] = out[t] - beta * x_smeared[t-1]
        where beta = sigmoid(raw_beta)

    - Initialized to beta ≈ 0 (no-op)
    - Only applied when use_smear=True in ModelConfig

Design notes:
    - Both ops use per-feature (channel-wise) scalars, NOT head-shared.
    - The sigmoid ensures the mixing coefficient stays in [0, 1].
    - Zero-padding of the first position ensures causal guarantees.
    - Both are O(S*D) — cheaper than any attention variant.

References:
    - nanochat architecture (token-mixing before/after attention)
    - RWKV-style token mixing (Peng et al., 2023) — conceptually similar
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import structlog
from flax import nnx

log = structlog.get_logger(__name__)


class Smear(nnx.Module):
    """Causal token-mixing: blend each position with its predecessor.

    Computes::

        x_smear[t] = (1 - alpha[f]) * x[t, f] + alpha[f] * x[t-1, f]

    for each feature f. The first position uses a zero predecessor (causal).
    alpha is per-feature and element-wise: shape (d_model,).

    Attributes:
        d_model: Feature dimension.
        raw_alpha: Unconstrained parameter; alpha = sigmoid(raw_alpha).
            Initialized to -10.0 → alpha ≈ 0.0 (near no-op at init).
    """

    def __init__(self, d_model: int, *, rngs: nnx.Rngs) -> None:
        """Initialize Smear.

        Args:
            d_model: Hidden dimension.
            rngs: Flax NNX RNG container.
        """
        self.d_model = d_model
        # Initialize raw_alpha to a large negative value so sigmoid(raw_alpha) ≈ 0.
        # This ensures smear starts as a no-op and learns to mix only as needed.
        self.raw_alpha = nnx.Param(jnp.full((d_model,), -10.0))  # [d_model]

        log.debug("smear.init", d_model=d_model)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply causal smear.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Smeared tensor of shape ``(batch, seq_len, d_model)``.
            Also returns x_prev (the shifted predecessor) for use by Backout.
        """
        # alpha ∈ [0, 1] per feature
        alpha = jax.nn.sigmoid(self.raw_alpha.get_value())  # [d_model]

        # Shift x right by 1 position with zero padding (causal predecessor)
        # x_prev[t] = x[t-1], x_prev[0] = 0
        x_prev = jnp.concatenate(
            [jnp.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1
        )  # [B, S, d_model]

        # Interpolate: (1 - alpha) * x + alpha * x_prev
        x_smeared = x + alpha * (x_prev - x)  # broadcast alpha over [B, S, d_model]

        return x_smeared, x_prev  # return both for Backout to use

    def __repr__(self) -> str:
        return f"Smear(d_model={self.d_model})"


class Backout(nnx.Module):
    """Remove smear residual from attention output.

    After attention operates on smeared inputs, its output contains a
    residual of the predecessor context. Backout subtracts a learned
    fraction of that predecessor to prevent double-counting:

        out_backed[t] = attn_out[t] - beta[f] * x_smeared_prev[t]

    where x_smeared_prev[t] = x_smeared[t-1] (zero-padded at t=0).

    Attributes:
        d_model: Feature dimension.
        raw_beta: Unconstrained parameter; beta = sigmoid(raw_beta).
            Initialized to -10.0 → beta ≈ 0.0 (near no-op at init).
    """

    def __init__(self, d_model: int, *, rngs: nnx.Rngs) -> None:
        """Initialize Backout.

        Args:
            d_model: Hidden dimension.
            rngs: Flax NNX RNG container.
        """
        self.d_model = d_model
        # Same near-zero initialization as Smear
        self.raw_beta = nnx.Param(jnp.full((d_model,), -10.0))  # [d_model]

        log.debug("backout.init", d_model=d_model)

    def __call__(
        self,
        attn_out: jax.Array,
        x_prev: jax.Array,
    ) -> jax.Array:
        """Apply backout correction to attention output.

        Args:
            attn_out: Attention output of shape ``(batch, seq_len, d_model)``.
            x_prev: Predecessor tensor from Smear (x shifted by 1, zero-padded)
                of shape ``(batch, seq_len, d_model)``.

        Returns:
            Corrected output of the same shape as *attn_out*.
        """
        # beta ∈ [0, 1] per feature
        beta = jax.nn.sigmoid(self.raw_beta.get_value())  # [d_model]

        # Remove learned fraction of the predecessor context
        return attn_out - beta * x_prev  # [B, S, d_model]

    def __repr__(self) -> str:
        return f"Backout(d_model={self.d_model})"
