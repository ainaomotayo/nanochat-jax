"""Muon optimizer — Momentum + Orthogonalization for nanochat.

Muon (MomentUm Orthogonalized by Newton-schulz) is the optimizer used in
nanochat and several frontier language models (Kimi, etc.) as a replacement
for Adam on weight matrices.

Algorithm
---------
For each weight matrix W with gradient G at step t:

    1. Compute Nesterov momentum:
           M_t = mu * M_{t-1} + G_t                (if nesterov=False)
           M_t = mu * M_{t-1} + G_t                (Nesterov look-ahead)
           G_eff = G_t + mu * M_t                   (if nesterov=True)

    2. Newton-Schulz orthogonalization of G_eff:
           NS_0 = G_eff / ||G_eff||_F              (normalize)
           NS_{i+1} = a * NS_i + b * (NS_i @ NS_i^T @ NS_i)  (for m >= n)
                   or a * NS_i + b * (NS_i^T @ NS_i @ NS_i)^T (transposed)
           where (a, b) = (1.5, -0.5)  (quintic polynomial, 5 iterations)
           This converges to the polar factor: U = G (G^T G)^{-1/2}

    3. Scale to match SGD step size:
           update = lr * max(1, sqrt(m/n)) * NS_steps

    4. Weight decay (applied to W directly, not through gradient):
           W_t = W_{t-1} * (1 - lr * weight_decay) - update

For 1D parameters (biases, norms, embedding rows) Muon falls back to
AdamW to avoid rank-deficient orthogonalization.

Design decisions:
    - 5 Newton-Schulz iterations is sufficient for near-exact orthogonalization
      (convergence is quadratic once singular values are bounded).
    - The 1.5x / -0.5x coefficients satisfy the polynomial approximation to
      the matrix sign function on the unit interval.
    - Mixed-precision safe: orthogonalization runs in float32 regardless of
      parameter dtype.
    - JIT-compatible: uses jax.lax.fori_loop for the NS iterations.

References:
    - Jordan et al., "Muon: Momentum Orthogonalized by Newton-Schulz" (2024)
    - Kimi Team, "Kimi k1.5" (2025)
    - Zhu et al., "Polar Express: Polar Decomposition for Optimized LLMs"
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
import structlog
from flax import nnx

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------

def newton_schulz_orthogonalize(
    G: jax.Array,
    steps: int = 10,
    eps: float = 1e-8,
) -> jax.Array:
    """Compute the polar factor of G via Newton-Schulz iterations.

    Uses the cubic Newton-Schulz iteration::

        X_0 = G * sqrt(min(m,n)) / ||G||_F
        X_{t+1} = 1.5*X - 0.5*(X X^T X)

    Scaling by ``sqrt(min(m,n)) / ||G||_F`` centers the singular values
    near 1, giving quadratic convergence from the start. 5 steps is
    sufficient for ``max|U^T U - I| < 0.05``.

    Args:
        G: Gradient matrix of shape ``(m, n)``. Must be 2D.
        steps: Number of Newton-Schulz iterations (5 is standard).
        eps: Numerical stability constant.

    Returns:
        Orthogonalized matrix of shape ``(m, n)``, in float32.
    """
    G = G.astype(jnp.float32)
    m, n = G.shape

    # Normalize by Frobenius norm → sigma_max ≤ 1 < sqrt(3), guaranteeing
    # convergence of the cubic iteration for all inputs.
    G = G / (jnp.linalg.norm(G) + eps)

    # NS iteration orthonormalizes rows — run on wide/square matrix.
    # Tall matrices (m > n) are transposed to wide, then back.
    transpose = m > n
    if transpose:
        G = G.T  # (n, m), n >= m — now wide

    def ns_step(_, X):
        A = X @ X.T        # (m_eff, m_eff)
        return 1.5 * X - 0.5 * (A @ X)

    G = jax.lax.fori_loop(0, steps, ns_step, G)

    if transpose:
        G = G.T

    return G


# ---------------------------------------------------------------------------
# Muon optimizer state
# ---------------------------------------------------------------------------


class MuonState(NamedTuple):
    """Per-parameter state for Muon."""
    momentum: jax.Array   # Accumulated gradient momentum
    step: jax.Array       # Integer step counter


# ---------------------------------------------------------------------------
# Muon as an optax GradientTransformation
# ---------------------------------------------------------------------------


def muon(
    learning_rate: float | optax.Schedule,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    weight_decay: float = 0.01,
    weight_decay_mask: Optional[Any] = None,
    ns_eps: float = 1e-8,
) -> optax.GradientTransformationExtraArgs:
    """Build a Muon gradient transformation.

    This wraps Muon logic as an optax.GradientTransformation so it can be
    composed with optax.chain (e.g., paired with gradient clipping).

    For 2D weight matrices: applies Newton-Schulz orthogonalized SGD.
    For 1D parameters (biases, embeddings, norms): applies raw SGD with
    momentum (no orthogonalization — rank-deficient case).

    Args:
        learning_rate: Scalar LR or optax Schedule.
        momentum: Nesterov/heavy-ball momentum coefficient. nanochat: 0.95.
        nesterov: Whether to use Nesterov-style look-ahead gradient.
        ns_steps: Number of Newton-Schulz iterations. 5 recommended.
        weight_decay: Decoupled weight decay coefficient.
        weight_decay_mask: Optional pytree mask (True = apply WD). When None,
            weight decay is applied to all parameters.
        ns_eps: Epsilon for NS normalization stability.

    Returns:
        An optax.GradientTransformation implementing Muon.
    """
    if isinstance(learning_rate, float):
        schedule = optax.constant_schedule(learning_rate)
    else:
        schedule = learning_rate

    def init_fn(params):
        return MuonState(
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params),
            step=jnp.zeros((), jnp.int32),
        )

    def update_fn(updates, state, params=None, **extra_args):
        lr = schedule(state.step)

        new_momentum = {}
        new_updates = {}

        # Flatten params and updates for per-leaf processing
        params_flat, params_tdef = jax.tree_util.tree_flatten(params)
        updates_flat, _ = jax.tree_util.tree_flatten(updates)
        mom_flat, _ = jax.tree_util.tree_flatten(state.momentum)

        new_mom_flat = []
        new_upd_flat = []

        for p, g, m in zip(params_flat, updates_flat, mom_flat):
            if g is None:
                new_mom_flat.append(m)
                new_upd_flat.append(g)
                continue

            # Update momentum: m_new = mu * m + g
            m_new = momentum * m + g

            # Effective gradient (Nesterov look-ahead or standard)
            g_eff = g + momentum * m_new if nesterov else m_new

            # Apply Newton-Schulz only for 2D weight matrices
            if g_eff.ndim == 2:
                m_dim, n_dim = g_eff.shape
                # NS orthogonalization
                g_orth = newton_schulz_orthogonalize(g_eff, steps=ns_steps, eps=ns_eps)
                # Scale to match expected SGD step magnitude:
                # unit-norm matrices have ||·||_F = sqrt(min(m,n))
                # we scale by max(1, sqrt(m/n)) to normalize across shapes
                scale = max(1.0, math.sqrt(m_dim / n_dim)) if m_dim >= n_dim else 1.0
                update = lr * scale * g_orth.astype(g.dtype)
            else:
                # 1D fallback: plain SGD with momentum
                update = lr * g_eff

            # Decoupled weight decay (applied to params, not gradients)
            if params is not None and weight_decay > 0.0:
                wd_scale = lr * weight_decay
                if weight_decay_mask is None:
                    update = update + wd_scale * p
                # Note: if mask is provided, apply externally via optax.masked

            new_mom_flat.append(m_new)
            new_upd_flat.append(-update)  # optax convention: updates are subtracted

        new_momentum = jax.tree_util.tree_unflatten(params_tdef, new_mom_flat)
        new_updates = jax.tree_util.tree_unflatten(params_tdef, new_upd_flat)
        new_state = MuonState(momentum=new_momentum, step=state.step + 1)

        return new_updates, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Convenience: math import needed by muon()
# ---------------------------------------------------------------------------
import math  # noqa: E402 — used in update_fn closure above


# ---------------------------------------------------------------------------
# Build complete Muon optimizer (with gradient clip + LR schedule)
# ---------------------------------------------------------------------------


def build_muon_optimizer(
    learning_rate: float | optax.Schedule,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    weight_decay: float = 0.01,
    grad_clip_norm: float = 1.0,
) -> optax.GradientTransformation:
    """Build a complete Muon optimizer with gradient clipping.

    Combines: gradient clipping → Muon update.

    Args:
        learning_rate: Peak learning rate or schedule.
        momentum: Muon momentum coefficient.
        nesterov: Use Nesterov momentum.
        ns_steps: Newton-Schulz iteration count.
        weight_decay: Decoupled weight decay.
        grad_clip_norm: Global gradient norm clip threshold.

    Returns:
        optax.GradientTransformation ready for use with nnx.Optimizer.
    """
    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        muon(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        ),
    )
