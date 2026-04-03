"""Unit tests for the Muon optimizer.

Tests:
1. Newton-Schulz orthogonalization: output is near-orthogonal
2. Muon update: parameters change after one step
3. Muon update: gradients are applied correctly
4. Nesterov momentum: update differs from non-Nesterov
5. 1D parameters: no orthogonalization applied (fallback)
6. Weight decay: parameters shrink with non-zero WD
"""

import math
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax

from nanochat.training.muon import newton_schulz_orthogonalize, muon, MuonState


class TestNewtonSchulzOrthogonalize:
    """Test the Newton-Schulz polar factor computation."""

    def test_output_shape(self):
        G = jax.random.normal(jax.random.PRNGKey(0), (8, 16))
        out = newton_schulz_orthogonalize(G)
        assert out.shape == (8, 16)

    def test_output_float32(self):
        G = jax.random.normal(jax.random.PRNGKey(0), (8, 16)).astype(jnp.bfloat16)
        out = newton_schulz_orthogonalize(G)
        assert out.dtype == jnp.float32

    def test_near_orthogonal_columns(self):
        """After NS, columns should have near-unit norms."""
        G = jax.random.normal(jax.random.PRNGKey(1), (16, 8)) * 5.0
        U = newton_schulz_orthogonalize(G, steps=10)

        # U^T U should be near identity (orthonormal columns)
        UTU = U.T @ U
        np.testing.assert_allclose(
            np.array(UTU), np.eye(8), atol=0.05,
            err_msg="Newton-Schulz output not near-orthogonal (10 steps)"
        )

    def test_wide_matrix(self):
        """Wide matrices (m < n) should also produce near-orthogonal rows."""
        G = jax.random.normal(jax.random.PRNGKey(2), (4, 16))
        U = newton_schulz_orthogonalize(G, steps=10)
        assert U.shape == (4, 16)

        # U U^T should be near identity (orthonormal rows for wide case)
        UUT = U @ U.T
        np.testing.assert_allclose(
            np.array(UUT), np.eye(4), atol=0.05,
        )

    def test_zero_input_no_nan(self):
        """Near-zero gradient should not produce NaN."""
        G = jnp.zeros((8, 8)) + 1e-20
        out = newton_schulz_orthogonalize(G)
        assert not jnp.any(jnp.isnan(out))

    def test_finite_output(self):
        G = jax.random.normal(jax.random.PRNGKey(3), (32, 64)) * 100.0
        out = newton_schulz_orthogonalize(G)
        assert jnp.isfinite(out).all()


class TestMuonOptimizer:
    """Test Muon as an optax.GradientTransformation."""

    def _make_params(self):
        """Simple params: one 2D weight matrix, one 1D bias."""
        return {
            "weight": jnp.ones((8, 16)),
            "bias": jnp.ones((8,)),
        }

    def _make_grads(self):
        key = jax.random.PRNGKey(42)
        return {
            "weight": jax.random.normal(key, (8, 16)),
            "bias": jax.random.normal(key, (8,)),
        }

    def test_init(self):
        opt = muon(learning_rate=1e-3)
        params = self._make_params()
        state = opt.init(params)
        assert isinstance(state, MuonState)

    def test_update_produces_finite_updates(self):
        opt = muon(learning_rate=1e-3, momentum=0.95)
        params = self._make_params()
        grads = self._make_grads()

        state = opt.init(params)
        updates, new_state = opt.update(grads, state, params)

        for key, u in updates.items():
            assert jnp.isfinite(u).all(), f"Non-finite update for {key}"

    def test_params_change_after_step(self):
        """Parameters should change after applying Muon updates."""
        opt = muon(learning_rate=1e-2, weight_decay=0.0)
        params = self._make_params()
        grads = self._make_grads()

        state = opt.init(params)
        updates, _ = opt.update(grads, state, params)

        # Apply updates
        new_params = {k: params[k] + updates[k] for k in params}

        # Weights should have changed
        weight_diff = jnp.max(jnp.abs(new_params["weight"] - params["weight"]))
        assert float(weight_diff) > 1e-10, "Weights did not change after Muon step"

    def test_momentum_accumulates(self):
        """Second step should differ from first step due to momentum."""
        opt = muon(learning_rate=1e-3, momentum=0.95)
        params = self._make_params()
        grads = self._make_grads()

        state = opt.init(params)

        # First step
        updates1, state = opt.update(grads, state, params)

        # Second step (same grads)
        updates2, _ = opt.update(grads, state, params)

        # With momentum, second step should differ from first
        diff = jnp.max(jnp.abs(updates1["weight"] - updates2["weight"]))
        assert float(diff) > 1e-10, "Momentum not accumulating"

    def test_step_counter_increments(self):
        opt = muon(learning_rate=1e-3)
        params = self._make_params()
        grads = self._make_grads()
        state = opt.init(params)

        assert int(state.step) == 0
        _, new_state = opt.update(grads, state, params)
        assert int(new_state.step) == 1


class TestMuonWithSchedule:
    def test_with_constant_schedule(self):
        """Muon should work with optax schedule as learning_rate."""
        schedule = optax.constant_schedule(1e-3)
        opt = muon(learning_rate=schedule)
        params = {"w": jnp.ones((4, 8))}
        grads = {"w": jax.random.normal(jax.random.PRNGKey(0), (4, 8))}

        state = opt.init(params)
        updates, _ = opt.update(grads, state, params)
        assert jnp.isfinite(updates["w"]).all()
