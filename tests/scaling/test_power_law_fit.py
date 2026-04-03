"""Test power law fitting on synthetic data."""
import numpy as np
from nanochat.scaling.analysis import fit_power_law


def test_fit_recovers_known_exponent():
    """Generate synthetic power law data and verify fitting recovers the exponent."""
    np.random.seed(42)
    true_a = 10.0
    true_alpha = 0.3

    xs = np.logspace(6, 10, 20)  # 1M to 10B
    ys = true_a * xs ** (-true_alpha) * (1 + 0.02 * np.random.randn(len(xs)))

    fit = fit_power_law(xs, ys)

    assert abs(fit["alpha"] - true_alpha) < 0.05, f"alpha={fit['alpha']:.3f}, expected ~{true_alpha}"
    assert fit["r_squared"] > 0.95, f"R²={fit['r_squared']:.3f}, expected > 0.95"
    assert fit["n_points"] == 20
