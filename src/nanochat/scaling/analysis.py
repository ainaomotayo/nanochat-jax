"""Scaling law analysis: power law fitting and Chinchilla-optimal computation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import structlog

logger = structlog.get_logger()


def _power_law(x: np.ndarray, a: float, alpha: float) -> np.ndarray:
    """Power law: L = a * x^(-alpha)."""
    return a * np.power(x, -alpha)


def fit_power_law(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    bootstrap_n: int = 1000,
) -> dict[str, float]:
    """Fit L = a * x^(-alpha) to data using log-space linear regression.

    Args:
        xs: Independent variable (e.g., n_params, n_tokens, flops)
        ys: Dependent variable (e.g., val_loss)
        bootstrap_n: Number of bootstrap samples for CI

    Returns:
        Dict with: a, alpha, alpha_ci_lo, alpha_ci_hi, r_squared, n_points
    """
    log_x = np.log(xs)
    log_y = np.log(ys)

    # Linear regression in log space: log(y) = log(a) - alpha * log(x)
    coeffs = np.polyfit(log_x, log_y, 1)
    alpha = -coeffs[0]
    a = np.exp(coeffs[1])

    # R-squared in log space
    y_pred = coeffs[0] * log_x + coeffs[1]
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)

    # Bootstrap confidence interval
    alphas = []
    rng = np.random.RandomState(42)
    for _ in range(bootstrap_n):
        idx = rng.choice(len(xs), size=len(xs), replace=True)
        try:
            c = np.polyfit(log_x[idx], log_y[idx], 1)
            alphas.append(-c[0])
        except (np.linalg.LinAlgError, ValueError):
            continue

    alphas = np.array(alphas)
    ci_lo = float(np.percentile(alphas, 5)) if len(alphas) > 0 else alpha
    ci_hi = float(np.percentile(alphas, 95)) if len(alphas) > 0 else alpha

    result = {
        "a": float(a),
        "alpha": float(alpha),
        "alpha_ci_lo": ci_lo,
        "alpha_ci_hi": ci_hi,
        "r_squared": float(r_squared),
        "n_points": len(xs),
    }
    logger.info("power_law_fit", **{k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})
    return result


def chinchilla_optimal(
    compute_budgets: np.ndarray,
    a_n: float = 406.4,
    a_d: float = 410.7,
    alpha: float = 0.34,
    beta: float = 0.28,
    irreducible_loss: float = 1.69,
) -> pd.DataFrame:
    """Compute Chinchilla-optimal (N*, D*) for each compute budget C.

    From Hoffmann et al. (2022):
        L(N, D) = E + A/N^alpha + B/D^beta
        N* ~ C^(beta/(alpha+beta)), D* ~ C^(alpha/(alpha+beta))

    Returns:
        DataFrame with columns: compute_flops, n_params, n_tokens, predicted_loss
    """
    # Chinchilla optimal scaling
    gamma = alpha / (alpha + beta)  # exponent for D*
    delta = beta / (alpha + beta)   # exponent for N*

    results = []
    for C in compute_budgets:
        # N* proportional to C^delta, D* = C / (6 * N*)
        N_star = int((C / 6.0) ** delta * 1e6 ** (1 - delta))  # rough scaling
        D_star = int(C / (6.0 * max(N_star, 1)))

        # Predicted loss
        L = irreducible_loss + a_n / max(N_star, 1) ** alpha + a_d / max(D_star, 1) ** beta

        results.append({
            "compute_flops": float(C),
            "n_params": N_star,
            "n_tokens": D_star,
            "predicted_loss": float(L),
        })

    return pd.DataFrame(results)
