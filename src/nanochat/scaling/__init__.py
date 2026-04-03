"""Scaling law experiment framework."""
from nanochat.scaling.runner import ScalingRunner, ScalingRunResult
from nanochat.scaling.analysis import fit_power_law, chinchilla_optimal
from nanochat.scaling.visualization import generate_full_report

__all__ = [
    "ScalingRunner", "ScalingRunResult",
    "fit_power_law", "chinchilla_optimal",
    "generate_full_report",
]
