"""Publication-quality scaling law visualizations."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import structlog

logger = structlog.get_logger()

# Style setup
sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = sns.color_palette("viridis", 10)


def plot_loss_vs_params(
    n_params: np.ndarray,
    val_losses: np.ndarray,
    fit: dict[str, float],
    output_path: str | Path,
    model_names: list[str] | None = None,
) -> None:
    """Log-log plot of val loss vs model size with fitted power law."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.scatter(n_params, val_losses, s=80, zorder=5, color=COLORS[0], label="Observed")

    if model_names:
        for x, y, name in zip(n_params, val_losses, model_names):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Fitted line
    x_fit = np.logspace(np.log10(n_params.min() * 0.5), np.log10(n_params.max() * 2), 200)
    y_fit = fit["a"] * x_fit ** (-fit["alpha"])
    ax.plot(x_fit, y_fit, "--", color=COLORS[3],
            label=f"Fit: L = {fit['a']:.1f} × N^(-{fit['alpha']:.3f})")

    # CI shading
    if "alpha_ci_lo" in fit and "alpha_ci_hi" in fit:
        y_lo = fit["a"] * x_fit ** (-fit["alpha_ci_hi"])
        y_hi = fit["a"] * x_fit ** (-fit["alpha_ci_lo"])
        ax.fill_between(x_fit, y_lo, y_hi, alpha=0.15, color=COLORS[3])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (N)", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Loss vs Model Size", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_saved", path=str(output_path))


def plot_loss_curves(
    results: list[dict],
    output_path: str | Path,
) -> None:
    """Training curves for all runs, x-axis = tokens seen."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for i, result in enumerate(results):
        name = result.get("model_size_name", f"Run {i}")
        losses = result.get("train_losses", [])
        if not losses:
            continue
        steps, loss_vals = zip(*losses)
        tokens = [s * result.get("n_tokens_trained", 1) / max(max(steps), 1) for s in steps]
        ax.plot(tokens, loss_vals, label=f"{name} (final: {loss_vals[-1]:.3f})",
                color=COLORS[i % len(COLORS)], linewidth=1.5)

    ax.set_xlabel("Tokens Seen", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Curves", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compute_scaling(
    flops: np.ndarray,
    val_losses: np.ndarray,
    fit: dict[str, float],
    output_path: str | Path,
) -> None:
    """Log-log plot of val loss vs total FLOPs."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.scatter(flops, val_losses, s=80, zorder=5, color=COLORS[1])

    x_fit = np.logspace(np.log10(flops.min() * 0.5), np.log10(flops.max() * 2), 200)
    y_fit = fit["a"] * x_fit ** (-fit["alpha"])
    ax.plot(x_fit, y_fit, "--", color=COLORS[4],
            label=f"Fit: L ∝ C^(-{fit['alpha']:.3f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute (FLOPs)", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Loss vs Compute", fontsize=14)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_full_report(
    results: list[dict],
    output_dir: str | Path,
) -> str:
    """Generate all plots and a markdown summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from nanochat.scaling.analysis import fit_power_law

    n_params = np.array([r["n_params"] for r in results])
    val_losses = np.array([r["final_val_loss"] for r in results])
    flops = np.array([r["flops_total"] for r in results])

    # Fit and plot
    if len(results) >= 2:
        fit_n = fit_power_law(n_params, val_losses)
        plot_loss_vs_params(n_params, val_losses, fit_n, output_dir / "loss_vs_params.png",
                           [r.get("model_size_name", "") for r in results])

        fit_c = fit_power_law(flops, val_losses)
        plot_compute_scaling(flops, val_losses, fit_c, output_dir / "loss_vs_compute.png")

        plot_loss_curves(results, output_dir / "training_curves.png")
    else:
        fit_n = {"alpha": 0.0, "a": 0.0, "r_squared": 0.0}
        fit_c = {"alpha": 0.0, "a": 0.0, "r_squared": 0.0}

    # Markdown summary
    summary_path = output_dir / "scaling_report.md"
    with open(summary_path, "w") as f:
        f.write("# Scaling Law Experiment Report\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Model | Params | Tokens | FLOPs | Val Loss | Val PPL |\n")
        f.write("|-------|--------|--------|-------|----------|--------|\n")
        for r in results:
            f.write(f"| {r.get('model_size_name', 'N/A')} | {r['n_params']:,} | "
                    f"{r['n_tokens_trained']:,} | {r['flops_total']:.2e} | "
                    f"{r['final_val_loss']:.4f} | {r.get('final_val_ppl', 0):.1f} |\n")
        f.write(f"\n## Fitted Exponents\n\n")
        f.write(f"- L(N): α = {fit_n.get('alpha', 0):.4f} (R² = {fit_n.get('r_squared', 0):.3f})\n")
        f.write(f"- L(C): α = {fit_c.get('alpha', 0):.4f} (R² = {fit_c.get('r_squared', 0):.3f})\n")
        f.write(f"- Reference: Kaplan α_N=0.076, Chinchilla α≈0.34\n")

    logger.info("report_generated", path=str(summary_path))
    return str(summary_path)
