"""Scaling experiment report generator.

Reads JSON result files produced by ScalingRunner and generates
markdown reports with summary tables, power-law fit results,
and environment metadata.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all JSON result files from a directory, sorted by n_params."""
    results = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            results.append(json.load(f))
    results.sort(key=lambda r: r.get("n_params", 0))
    return results


def _format_params(n: int) -> str:
    """Format a parameter count as a human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _format_tokens(n: int) -> str:
    """Format a token count as a human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def generate_comparison_table(results: list[dict[str, Any]]) -> str:
    """Generate a markdown comparison table from a list of result dicts.

    Args:
        results: List of ScalingRunResult-like dicts with keys such as
            model_size_name, n_params, n_tokens_trained, final_val_loss,
            final_val_ppl, tokens_per_second, mfu.

    Returns:
        A markdown table string.
    """
    lines = [
        "| Model | Params | Tokens | Val Loss | Val PPL | Tok/s | MFU |",
        "|-------|--------|--------|----------|---------|-------|-----|",
    ]
    for r in results:
        name = r.get("model_size_name", "N/A")
        params = _format_params(r.get("n_params", 0))
        tokens = _format_tokens(r.get("n_tokens_trained", 0))
        loss = r.get("final_val_loss", 0.0)
        ppl = r.get("final_val_ppl", 0.0)
        tps = r.get("tokens_per_second", 0.0)
        mfu = r.get("mfu", 0.0)
        lines.append(
            f"| {name} | {params} | {tokens} | {loss:.4f} | {ppl:.1f} | "
            f"{tps:,.0f} | {mfu:.2%} |"
        )
    return "\n".join(lines)


def generate_report(
    results_dir: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """Generate a full markdown report from scaling experiment JSON results.

    Reads all ``*.json`` files in *results_dir*, produces a markdown
    report with summary tables, power-law fit statistics, and
    environment metadata.

    Args:
        results_dir: Directory containing ScalingRunResult JSON files.
        output_path: Where to write the markdown report. If None,
            defaults to ``results_dir / scaling_report.md``.

    Returns:
        The string content of the generated report.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    results = _load_results(results_dir)
    if not results:
        raise ValueError(f"No JSON result files found in {results_dir}")

    if output_path is None:
        output_path = results_dir / "scaling_report.md"
    output_path = Path(output_path)

    sections: list[str] = []

    # ── Header ────────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections.append(f"# Scaling Law Experiment Report\n\nGenerated: {timestamp}\n")

    # ── Summary table ─────────────────────────────────────────────
    sections.append("## Results Summary\n")
    sections.append(generate_comparison_table(results))
    sections.append("")

    # ── Power-law fits ────────────────────────────────────────────
    n_params = np.array([r["n_params"] for r in results])
    val_losses = np.array([r["final_val_loss"] for r in results])

    if len(results) >= 2:
        from nanochat.scaling.analysis import fit_power_law

        fit_n = fit_power_law(n_params, val_losses)
        sections.append("## Power Law Fits\n")
        sections.append(
            f"- **L(N)**: alpha = {fit_n['alpha']:.4f}, "
            f"a = {fit_n['a']:.4f}, "
            f"R^2 = {fit_n['r_squared']:.4f}"
        )
        sections.append(
            f"  - 90% CI for alpha: [{fit_n['alpha_ci_lo']:.4f}, "
            f"{fit_n['alpha_ci_hi']:.4f}]"
        )

        flops = np.array([r.get("flops_total", 0.0) for r in results])
        if np.all(flops > 0):
            fit_c = fit_power_law(flops, val_losses)
            sections.append(
                f"- **L(C)**: alpha = {fit_c['alpha']:.4f}, "
                f"a = {fit_c['a']:.4f}, "
                f"R^2 = {fit_c['r_squared']:.4f}"
            )
        sections.append(
            "\n- Reference: Kaplan et al. alpha_N ~ 0.076, "
            "Chinchilla alpha ~ 0.34\n"
        )
    else:
        sections.append(
            "## Power Law Fits\n\nInsufficient data points "
            "(need >= 2) for power law fitting.\n"
        )

    # ── Environment ───────────────────────────────────────────────
    sections.append("## Environment\n")
    try:
        import jax
        jax_version = jax.__version__
    except ImportError:
        jax_version = "N/A"
    sections.append(f"- Platform: {platform.system()} {platform.release()}")
    sections.append(f"- Python: {platform.python_version()}")
    sections.append(f"- JAX: {jax_version}")

    # ── Reproducibility ───────────────────────────────────────────
    sections.append("\n## Reproducibility\n")
    first_config = results[0].get("config_snapshot", {})
    seed_info = "N/A"
    if first_config:
        training_snap = first_config.get("training", {})
        seed_info = str(training_snap.get("seed", "default (42)"))
    sections.append(f"- Random seed: {seed_info}")
    sections.append(f"- Number of runs: {len(results)}")
    sections.append(f"- Results directory: `{results_dir}`\n")

    report_text = "\n".join(sections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text)

    logger.info("report_generated", path=str(output_path), n_runs=len(results))
    return report_text
