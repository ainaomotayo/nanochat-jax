"""Tests for the scaling report generator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanochat.scaling.report import generate_report, generate_comparison_table


@pytest.fixture
def sample_results() -> list[dict]:
    """Create sample scaling run results for testing."""
    return [
        {
            "run_id": "run_small",
            "model_size_name": "d128_l4",
            "n_params": 500_000,
            "n_tokens_trained": 100_000,
            "flops_total": 3e11,
            "final_val_loss": 4.5,
            "final_val_ppl": 90.0,
            "tokens_per_second": 50_000.0,
            "mfu": 0.12,
            "wall_time_seconds": 2.0,
            "train_losses": [(0, 6.0), (100, 5.0), (200, 4.5)],
            "val_losses": [(0, 6.0), (100, 5.0), (200, 4.5)],
            "config_snapshot": {
                "model": {"d_model": 128, "n_layers": 4},
                "training": {"seed": 42},
            },
        },
        {
            "run_id": "run_medium",
            "model_size_name": "d256_l6",
            "n_params": 5_000_000,
            "n_tokens_trained": 200_000,
            "flops_total": 6e12,
            "final_val_loss": 3.8,
            "final_val_ppl": 44.7,
            "tokens_per_second": 30_000.0,
            "mfu": 0.08,
            "wall_time_seconds": 6.7,
            "train_losses": [(0, 5.5), (100, 4.2), (200, 3.8)],
            "val_losses": [(0, 5.5), (100, 4.2), (200, 3.8)],
            "config_snapshot": {
                "model": {"d_model": 256, "n_layers": 6},
                "training": {"seed": 42},
            },
        },
        {
            "run_id": "run_large",
            "model_size_name": "d512_l6",
            "n_params": 50_000_000,
            "n_tokens_trained": 500_000,
            "flops_total": 1.5e14,
            "final_val_loss": 3.2,
            "final_val_ppl": 24.5,
            "tokens_per_second": 15_000.0,
            "mfu": 0.05,
            "wall_time_seconds": 33.3,
            "train_losses": [(0, 5.0), (100, 3.8), (200, 3.2)],
            "val_losses": [(0, 5.0), (100, 3.8), (200, 3.2)],
            "config_snapshot": {
                "model": {"d_model": 512, "n_layers": 6},
                "training": {"seed": 42},
            },
        },
    ]


class TestGenerateReport:
    """Test generate_report from JSON result files."""

    def test_generate_report_from_json(self, tmp_path: Path, sample_results: list[dict]):
        # Write sample result files
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        for r in sample_results:
            with open(results_dir / f"{r['run_id']}.json", "w") as f:
                json.dump(r, f)

        report = generate_report(results_dir)

        assert isinstance(report, str)
        assert "# Scaling Law Experiment Report" in report
        assert "Results Summary" in report
        assert "d128_l4" in report
        assert "d256_l6" in report
        assert "d512_l6" in report
        assert "Power Law Fits" in report
        assert "alpha" in report
        assert "Environment" in report
        assert "Reproducibility" in report

        # Verify the file was written
        report_path = results_dir / "scaling_report.md"
        assert report_path.exists()
        assert report_path.read_text() == report

    def test_generate_report_custom_output_path(self, tmp_path: Path, sample_results: list[dict]):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        for r in sample_results:
            with open(results_dir / f"{r['run_id']}.json", "w") as f:
                json.dump(r, f)

        custom_path = tmp_path / "custom" / "report.md"
        report = generate_report(results_dir, output_path=custom_path)

        assert custom_path.exists()
        assert "Scaling Law" in report

    def test_generate_report_no_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            generate_report(tmp_path / "nonexistent")

    def test_generate_report_empty_dir_raises(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No JSON result files"):
            generate_report(empty_dir)


class TestComparisonTable:
    """Test generate_comparison_table markdown formatting."""

    def test_comparison_table_format(self, sample_results: list[dict]):
        table = generate_comparison_table(sample_results)

        assert isinstance(table, str)
        lines = table.strip().split("\n")
        # Header + separator + 3 data rows
        assert len(lines) == 5

        # Check header row
        assert "Model" in lines[0]
        assert "Params" in lines[0]
        assert "Tok/s" in lines[0]
        assert "MFU" in lines[0]

        # Check separator
        assert lines[1].startswith("|---")

        # Check model names appear in table
        assert "d128_l4" in table
        assert "d256_l6" in table
        assert "d512_l6" in table

    def test_comparison_table_param_formatting(self, sample_results: list[dict]):
        table = generate_comparison_table(sample_results)
        # 500K params
        assert "500.0K" in table
        # 5M params
        assert "5.0M" in table
        # 50M params
        assert "50.0M" in table

    def test_comparison_table_empty(self):
        table = generate_comparison_table([])
        lines = table.strip().split("\n")
        # Just header and separator, no data rows
        assert len(lines) == 2
