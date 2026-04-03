#!/usr/bin/env python3
"""Launch scaling law experiments.

Usage:
    python scripts/run_scaling.py --experiment scale_n
    python scripts/run_scaling.py --experiment scale_d
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import structlog
import numpy as np
from nanochat.config import ModelConfig
from nanochat.scaling.runner import ScalingRunner
from nanochat.scaling.analysis import fit_power_law
from nanochat.scaling.visualization import generate_full_report

logger = structlog.get_logger()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run scaling law experiments")
    parser.add_argument("--experiment", type=str, default="scale_n",
                       choices=["scale_n", "scale_d", "scale_c"])
    parser.add_argument("--output-dir", type=str, default="outputs/scaling")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per run (for quick tests)")
    args = parser.parse_args()

    runner = ScalingRunner(output_dir=args.output_dir)

    if args.experiment == "scale_n":
        # Vary model size at fixed token budget
        configs = [
            ModelConfig.for_scale("nano"),
            ModelConfig.for_scale("small"),
        ]
        token_budget = args.max_steps * 8 * configs[0].max_seq_len  # rough
        results = runner.run_grid("scale_n", model_configs=configs,
                                  token_budgets=[token_budget])
    elif args.experiment == "scale_d":
        # Vary token budget at fixed model size
        cfg = ModelConfig.for_scale("nano")
        budgets = [args.max_steps * 8 * cfg.max_seq_len * m for m in [1, 2, 4]]
        results = runner.run_grid("scale_d", model_configs=[cfg],
                                  token_budgets=budgets)
    else:
        logger.info("experiment_type_not_yet_implemented", type=args.experiment)
        return

    # Analysis and plots
    if len(results) >= 2:
        from dataclasses import asdict
        result_dicts = [asdict(r) for r in results]
        report_path = generate_full_report(result_dicts, args.output_dir)
        logger.info("scaling_report", path=report_path)
    else:
        logger.warning("insufficient_results", n=len(results))


if __name__ == "__main__":
    main()
