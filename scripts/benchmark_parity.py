#!/usr/bin/env python3
"""Benchmark comparison script for nanochat-jax.

Measures throughput (tokens/sec) across model sizes and optionally runs
a scaling experiment, writing results as JSON and markdown summary.

Usage:
    python scripts/benchmark_parity.py --device cpu --model_sizes nano,small --output_dir results/bench
    python scripts/benchmark_parity.py --device gpu --model_sizes nano,small,medium --run_scaling
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
import numpy as np
import structlog
from flax import nnx

from nanochat.config import ModelConfig, TrainingConfig
from nanochat.core.device import setup_device, device_info
from nanochat.model.transformer import TransformerLM
from nanochat.model.param_count import count_params, estimate_flops_per_token, compute_mfu
from nanochat.training.loss import cross_entropy_loss

logger = structlog.get_logger()

# RTX 3050 FP32 peak ~9 TFLOPS; override with --peak_flops
_DEFAULT_PEAK_FLOPS = 9.0e12


def _benchmark_throughput(
    model_cfg: ModelConfig,
    batch_size: int,
    n_steps: int,
    warmup_steps: int,
    peak_flops: float,
) -> dict:
    """Benchmark forward+backward throughput for a model config.

    Returns a dict with model info, tokens/sec, and MFU.
    """
    rngs = nnx.Rngs(params=42, dropout=43)
    model = TransformerLM(model_cfg, rngs=rngs)
    n_params = count_params(model).get("total", 0)
    flops_per_token = estimate_flops_per_token(model_cfg)

    # Dummy optimizer for realistic memory/compute
    import optax
    tx = optax.adamw(learning_rate=1e-4)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(m):
            logits, _ = m(batch["input_ids"], deterministic=False)
            loss, metrics = cross_entropy_loss(logits[:, :-1, :], batch["labels"][:, :-1])
            return loss, metrics

        (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return loss

    rng = jax.random.PRNGKey(0)

    def make_batch():
        nonlocal rng
        rng, k = jax.random.split(rng)
        ids = jax.random.randint(k, (batch_size, model_cfg.max_seq_len + 1), 0, model_cfg.vocab_size)
        return {"input_ids": ids[:, :-1], "labels": ids[:, 1:]}

    # Warmup
    for _ in range(warmup_steps):
        batch = make_batch()
        loss = train_step(model, optimizer, batch)
    jax.block_until_ready(loss)

    # Timed run
    start = time.perf_counter()
    for _ in range(n_steps):
        batch = make_batch()
        loss = train_step(model, optimizer, batch)
    jax.block_until_ready(loss)
    elapsed = time.perf_counter() - start

    tokens_total = n_steps * batch_size * model_cfg.max_seq_len
    tps = tokens_total / max(elapsed, 1e-9)

    try:
        mfu = compute_mfu(tps, model_cfg, peak_flops=peak_flops)
    except Exception:
        mfu = 0.0

    return {
        "model_size_name": f"d{model_cfg.d_model}_l{model_cfg.n_layers}",
        "d_model": model_cfg.d_model,
        "n_layers": model_cfg.n_layers,
        "n_params": n_params,
        "batch_size": batch_size,
        "seq_len": model_cfg.max_seq_len,
        "n_steps": n_steps,
        "wall_time_s": round(elapsed, 3),
        "tokens_per_second": round(tps, 1),
        "flops_per_token": flops_per_token,
        "mfu": round(mfu, 4),
        "final_loss": float(loss),
    }


def _write_markdown_summary(results: list[dict], output_dir: Path, dev_info: dict) -> Path:
    """Write a markdown summary of benchmark results."""
    md_path = output_dir / "benchmark_summary.md"
    with open(md_path, "w") as f:
        f.write("# nanochat-jax Benchmark Results\n\n")
        f.write("## Environment\n\n")
        for k, v in dev_info.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Throughput\n\n")
        f.write("| Model | Params | Seq Len | Batch | Tok/s | MFU | Wall (s) |\n")
        f.write("|-------|--------|---------|-------|-------|-----|----------|\n")
        for r in results:
            params = r["n_params"]
            if params >= 1e6:
                params_str = f"{params / 1e6:.1f}M"
            elif params >= 1e3:
                params_str = f"{params / 1e3:.1f}K"
            else:
                params_str = str(params)
            f.write(
                f"| {r['model_size_name']} | {params_str} | {r['seq_len']} | "
                f"{r['batch_size']} | {r['tokens_per_second']:,.0f} | "
                f"{r['mfu']:.2%} | {r['wall_time_s']:.1f} |\n"
            )
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="nanochat-jax throughput benchmark")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "tpu"],
        help="Device to benchmark on (default: cpu).",
    )
    parser.add_argument(
        "--model_sizes",
        type=str,
        default="nano,small",
        help="Comma-separated model scale presets (default: nano,small).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for benchmarking (default: 4).",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=20,
        help="Number of timed training steps (default: 20).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=3,
        help="Number of warmup steps before timing (default: 3).",
    )
    parser.add_argument(
        "--peak_flops",
        type=float,
        default=_DEFAULT_PEAK_FLOPS,
        help=f"Hardware peak FLOPS for MFU (default: {_DEFAULT_PEAK_FLOPS:.1e}).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/benchmark",
        help="Directory to write results (default: results/benchmark).",
    )
    parser.add_argument(
        "--run_scaling",
        action="store_true",
        help="Also run a small scaling experiment after throughput benchmarks.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset HDF5 path (optional; uses synthetic data if not set).",
    )
    args = parser.parse_args()

    setup_device(args.device)
    dev_info = device_info()
    logger.info("benchmark_start", **dev_info)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_sizes = [s.strip() for s in args.model_sizes.split(",")]

    # ── Throughput benchmarks ─────────────────────────────────────
    results: list[dict] = []
    for size_name in model_sizes:
        logger.info("benchmarking", model_size=size_name)
        model_cfg = ModelConfig.for_scale(size_name)
        result = _benchmark_throughput(
            model_cfg=model_cfg,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            warmup_steps=args.warmup_steps,
            peak_flops=args.peak_flops,
        )
        results.append(result)
        logger.info(
            "benchmark_result",
            model=result["model_size_name"],
            n_params=result["n_params"],
            tok_per_sec=int(result["tokens_per_second"]),
            mfu=round(result["mfu"], 4),
        )

    # Write JSON results
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump({"device_info": dev_info, "results": results}, f, indent=2, default=str)

    # Write markdown summary
    md_path = _write_markdown_summary(results, output_dir, dev_info)

    print(f"\nBenchmark complete.")
    print(f"  JSON:     {json_path}")
    print(f"  Markdown: {md_path}")
    for r in results:
        print(f"  {r['model_size_name']}: {r['tokens_per_second']:,.0f} tok/s, MFU={r['mfu']:.2%}")

    # ── Optional scaling experiment ───────────────────────────────
    if args.run_scaling:
        from nanochat.scaling.runner import ScalingRunner

        logger.info("scaling_experiment_start")
        scaling_dir = output_dir / "scaling"
        runner = ScalingRunner(output_dir=scaling_dir)

        model_cfgs = [ModelConfig.for_scale(s) for s in model_sizes]
        scaling_results = runner.run_grid(
            experiment_type="scale_n",
            model_configs=model_cfgs,
            token_budgets=[50_000],
            peak_flops=args.peak_flops,
        )

        # Generate scaling report
        from nanochat.scaling.report import generate_report
        report_text = generate_report(scaling_dir)
        print(f"\n  Scaling report: {scaling_dir / 'scaling_report.md'}")


if __name__ == "__main__":
    main()
