#!/usr/bin/env python3
"""Evaluation entry point for nanochat-jax.

Usage:
    python scripts/evaluate.py --model-size nano
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
import structlog
from flax import nnx

from nanochat.config import ModelConfig, TrainingConfig
from nanochat.model.transformer import TransformerLM
from nanochat.model.param_count import count_params
from nanochat.evaluation.evaluator import Evaluator
from nanochat.evaluation.throughput import benchmark_training_throughput

logger = structlog.get_logger()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate nanochat-jax model")
    parser.add_argument("--model-size", type=str, default="nano")
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--benchmark", action="store_true", help="Run throughput benchmark")
    args = parser.parse_args()

    cfg = ModelConfig.for_scale(args.model_size)
    model = TransformerLM(cfg, rngs=nnx.Rngs(params=42, dropout=43))

    logger.info("model_loaded", size=args.model_size, params=count_params(model).get("total", 0))

    # Synthetic eval data
    def eval_loader():
        rng = jax.random.PRNGKey(99)
        while True:
            rng, k = jax.random.split(rng)
            ids = jax.random.randint(k, (2, cfg.max_seq_len), 0, cfg.vocab_size)
            yield {"input_ids": ids, "labels": ids, "attention_mask": jnp.ones_like(ids)}

    evaluator = Evaluator(model, eval_steps=args.eval_steps)
    results = evaluator.evaluate(eval_loader())
    logger.info("evaluation_results", **{k: round(v, 4) for k, v in results.items()})

    if args.benchmark:
        train_cfg = TrainingConfig(batch_size=2, total_steps=100, dtype="float32")
        report = benchmark_training_throughput(model, cfg, train_cfg, n_warmup=2, n_benchmark=5)
        logger.info("throughput", tokens_per_sec=int(report.tokens_per_second),
                    avg_step_ms=round(report.avg_step_time_ms, 1))


if __name__ == "__main__":
    main()
