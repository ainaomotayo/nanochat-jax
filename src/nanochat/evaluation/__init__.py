"""Evaluation module for nanochat-jax."""
from nanochat.evaluation.metrics import perplexity, bits_per_byte, token_accuracy
from nanochat.evaluation.evaluator import Evaluator
from nanochat.evaluation.throughput import ThroughputReport, benchmark_training_throughput
from nanochat.evaluation.suite import QUICK_TASKS, STANDARD_TASKS, run_eval_suite

__all__ = [
    "perplexity", "bits_per_byte", "token_accuracy",
    "Evaluator", "ThroughputReport", "benchmark_training_throughput",
    "QUICK_TASKS", "STANDARD_TASKS", "run_eval_suite",
]
