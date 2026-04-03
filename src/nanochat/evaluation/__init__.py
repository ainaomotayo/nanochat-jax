"""Evaluation module for nanochat-jax."""
from nanochat.evaluation.metrics import perplexity, bits_per_byte, token_accuracy
from nanochat.evaluation.evaluator import Evaluator
from nanochat.evaluation.throughput import ThroughputReport, benchmark_training_throughput

__all__ = [
    "perplexity", "bits_per_byte", "token_accuracy",
    "Evaluator", "ThroughputReport", "benchmark_training_throughput",
]
