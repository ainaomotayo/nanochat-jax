"""Throughput benchmarking for training and inference."""
from __future__ import annotations
import time
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import structlog
from flax import nnx
from nanochat.config import ModelConfig, TrainingConfig
from nanochat.model.transformer import TransformerLM
from nanochat.model.param_count import estimate_flops_per_token, compute_mfu
from nanochat.training.optimizer import build_optimizer
from nanochat.training.loss import cross_entropy_loss

logger = structlog.get_logger()


@dataclass
class ThroughputReport:
    """Results from a throughput benchmark."""
    tokens_per_second: float
    samples_per_second: float
    mfu: float
    peak_memory_gb: float
    avg_step_time_ms: float
    n_steps: int


def benchmark_training_throughput(
    model: TransformerLM,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    *,
    n_warmup: int = 3,
    n_benchmark: int = 10,
    peak_device_flops: float = 312e12,  # A100 bf16 peak
) -> ThroughputReport:
    """Benchmark training throughput.

    Warms up JIT compilation, then measures stable throughput.
    """
    batch_size = train_cfg.batch_size
    seq_len = model_cfg.max_seq_len

    # Build optimizer
    tx = build_optimizer(train_cfg)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Dummy batch
    rng = jax.random.PRNGKey(0)
    dummy_ids = jax.random.randint(rng, (batch_size, seq_len), 0, model_cfg.vocab_size)
    batch = {
        "input_ids": dummy_ids,
        "labels": dummy_ids,
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
    }

    @nnx.jit
    def step(model, optimizer, batch):
        def loss_fn(model):
            logits, _ = model(batch["input_ids"], deterministic=False)
            loss, _ = cross_entropy_loss(logits[:, :-1], batch["labels"][:, 1:])
            return loss
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Warmup
    for _ in range(n_warmup):
        _ = step(model, optimizer, batch)
    jax.effects_barrier()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_benchmark):
        _ = step(model, optimizer, batch)
    jax.effects_barrier()
    elapsed = time.perf_counter() - start

    tokens_per_step = batch_size * seq_len
    total_tokens = n_benchmark * tokens_per_step
    tps = total_tokens / elapsed
    sps = n_benchmark * batch_size / elapsed
    avg_step_ms = (elapsed / n_benchmark) * 1000

    mfu_val = compute_mfu(tps, model_cfg, peak_device_flops)

    report = ThroughputReport(
        tokens_per_second=tps,
        samples_per_second=sps,
        mfu=mfu_val,
        peak_memory_gb=0.0,  # JAX doesn't easily expose this
        avg_step_time_ms=avg_step_ms,
        n_steps=n_benchmark,
    )

    logger.info("throughput_benchmark", tokens_per_sec=int(tps), mfu=round(mfu_val, 3),
                avg_step_ms=round(avg_step_ms, 1))
    return report
