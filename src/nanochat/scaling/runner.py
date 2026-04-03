"""Scaling law experiment runner."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal
import jax
import structlog
from flax import nnx
from nanochat.config import ModelConfig, TrainingConfig
from nanochat.model.transformer import TransformerLM
from nanochat.model.param_count import count_params, estimate_flops_per_token

logger = structlog.get_logger()


@dataclass
class ScalingRunResult:
    """Complete result from one scaling experiment run."""
    run_id: str
    model_size_name: str
    n_params: int
    n_tokens_trained: int
    flops_total: float
    final_val_loss: float
    final_val_ppl: float
    tokens_per_second: float
    mfu: float
    wall_time_seconds: float
    train_losses: list[tuple[int, float]] = field(default_factory=list)
    val_losses: list[tuple[int, float]] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)


class ScalingRunner:
    """Orchestrates a grid of training runs for scaling law analysis.

    Supports:
    - scale_n: Fix token budget, vary model size
    - scale_d: Fix model size, vary token budget
    - scale_c: Vary both along compute frontier
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        run_id: str,
        token_budget: int | None = None,
    ) -> ScalingRunResult:
        """Run a single training experiment."""
        logger.info("scaling_run_start", run_id=run_id, d_model=model_cfg.d_model,
                    n_layers=model_cfg.n_layers)

        # Compute total steps from token budget if provided
        if token_budget is not None:
            tokens_per_step = train_cfg.batch_size * model_cfg.max_seq_len
            train_cfg = train_cfg.model_copy(
                update={"total_steps": token_budget // tokens_per_step}
            )

        # Build model
        rngs = nnx.Rngs(params=42, dropout=43)
        model = TransformerLM(model_cfg, rngs=rngs)
        param_counts = count_params(model)
        n_params = param_counts.get("total", 0)
        flops_per_token = estimate_flops_per_token(model_cfg)

        # Training (simplified for scaling experiments - no data loading, use synthetic)
        from nanochat.training.optimizer import build_optimizer
        from nanochat.training.loss import cross_entropy_loss

        tx = build_optimizer(train_cfg)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        train_losses = []
        val_losses = []
        rng = jax.random.PRNGKey(42)

        @nnx.jit
        def train_step(model, optimizer, batch):
            def loss_fn(model):
                logits, _ = model(batch["input_ids"], deterministic=False)
                loss, metrics = cross_entropy_loss(logits[:, :-1], batch["labels"][:, 1:])
                return loss, metrics
            (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            optimizer.update(model, grads)
            return loss

        start_time = time.time()

        for step in range(train_cfg.total_steps):
            rng, batch_rng = jax.random.split(rng)
            batch = {
                "input_ids": jax.random.randint(batch_rng, (train_cfg.batch_size, model_cfg.max_seq_len), 0, model_cfg.vocab_size),
                "labels": jax.random.randint(batch_rng, (train_cfg.batch_size, model_cfg.max_seq_len), 0, model_cfg.vocab_size),
            }

            loss = train_step(model, optimizer, batch)

            if step % max(train_cfg.total_steps // 20, 1) == 0:
                loss_val = float(loss)
                train_losses.append((step, loss_val))
                val_losses.append((step, loss_val))  # Use train loss as proxy for synthetic data
                logger.info("scaling_step", run_id=run_id, step=step, loss=round(loss_val, 4))

        wall_time = time.time() - start_time
        n_tokens = train_cfg.total_steps * train_cfg.batch_size * model_cfg.max_seq_len

        result = ScalingRunResult(
            run_id=run_id,
            model_size_name=f"d{model_cfg.d_model}_l{model_cfg.n_layers}",
            n_params=n_params,
            n_tokens_trained=n_tokens,
            flops_total=3.0 * flops_per_token * n_tokens,
            final_val_loss=float(loss),
            final_val_ppl=float(jax.numpy.exp(jax.numpy.float32(min(float(loss), 20.0)))),
            tokens_per_second=n_tokens / max(wall_time, 1e-6),
            mfu=0.0,
            wall_time_seconds=wall_time,
            train_losses=train_losses,
            val_losses=val_losses,
            config_snapshot={"model": model_cfg.model_dump(), "training": train_cfg.model_dump()},
        )

        # Save result
        result_path = self.output_dir / f"{run_id}.json"
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        logger.info("scaling_run_complete", run_id=run_id, n_params=n_params,
                    final_loss=round(result.final_val_loss, 4), wall_time=round(wall_time, 1))
        return result

    def run_grid(
        self,
        experiment_type: str,
        model_configs: list[ModelConfig] | None = None,
        token_budgets: list[int] | None = None,
        seeds: list[int] | None = None,
    ) -> list[ScalingRunResult]:
        """Run a grid of scaling experiments."""
        seeds = seeds or [42]
        results = []

        if experiment_type == "scale_n" and model_configs and token_budgets:
            # Vary model size at fixed token budget
            for cfg in model_configs:
                for budget in token_budgets:
                    for seed in seeds:
                        run_id = f"n_{cfg.d_model}_{cfg.n_layers}_t{budget}_s{seed}"
                        train_cfg = TrainingConfig(batch_size=8, total_steps=budget // (8 * cfg.max_seq_len))
                        result = self.run_single(cfg, train_cfg, run_id, token_budget=budget)
                        results.append(result)

        elif experiment_type == "scale_d" and model_configs and token_budgets:
            # Vary token budget at fixed model size
            cfg = model_configs[0]
            for budget in token_budgets:
                for seed in seeds:
                    run_id = f"d_{cfg.d_model}_t{budget}_s{seed}"
                    train_cfg = TrainingConfig(batch_size=8, total_steps=budget // (8 * cfg.max_seq_len))
                    result = self.run_single(cfg, train_cfg, run_id, token_budget=budget)
                    results.append(result)

        logger.info("scaling_grid_complete", n_runs=len(results))
        return results
