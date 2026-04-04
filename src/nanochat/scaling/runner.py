"""Scaling law experiment runner."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator, Literal
import numpy as np
import jax
import jax.numpy as jnp
import structlog
from flax import nnx
from nanochat.config import ModelConfig, TrainingConfig
from nanochat.model.transformer import TransformerLM
from nanochat.model.param_count import count_params, estimate_flops_per_token, compute_mfu
from nanochat.training.optimizer import build_optimizer
from nanochat.training.loss import cross_entropy_loss

logger = structlog.get_logger()

# RTX 3050 peak FP32 TFLOPS ≈ 9.0 × 10^12. Override via run_single(peak_flops=...).
_DEFAULT_PEAK_FLOPS = 9.0e12


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


def _build_model_for_target_params(
    target_params: int,
    vocab_size: int = 256,
    max_seq_len: int = 128,
) -> ModelConfig:
    """Return a ModelConfig whose total parameter count is closest to target_params.

    Searches over a grid of (d_model, n_layers) combinations using the
    approximate formula N ≈ 12 * L * d² (attention + FFN, ignoring vocab).
    """
    best_cfg: ModelConfig | None = None
    best_delta = float("inf")

    for n_layers in [2, 3, 4, 5, 6, 8, 10, 12]:
        for d_model in [32, 48, 64, 96, 128, 192, 256, 384, 512, 768]:
            # Quick param estimate: 12 * L * d^2  (ignores vocab/head counts)
            approx = 12 * n_layers * d_model * d_model
            delta = abs(approx - target_params)
            if delta < best_delta:
                best_delta = delta
                n_heads = max(1, d_model // 64) if d_model >= 64 else max(1, d_model // 32)
                # Ensure heads divides d_model
                while d_model % n_heads != 0 and n_heads > 1:
                    n_heads -= 1
                best_cfg = ModelConfig(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    n_kv_heads=n_heads,
                    max_seq_len=max_seq_len,
                )

    assert best_cfg is not None
    return best_cfg


class ScalingRunner:
    """Orchestrates a grid of training runs for scaling law analysis.

    Supports three experiment modes:

    * **scale_n** – Fix token budget, vary model size. Reveals how loss
      scales with parameter count N at constant compute.
    * **scale_d** – Fix model size, vary token budget. Reveals how loss
      scales with data D.
    * **scale_c** – Vary both N and D along the Chinchilla compute
      frontier (C = 6·N·D = const). Reveals the envelope of optimal
      loss vs. compute.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def run_single(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        run_id: str,
        token_budget: int | None = None,
        data_loader: Iterator[dict[str, Any]] | None = None,
        peak_flops: float = _DEFAULT_PEAK_FLOPS,
    ) -> ScalingRunResult:
        """Execute one training experiment.

        Args:
            model_cfg: Model architecture.
            train_cfg: Training hyperparameters.
            run_id: Unique experiment identifier.
            token_budget: If set, overrides train_cfg.total_steps.
            data_loader: Optional real data iterator. If None, uses
                synthetic random tokens.
            peak_flops: Hardware peak FLOPS for MFU computation.

        Returns:
            :class:`ScalingRunResult` with training curves and metrics.
        """
        logger.info("scaling_run_start", run_id=run_id,
                    d_model=model_cfg.d_model, n_layers=model_cfg.n_layers)

        # Override steps from token budget
        if token_budget is not None:
            tokens_per_step = train_cfg.batch_size * model_cfg.max_seq_len
            train_cfg = train_cfg.model_copy(
                update={"total_steps": max(1, token_budget // tokens_per_step)}
            )

        # Build model + optimizer
        rngs = nnx.Rngs(params=42, dropout=43)
        model = TransformerLM(model_cfg, rngs=rngs)
        n_params = count_params(model).get("total", 0)
        flops_per_token = estimate_flops_per_token(model_cfg)

        tx = build_optimizer(train_cfg)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        # ── JIT-compiled train step (fixed label convention) ──────────
        @nnx.jit
        def train_step(model, optimizer, batch):
            def loss_fn(m):
                logits, _ = m(batch["input_ids"], deterministic=False)
                # Data contract: labels are pre-shifted (labels[t] = next token).
                # logits[:, :-1, :] paired with labels[:, :-1] is correct.
                loss, metrics = cross_entropy_loss(
                    logits[:, :-1, :], batch["labels"][:, :-1]
                )
                return loss, metrics

            (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            optimizer.update(model, grads)
            return loss

        # ── Data source ───────────────────────────────────────────────
        rng = jax.random.PRNGKey(42)

        def _synthetic_batch():
            nonlocal rng
            rng, k = jax.random.split(rng)
            # Generate S+1 tokens so labels are a proper shifted window
            ids = jax.random.randint(
                k,
                (train_cfg.batch_size, model_cfg.max_seq_len + 1),
                0,
                model_cfg.vocab_size,
            )
            return {"input_ids": ids[:, :-1], "labels": ids[:, 1:]}

        def _next_batch():
            if data_loader is not None:
                return next(data_loader)
            return _synthetic_batch()

        # ── Training loop ─────────────────────────────────────────────
        train_losses: list[tuple[int, float]] = []
        val_losses: list[tuple[int, float]] = []
        log_every = max(train_cfg.total_steps // 20, 1)
        start_time = time.time()
        loss = jnp.array(0.0)

        for step in range(train_cfg.total_steps):
            batch = _next_batch()
            loss = train_step(model, optimizer, batch)

            if step % log_every == 0:
                lv = float(loss)
                train_losses.append((step, lv))
                val_losses.append((step, lv))
                logger.info("scaling_step", run_id=run_id, step=step,
                            loss=round(lv, 4))

        wall_time = time.time() - start_time
        n_tokens = train_cfg.total_steps * train_cfg.batch_size * model_cfg.max_seq_len
        tps = n_tokens / max(wall_time, 1e-6)

        # ── MFU ───────────────────────────────────────────────────────
        try:
            mfu_val = compute_mfu(tps, model_cfg, peak_flops=peak_flops)
        except Exception:
            mfu_val = 0.0

        final_loss = float(loss)
        result = ScalingRunResult(
            run_id=run_id,
            model_size_name=f"d{model_cfg.d_model}_l{model_cfg.n_layers}",
            n_params=n_params,
            n_tokens_trained=n_tokens,
            flops_total=3.0 * flops_per_token * n_tokens,
            final_val_loss=final_loss,
            final_val_ppl=float(jnp.exp(jnp.float32(min(final_loss, 20.0)))),
            tokens_per_second=tps,
            mfu=mfu_val,
            wall_time_seconds=wall_time,
            train_losses=train_losses,
            val_losses=val_losses,
            config_snapshot={
                "model": model_cfg.model_dump(),
                "training": train_cfg.model_dump(),
            },
        )

        result_path = self.output_dir / f"{run_id}.json"
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        logger.info("scaling_run_complete", run_id=run_id, n_params=n_params,
                    final_loss=round(final_loss, 4),
                    mfu=round(mfu_val, 4),
                    tok_per_sec=int(tps),
                    wall_s=round(wall_time, 1))
        return result

    # ------------------------------------------------------------------
    # Grid experiments
    # ------------------------------------------------------------------

    def run_grid(
        self,
        experiment_type: Literal["scale_n", "scale_d", "scale_c"],
        model_configs: list[ModelConfig] | None = None,
        token_budgets: list[int] | None = None,
        compute_budgets: list[float] | None = None,
        seeds: list[int] | None = None,
        peak_flops: float = _DEFAULT_PEAK_FLOPS,
        data_loader_factory: Any | None = None,
    ) -> list[ScalingRunResult]:
        """Run a structured grid of scaling experiments.

        Args:
            experiment_type: One of ``"scale_n"``, ``"scale_d"``,
                ``"scale_c"``.
            model_configs: List of :class:`ModelConfig` (used by
                scale_n and scale_c).
            token_budgets: Token budget(s) for scale_n / scale_d.
            compute_budgets: FLOPs budgets for scale_c (e.g.
                ``[1e13, 1e14, 1e15]``).
            seeds: Random seeds; results averaged implicitly.
            peak_flops: Hardware peak FLOPS for MFU.
            data_loader_factory: Callable ``(model_cfg, train_cfg) ->
                Iterator``. If None, uses synthetic data.

        Returns:
            List of :class:`ScalingRunResult`, one per run.
        """
        seeds = seeds or [42]
        results: list[ScalingRunResult] = []

        if experiment_type == "scale_n":
            # Vary model size at fixed token budget
            cfgs = model_configs or []
            budgets = token_budgets or [100_000]
            for cfg in cfgs:
                for budget in budgets:
                    for seed in seeds:
                        run_id = f"scaleN_d{cfg.d_model}_l{cfg.n_layers}_t{budget}_s{seed}"
                        train_cfg = TrainingConfig(
                            batch_size=8,
                            optimizer="adamw",
                            total_steps=max(1, budget // (8 * cfg.max_seq_len)),
                        )
                        dl = data_loader_factory(cfg, train_cfg) if data_loader_factory else None
                        r = self.run_single(cfg, train_cfg, run_id,
                                            token_budget=budget,
                                            data_loader=dl,
                                            peak_flops=peak_flops)
                        results.append(r)

        elif experiment_type == "scale_d":
            # Vary token budget at fixed model size
            cfg = (model_configs or [None])[0]
            if cfg is None:
                raise ValueError("scale_d requires at least one model_config")
            budgets = token_budgets or [50_000, 100_000, 200_000]
            for budget in budgets:
                for seed in seeds:
                    run_id = f"scaleD_d{cfg.d_model}_t{budget}_s{seed}"
                    train_cfg = TrainingConfig(
                        batch_size=8,
                        optimizer="adamw",
                        total_steps=max(1, budget // (8 * cfg.max_seq_len)),
                    )
                    dl = data_loader_factory(cfg, train_cfg) if data_loader_factory else None
                    r = self.run_single(cfg, train_cfg, run_id,
                                        token_budget=budget,
                                        data_loader=dl,
                                        peak_flops=peak_flops)
                    results.append(r)

        elif experiment_type == "scale_c":
            # Compute frontier: Chinchilla-optimal N* and D* for each budget C
            from nanochat.scaling.analysis import chinchilla_optimal

            budgets = np.array(compute_budgets or [1e12, 3e12, 1e13, 3e13, 1e14])
            df = chinchilla_optimal(budgets)
            base_vocab = (model_configs[0].vocab_size if model_configs else 256)

            for _, row in df.iterrows():
                N_target = int(row["n_params"])
                D_target = max(int(row["n_tokens"]), 1000)
                cfg = _build_model_for_target_params(N_target, vocab_size=base_vocab)
                actual_n = count_params(
                    TransformerLM(cfg, rngs=nnx.Rngs(params=0, dropout=1))
                ).get("total", 0)

                run_id = (f"scaleC_C{row['compute_flops']:.1e}"
                          f"_N{actual_n}_D{D_target}")
                train_cfg = TrainingConfig(
                    batch_size=8,
                    optimizer="adamw",
                    total_steps=max(1, D_target // (8 * cfg.max_seq_len)),
                )
                dl = data_loader_factory(cfg, train_cfg) if data_loader_factory else None
                r = self.run_single(cfg, train_cfg, run_id,
                                    data_loader=dl,
                                    peak_flops=peak_flops)
                results.append(r)

        else:
            raise ValueError(
                f"Unknown experiment_type={experiment_type!r}. "
                "Choose from 'scale_n', 'scale_d', 'scale_c'."
            )

        logger.info("scaling_grid_complete",
                    experiment_type=experiment_type,
                    n_runs=len(results))
        return results
