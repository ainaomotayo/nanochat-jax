"""Enterprise-grade training loop for TransformerLM."""
from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import optax
import structlog
from flax import nnx

from nanochat.config import ModelConfig, TrainingConfig
from nanochat.model.transformer import TransformerLM
from nanochat.model.param_count import count_params, estimate_flops_per_token, compute_mfu
from nanochat.training.loss import cross_entropy_loss
from nanochat.training.optimizer import build_optimizer
from nanochat.training.checkpoint import CheckpointManager

logger = structlog.get_logger()

# Constants
DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


class Trainer:
    """Enterprise-grade training loop for TransformerLM.

    Handles: training steps, gradient accumulation, evaluation,
    checkpointing, logging, throughput monitoring, and graceful interruption.
    """

    def __init__(
        self,
        model: TransformerLM,
        train_loader: Iterator[dict[str, jax.Array]],
        val_loader: Iterator[dict[str, jax.Array]] | None,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        # Build optimizer
        tx = build_optimizer(train_cfg)
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        # Checkpoint manager
        self.ckpt_manager = CheckpointManager(
            train_cfg.checkpoint_dir,
            keep_last_n=train_cfg.keep_last_n,
        )

        # Resume from checkpoint if requested
        if train_cfg.resume_from:
            resume_path = Path(train_cfg.resume_from)
            if resume_path.exists():
                self.global_step = self.ckpt_manager.load(
                    resume_path, model, optimizer=self.optimizer
                )
                logger.info("resumed_from_checkpoint",
                            step=self.global_step, path=str(resume_path))
            else:
                logger.warning("resume_path_not_found", path=str(resume_path))

        # Compute dtype
        self.compute_dtype = DTYPE_MAP.get(train_cfg.dtype, jnp.bfloat16)

        # Param counts and FLOP estimates
        self.param_counts = count_params(model)
        self.flops_per_token = estimate_flops_per_token(model_cfg)

        # State
        self.global_step = 0
        self.best_val_loss = float("inf")
        self._interrupted = False

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)

        logger.info(
            "trainer_initialized",
            total_params=self.param_counts.get("total", 0),
            flops_per_token=self.flops_per_token,
            total_steps=train_cfg.total_steps,
            dtype=train_cfg.dtype,
        )

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle SIGINT by setting interrupt flag."""
        logger.warning("interrupt_received", step=self.global_step)
        self._interrupted = True

    @staticmethod
    @nnx.jit
    def _train_step_jit(
        model: TransformerLM,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """JIT-compiled single training step."""
        def loss_fn(model: TransformerLM) -> tuple[jax.Array, dict[str, jax.Array]]:
            logits, _ = model(
                batch["input_ids"],
                deterministic=False,
            )
            # Data contract: batch["labels"][t] = target for logit[t] (pre-shifted).
            # TokenDataset:   labels = window[1:]  (shifted by 1 at load time)
            # PackedBatch:    labels[t] = next token or IGNORE_INDEX
            # Synthetic:      labels = ids_full[:, 1:] (see train.py)
            # So: logits[:, :-1, :] paired with labels[:, :-1] is correct.
            labels_src = batch["labels"] if "labels" in batch else batch["input_ids"]
            loss, metrics = cross_entropy_loss(
                logits=logits[:, :-1, :],
                labels=labels_src[:, :-1],
            )
            return loss, metrics

        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        # Compute grad norm before optimizer step
        grad_norm = optax.global_norm(jax.tree.leaves(grads))

        optimizer.update(model, grads)

        return {
            "loss": loss,
            "ce_loss": metrics["ce_loss"],
            "z_loss": metrics["z_loss"],
            "n_tokens": metrics["n_tokens"],
            "grad_norm": grad_norm,
        }

    def train_step(self, batch: dict[str, jax.Array]) -> dict[str, float]:
        """Execute one training step and return metrics as Python floats."""
        metrics = self._train_step_jit(self.model, self.optimizer, batch)
        return {k: float(v) for k, v in metrics.items()}

    def evaluate(self) -> dict[str, float]:
        """Run evaluation loop on val_loader."""
        if self.val_loader is None:
            return {}

        total_loss = 0.0
        total_tokens = 0.0
        n_batches = 0

        for batch in self.val_loader:
            logits, _ = self.model(batch["input_ids"], deterministic=True)
            labels_src = batch.get("labels", batch["input_ids"])
            loss, metrics = cross_entropy_loss(
                logits=logits[:, :-1, :],
                labels=labels_src[:, :-1],
            )
            total_loss += float(loss) * float(metrics["n_tokens"])
            total_tokens += float(metrics["n_tokens"])
            n_batches += 1
            if n_batches >= self.train_cfg.eval_steps:
                break

        if total_tokens == 0:
            return {"val_loss": float("inf"), "val_ppl": float("inf")}

        val_loss = total_loss / total_tokens
        val_ppl = min(float(jnp.exp(jnp.float32(val_loss))), 1e6)

        return {"val_loss": val_loss, "val_ppl": val_ppl}

    def train(self) -> dict[str, Any]:
        """Main training loop.

        Returns:
            Final metrics dict with training summary.
        """
        logger.info("training_start", total_steps=self.train_cfg.total_steps)

        start_time = time.time()
        step_start = time.time()

        for step in range(self.global_step, self.train_cfg.total_steps):
            if self._interrupted:
                logger.info("training_interrupted", step=step)
                self.ckpt_manager.save(step, self.model, {"interrupted": True},
                                       optimizer=self.optimizer)
                break

            self.global_step = step

            # Get batch
            batch = next(self.train_loader)

            # Training step
            step_start = time.time()
            metrics = self.train_step(batch)
            step_time = time.time() - step_start

            # Compute throughput
            tokens_in_step = int(metrics.get("n_tokens", 0))
            tokens_per_sec = tokens_in_step / max(step_time, 1e-6)

            # Log metrics
            if step % 100 == 0:
                logger.info(
                    "train_step",
                    step=step,
                    loss=round(metrics["loss"], 4),
                    grad_norm=round(metrics["grad_norm"], 4),
                    tokens_per_sec=int(tokens_per_sec),
                    step_time_ms=round(step_time * 1000, 1),
                )

            # Evaluate
            if self.val_loader and step > 0 and step % self.train_cfg.eval_every_steps == 0:
                val_metrics = self.evaluate()
                logger.info("eval", step=step, **{k: round(v, 4) for k, v in val_metrics.items()})

                if val_metrics.get("val_loss", float("inf")) < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]

            # Save checkpoint
            if step > 0 and step % self.train_cfg.save_every_steps == 0:
                self.ckpt_manager.save(step, self.model, metrics,
                                       optimizer=self.optimizer)

        # Final save
        total_time = time.time() - start_time
        final_metrics = {
            "final_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "total_time_seconds": total_time,
            "total_params": self.param_counts.get("total", 0),
        }
        self.ckpt_manager.save(self.global_step, self.model, final_metrics,
                               optimizer=self.optimizer)

        logger.info("training_complete", **final_metrics)
        return final_metrics
