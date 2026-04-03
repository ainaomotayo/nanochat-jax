"""Evaluation harness for language models."""
from __future__ import annotations
from typing import Iterator
import jax
import jax.numpy as jnp
import structlog
from nanochat.model.transformer import TransformerLM
from nanochat.training.loss import cross_entropy_loss
from nanochat.evaluation.metrics import perplexity, bits_per_byte, token_accuracy

logger = structlog.get_logger()


class Evaluator:
    """Runs evaluation loop on a validation dataset.

    Computes loss, perplexity, BPB, and token accuracy over eval_steps batches.
    """

    def __init__(self, model: TransformerLM, eval_steps: int = 100):
        self.model = model
        self.eval_steps = eval_steps

    def evaluate(self, val_loader: Iterator[dict[str, jax.Array]]) -> dict[str, float]:
        """Run evaluation and return metrics dict.

        Returns:
            Dict with keys: val_loss, val_ppl, val_bpb, val_accuracy
        """
        total_loss = 0.0
        total_tokens = 0.0
        total_correct = 0
        total_positions = 0

        for step_idx in range(self.eval_steps):
            try:
                batch = next(val_loader)
            except StopIteration:
                break

            logits, _ = self.model(batch["input_ids"], deterministic=True)
            shifted_logits = logits[:, :-1, :]
            shifted_labels = batch.get("labels", batch["input_ids"][:, 1:])
            if shifted_labels.shape[1] > shifted_logits.shape[1]:
                shifted_labels = shifted_labels[:, :shifted_logits.shape[1]]

            loss, metrics = cross_entropy_loss(shifted_logits, shifted_labels)
            n_tok = float(metrics["n_tokens"])
            total_loss += float(loss) * n_tok
            total_tokens += n_tok

            # Token accuracy
            preds = jnp.argmax(shifted_logits, axis=-1)
            mask = (shifted_labels != -100)
            correct = ((preds == shifted_labels) & mask).sum()
            total_correct += int(correct)
            total_positions += int(mask.sum())

        if total_tokens == 0:
            return {"val_loss": float("inf"), "val_ppl": float("inf"),
                    "val_bpb": float("inf"), "val_accuracy": 0.0}

        avg_loss = total_loss / total_tokens
        results = {
            "val_loss": avg_loss,
            "val_ppl": perplexity(avg_loss),
            "val_bpb": bits_per_byte(avg_loss),
            "val_accuracy": total_correct / max(total_positions, 1),
        }
        logger.info("evaluation_complete", **{k: round(v, 4) for k, v in results.items()})
        return results
