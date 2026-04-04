"""Supervised Fine-Tuning (SFT) trainer for nanochat-jax.

Trains a pretrained TransformerLM on conversation data using a response_mask
to compute loss only on assistant tokens (not system/user prompts).

Supports ChatML template::

    <|system|>
    You are helpful.
    <|end|><|user|>
    What is 2+2?
    <|end|><|assistant|>
    4
    <|end|>

The SFTDataset converts conversations (list of {role, content} dicts) into
input_ids, labels, response_mask, and attention_mask tensors.

Usage::

    dataset = SFTDataset(conversations, tokenizer, max_seq_len=512)
    trainer = SFTTrainer(model, dataset, train_cfg, model_cfg)
    trainer.train(num_steps=1000)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax
import structlog
from flax import nnx
from flax.nnx.transforms.autodiff import DiffState

from nanochat.config import ModelConfig, TrainingConfig
from nanochat.model.transformer import TransformerLM
from nanochat.training.loss import cross_entropy_loss, IGNORE_INDEX
from nanochat.training.optimizer import build_optimizer
from nanochat.training.checkpoint import CheckpointManager
from nanochat.training.lora import LoRAParam, get_lora_params

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Synthetic conversation data for testing
# ---------------------------------------------------------------------------

SYNTHETIC_CONVERSATIONS: list[list[dict[str, str]]] = [
    [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
    ],
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ],
    [
        {"role": "system", "content": "You answer questions concisely."},
        {"role": "user", "content": "What color is the sky?"},
        {"role": "assistant", "content": "Blue."},
    ],
    [
        {"role": "user", "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did the scarecrow win an award? Because he was outstanding in his field!"},
    ],
    [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Write hello world in Python."},
        {"role": "assistant", "content": "print('Hello, World!')"},
    ],
    [
        {"role": "user", "content": "What is 10 times 5?"},
        {"role": "assistant", "content": "10 times 5 is 50."},
    ],
    [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Name three colors."},
        {"role": "assistant", "content": "Red, green, and blue."},
    ],
]


# ---------------------------------------------------------------------------
# Simple tokenizer for testing (when no real tokenizer is available)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """Minimal character-level tokenizer for SFT testing.

    Maps each byte to an integer ID with special token support.
    This is only used when no real tokenizer (e.g., BPETokenizer) is provided.
    """

    SPECIAL_TOKENS = {
        "<|system|>": 250,
        "<|user|>": 251,
        "<|assistant|>": 252,
        "<|end|>": 253,
        "<|pad|>": 254,
        "<|bos|>": 255,
    }

    def __init__(self, vocab_size: int = 256) -> None:
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_id(self) -> int:
        return self.SPECIAL_TOKENS["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.SPECIAL_TOKENS["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.SPECIAL_TOKENS["<|end|>"]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to token IDs, handling special tokens."""
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)

        remaining = text
        while remaining:
            found = False
            for token_str, token_id in self.SPECIAL_TOKENS.items():
                if remaining.startswith(token_str):
                    ids.append(token_id)
                    remaining = remaining[len(token_str):]
                    found = True
                    break
            if not found:
                # Encode as byte value (0-249 range)
                ids.append(min(ord(remaining[0]), 249))
                remaining = remaining[1:]

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        special_ids = set(self.SPECIAL_TOKENS.values()) if skip_special_tokens else set()
        chars = []
        for i in ids:
            if i in special_ids:
                continue
            if 0 <= i < 128:
                chars.append(chr(i))
            else:
                chars.append("?")
        return "".join(chars)


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

class SFTDataset:
    """Dataset for supervised fine-tuning on conversation data.

    Converts conversations (list of {role, content} message dicts) into
    padded/truncated sequences with a response_mask that is 1 only on
    assistant response tokens.

    Each item returns:
        input_ids:     [max_seq_len] int32 — token IDs
        labels:        [max_seq_len] int32 — shifted labels (IGNORE_INDEX for masked)
        response_mask: [max_seq_len] int32 — 1 for assistant tokens, 0 elsewhere
        attention_mask:[max_seq_len] int32 — 1 for real tokens, 0 for padding

    Args:
        conversations: List of conversations. Each conversation is a list of
            {role: str, content: str} dicts.
        tokenizer: Tokenizer with encode() and special token support.
        max_seq_len: Maximum sequence length (truncation + padding target).
    """

    def __init__(
        self,
        conversations: list[list[dict[str, str]]] | None,
        tokenizer: Any,
        max_seq_len: int = 512,
    ) -> None:
        if conversations is None:
            conversations = SYNTHETIC_CONVERSATIONS

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.items: list[dict[str, np.ndarray]] = []

        for conv in conversations:
            item = self._process_conversation(conv)
            if item is not None:
                self.items.append(item)

        logger.info(
            "sft_dataset_built",
            n_conversations=len(conversations),
            n_valid_items=len(self.items),
            max_seq_len=max_seq_len,
        )

    def _process_conversation(
        self, messages: list[dict[str, str]]
    ) -> dict[str, np.ndarray] | None:
        """Tokenize a single conversation and build response mask.

        The ChatML template is applied manually so we can track which
        token positions correspond to assistant responses.

        Returns None if the conversation produces no tokens.
        """
        all_ids: list[int] = []
        all_mask: list[int] = []  # 1 = assistant token, 0 = other

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Encode role tag: <|role|>\n
            role_tag = f"<|{role}|>\n"
            role_ids = self.tokenizer.encode(role_tag)

            # Encode content + end tag: content\n<|end|>
            content_text = f"{content}\n<|end|>"
            content_ids = self.tokenizer.encode(content_text)

            # Role tag is never trained on
            all_ids.extend(role_ids)
            all_mask.extend([0] * len(role_ids))

            # Content: only assistant content is trained on
            all_ids.extend(content_ids)
            if role == "assistant":
                all_mask.extend([1] * len(content_ids))
            else:
                all_mask.extend([0] * len(content_ids))

        if len(all_ids) == 0:
            return None

        # Truncate
        if len(all_ids) > self.max_seq_len:
            all_ids = all_ids[: self.max_seq_len]
            all_mask = all_mask[: self.max_seq_len]

        seq_len = len(all_ids)

        # Build labels: shifted by 1 (predict next token)
        # labels[t] = input_ids[t+1], with IGNORE_INDEX for non-response and last position
        labels = [IGNORE_INDEX] * seq_len
        for t in range(seq_len - 1):
            if all_mask[t + 1] == 1:
                labels[t] = all_ids[t + 1]

        # Build attention mask (1 for real tokens)
        attention_mask = [1] * seq_len

        # Pad to max_seq_len
        pad_len = self.max_seq_len - seq_len
        pad_id = getattr(self.tokenizer, "pad_id", 0)

        all_ids = all_ids + [pad_id] * pad_len
        labels = labels + [IGNORE_INDEX] * pad_len
        all_mask = all_mask + [0] * pad_len
        attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": np.array(all_ids, dtype=np.int32),
            "labels": np.array(labels, dtype=np.int32),
            "response_mask": np.array(all_mask, dtype=np.int32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.items[idx]

    def make_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Iterator[dict[str, jax.Array]]:
        """Create an infinite batch iterator over the dataset.

        Args:
            batch_size: Number of examples per batch.
            shuffle: Whether to shuffle between epochs.
            seed: Random seed for shuffling.

        Yields:
            Dict with input_ids, labels, response_mask, attention_mask
            as JAX arrays of shape (batch_size, max_seq_len).
        """
        rng = np.random.RandomState(seed)
        indices = np.arange(len(self.items))

        while True:
            if shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices) - batch_size + 1, batch_size):
                batch_indices = indices[start : start + batch_size]
                batch = {
                    key: jnp.array(
                        np.stack([self.items[i][key] for i in batch_indices])
                    )
                    for key in ("input_ids", "labels", "response_mask", "attention_mask")
                }
                yield batch


# ---------------------------------------------------------------------------
# JIT-compiled train step factory
# ---------------------------------------------------------------------------

def _make_train_step_fn(wrt: type) -> "Callable":
    """Create a JIT-compiled SFT train step that computes gradients wrt *wrt*."""

    @nnx.jit
    def _train_step_jit(
        model: "TransformerLM",
        optimizer: nnx.Optimizer,
        batch: "dict[str, jax.Array]",
    ) -> "dict[str, jax.Array]":
        def loss_fn(m: "TransformerLM") -> "tuple[jax.Array, dict[str, jax.Array]]":
            logits, _ = m(
                batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                deterministic=False,
            )
            loss, metrics = cross_entropy_loss(
                logits=logits[:, :-1, :],
                labels=batch["labels"][:, :-1],
            )
            return loss, metrics

        (loss, metrics), grads = nnx.value_and_grad(
            loss_fn, has_aux=True, argnums=DiffState(0, wrt)
        )(model)

        grad_norm = optax.global_norm(jax.tree.leaves(grads))
        optimizer.update(model, grads)

        return {
            "loss": loss,
            "ce_loss": metrics["ce_loss"],
            "z_loss": metrics["z_loss"],
            "n_tokens": metrics["n_tokens"],
            "grad_norm": grad_norm,
        }

    return _train_step_jit


# ---------------------------------------------------------------------------
# SFT Trainer
# ---------------------------------------------------------------------------

class SFTTrainer:
    """Supervised Fine-Tuning trainer for TransformerLM.

    Computes cross-entropy loss only on assistant response tokens using
    the response_mask from SFTDataset. Supports both full fine-tuning
    and LoRA fine-tuning (when LoRA has been applied to the model).

    Args:
        model: TransformerLM model (optionally with LoRA applied).
        dataset: SFTDataset instance.
        train_cfg: Training configuration.
        model_cfg: Model configuration.
        use_lora: If True, only optimize LoRA parameters.
    """

    def __init__(
        self,
        model: TransformerLM,
        dataset: SFTDataset,
        train_cfg: TrainingConfig,
        model_cfg: ModelConfig,
        use_lora: bool = False,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.use_lora = use_lora

        # Build optimizer
        tx = build_optimizer(train_cfg)
        if use_lora:
            self.optimizer = nnx.Optimizer(model, tx, wrt=LoRAParam)
        else:
            self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        # Build data loader
        self.train_loader = dataset.make_loader(
            batch_size=train_cfg.batch_size,
            shuffle=True,
            seed=42,
        )

        # Checkpoint manager
        self.ckpt_manager = CheckpointManager(
            train_cfg.checkpoint_dir,
            keep_last_n=train_cfg.keep_last_n,
        )

        # State
        self.global_step = 0

        # Build JIT-compiled train step with correct wrt type
        _wrt = LoRAParam if use_lora else nnx.Param
        self._train_step_jit = _make_train_step_fn(_wrt)

        logger.info(
            "sft_trainer_initialized",
            n_examples=len(dataset),
            batch_size=train_cfg.batch_size,
            total_steps=train_cfg.total_steps,
            use_lora=use_lora,
            learning_rate=train_cfg.learning_rate,
        )

    def train_step(self, batch: dict[str, jax.Array]) -> dict[str, float]:
        """Execute one SFT training step, returning metrics as Python floats."""
        metrics = self._train_step_jit(self.model, self.optimizer, batch)
        return {k: float(v) for k, v in metrics.items()}

    def train(self, num_steps: int | None = None) -> dict[str, Any]:
        """Main SFT training loop.

        Args:
            num_steps: Number of steps to train. Defaults to train_cfg.total_steps.

        Returns:
            Final metrics dict.
        """
        total_steps = num_steps or self.train_cfg.total_steps
        logger.info("sft_training_start", total_steps=total_steps)

        start_time = time.time()

        for step in range(total_steps):
            self.global_step = step

            batch = next(self.train_loader)
            step_start = time.time()
            metrics = self.train_step(batch)
            step_time = time.time() - step_start

            tokens_in_step = int(metrics.get("n_tokens", 0))
            tokens_per_sec = tokens_in_step / max(step_time, 1e-6)

            if step % 100 == 0:
                logger.info(
                    "sft_train_step",
                    step=step,
                    loss=round(metrics["loss"], 4),
                    grad_norm=round(metrics["grad_norm"], 4),
                    n_response_tokens=tokens_in_step,
                    tokens_per_sec=int(tokens_per_sec),
                    step_time_ms=round(step_time * 1000, 1),
                )

            # Save checkpoint
            if step > 0 and step % self.train_cfg.save_every_steps == 0:
                self.ckpt_manager.save(step, self.model, metrics)

        # Final save
        total_time = time.time() - start_time
        final_metrics = {
            "final_step": self.global_step,
            "total_time_seconds": total_time,
        }
        self.ckpt_manager.save(self.global_step, self.model, final_metrics)

        logger.info("sft_training_complete", **final_metrics)
        return final_metrics
