"""lm-eval harness adapter for nanochat-jax models.

Wraps an InferenceEngine + TransformerLM so that the lm-evaluation-harness
can call loglikelihood, loglikelihood_rolling, and generate_until.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import structlog

from nanochat.config import ModelConfig
from nanochat.inference.engine import InferenceEngine
from nanochat.model.transformer import TransformerLM
from nanochat.tokenizer.base import BaseTokenizer

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Graceful import of lm_eval
# ---------------------------------------------------------------------------
try:
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance

    _LM_EVAL_AVAILABLE = True
except ImportError:
    _LM_EVAL_AVAILABLE = False

    class LM:  # type: ignore[no-redef]
        """Stub so the class definition does not crash when lm-eval is absent."""

    class Instance:  # type: ignore[no-redef]
        """Stub."""


def _check_lm_eval() -> None:
    """Raise a helpful error when lm-eval is not installed."""
    if not _LM_EVAL_AVAILABLE:
        raise ImportError(
            "lm-eval is required for NanoChatJAXModel. "
            "Install it with: pip install lm-eval>=0.4"
        )


# ---------------------------------------------------------------------------
# Helper: group requests by similar length for efficient batching
# ---------------------------------------------------------------------------

def _group_by_length(items: list[tuple[int, list[int]]], batch_size: int):
    """Yield batches of (original_index, token_ids) grouped by similar length.

    Returns batches as ``(indices, padded_array, lengths)`` where
    *padded_array* has shape ``(B, max_len_in_batch)`` and *lengths*
    records each sequence's true length.
    """
    indexed = sorted(enumerate(items), key=lambda t: len(t[1][1]))
    for chunk_start in range(0, len(indexed), batch_size):
        chunk = indexed[chunk_start : chunk_start + batch_size]
        indices = [c[0] for c in chunk]
        seqs = [c[1][1] for c in chunk]
        max_len = max(len(s) for s in seqs)
        pad_id = 0  # will be overridden per-model, but 0 works for padding
        padded = [s + [pad_id] * (max_len - len(s)) for s in seqs]
        lengths = [len(s) for s in seqs]
        yield indices, np.array(padded, dtype=np.int32), lengths


class NanoChatJAXModel(LM):
    """lm-eval harness adapter for nanochat-jax TransformerLM.

    Provides the three methods the harness needs:

    - :meth:`loglikelihood` -- log P(continuation | context)
    - :meth:`loglikelihood_rolling` -- sliding-window perplexity
    - :meth:`generate_until` -- autoregressive generation until stop string
    """

    def __init__(
        self,
        model: TransformerLM,
        tokenizer: BaseTokenizer,
        cfg: ModelConfig,
        batch_size: int = 8,
    ):
        _check_lm_eval()
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self._batch_size = batch_size
        self.engine = InferenceEngine(model, tokenizer, cfg)
        logger.info(
            "lm_eval_adapter_init",
            vocab_size=cfg.vocab_size,
            max_seq_len=cfg.max_seq_len,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
        batch_size: int = 8,
        tokenizer: Optional[BaseTokenizer] = None,
        cfg: Optional[ModelConfig] = None,
    ) -> "NanoChatJAXModel":
        """Load a nanochat-jax model from a checkpoint directory.

        Args:
            path: Path to the checkpoint directory (containing model.pkl and metadata.json).
            device: One of ``"cpu"``, ``"gpu"``, ``"tpu"``.
            batch_size: Batch size for evaluation.
            tokenizer: Pre-built tokenizer. If None, creates a BPETokenizer.
            cfg: Model config. If None, tries to load from checkpoint or uses default.
        """
        import json
        import pickle
        from flax import nnx
        from nanochat.core.device import setup_device
        from nanochat.training.checkpoint import CheckpointManager

        setup_device(device)

        path = Path(path)

        # Load config from checkpoint metadata if available
        if cfg is None:
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = ModelConfig(**json.load(f))
            else:
                cfg = ModelConfig()

        # Build tokenizer
        if tokenizer is None:
            from nanochat.tokenizer.bpe import BPETokenizer
            tokenizer = BPETokenizer()

        # Build model and load weights
        model = TransformerLM(cfg, rngs=nnx.Rngs(params=0, dropout=1))
        ckpt_mgr = CheckpointManager(path.parent)
        ckpt_mgr.load(path, model)

        return cls(model, tokenizer, cfg, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def tok_encode(self, text: str, add_bos: bool = False) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_bos=add_bos)

    def tok_decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(ids)

    # ------------------------------------------------------------------
    # Core: loglikelihood
    # ------------------------------------------------------------------

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute log P(continuation | context) for each request.

        Each request carries ``(context, continuation)`` in its ``.args``.
        Returns a list of ``(log_likelihood, is_greedy)`` tuples.
        """
        results: list[Optional[tuple[float, bool]]] = [None] * len(requests)

        # Pre-encode all requests
        encoded: list[tuple[int, int, list[int], int]] = []
        for idx, req in enumerate(requests):
            context_str, continuation_str = req.args
            ctx_ids = self.tok_encode(context_str)
            cont_ids = self.tok_encode(continuation_str)
            # Truncate from the left if total exceeds max_seq_len
            full_ids = ctx_ids + cont_ids
            if len(full_ids) > self.cfg.max_seq_len:
                overflow = len(full_ids) - self.cfg.max_seq_len
                full_ids = full_ids[overflow:]
                # Adjust continuation length (it stays the same unless context was too short)
                cont_len = len(cont_ids)
            else:
                cont_len = len(cont_ids)
            encoded.append((idx, cont_len, full_ids, len(full_ids)))

        # Sort by length for efficient batching
        encoded.sort(key=lambda t: t[3])

        # Process in batches
        for chunk_start in range(0, len(encoded), self._batch_size):
            chunk = encoded[chunk_start : chunk_start + self._batch_size]
            batch_indices = [c[0] for c in chunk]
            batch_cont_lens = [c[1] for c in chunk]
            batch_seqs = [c[2] for c in chunk]
            max_len = max(c[3] for c in chunk)

            # Pad sequences
            pad_id = self.tokenizer.pad_id
            padded = [s + [pad_id] * (max_len - len(s)) for s in batch_seqs]
            input_ids = jnp.array(padded, dtype=jnp.int32)  # [B, L]

            # Forward pass -- get logits for all positions
            logits, _ = self.model(input_ids, deterministic=True)
            # logits: [B, L, vocab]

            # For each item in the batch, extract continuation log-probs
            for i, (orig_idx, cont_len, seq, seq_len) in enumerate(chunk):
                # The continuation tokens are the last cont_len tokens of seq
                # Logits at position t predict token at position t+1
                # So for continuation starting at position (seq_len - cont_len),
                # we need logits at positions (seq_len - cont_len - 1) through (seq_len - 2)
                # and targets are tokens at positions (seq_len - cont_len) through (seq_len - 1)
                start_pos = seq_len - cont_len
                # Logit positions: start_pos - 1 to seq_len - 2 (inclusive)
                # These predict tokens at positions: start_pos to seq_len - 1
                logit_slice = logits[i, start_pos - 1 : seq_len - 1, :]  # [cont_len, vocab]
                target_tokens = jnp.array(seq[start_pos:seq_len], dtype=jnp.int32)  # [cont_len]

                # Log-softmax for numerical stability
                log_probs = jax.nn.log_softmax(logit_slice, axis=-1)  # [cont_len, vocab]

                # Gather log-probs at target positions
                token_log_probs = log_probs[jnp.arange(cont_len), target_tokens]  # [cont_len]
                total_ll = float(jnp.sum(token_log_probs))

                # Check if every continuation token was the greedy (argmax) choice
                greedy_tokens = jnp.argmax(logit_slice, axis=-1)  # [cont_len]
                is_greedy = bool(jnp.all(greedy_tokens == target_tokens))

                results[orig_idx] = (total_ll, is_greedy)

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Core: loglikelihood_rolling (sliding-window PPL)
    # ------------------------------------------------------------------

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute rolling (unconditional) log-likelihood for each request.

        Used for perplexity calculations where the entire text is scored.
        Uses a sliding window of size ``max_seq_len`` to handle long texts.
        """
        results: list[tuple[float, bool]] = []

        for req in requests:
            (text,) = req.args
            token_ids = self.tok_encode(text)

            if len(token_ids) == 0:
                results.append((0.0, True))
                continue

            total_ll = 0.0
            all_greedy = True
            window_size = self.cfg.max_seq_len

            # Slide a window across the token sequence
            offset = 0
            while offset < len(token_ids):
                end = min(offset + window_size, len(token_ids))
                window_ids = token_ids[offset:end]
                seq_len = len(window_ids)

                if seq_len < 2:
                    # Need at least 2 tokens (one input, one target)
                    break

                input_arr = jnp.array([window_ids], dtype=jnp.int32)  # [1, seq_len]
                logits, _ = self.model(input_arr, deterministic=True)

                # Score positions 1..seq_len-1 (predicted from 0..seq_len-2)
                # For the first window, score all positions.
                # For subsequent windows, only score positions beyond the overlap.
                if offset == 0:
                    score_start = 0
                else:
                    # Only score the new (non-overlapping) portion
                    overlap = window_size // 2
                    score_start = overlap

                logit_slice = logits[0, score_start : seq_len - 1, :]  # [n_score, vocab]
                targets = jnp.array(window_ids[score_start + 1 : seq_len], dtype=jnp.int32)

                if logit_slice.shape[0] == 0:
                    break

                log_probs = jax.nn.log_softmax(logit_slice, axis=-1)
                token_log_probs = log_probs[jnp.arange(logit_slice.shape[0]), targets]
                total_ll += float(jnp.sum(token_log_probs))

                greedy_tokens = jnp.argmax(logit_slice, axis=-1)
                if not bool(jnp.all(greedy_tokens == targets)):
                    all_greedy = False

                # Advance with 50% overlap for context
                if end == len(token_ids):
                    break
                offset += window_size // 2

            results.append((total_ll, all_greedy))

        return results

    # ------------------------------------------------------------------
    # Core: generate_until
    # ------------------------------------------------------------------

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate text until a stop string is hit.

        Each request carries ``(context, gen_kwargs)`` in its ``.args``.
        ``gen_kwargs`` may include ``until`` (list of stop strings),
        ``max_gen_toks``, ``temperature``, etc.
        """
        results: list[Optional[str]] = [None] * len(requests)

        for idx, req in enumerate(requests):
            context_str, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
            temperature = gen_kwargs.get("temperature", 0.0)
            top_k = gen_kwargs.get("top_k", 0)
            top_p = gen_kwargs.get("top_p", 1.0)

            generated = self.engine.generate(
                context_str,
                max_new_tokens=max_gen_toks,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=0,
            )

            # Truncate at first stop string
            for stop_str in until:
                if stop_str in generated:
                    generated = generated[: generated.index(stop_str)]

            results[idx] = generated

        return results  # type: ignore[return-value]
