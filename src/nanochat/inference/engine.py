"""Autoregressive inference engine with KV caching."""
from __future__ import annotations
from typing import Generator, Iterator
import jax
import jax.numpy as jnp
import numpy as np
import structlog
from nanochat.config import ModelConfig
from nanochat.model.transformer import TransformerLM
from nanochat.tokenizer.base import BaseTokenizer
from nanochat.inference.kv_cache import KVCache, init_kv_caches
from nanochat.inference.sampling import combined_sample

logger = structlog.get_logger()


class InferenceEngine:
    """Autoregressive inference with KV cache.

    Two-phase decoding:

    1. **Prefill** – process full prompt in parallel, populate KV cache.
    2. **Decode** – generate one token at a time using cached K/V.

    Example::

        engine = InferenceEngine(model, tokenizer, cfg)
        text = engine.generate("Once upon a time", max_new_tokens=100)

        # Streaming – yields text fragments as they arrive
        for fragment in engine.generate("Hello", max_new_tokens=50, stream=True):
            print(fragment, end="", flush=True)
    """

    def __init__(self, model: TransformerLM, tokenizer: BaseTokenizer, cfg: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str | list[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        seed: int = 0,
        stream: bool = False,
    ) -> str | list[str] | Generator[str, None, None]:
        """Generate text from prompt(s).

        Args:
            prompt: Single string or list of strings.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k filtering (0 = disabled).
            top_p: Nucleus sampling threshold.
            repetition_penalty: Penalty for repeated tokens (1.0 = none).
            seed: RNG seed for reproducibility.
            stream: If True, return a generator that yields decoded text
                fragments one token at a time (single-prompt only).

        Returns:
            * ``str`` – if a single prompt was given and ``stream=False``.
            * ``list[str]`` – if a list of prompts and ``stream=False``.
            * ``Generator[str, None, None]`` – if ``stream=True`` (single
              prompt only; batch is ignored).
        """
        is_batch = isinstance(prompt, list)

        if stream:
            # Stream mode: always single-prompt
            single = prompt[0] if is_batch else prompt
            return self._stream_generate(
                single,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )

        prompts = prompt if is_batch else [prompt]
        results = self._batch_generate(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        return results if is_batch else results[0]

    # ------------------------------------------------------------------
    # Internal: non-streaming batch generation
    # ------------------------------------------------------------------

    def _batch_generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: int,
    ) -> list[str]:
        """Generate for a batch of prompts (no streaming)."""
        # Encode + pad
        encoded = [self.tokenizer.encode(p, add_bos=True) for p in prompts]
        max_prompt_len = max(len(e) for e in encoded)
        pad_id = self.tokenizer.pad_id
        padded = [e + [pad_id] * (max_prompt_len - len(e)) for e in encoded]
        input_ids = jnp.array(padded, dtype=jnp.int32)  # [B, L]
        batch_size = input_ids.shape[0]

        rng = jax.random.PRNGKey(seed)
        eos_id = self.tokenizer.eos_id
        finished = jnp.zeros(batch_size, dtype=jnp.bool_)

        # Prefill
        logits, kv_caches = self.model(input_ids, deterministic=True)
        next_logits = logits[:, -1, :]  # [B, vocab]

        generated_tokens: list[jax.Array] = []

        for _ in range(max_new_tokens):
            rng, sample_rng = jax.random.split(rng)

            if temperature == 0:
                next_token = jnp.argmax(next_logits, axis=-1)
            else:
                next_token = combined_sample(
                    next_logits, sample_rng,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

            generated_tokens.append(next_token)
            finished = finished | (next_token == eos_id)
            if jnp.all(finished):
                break

            token_input = next_token[:, None]  # [B, 1]
            logits, kv_caches = self.model(
                token_input, kv_caches=kv_caches, deterministic=True
            )
            next_logits = logits[:, -1, :]

        if not generated_tokens:
            return [""] * batch_size

        all_tokens = jnp.stack(generated_tokens, axis=1)  # [B, gen_len]
        results = []
        for i in range(batch_size):
            tokens = all_tokens[i].tolist()
            if eos_id in tokens:
                tokens = tokens[: tokens.index(eos_id)]
            results.append(self.tokenizer.decode(tokens))
        return results

    # ------------------------------------------------------------------
    # Internal: streaming generation (single prompt)
    # ------------------------------------------------------------------

    def _stream_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: int,
    ) -> Generator[str, None, None]:
        """Generator that yields decoded text fragments one token at a time."""
        encoded = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = jnp.array([encoded], dtype=jnp.int32)  # [1, L]

        rng = jax.random.PRNGKey(seed)
        eos_id = self.tokenizer.eos_id

        # Prefill
        logits, kv_caches = self.model(input_ids, deterministic=True)
        next_logits = logits[:, -1, :]  # [1, vocab]

        for _ in range(max_new_tokens):
            rng, sample_rng = jax.random.split(rng)

            if temperature == 0:
                next_token = jnp.argmax(next_logits, axis=-1)  # [1]
            else:
                next_token = combined_sample(
                    next_logits, sample_rng,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

            tok_id = int(next_token[0])
            if tok_id == eos_id:
                break

            # Yield decoded fragment immediately
            fragment = self.tokenizer.decode([tok_id], skip_special_tokens=True)
            if fragment:
                yield fragment

            token_input = next_token[:, None]  # [1, 1]
            logits, kv_caches = self.model(
                token_input, kv_caches=kv_caches, deterministic=True
            )
            next_logits = logits[:, -1, :]
