"""Autoregressive inference engine with KV caching."""
from __future__ import annotations
from typing import Generator
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
    1. Prefill: process full prompt in parallel, fill KV cache
    2. Decode: generate one token at a time using cached K/V
    """

    def __init__(self, model: TransformerLM, tokenizer: BaseTokenizer, cfg: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

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
            prompt: Single string or list of strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens (1.0 = none)
            seed: RNG seed for reproducibility
            stream: If True, return generator yielding tokens

        Returns:
            Generated text (string or list of strings, or generator if stream=True)
        """
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        # Encode prompts
        encoded = [self.tokenizer.encode(p, add_bos=True) for p in prompts]
        max_prompt_len = max(len(e) for e in encoded)

        # Pad prompts to same length
        pad_id = self.tokenizer.pad_id
        padded = [e + [pad_id] * (max_prompt_len - len(e)) for e in encoded]
        input_ids = jnp.array(padded, dtype=jnp.int32)  # [batch, prompt_len]

        batch_size = input_ids.shape[0]
        rng = jax.random.PRNGKey(seed)

        # Prefill: process full prompt
        logits, kv_caches = self.model(input_ids, deterministic=True)
        # Take logits at last real position for each sequence
        next_logits = logits[:, -1, :]  # [batch, vocab]

        # Decode loop
        generated_tokens = []
        eos_id = self.tokenizer.eos_id
        finished = jnp.zeros(batch_size, dtype=jnp.bool_)

        for step in range(max_new_tokens):
            rng, sample_rng = jax.random.split(rng)

            # Sample next token
            if temperature == 0:
                next_token = jnp.argmax(next_logits, axis=-1)  # [batch]
            else:
                next_token = combined_sample(
                    next_logits, sample_rng,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

            generated_tokens.append(next_token)

            # Check for EOS
            finished = finished | (next_token == eos_id)
            if jnp.all(finished):
                break

            # Decode step: forward single token with KV cache
            token_input = next_token[:, None]  # [batch, 1]
            logits, kv_caches = self.model(
                token_input, kv_caches=kv_caches, deterministic=True
            )
            next_logits = logits[:, -1, :]  # [batch, vocab]

        # Decode tokens to text
        all_tokens = jnp.stack(generated_tokens, axis=1)  # [batch, gen_len]
        results = []
        for i in range(batch_size):
            tokens = all_tokens[i].tolist()
            # Truncate at EOS
            if eos_id in tokens:
                tokens = tokens[:tokens.index(eos_id)]
            text = self.tokenizer.decode(tokens)
            results.append(text)

        if is_batch:
            return results
        return results[0]
