"""Sampling strategies for text generation. All pure JAX, XLA-compatible."""
from __future__ import annotations
import jax
import jax.numpy as jnp


def greedy_sample(logits: jax.Array) -> jax.Array:
    """Deterministic argmax sampling.
    logits: [batch, vocab] -> tokens: [batch]
    """
    return jnp.argmax(logits, axis=-1)


def temperature_sample(logits: jax.Array, rng: jax.Array, temperature: float = 1.0) -> jax.Array:
    """Sample with temperature scaling.
    logits: [batch, vocab] -> tokens: [batch]
    """
    logits = logits / jnp.maximum(temperature, 1e-6)
    return jax.random.categorical(rng, logits, axis=-1)


def top_k_sample(logits: jax.Array, rng: jax.Array, k: int, temperature: float = 1.0) -> jax.Array:
    """Top-k filtering then temperature sampling.
    Zero out logits outside top-k, then sample.
    logits: [batch, vocab] -> tokens: [batch]
    """
    logits = logits / jnp.maximum(temperature, 1e-6)
    top_k_values = jax.lax.top_k(logits, k)[0]  # [batch, k]
    threshold = top_k_values[:, -1:]  # [batch, 1] - kth largest value
    logits = jnp.where(logits < threshold, -1e10, logits)
    return jax.random.categorical(rng, logits, axis=-1)


def top_p_sample(logits: jax.Array, rng: jax.Array, p: float = 0.95,
                 temperature: float = 1.0) -> jax.Array:
    """Nucleus (top-p) sampling.
    Keep smallest set of tokens whose cumulative probability >= p.
    logits: [batch, vocab] -> tokens: [batch]
    """
    logits = logits / jnp.maximum(temperature, 1e-6)

    # Sort logits descending
    sorted_indices = jnp.argsort(-logits, axis=-1)  # [batch, vocab]
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)  # [batch, vocab]

    # Compute cumulative probabilities
    sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)  # [batch, vocab]
    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)  # [batch, vocab]

    # Remove tokens with cumulative prob above threshold (keep first that crosses)
    sorted_mask = cumulative_probs - sorted_probs > p  # [batch, vocab]
    sorted_logits = jnp.where(sorted_mask, -1e10, sorted_logits)  # [batch, vocab]

    # Unsort back to original order
    unsort_indices = jnp.argsort(sorted_indices, axis=-1)
    logits = jnp.take_along_axis(sorted_logits, unsort_indices, axis=-1)

    return jax.random.categorical(rng, logits, axis=-1)


def combined_sample(
    logits: jax.Array,
    rng: jax.Array,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generated_ids: jax.Array | None = None,
) -> jax.Array:
    """Combined sampling: repetition penalty -> top-k -> top-p -> temperature -> sample.
    logits: [batch, vocab] -> tokens: [batch]
    """
    # Repetition penalty
    if repetition_penalty != 1.0 and generated_ids is not None:
        # For tokens already generated, divide positive logits and multiply negative logits
        vocab_size = logits.shape[-1]
        # Create mask of generated token IDs
        penalty_mask = jnp.zeros_like(logits)
        # Use scatter to mark generated positions (simplified: check each position)
        for_penalty = jnp.take_along_axis(logits, generated_ids, axis=-1)  # [batch, gen_len]
        # Apply penalty
        penalized = jnp.where(for_penalty > 0, for_penalty / repetition_penalty,
                              for_penalty * repetition_penalty)
        logits = logits.at[jnp.arange(logits.shape[0])[:, None], generated_ids].set(penalized)

    # Temperature
    logits = logits / jnp.maximum(temperature, 1e-6)

    # Top-k
    if top_k > 0:
        top_k_values = jax.lax.top_k(logits, top_k)[0]
        threshold = top_k_values[:, -1:]
        logits = jnp.where(logits < threshold, -1e10, logits)

    # Top-p
    if top_p < 1.0:
        sorted_indices = jnp.argsort(-logits, axis=-1)
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumulative = jnp.cumsum(sorted_probs, axis=-1)
        mask = cumulative - sorted_probs > top_p
        sorted_logits = jnp.where(mask, -1e10, sorted_logits)
        unsort = jnp.argsort(sorted_indices, axis=-1)
        logits = jnp.take_along_axis(sorted_logits, unsort, axis=-1)

    return jax.random.categorical(rng, logits, axis=-1)
