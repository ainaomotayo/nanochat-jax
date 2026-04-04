"""GRPO (Group Relative Policy Optimization) trainer for nanochat-jax.

Implements the GRPO algorithm from DeepSeek-R1:
1. For each prompt, generate a GROUP of completions from the policy.
2. Score completions with a reward function.
3. Normalize rewards within each group to get advantages.
4. Compute PPO-clip loss with KL penalty against a frozen reference model.

Key design: generation happens OUTSIDE jit (dynamic lengths), while log
prob computation and loss are INSIDE jit (fixed shapes after padding).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import optax
import structlog
from flax import nnx

from nanochat.config import ModelConfig, TrainingConfig
from nanochat.model.transformer import TransformerLM
from nanochat.tokenizer.base import BaseTokenizer
from nanochat.training.optimizer import build_optimizer
from nanochat.inference.engine import InferenceEngine

logger = structlog.get_logger()


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class GRPOConfig:
    """GRPO-specific hyperparameters."""

    group_size: int = 8
    """Number of completions sampled per prompt."""

    epsilon: float = 0.2
    """PPO clipping ratio."""

    kl_beta: float = 0.01
    """KL divergence penalty weight against reference model."""

    max_completion_len: int = 256
    """Maximum number of tokens to generate per completion."""

    temperature: float = 0.8
    """Sampling temperature for completion generation."""


# ======================================================================
# Reward functions
# ======================================================================

class RewardFunction:
    """Static reward functions for GRPO training."""

    @staticmethod
    def gsm8k_numeric(prompt: str, completion: str, answer: str) -> float:
        """Binary reward: +1.0 if the last number in completion matches the answer.

        Extracts all numbers (including negatives and decimals) from the
        completion text, then checks if the final one matches the expected
        answer string after stripping whitespace.
        """
        numbers = re.findall(r'-?\d+\.?\d*', completion)
        if not numbers:
            return 0.0
        answer_numbers = re.findall(r'-?\d+\.?\d*', answer.strip())
        if not answer_numbers:
            return 0.0
        return 1.0 if numbers[-1] == answer_numbers[-1] else 0.0

    @staticmethod
    def gsm8k_format(prompt: str, completion: str, answer: str) -> float:
        """Format reward: +0.5 for '####' separator, +0.5 for numeric ending.

        Encourages the model to follow GSM8K answer formatting conventions.
        """
        score = 0.0
        if "####" in completion:
            score += 0.5
        numbers = re.findall(r'-?\d+\.?\d*', completion)
        if numbers:
            # Check that the completion ends (approximately) with a number
            stripped = completion.rstrip()
            if stripped and re.search(r'-?\d+\.?\d*\s*$', stripped):
                score += 0.5
        return score

    @staticmethod
    def length_penalty(
        prompt: str,
        completion: str,
        answer: str,
        min_len: int = 10,
        max_len: int = 200,
    ) -> float:
        """Reward that penalizes completions shorter than min_len or longer than max_len.

        Returns 1.0 for completions within [min_len, max_len] word count.
        Linearly decays outside those bounds down to 0.0.
        """
        words = completion.split()
        n = len(words)
        if n < min_len:
            return max(0.0, n / max(min_len, 1))
        if n > max_len:
            overshoot = n - max_len
            decay = max(0.0, 1.0 - overshoot / max(max_len, 1))
            return decay
        return 1.0


# ======================================================================
# GRPO Trainer
# ======================================================================

RewardFn = Callable[[str, str, str], float]


class GRPOTrainer:
    """Group Relative Policy Optimization trainer.

    Holds a trainable policy model and a frozen reference model.
    Each train_step:
        1. Generates group_size completions per prompt (outside jit).
        2. Scores them with a reward function.
        3. Computes group-normalized advantages.
        4. Computes GRPO loss with PPO clipping and KL penalty (inside jit).
        5. Updates the policy model.
    """

    def __init__(
        self,
        policy_model: TransformerLM,
        ref_model: TransformerLM,
        tokenizer: BaseTokenizer,
        grpo_config: GRPOConfig,
        train_config: TrainingConfig,
        rngs: nnx.Rngs,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.grpo_config = grpo_config
        self.train_config = train_config
        self.rngs = rngs

        # Build optimizer for policy model only
        tx = build_optimizer(train_config)
        self.optimizer = nnx.Optimizer(policy_model, tx, wrt=nnx.Param)

        # Inference engine for generation (uses policy model)
        self.engine = InferenceEngine(
            policy_model,
            tokenizer,
            policy_model.cfg,
        )

        self.global_step = 0

        logger.info(
            "grpo_trainer_initialized",
            group_size=grpo_config.group_size,
            epsilon=grpo_config.epsilon,
            kl_beta=grpo_config.kl_beta,
            max_completion_len=grpo_config.max_completion_len,
            temperature=grpo_config.temperature,
        )

    # ------------------------------------------------------------------
    # Log probability computation (jit-compiled)
    # ------------------------------------------------------------------

    @staticmethod
    @nnx.jit
    def _compute_log_probs(
        model: TransformerLM,
        input_ids: jax.Array,
        response_mask: jax.Array,
    ) -> jax.Array:
        """Compute per-token log probabilities under a model.

        Args:
            model: TransformerLM to evaluate.
            input_ids: Token IDs [batch, seq_len].
            response_mask: Binary mask [batch, seq_len] indicating response
                tokens (1 = response, 0 = prompt/padding).

        Returns:
            log_probs: Per-token log probs [batch, seq_len-1] for each
                next-token prediction, masked by response_mask[:, 1:].
        """
        logits, _ = model(input_ids, deterministic=True)
        # logits[:, t, :] predicts token at position t+1
        # So logits[:, :-1, :] predicts input_ids[:, 1:]
        log_probs_all = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
        # Gather the log prob of the actual next token
        target_ids = input_ids[:, 1:]  # [batch, seq_len-1]
        # One-hot gather
        log_probs = jnp.take_along_axis(
            log_probs_all,
            target_ids[:, :, None],
            axis=-1,
        )[:, :, 0]  # [batch, seq_len-1]
        # Mask to only response tokens (shifted by 1 to match logits alignment)
        mask = response_mask[:, 1:]  # [batch, seq_len-1]
        return log_probs * mask

    # ------------------------------------------------------------------
    # GRPO loss (pure function, called inside jit)
    # ------------------------------------------------------------------

    @staticmethod
    def _grpo_loss(
        policy_log_probs: jax.Array,
        old_log_probs: jax.Array,
        ref_log_probs: jax.Array,
        advantages: jax.Array,
        mask: jax.Array,
        epsilon: float,
        kl_beta: float,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute the GRPO loss.

        Args:
            policy_log_probs: Current policy log probs [batch, seq].
            old_log_probs: Log probs from the sampling policy [batch, seq].
            ref_log_probs: Reference model log probs [batch, seq].
            advantages: Group-normalized advantages [batch, 1] broadcast over seq.
            mask: Response mask [batch, seq].
            epsilon: PPO clip ratio.
            kl_beta: KL penalty weight.

        Returns:
            (total_loss, metrics_dict)
        """
        # Importance sampling ratio
        ratio = jnp.exp(policy_log_probs - old_log_probs)

        # PPO clipped surrogate
        clipped = jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
        # advantages is [batch, 1], broadcasts over seq dim
        surr1 = ratio * advantages
        surr2 = clipped * advantages
        policy_loss_per_token = -jnp.minimum(surr1, surr2) * mask
        n_tokens = jnp.maximum(mask.sum(), 1.0)
        policy_loss = policy_loss_per_token.sum() / n_tokens

        # KL divergence (unbiased estimator): E[exp(log_pi - log_ref) - 1 - (log_pi - log_ref)]
        log_ratio_kl = policy_log_probs - ref_log_probs
        kl_per_token = (jnp.exp(log_ratio_kl) - 1.0 - log_ratio_kl) * mask
        kl = kl_per_token.sum() / n_tokens

        total_loss = policy_loss + kl_beta * kl

        metrics = {
            "policy_loss": policy_loss,
            "kl": kl,
            "total_loss": total_loss,
            "mean_ratio": (ratio * mask).sum() / n_tokens,
        }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        prompts: list[str],
        answers: list[str],
        reward_fn: RewardFn | None = None,
        seed: int = 0,
    ) -> dict[str, float]:
        """Execute one GRPO training step.

        Args:
            prompts: List of prompt strings.
            answers: Corresponding ground-truth answers for reward scoring.
            reward_fn: Reward function (prompt, completion, answer) -> float.
                Defaults to RewardFunction.gsm8k_numeric.
            seed: RNG seed for generation.

        Returns:
            Metrics dict with float values.
        """
        if reward_fn is None:
            reward_fn = RewardFunction.gsm8k_numeric

        cfg = self.grpo_config
        n_prompts = len(prompts)

        # ----- Phase 1: Generate completions (outside jit) -----
        all_completions: list[str] = []
        all_rewards: list[float] = []

        for p_idx, (prompt, answer) in enumerate(zip(prompts, answers)):
            # Generate group_size completions for this prompt
            batch_prompts = [prompt] * cfg.group_size
            completions = self.engine.generate(
                batch_prompts,
                max_new_tokens=cfg.max_completion_len,
                temperature=cfg.temperature,
                seed=seed + p_idx * cfg.group_size,
            )
            # Score each completion
            rewards = [reward_fn(prompt, c, answer) for c in completions]
            all_completions.extend(completions)
            all_rewards.extend(rewards)

        # ----- Phase 2: Compute group-normalized advantages -----
        rewards_array = jnp.array(all_rewards, dtype=jnp.float32)
        # Reshape to [n_prompts, group_size] for per-group normalization
        rewards_grouped = rewards_array.reshape(n_prompts, cfg.group_size)
        group_mean = jnp.mean(rewards_grouped, axis=1, keepdims=True)
        group_std = jnp.std(rewards_grouped, axis=1, keepdims=True)
        advantages_grouped = (rewards_grouped - group_mean) / (group_std + 1e-8)
        advantages_flat = advantages_grouped.reshape(-1)  # [n_prompts * group_size]

        # ----- Phase 3: Tokenize and pad sequences -----
        total_samples = n_prompts * cfg.group_size
        # Build full sequences: prompt + completion
        all_input_ids = []
        all_response_masks = []

        for p_idx, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
            for g_idx in range(cfg.group_size):
                comp_idx = p_idx * cfg.group_size + g_idx
                completion = all_completions[comp_idx]
                comp_ids = self.tokenizer.encode(completion)
                full_ids = prompt_ids + comp_ids
                # Response mask: 0 for prompt, 1 for completion
                resp_mask = [0] * len(prompt_ids) + [1] * len(comp_ids)
                all_input_ids.append(full_ids)
                all_response_masks.append(resp_mask)

        # Pad to uniform length
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_id
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(all_input_ids, all_response_masks):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_masks.append(mask + [0] * pad_len)

        input_ids = jnp.array(padded_ids, dtype=jnp.int32)     # [total, max_len]
        response_mask = jnp.array(padded_masks, dtype=jnp.float32)  # [total, max_len]
        advantages = advantages_flat[:, None]  # [total, 1] for broadcasting

        # ----- Phase 4: Compute log probs (inside jit) -----
        # Old log probs (from the current policy, before update — these are the
        # "sampling" log probs since we just generated from this policy)
        old_log_probs = self._compute_log_probs(
            self.policy_model, input_ids, response_mask,
        )
        old_log_probs = jax.lax.stop_gradient(old_log_probs)

        # Reference model log probs (frozen, no grad)
        ref_log_probs = self._compute_log_probs(
            self.ref_model, input_ids, response_mask,
        )
        ref_log_probs = jax.lax.stop_gradient(ref_log_probs)

        # ----- Phase 5: Compute loss and update (inside jit) -----
        mask_for_loss = response_mask[:, 1:]  # align with log_probs shape

        metrics = self._update_step(
            self.policy_model,
            self.optimizer,
            input_ids,
            response_mask,
            old_log_probs,
            ref_log_probs,
            advantages,
            mask_for_loss,
            self.grpo_config.epsilon,
            self.grpo_config.kl_beta,
        )

        self.global_step += 1

        result = {k: float(v) for k, v in metrics.items()}
        result["mean_reward"] = float(jnp.mean(rewards_array))
        result["std_reward"] = float(jnp.std(rewards_array))
        result["step"] = self.global_step

        logger.info(
            "grpo_step",
            step=self.global_step,
            mean_reward=round(result["mean_reward"], 4),
            policy_loss=round(result["policy_loss"], 4),
            kl=round(result["kl"], 6),
        )

        return result

    @staticmethod
    @nnx.jit
    def _update_step(
        model: TransformerLM,
        optimizer: nnx.Optimizer,
        input_ids: jax.Array,
        response_mask: jax.Array,
        old_log_probs: jax.Array,
        ref_log_probs: jax.Array,
        advantages: jax.Array,
        mask_for_loss: jax.Array,
        epsilon: float,
        kl_beta: float,
    ) -> dict[str, jax.Array]:
        """JIT-compiled forward + backward + optimizer update."""

        def loss_fn(model: TransformerLM) -> tuple[jax.Array, dict[str, jax.Array]]:
            # Inline log prob computation (cannot call @nnx.jit from inside @nnx.jit)
            logits, _ = model(input_ids, deterministic=True)
            log_probs_all = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
            target_ids = input_ids[:, 1:]
            policy_log_probs = jnp.take_along_axis(
                log_probs_all, target_ids[:, :, None], axis=-1,
            )[:, :, 0]
            policy_log_probs = policy_log_probs * response_mask[:, 1:]

            total_loss, metrics = GRPOTrainer._grpo_loss(
                policy_log_probs,
                old_log_probs,
                ref_log_probs,
                advantages,
                mask_for_loss,
                epsilon,
                kl_beta,
            )
            return total_loss, metrics

        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        grad_norm = optax.global_norm(jax.tree.leaves(grads))
        optimizer.update(model, grads)

        metrics["grad_norm"] = grad_norm
        return metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        prompts: list[str],
        answers: list[str],
        reward_fn: RewardFn | None = None,
        total_steps: int | None = None,
        batch_size: int = 2,
        seed: int = 0,
    ) -> dict[str, float]:
        """Run the full GRPO training loop over a dataset of prompts.

        Args:
            prompts: Full list of prompt strings.
            answers: Corresponding ground-truth answers.
            reward_fn: Reward function to use.
            total_steps: Override number of steps (defaults to train_config).
            batch_size: Number of prompts per step.
            seed: Base RNG seed for generation.

        Returns:
            Final metrics summary.
        """
        if reward_fn is None:
            reward_fn = RewardFunction.gsm8k_numeric

        steps = total_steps if total_steps is not None else self.train_config.total_steps
        n_prompts = len(prompts)

        logger.info(
            "grpo_training_start",
            n_prompts=n_prompts,
            total_steps=steps,
            batch_size=batch_size,
        )

        start_time = time.time()
        all_metrics: list[dict[str, float]] = []

        for step in range(steps):
            # Cycle through prompts
            start_idx = (step * batch_size) % n_prompts
            end_idx = start_idx + batch_size
            if end_idx > n_prompts:
                # Wrap around
                batch_prompts = prompts[start_idx:] + prompts[:end_idx - n_prompts]
                batch_answers = answers[start_idx:] + answers[:end_idx - n_prompts]
            else:
                batch_prompts = prompts[start_idx:end_idx]
                batch_answers = answers[start_idx:end_idx]

            step_metrics = self.train_step(
                batch_prompts,
                batch_answers,
                reward_fn=reward_fn,
                seed=seed + step,
            )
            all_metrics.append(step_metrics)

        total_time = time.time() - start_time

        # Summarize
        final_metrics = {
            "total_steps": steps,
            "total_time_seconds": total_time,
            "final_mean_reward": all_metrics[-1]["mean_reward"] if all_metrics else 0.0,
            "final_policy_loss": all_metrics[-1]["policy_loss"] if all_metrics else 0.0,
            "final_kl": all_metrics[-1]["kl"] if all_metrics else 0.0,
        }
        logger.info("grpo_training_complete", **{k: round(v, 6) for k, v in final_metrics.items()})
        return final_metrics
