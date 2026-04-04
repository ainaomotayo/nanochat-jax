"""Tests for GRPO RL trainer."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from nanochat.config import ModelConfig, TrainingConfig
from nanochat.model.transformer import TransformerLM
from nanochat.tokenizer.char import CharTokenizer
from nanochat.training.rl_trainer import (
    GRPOConfig,
    GRPOTrainer,
    RewardFunction,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def char_tokenizer() -> CharTokenizer:
    """Build a CharTokenizer covering ASCII printable + whitespace."""
    all_chars = "".join(chr(i) for i in range(32, 127)) + "\n\t"
    return CharTokenizer.from_text(all_chars)


@pytest.fixture
def tiny_model_cfg() -> ModelConfig:
    """Tiny model config for fast CPU tests."""
    return ModelConfig.for_scale("nano")


@pytest.fixture
def tiny_policy_model(tiny_model_cfg: ModelConfig) -> TransformerLM:
    rngs = nnx.Rngs(params=0, dropout=1)
    return TransformerLM(tiny_model_cfg, rngs=rngs)


@pytest.fixture
def tiny_ref_model(tiny_model_cfg: ModelConfig) -> TransformerLM:
    rngs = nnx.Rngs(params=0, dropout=1)
    return TransformerLM(tiny_model_cfg, rngs=rngs)


# ======================================================================
# RewardFunction tests
# ======================================================================

class TestGSM8KNumericReward:
    def test_correct_integer(self):
        reward = RewardFunction.gsm8k_numeric(
            prompt="What is 2+2?",
            completion="The answer is 4",
            answer="4",
        )
        assert reward == 1.0

    def test_correct_negative(self):
        reward = RewardFunction.gsm8k_numeric(
            prompt="What is 3-5?",
            completion="3 minus 5 equals -2",
            answer="-2",
        )
        assert reward == 1.0

    def test_correct_decimal(self):
        reward = RewardFunction.gsm8k_numeric(
            prompt="Half of 7?",
            completion="That gives us 3.5",
            answer="3.5",
        )
        assert reward == 1.0

    def test_incorrect(self):
        reward = RewardFunction.gsm8k_numeric(
            prompt="What is 2+2?",
            completion="The answer is 5",
            answer="4",
        )
        assert reward == 0.0

    def test_no_numbers_in_completion(self):
        reward = RewardFunction.gsm8k_numeric(
            prompt="What is 2+2?",
            completion="I don't know the answer",
            answer="4",
        )
        assert reward == 0.0

    def test_multiple_numbers_uses_last(self):
        reward = RewardFunction.gsm8k_numeric(
            prompt="Compute step by step",
            completion="First 10, then 20, finally 30",
            answer="30",
        )
        assert reward == 1.0

    def test_answer_with_hash_separator(self):
        reward = RewardFunction.gsm8k_numeric(
            prompt="How much?",
            completion="Step: 5+13=18 #### 18",
            answer="#### 18",
        )
        assert reward == 1.0


class TestGSM8KFormatReward:
    def test_full_format(self):
        reward = RewardFunction.gsm8k_format(
            prompt="Q",
            completion="Working... #### 42",
            answer="42",
        )
        assert reward == 1.0

    def test_separator_only(self):
        reward = RewardFunction.gsm8k_format(
            prompt="Q",
            completion="Working... #### the answer is unclear",
            answer="42",
        )
        assert reward == 0.5

    def test_numeric_ending_only(self):
        reward = RewardFunction.gsm8k_format(
            prompt="Q",
            completion="The total is 42",
            answer="42",
        )
        assert reward == 0.5

    def test_no_format(self):
        reward = RewardFunction.gsm8k_format(
            prompt="Q",
            completion="I have no idea",
            answer="42",
        )
        assert reward == 0.0


class TestLengthPenalty:
    def test_within_range(self):
        completion = " ".join(["word"] * 50)
        reward = RewardFunction.length_penalty("p", completion, "a", min_len=10, max_len=200)
        assert reward == 1.0

    def test_too_short(self):
        completion = "short"
        reward = RewardFunction.length_penalty("p", completion, "a", min_len=10, max_len=200)
        assert 0.0 <= reward < 1.0

    def test_too_long(self):
        completion = " ".join(["word"] * 300)
        reward = RewardFunction.length_penalty("p", completion, "a", min_len=10, max_len=200)
        assert 0.0 <= reward < 1.0

    def test_empty(self):
        reward = RewardFunction.length_penalty("p", "", "a", min_len=10, max_len=200)
        assert reward == 0.0


# ======================================================================
# Advantage normalization test
# ======================================================================

def test_group_normalize_advantages():
    """Verify group-level normalization produces zero-mean, unit-variance per group."""
    rewards = jnp.array([
        [1.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=jnp.float32)  # [2 prompts, 4 group_size]

    group_mean = jnp.mean(rewards, axis=1, keepdims=True)
    group_std = jnp.std(rewards, axis=1, keepdims=True)
    advantages = (rewards - group_mean) / (group_std + 1e-8)

    # Each group should have mean ~0
    for i in range(2):
        assert abs(float(jnp.mean(advantages[i]))) < 1e-5

    # Each group should have std ~1 (if not all same)
    assert float(jnp.std(advantages[0])) > 0.9
    assert float(jnp.std(advantages[1])) > 0.9


# ======================================================================
# GRPO loss shape test
# ======================================================================

def test_grpo_loss_shape():
    """Verify GRPO loss returns scalar and correct metric keys."""
    batch, seq = 4, 16
    policy_lp = jnp.zeros((batch, seq), dtype=jnp.float32)
    old_lp = jnp.zeros((batch, seq), dtype=jnp.float32)
    ref_lp = jnp.zeros((batch, seq), dtype=jnp.float32)
    advantages = jnp.ones((batch, 1), dtype=jnp.float32)
    mask = jnp.ones((batch, seq), dtype=jnp.float32)

    loss, metrics = GRPOTrainer._grpo_loss(
        policy_lp, old_lp, ref_lp, advantages, mask,
        epsilon=0.2, kl_beta=0.01,
    )

    assert loss.shape == ()
    assert "policy_loss" in metrics
    assert "kl" in metrics
    assert "total_loss" in metrics
    assert "mean_ratio" in metrics
    # When policy == old, ratio should be 1.0
    assert abs(float(metrics["mean_ratio"]) - 1.0) < 1e-5
    # When policy == ref, KL should be 0
    assert abs(float(metrics["kl"])) < 1e-5


def test_grpo_loss_nonzero_kl():
    """KL should be positive when policy differs from reference."""
    batch, seq = 4, 16
    policy_lp = jnp.ones((batch, seq), dtype=jnp.float32) * -1.0
    old_lp = jnp.ones((batch, seq), dtype=jnp.float32) * -1.0
    ref_lp = jnp.ones((batch, seq), dtype=jnp.float32) * -2.0  # Different from policy
    advantages = jnp.ones((batch, 1), dtype=jnp.float32)
    mask = jnp.ones((batch, seq), dtype=jnp.float32)

    loss, metrics = GRPOTrainer._grpo_loss(
        policy_lp, old_lp, ref_lp, advantages, mask,
        epsilon=0.2, kl_beta=0.01,
    )

    assert float(metrics["kl"]) > 0.0


# ======================================================================
# Smoke test: full trainer on CPU with tiny model
# ======================================================================

def test_grpo_trainer_smoke(char_tokenizer, tiny_model_cfg, tiny_policy_model, tiny_ref_model):
    """End-to-end smoke test: 3 GRPO steps on CPU with tiny model and 2 synthetic prompts."""
    grpo_cfg = GRPOConfig(
        group_size=2,
        epsilon=0.2,
        kl_beta=0.01,
        max_completion_len=8,
        temperature=0.8,
    )

    train_cfg = TrainingConfig(
        learning_rate=1e-4,
        total_steps=3,
        warmup_steps=0,
        batch_size=1,
        optimizer="adamw",
        weight_decay=0.01,
    )

    rngs = nnx.Rngs(params=42, dropout=43)

    trainer = GRPOTrainer(
        policy_model=tiny_policy_model,
        ref_model=tiny_ref_model,
        tokenizer=char_tokenizer,
        grpo_config=grpo_cfg,
        train_config=train_cfg,
        rngs=rngs,
    )

    # Simple synthetic prompts (short, within nano vocab)
    prompts = ["2+2=", "3+5="]
    answers = ["4", "8"]

    # Custom reward that always gives something
    def simple_reward(prompt: str, completion: str, answer: str) -> float:
        if answer in completion:
            return 1.0
        return 0.1

    results = trainer.train(
        prompts=prompts,
        answers=answers,
        reward_fn=simple_reward,
        total_steps=3,
        batch_size=2,
        seed=0,
    )

    assert "total_steps" in results
    assert results["total_steps"] == 3
    assert "final_mean_reward" in results
    assert "final_policy_loss" in results
    assert "final_kl" in results
    assert "total_time_seconds" in results
    assert results["total_time_seconds"] > 0
