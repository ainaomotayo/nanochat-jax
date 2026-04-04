#!/usr/bin/env python3
"""GRPO RL training entry point for nanochat-jax.

Usage:
    python scripts/rl_train.py --device cpu --dataset gsm8k --reward gsm8k_numeric
    python scripts/rl_train.py --policy_checkpoint ckpt/ --ref_checkpoint ckpt/ --steps 100
    python scripts/rl_train.py --device cpu --max_samples 10 --group_size 4 --steps 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import structlog
from flax import nnx

from nanochat.config import ModelConfig, TrainingConfig
from nanochat.core.device import setup_device
from nanochat.model.transformer import TransformerLM
from nanochat.training.rl_trainer import GRPOConfig, GRPOTrainer, RewardFunction
from nanochat.training.checkpoint import CheckpointManager

logger = structlog.get_logger()


def load_gsm8k_data(max_samples: int | None = None) -> tuple[list[str], list[str]]:
    """Load GSM8K-style training data.

    Attempts to load from HuggingFace datasets. Falls back to synthetic
    examples if the library is not available.

    Returns:
        (prompts, answers) lists.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        prompts = []
        answers = []
        limit = max_samples if max_samples is not None else len(ds)
        for i, example in enumerate(ds):
            if i >= limit:
                break
            prompts.append(example["question"])
            # Extract the final numeric answer after ####
            answer_text = example["answer"]
            if "####" in answer_text:
                answer_text = answer_text.split("####")[-1].strip()
            answers.append(answer_text)
        logger.info("gsm8k_loaded", n_samples=len(prompts), source="huggingface")
        return prompts, answers
    except (ImportError, Exception) as exc:
        logger.warning("gsm8k_fallback_to_synthetic", reason=str(exc))
        # Synthetic GSM8K-like prompts for testing
        synthetic_prompts = [
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
            "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
            "Every day, Wendi feeds each of her 6 chickens 3 cups of mixed chicken feed, containing seeds, mealworms and vegetables. She gives the chickens their feed in three separate meals. How many cups of feed does she need to carry for each meal?",
        ]
        synthetic_answers = ["18", "3", "45000", "624", "6"]
        if max_samples is not None:
            synthetic_prompts = synthetic_prompts[:max_samples]
            synthetic_answers = synthetic_answers[:max_samples]
        logger.info("gsm8k_loaded", n_samples=len(synthetic_prompts), source="synthetic")
        return synthetic_prompts, synthetic_answers


def build_reward_fn(name: str):
    """Look up a named reward function."""
    reward_fns = {
        "gsm8k_numeric": RewardFunction.gsm8k_numeric,
        "gsm8k_format": RewardFunction.gsm8k_format,
        "length_penalty": RewardFunction.length_penalty,
    }
    if name not in reward_fns:
        raise ValueError(f"Unknown reward function '{name}'. Available: {list(reward_fns.keys())}")
    return reward_fns[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO RL training for nanochat-jax")

    # Device and checkpoints
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--policy_checkpoint", type=str, default=None,
                        help="Path to policy model checkpoint directory")
    parser.add_argument("--ref_checkpoint", type=str, default=None,
                        help="Path to reference model checkpoint directory")

    # Dataset and reward
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--reward", type=str, default="gsm8k_numeric",
                        choices=["gsm8k_numeric", "gsm8k_format", "length_penalty"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size for testing")

    # GRPO hyperparameters
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--kl_beta", type=float, default=0.01)
    parser.add_argument("--max_completion_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)

    # Training hyperparameters
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Number of prompts per training step")

    # Model config (for fresh model if no checkpoint)
    parser.add_argument("--model_size", type=str, default="nano",
                        choices=["nano", "small", "medium", "large", "xlarge"])

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/grpo/")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup device
    setup_device(args.device)
    logger.info("device_configured", device=args.device)

    # Build or load models
    seed = args.seed
    rngs = nnx.Rngs(params=seed, dropout=seed + 1)

    ckpt_manager = CheckpointManager(args.output_dir)

    if args.policy_checkpoint:
        # Load model config from checkpoint metadata
        meta_path = Path(args.policy_checkpoint) / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            model_cfg = ModelConfig(**meta.get("model_config", {}))
        else:
            model_cfg = ModelConfig.for_scale(args.model_size)

        policy_model = TransformerLM(model_cfg, rngs=rngs)
        ckpt_manager_load = CheckpointManager(str(Path(args.policy_checkpoint).parent))
        ckpt_manager_load.load(Path(args.policy_checkpoint), policy_model)
        logger.info("policy_model_loaded", path=args.policy_checkpoint)
    else:
        model_cfg = ModelConfig.for_scale(args.model_size)
        policy_model = TransformerLM(model_cfg, rngs=rngs)
        logger.info("policy_model_initialized", size=args.model_size)

    if args.ref_checkpoint:
        ref_rngs = nnx.Rngs(params=seed + 100, dropout=seed + 101)
        ref_model = TransformerLM(model_cfg, rngs=ref_rngs)
        ref_ckpt = CheckpointManager(str(Path(args.ref_checkpoint).parent))
        ref_ckpt.load(Path(args.ref_checkpoint), ref_model)
        logger.info("ref_model_loaded", path=args.ref_checkpoint)
    else:
        # Clone policy model weights for reference (fresh copy, never updated)
        ref_rngs = nnx.Rngs(params=seed, dropout=seed + 1)
        ref_model = TransformerLM(model_cfg, rngs=ref_rngs)
        logger.info("ref_model_initialized_from_same_weights")

    # Build tokenizer (CharTokenizer covers the nano vocab_size=256 preset)
    from nanochat.tokenizer.char import CharTokenizer
    # Build a character tokenizer that covers ASCII + specials
    all_chars = "".join(chr(i) for i in range(32, 127)) + "\n\t"
    tokenizer = CharTokenizer.from_text(all_chars)
    logger.info("tokenizer_loaded", vocab_size=tokenizer.vocab_size)

    # Load dataset
    if args.dataset == "gsm8k":
        prompts, answers = load_gsm8k_data(max_samples=args.max_samples)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Build configs
    grpo_config = GRPOConfig(
        group_size=args.group_size,
        epsilon=args.epsilon,
        kl_beta=args.kl_beta,
        max_completion_len=args.max_completion_len,
        temperature=args.temperature,
    )

    train_config = TrainingConfig(
        learning_rate=args.learning_rate,
        total_steps=args.steps,
        warmup_steps=min(10, args.steps // 10),
        batch_size=1,  # GRPO handles batching internally
        checkpoint_dir=args.output_dir,
        optimizer="adamw",
        weight_decay=0.01,
    )

    # Build trainer
    trainer = GRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        grpo_config=grpo_config,
        train_config=train_config,
        rngs=rngs,
    )

    # Build reward function
    reward_fn = build_reward_fn(args.reward)

    # Train
    results = trainer.train(
        prompts=prompts,
        answers=answers,
        reward_fn=reward_fn,
        total_steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Save final checkpoint
    ckpt_manager.save(
        trainer.global_step,
        policy_model,
        results,
    )
    logger.info("grpo_training_finished", output_dir=args.output_dir, **results)


if __name__ == "__main__":
    main()
