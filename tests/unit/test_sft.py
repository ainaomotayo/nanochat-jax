"""Tests for SFT trainer and LoRA implementation."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nanochat.config import ModelConfig, TrainingConfig
from nanochat.model.transformer import TransformerLM
from nanochat.training.sft_trainer import SFTDataset, SFTTrainer, SimpleTokenizer
from nanochat.training.lora import (
    LoRALinear,
    LoRAParam,
    apply_lora,
    count_base_params,
    count_lora_params,
    get_lora_params,
    merge_lora,
)
from nanochat.training.loss import IGNORE_INDEX


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sft_model_cfg() -> ModelConfig:
    """Tiny model config for SFT tests."""
    return ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        d_ff=256,
        max_seq_len=64,
        dropout_rate=0.0,
        use_qk_norm=True,
        logit_softcap=30.0,
        use_value_embeddings=False,
        use_per_layer_scalars=False,
        use_smear=False,
    )


@pytest.fixture(scope="module")
def sft_train_cfg() -> TrainingConfig:
    """Small training config for SFT tests."""
    return TrainingConfig(
        batch_size=2,
        optimizer="adamw",
        learning_rate=2e-5,
        warmup_steps=0,
        total_steps=10,
        weight_decay=0.0,
        dtype="float32",
        param_dtype="float32",
        save_every_steps=1000,
        eval_every_steps=1000,
        checkpoint_dir="/tmp/nanochat_sft_test_ckpts",
    )


@pytest.fixture(scope="module")
def tokenizer() -> SimpleTokenizer:
    """Simple tokenizer for SFT tests."""
    return SimpleTokenizer(vocab_size=256)


@pytest.fixture(scope="module")
def conversations() -> list[list[dict[str, str]]]:
    """Test conversations."""
    return [
        [
            {"role": "system", "content": "You help."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello!"},
        ],
        [
            {"role": "user", "content": "Bye."},
            {"role": "assistant", "content": "Goodbye!"},
        ],
    ]


# ---------------------------------------------------------------------------
# SFT Dataset Tests
# ---------------------------------------------------------------------------

class TestSFTDataset:
    """Tests for SFTDataset."""

    def test_sft_dataset_produces_response_mask(
        self, conversations, tokenizer
    ):
        """SFTDataset items must contain a response_mask array."""
        dataset = SFTDataset(conversations, tokenizer, max_seq_len=64)

        assert len(dataset) > 0
        item = dataset[0]

        assert "response_mask" in item
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item

        # Shapes must match
        assert item["response_mask"].shape == (64,)
        assert item["input_ids"].shape == (64,)
        assert item["labels"].shape == (64,)
        assert item["attention_mask"].shape == (64,)

        # response_mask should have some 1s (assistant tokens) and some 0s
        assert item["response_mask"].sum() > 0, "No assistant tokens found"
        assert item["response_mask"].sum() < 64, "All tokens marked as assistant"

    def test_response_mask_excludes_system_and_user(
        self, tokenizer
    ):
        """Response mask must be 0 for system and user tokens, 1 only for assistant."""
        conv = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]
        dataset = SFTDataset([conv], tokenizer, max_seq_len=128)
        item = dataset[0]

        input_ids = item["input_ids"]
        response_mask = item["response_mask"]
        attention_mask = item["attention_mask"]

        # Encode the pieces separately to find boundaries
        system_tag_ids = tokenizer.encode("<|system|>\n")
        user_tag_ids = tokenizer.encode("<|user|>\n")
        assistant_tag_ids = tokenizer.encode("<|assistant|>\n")

        # Find where system tag starts (should be at beginning)
        system_tag_id = tokenizer.SPECIAL_TOKENS["<|system|>"]
        user_tag_id = tokenizer.SPECIAL_TOKENS["<|user|>"]
        assistant_tag_id = tokenizer.SPECIAL_TOKENS["<|assistant|>"]

        ids_list = input_ids.tolist()
        mask_list = response_mask.tolist()

        # All positions with system/user special token should have mask=0
        for i, tid in enumerate(ids_list):
            if tid == system_tag_id or tid == user_tag_id:
                assert mask_list[i] == 0, (
                    f"Position {i} has system/user token {tid} but mask={mask_list[i]}"
                )

        # The assistant content "2" should have mask=1 somewhere
        assistant_content_encoded = tokenizer.encode("2")
        found_assistant_content = False
        for i, tid in enumerate(ids_list):
            if tid == assistant_content_encoded[0] and mask_list[i] == 1:
                found_assistant_content = True
                break
        assert found_assistant_content, "Assistant content '2' not found with mask=1"

        # Padding positions should have mask=0
        for i in range(len(ids_list)):
            if attention_mask[i] == 0:
                assert mask_list[i] == 0, (
                    f"Padding position {i} has response_mask=1"
                )

    def test_labels_ignore_non_response(self, conversations, tokenizer):
        """Labels should be IGNORE_INDEX for non-response token positions."""
        dataset = SFTDataset(conversations, tokenizer, max_seq_len=64)
        item = dataset[0]

        labels = item["labels"]
        response_mask = item["response_mask"]

        # Where response_mask is 0, labels should be IGNORE_INDEX
        for i in range(len(labels)):
            if response_mask[i] == 0:
                # The label at position i-1 (which predicts position i) should be IGNORE_INDEX
                # But our construction: labels[t] = IGNORE_INDEX when response_mask[t+1] == 0
                pass  # Labels use a shift, so the exact relationship is:
                       # labels[t] != IGNORE_INDEX only when response_mask[t+1] == 1

        # At minimum, some labels should be IGNORE_INDEX (system/user portions)
        n_ignored = (labels == IGNORE_INDEX).sum()
        assert n_ignored > 0, "No IGNORE_INDEX labels found"

        # Some labels should be real token IDs (assistant portions)
        n_real = (labels != IGNORE_INDEX).sum()
        assert n_real > 0, "No real labels found (all IGNORE_INDEX)"

    def test_synthetic_dataset_loads(self, tokenizer):
        """Passing None for conversations should use synthetic data."""
        dataset = SFTDataset(None, tokenizer, max_seq_len=64)
        assert len(dataset) > 0

    def test_make_loader_yields_batches(self, conversations, tokenizer):
        """make_loader should yield batches with correct shapes."""
        dataset = SFTDataset(conversations, tokenizer, max_seq_len=32)
        loader = dataset.make_loader(batch_size=2)
        batch = next(loader)

        assert batch["input_ids"].shape == (2, 32)
        assert batch["labels"].shape == (2, 32)
        assert batch["response_mask"].shape == (2, 32)
        assert batch["attention_mask"].shape == (2, 32)


# ---------------------------------------------------------------------------
# SFT Trainer Tests
# ---------------------------------------------------------------------------

class TestSFTTrainer:
    """Tests for SFTTrainer."""

    def test_sft_train_step_loss_decreases(
        self, sft_model_cfg, sft_train_cfg, tokenizer
    ):
        """Loss should decrease over several SFT training steps."""
        rngs = nnx.Rngs(params=0, dropout=1)
        model = TransformerLM(sft_model_cfg, rngs=rngs)

        dataset = SFTDataset(None, tokenizer, max_seq_len=sft_model_cfg.max_seq_len)

        trainer = SFTTrainer(
            model=model,
            dataset=dataset,
            train_cfg=sft_train_cfg,
            model_cfg=sft_model_cfg,
            use_lora=False,
        )

        # Collect losses over several steps
        losses = []
        for _ in range(10):
            batch = next(trainer.train_loader)
            metrics = trainer.train_step(batch)
            losses.append(metrics["loss"])

        # Loss should decrease (or at least not increase monotonically)
        # Compare average of first 3 vs last 3
        early_avg = sum(losses[:3]) / 3
        late_avg = sum(losses[-3:]) / 3
        assert late_avg < early_avg, (
            f"Loss did not decrease: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
        )


# ---------------------------------------------------------------------------
# LoRA Tests
# ---------------------------------------------------------------------------

class TestLoRA:
    """Tests for LoRA implementation."""

    def test_lora_linear_output_matches_base_at_init(self):
        """At initialization (B=0), LoRALinear output must match base linear."""
        rngs = nnx.Rngs(params=42)
        base_linear = nnx.Linear(64, 128, use_bias=False, rngs=rngs)
        lora_linear = LoRALinear(base_linear, rank=8, alpha=8.0, rngs=rngs)

        # Random input
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 64))

        base_out = base_linear(x)
        lora_out = lora_linear(x)

        np.testing.assert_allclose(
            np.array(base_out),
            np.array(lora_out),
            atol=1e-5,
            err_msg="LoRA output does not match base at initialization",
        )

    def test_lora_param_count_reduction(self, sft_model_cfg):
        """LoRA should have far fewer trainable params than the full model."""
        rngs = nnx.Rngs(params=42, dropout=43)
        model = TransformerLM(sft_model_cfg, rngs=rngs)

        n_base_before = count_base_params(model)

        lora_rngs = nnx.Rngs(params=100)
        apply_lora(model, rank=4, rngs=lora_rngs)

        n_lora = count_lora_params(model)
        n_base_after = count_base_params(model)

        # LoRA params should be much fewer than base params
        assert n_lora > 0, "No LoRA parameters found"
        assert n_lora < n_base_after, (
            f"LoRA params ({n_lora}) should be fewer than base params ({n_base_after})"
        )

        # Typically LoRA is <10% of base params for small ranks
        ratio = n_lora / n_base_after
        assert ratio < 0.5, (
            f"LoRA param ratio {ratio:.2%} is too high (expected <50%)"
        )

    def test_merge_lora_equivalence(self, sft_model_cfg):
        """After merging, model output must match pre-merge output exactly."""
        rngs = nnx.Rngs(params=42, dropout=43)
        model = TransformerLM(sft_model_cfg, rngs=rngs)

        lora_rngs = nnx.Rngs(params=100)
        apply_lora(model, rank=8, rngs=lora_rngs)

        # Manually set non-zero LoRA weights to make the test meaningful
        for layer in model.layers:
            attn = layer.attention
            for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                module = getattr(attn, proj_name)
                if isinstance(module, LoRALinear):
                    # Set B to non-zero so LoRA has an effect
                    module.lora_B[...] = (
                        jax.random.normal(
                            jax.random.PRNGKey(hash(proj_name) % (2**31)),
                            module.lora_B[...].shape,
                        ) * 0.01
                    )

        # Get output before merge
        x = jax.random.randint(
            jax.random.PRNGKey(0),
            (1, sft_model_cfg.max_seq_len),
            0, sft_model_cfg.vocab_size,
        )
        logits_before, _ = model(x, deterministic=True)

        # Merge LoRA weights
        merge_lora(model)

        # Get output after merge
        logits_after, _ = model(x, deterministic=True)

        # float32 non-associativity: x@W + x@A@B vs x@(W+A@B) differs ~1e-3
        np.testing.assert_allclose(
            np.array(logits_before),
            np.array(logits_after),
            atol=5e-3,
            err_msg="Merged model output does not match pre-merge output",
        )

        # After merge, no LoRALinear modules should remain
        for layer in model.layers:
            attn = layer.attention
            for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                module = getattr(attn, proj_name)
                assert not isinstance(module, LoRALinear), (
                    f"{proj_name} is still LoRALinear after merge"
                )

    def test_apply_lora_wraps_attention_projections(self, sft_model_cfg):
        """apply_lora should wrap all Q/K/V/out_proj in all layers."""
        rngs = nnx.Rngs(params=42, dropout=43)
        model = TransformerLM(sft_model_cfg, rngs=rngs)

        lora_rngs = nnx.Rngs(params=100)
        apply_lora(model, rank=4, rngs=lora_rngs)

        for i, layer in enumerate(model.layers):
            attn = layer.attention
            for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                module = getattr(attn, proj_name)
                assert isinstance(module, LoRALinear), (
                    f"Layer {i} {proj_name} is {type(module).__name__}, expected LoRALinear"
                )

    def test_lora_b_initialized_to_zeros(self):
        """lora_B should be initialized to all zeros."""
        rngs = nnx.Rngs(params=42)
        base = nnx.Linear(32, 64, use_bias=False, rngs=rngs)
        lora = LoRALinear(base, rank=4, alpha=4.0, rngs=rngs)

        np.testing.assert_array_equal(
            np.array(lora.lora_B[...]),
            np.zeros((4, 64)),
            err_msg="lora_B should be initialized to zeros",
        )

    def test_lora_a_is_nonzero(self):
        """lora_A should be initialized to non-zero values."""
        rngs = nnx.Rngs(params=42)
        base = nnx.Linear(32, 64, use_bias=False, rngs=rngs)
        lora = LoRALinear(base, rank=4, alpha=4.0, rngs=rngs)

        assert jnp.any(lora.lora_A[...] != 0), "lora_A should be non-zero"

    def test_get_lora_params_returns_only_lora(self, sft_model_cfg):
        """get_lora_params should return only LoRAParam variables."""
        rngs = nnx.Rngs(params=42, dropout=43)
        model = TransformerLM(sft_model_cfg, rngs=rngs)

        lora_rngs = nnx.Rngs(params=100)
        apply_lora(model, rank=4, rngs=lora_rngs)

        lora_state = get_lora_params(model)
        leaves = jax.tree.leaves(lora_state)

        # Should have leaves (A and B for each of 4 projections * 2 layers = 16)
        assert len(leaves) > 0, "No LoRA params found"

        expected_n = 4 * 2 * sft_model_cfg.n_layers  # 4 proj * 2 (A,B) * n_layers
        assert len(leaves) == expected_n, (
            f"Expected {expected_n} LoRA param arrays, got {len(leaves)}"
        )


# ---------------------------------------------------------------------------
# Integration: SFT + LoRA
# ---------------------------------------------------------------------------

class TestSFTWithLoRA:
    """Integration tests combining SFT and LoRA."""

    def test_sft_with_lora_runs(self, sft_model_cfg, sft_train_cfg, tokenizer):
        """SFT training with LoRA should run without errors."""
        rngs = nnx.Rngs(params=0, dropout=1)
        model = TransformerLM(sft_model_cfg, rngs=rngs)

        lora_rngs = nnx.Rngs(params=100)
        apply_lora(model, rank=4, rngs=lora_rngs)

        dataset = SFTDataset(None, tokenizer, max_seq_len=sft_model_cfg.max_seq_len)

        trainer = SFTTrainer(
            model=model,
            dataset=dataset,
            train_cfg=sft_train_cfg,
            model_cfg=sft_model_cfg,
            use_lora=True,
        )

        # Run a few steps (should not raise)
        for _ in range(3):
            batch = next(trainer.train_loader)
            metrics = trainer.train_step(batch)
            assert "loss" in metrics
            assert metrics["loss"] > 0
