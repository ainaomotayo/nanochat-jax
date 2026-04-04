"""Tests for ModelConfig and TrainingConfig: validation, presets, serialization."""
from __future__ import annotations
import pytest
from pydantic import ValidationError

from nanochat.config import ModelConfig, TrainingConfig
from nanochat.config.data_config import DataConfig


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfigDefaults:
    def test_instantiates_with_defaults(self):
        cfg = ModelConfig()
        assert cfg.vocab_size > 0
        assert cfg.d_model > 0
        assert cfg.n_layers > 0

    def test_d_head_computed(self):
        cfg = ModelConfig(d_model=128, n_heads=4)
        assert cfg.d_head == 32  # 128 // 4

    def test_d_ff_defaults_to_4x_dmodel(self):
        cfg = ModelConfig(d_model=64, d_ff=None, ffn_type="relu2")
        assert cfg.d_ff == 256  # 4 * 64

    def test_nanochat_defaults(self):
        """Verify nanochat-faithful defaults are set."""
        cfg = ModelConfig()
        assert cfg.ffn_type == "relu2"
        assert cfg.use_qk_norm is True
        assert cfg.logit_softcap == pytest.approx(30.0)
        assert cfg.rope_base == pytest.approx(100_000.0)
        assert cfg.tie_embeddings is False
        assert cfg.use_value_embeddings is True
        assert cfg.use_per_layer_scalars is True
        assert cfg.use_smear is True


class TestModelConfigValidation:
    def test_d_head_is_floor_division(self):
        # ModelConfig floors d_model // n_heads (does not require exact divisibility)
        cfg = ModelConfig(d_model=64, n_heads=3)
        assert cfg.d_head == 64 // 3  # 21

    def test_n_kv_heads_cannot_exceed_n_heads(self):
        # n_kv_heads > n_heads is invalid for GQA
        with pytest.raises(ValidationError):
            ModelConfig(n_heads=4, n_kv_heads=8)

    def test_negative_vocab_size_rejected(self):
        with pytest.raises(ValidationError):
            ModelConfig(vocab_size=-1)


class TestModelConfigPresets:
    @pytest.mark.parametrize("size", ["nano", "small", "medium"])
    def test_for_scale_returns_valid_config(self, size):
        cfg = ModelConfig.for_scale(size)
        assert isinstance(cfg, ModelConfig)
        assert cfg.d_model > 0
        assert cfg.n_layers > 0
        assert cfg.n_heads > 0

    def test_larger_scale_has_more_params(self):
        from nanochat.model.transformer import TransformerLM
        from nanochat.model.param_count import count_params
        from flax import nnx

        nano = ModelConfig.for_scale("nano")
        small = ModelConfig.for_scale("small")

        m_nano = TransformerLM(nano, rngs=nnx.Rngs(params=0, dropout=1))
        m_small = TransformerLM(small, rngs=nnx.Rngs(params=0, dropout=1))

        n_nano = count_params(m_nano)["total"]
        n_small = count_params(m_small)["total"]
        assert n_small > n_nano

    def test_unknown_scale_raises(self):
        with pytest.raises((ValueError, KeyError)):
            ModelConfig.for_scale("gigantic")


class TestModelConfigSerialization:
    def test_model_dump_round_trip(self):
        cfg = ModelConfig.for_scale("nano")
        d = cfg.model_dump()
        cfg2 = ModelConfig(**d)
        assert cfg2.d_model == cfg.d_model
        assert cfg2.n_layers == cfg.n_layers
        assert cfg2.ffn_type == cfg.ffn_type

    def test_json_round_trip(self):
        cfg = ModelConfig.for_scale("nano")
        js = cfg.model_dump_json()
        cfg2 = ModelConfig.model_validate_json(js)
        assert cfg2 == cfg


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfigDefaults:
    def test_instantiates(self):
        cfg = TrainingConfig()
        assert cfg.batch_size > 0
        assert cfg.learning_rate > 0
        assert cfg.total_steps > 0

    def test_optimizer_defaults_to_muon(self):
        cfg = TrainingConfig()
        assert cfg.optimizer == "muon"

    def test_effective_batch_size(self):
        cfg = TrainingConfig(batch_size=8, gradient_accumulation_steps=4)
        assert cfg.effective_batch_size == 32

    def test_lr_decay_steps_defaults_to_total_steps(self):
        cfg = TrainingConfig(total_steps=1000, lr_decay_steps=None)
        assert cfg.lr_decay_steps == 1000


class TestTrainingConfigValidation:
    def test_negative_lr_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-0.001)

    def test_batch_size_zero_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)

    def test_total_steps_zero_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(total_steps=0)


class TestComputeTotalSteps:
    def test_basic_calculation(self):
        steps = TrainingConfig.compute_total_steps(
            token_budget=1_000_000, seq_len=256, batch_size=8
        )
        assert steps == 489  # ceil(1_000_000 / (256 * 8))

    def test_zero_budget_raises(self):
        with pytest.raises(ValueError):
            TrainingConfig.compute_total_steps(0, 256, 8)

    def test_for_scale_experiment(self):
        cfg = TrainingConfig.for_scale_experiment(
            token_budget=500_000, seq_len=128, batch_size=16
        )
        assert isinstance(cfg, TrainingConfig)
        assert cfg.total_steps > 0
        assert cfg.warmup_steps >= 100


class TestTrainingConfigSerialization:
    def test_model_dump_round_trip(self):
        cfg = TrainingConfig(batch_size=16, optimizer="adamw")
        d = cfg.model_dump()
        cfg2 = TrainingConfig(**{k: v for k, v in d.items()
                                  if k != "effective_batch_size"})
        assert cfg2.batch_size == cfg.batch_size
        assert cfg2.optimizer == cfg.optimizer


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

class TestDataConfig:
    def test_instantiates(self):
        cfg = DataConfig()
        assert isinstance(cfg, DataConfig)
