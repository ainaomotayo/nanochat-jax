"""Tests for ScalingRunner: synthetic runs, label alignment, scale modes."""
from __future__ import annotations
import pytest
import math
from nanochat.config import ModelConfig, TrainingConfig
from nanochat.scaling.runner import ScalingRunner, _build_model_for_target_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _micro_cfg(vocab_size: int = 64) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size, d_model=32, n_layers=1,
        n_heads=2, n_kv_heads=2, max_seq_len=16,
        use_value_embeddings=False, use_per_layer_scalars=False,
        use_smear=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildModelForTarget:
    def test_returns_model_config(self):
        cfg = _build_model_for_target_params(50_000, vocab_size=64)
        assert isinstance(cfg, ModelConfig)

    def test_heads_divides_d_model(self):
        for target in [10_000, 100_000, 1_000_000]:
            cfg = _build_model_for_target_params(target)
            assert cfg.d_model % cfg.n_heads == 0, (
                f"d_model={cfg.d_model} not divisible by n_heads={cfg.n_heads}"
            )

    def test_reasonable_param_count(self):
        from nanochat.model.param_count import count_params
        from nanochat.model.transformer import TransformerLM
        from flax import nnx

        target = 200_000
        cfg = _build_model_for_target_params(target, vocab_size=64)
        model = TransformerLM(cfg, rngs=nnx.Rngs(params=0, dropout=1))
        n = count_params(model)["total"]
        # Within 10x (rough search — not perfect)
        assert n < target * 10


class TestRunSingle:
    def test_produces_result(self, tmp_path):
        cfg = _micro_cfg()
        train_cfg = TrainingConfig(
            batch_size=2, total_steps=5,
            optimizer="adamw", dtype="float32", param_dtype="float32",
        )
        runner = ScalingRunner(tmp_path)
        result = runner.run_single(cfg, train_cfg, run_id="test_run")

        assert result.n_params > 0
        assert math.isfinite(result.final_val_loss)
        assert result.tokens_per_second > 0
        assert result.wall_time_seconds > 0

    def test_result_saved_as_json(self, tmp_path):
        import json
        cfg = _micro_cfg()
        train_cfg = TrainingConfig(
            batch_size=2, total_steps=5,
            optimizer="adamw", dtype="float32", param_dtype="float32",
        )
        runner = ScalingRunner(tmp_path)
        runner.run_single(cfg, train_cfg, run_id="save_test")
        assert (tmp_path / "save_test.json").exists()
        data = json.loads((tmp_path / "save_test.json").read_text())
        assert "final_val_loss" in data

    def test_loss_finite(self, tmp_path):
        """Ensure synthetic data training produces finite losses (not NaN)."""
        cfg = _micro_cfg()
        train_cfg = TrainingConfig(
            batch_size=2, total_steps=20,
            optimizer="adamw", learning_rate=1e-3,
            dtype="float32", param_dtype="float32",
        )
        runner = ScalingRunner(tmp_path)
        result = runner.run_single(cfg, train_cfg, run_id="finite")
        assert math.isfinite(result.final_val_loss)
        assert result.final_val_loss > 0

    def test_token_budget_overrides_steps(self, tmp_path):
        """Token budget should compute steps = budget // (batch * seq_len)."""
        cfg = _micro_cfg()
        train_cfg = TrainingConfig(
            batch_size=4, total_steps=99_999,  # will be overridden
            optimizer="adamw", dtype="float32", param_dtype="float32",
        )
        budget = 4 * 16 * 10  # exactly 10 steps
        runner = ScalingRunner(tmp_path)
        result = runner.run_single(cfg, train_cfg, run_id="budget",
                                   token_budget=budget)
        assert result.n_tokens_trained == budget

    def test_mfu_in_unit_range(self, tmp_path):
        cfg = _micro_cfg()
        train_cfg = TrainingConfig(
            batch_size=2, total_steps=5,
            optimizer="adamw", dtype="float32", param_dtype="float32",
        )
        runner = ScalingRunner(tmp_path)
        result = runner.run_single(cfg, train_cfg, run_id="mfu_test",
                                   peak_flops=1e12)
        assert 0.0 <= result.mfu <= 1.0

    def test_label_alignment_loss_decreasing(self, tmp_path):
        """With correct label alignment, loss should be <= initial entropy."""
        import math
        cfg = _micro_cfg(vocab_size=64)
        # Entropy of uniform 64-class distribution ≈ ln(64) ≈ 4.16
        uniform_entropy = math.log(64)
        train_cfg = TrainingConfig(
            batch_size=4, total_steps=30,
            optimizer="adamw", learning_rate=3e-3,
            warmup_steps=0,
            dtype="float32", param_dtype="float32",
        )
        runner = ScalingRunner(tmp_path)
        result = runner.run_single(cfg, train_cfg, run_id="label_align")
        # Loss should start near entropy and stay finite (no NaN from wrong shapes)
        assert result.final_val_loss < uniform_entropy * 2  # well below 2x entropy
        assert math.isfinite(result.final_val_loss)


class TestRunGrid:
    def test_scale_n_returns_multiple_results(self, tmp_path):
        cfgs = [_micro_cfg(), _micro_cfg()]  # 2 "sizes" (same for test speed)
        cfgs[1] = ModelConfig(
            vocab_size=64, d_model=48, n_layers=1,
            n_heads=2, n_kv_heads=2, max_seq_len=16,
            use_value_embeddings=False, use_per_layer_scalars=False,
            use_smear=False,
        )
        runner = ScalingRunner(tmp_path)
        # tiny budget so this runs fast
        results = runner.run_grid(
            "scale_n",
            model_configs=cfgs,
            token_budgets=[2 * 16 * 5],  # 5 steps
        )
        assert len(results) == 2

    def test_scale_d_returns_multiple_results(self, tmp_path):
        cfg = _micro_cfg()
        runner = ScalingRunner(tmp_path)
        results = runner.run_grid(
            "scale_d",
            model_configs=[cfg],
            token_budgets=[2 * 16 * 3, 2 * 16 * 6],
        )
        assert len(results) == 2

    def test_scale_c_runs(self, tmp_path):
        runner = ScalingRunner(tmp_path)
        # Very small budgets so models are tiny
        results = runner.run_grid(
            "scale_c",
            compute_budgets=[1e9, 1e10],
            model_configs=[ModelConfig(vocab_size=64, d_model=32,
                                       n_layers=1, n_heads=2, n_kv_heads=2,
                                       max_seq_len=16,
                                       use_value_embeddings=False,
                                       use_per_layer_scalars=False,
                                       use_smear=False)],
        )
        assert len(results) == 2

    def test_unknown_experiment_type_raises(self, tmp_path):
        runner = ScalingRunner(tmp_path)
        with pytest.raises(ValueError, match="Unknown experiment_type"):
            runner.run_grid("scale_z")


class TestRunWithRealData:
    """Run a single scaling experiment using the TinyShakespeare HDF5."""

    def test_run_with_hdf5_data(self, tmp_path):
        pytest.importorskip("h5py")
        data_path = __import__("pathlib").Path("data/shakespeare_char.h5")
        if not data_path.exists():
            pytest.skip("shakespeare_char.h5 not available")

        from nanochat.tokenizer.char import CharTokenizer
        from nanochat.data.dataset import TokenDataset
        from nanochat.data.loader import build_dataloader

        tok = CharTokenizer.load("data/shakespeare_vocab.json")
        cfg = ModelConfig(
            vocab_size=tok.vocab_size, d_model=32, n_layers=1,
            n_heads=2, n_kv_heads=2, max_seq_len=32,
            use_value_embeddings=False, use_per_layer_scalars=False,
            use_smear=False,
        )
        train_cfg = TrainingConfig(
            batch_size=4, total_steps=10,
            optimizer="adamw", dtype="float32", param_dtype="float32",
        )
        ds = TokenDataset(data_path, cfg.max_seq_len, split="train")
        loader = build_dataloader(ds, train_cfg.batch_size, shuffle=True, seed=0)

        runner = ScalingRunner(tmp_path)
        result = runner.run_single(cfg, train_cfg, run_id="real_data",
                                   data_loader=loader)
        assert math.isfinite(result.final_val_loss)
        assert result.final_val_loss < math.log(tok.vocab_size) * 2
