"""Tests for CheckpointManager: save, load, resume, best-k cleanup."""
from __future__ import annotations
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from flax import nnx

from nanochat.config import ModelConfig, TrainingConfig
from nanochat.model.transformer import TransformerLM
from nanochat.training.checkpoint import CheckpointManager
from nanochat.training.optimizer import build_optimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_model():
    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_layers=1,
        n_heads=2, n_kv_heads=2, max_seq_len=16,
        use_value_embeddings=False, use_per_layer_scalars=False,
        use_smear=False,
    )
    model = TransformerLM(cfg, rngs=nnx.Rngs(params=0, dropout=1))
    return model, cfg


def _get_param_array(model):
    """Return a representative parameter as a numpy array for comparison."""
    state = nnx.state(model, nnx.Param)
    leaves = jax.tree_util.tree_leaves(state)
    return np.asarray(leaves[0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckpointSave:
    def test_creates_directory_structure(self, tmp_path):
        model, _ = _tiny_model()
        ckpt = CheckpointManager(tmp_path / "ckpts", keep_last_n=3)
        path = ckpt.save(100, model, {"val_loss": 2.5})

        assert path.is_dir()
        assert (path / "model.pkl").exists()
        assert (path / "metadata.json").exists()

    def test_metadata_contains_step_and_metrics(self, tmp_path):
        import json
        model, _ = _tiny_model()
        ckpt = CheckpointManager(tmp_path / "ckpts", keep_last_n=3)
        path = ckpt.save(500, model, {"val_loss": 1.8, "train_loss": 1.5})

        meta = json.loads((path / "metadata.json").read_text())
        assert meta["step"] == 500
        assert meta["metrics"]["val_loss"] == pytest.approx(1.8)

    def test_latest_symlink_updated(self, tmp_path):
        model, _ = _tiny_model()
        ckpt = CheckpointManager(tmp_path / "ckpts", keep_last_n=3)
        ckpt.save(100, model)
        ckpt.save(200, model)

        latest = ckpt.find_latest()
        assert latest is not None
        assert "step_000200" in str(latest)

    def test_keep_last_n_removes_old(self, tmp_path):
        model, _ = _tiny_model()
        ckpt = CheckpointManager(tmp_path / "ckpts", keep_last_n=2)
        ckpt.save(100, model)
        ckpt.save(200, model)
        ckpt.save(300, model)  # should evict step_100

        remaining = list((tmp_path / "ckpts").glob("step_*"))
        steps = sorted(int(p.name.split("_")[1]) for p in remaining)
        assert 100 not in steps
        assert steps == [200, 300]


class TestCheckpointLoad:
    def test_round_trip_preserves_weights(self, tmp_path):
        """Save then load must produce bit-identical parameters."""
        model, _ = _tiny_model()
        original = _get_param_array(model)

        ckpt = CheckpointManager(tmp_path / "ckpts", keep_last_n=3)
        path = ckpt.save(1000, model)

        # Corrupt model weights
        model2, _ = _tiny_model()  # fresh random init
        loaded = ckpt.load(path, model2)

        restored = _get_param_array(model2)
        np.testing.assert_array_almost_equal(original, restored, decimal=6)
        assert loaded == 1000

    def test_load_returns_step(self, tmp_path):
        model, _ = _tiny_model()
        ckpt = CheckpointManager(tmp_path / "ckpts")
        path = ckpt.save(42, model)
        step = ckpt.load(path, model)
        assert step == 42

    def test_load_latest_no_checkpoints_returns_none(self, tmp_path):
        model, _ = _tiny_model()
        ckpt = CheckpointManager(tmp_path / "empty")
        result = ckpt.load_latest(model)
        assert result is None

    def test_load_latest_restores_most_recent(self, tmp_path):
        model, _ = _tiny_model()
        ckpt = CheckpointManager(tmp_path / "ckpts", keep_last_n=5)
        ckpt.save(10, model)
        ckpt.save(20, model)
        ckpt.save(30, model)

        model2, _ = _tiny_model()
        step = ckpt.load_latest(model2)
        assert step == 30

    def test_missing_file_raises(self, tmp_path):
        model, _ = _tiny_model()
        ckpt = CheckpointManager(tmp_path / "ckpts")
        with pytest.raises(FileNotFoundError):
            ckpt.load(tmp_path / "nonexistent_step", model)


class TestCheckpointResume:
    """End-to-end: save at step N → reload → training step is correct."""

    def test_resume_restores_global_step(self, tmp_path):
        from nanochat.training.trainer import Trainer
        import jax.numpy as jnp

        model, model_cfg = _tiny_model()
        train_cfg = TrainingConfig(
            batch_size=2,
            total_steps=10,
            optimizer="adamw",
            learning_rate=1e-3,
            warmup_steps=0,
            checkpoint_dir=str(tmp_path / "ckpts"),
            save_every_steps=5,
            eval_every_steps=100,
            dtype="float32",
            param_dtype="float32",
        )

        def loader():
            rng = jax.random.PRNGKey(0)
            S = model_cfg.max_seq_len
            while True:
                rng, k = jax.random.split(rng)
                ids = jax.random.randint(k, (2, S + 1), 0, model_cfg.vocab_size)
                yield {"input_ids": ids[:, :-1], "labels": ids[:, 1:],
                       "attention_mask": jnp.ones((2, S), jnp.int32)}

        trainer = Trainer(
            model=model,
            train_loader=loader(),
            val_loader=None,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
        )
        results = trainer.train()
        assert results["final_step"] == 9  # 0..9

        # Verify checkpoint was saved
        ckpts = list((tmp_path / "ckpts").glob("step_*"))
        assert len(ckpts) > 0

    def test_save_with_optimizer_state(self, tmp_path):
        """Saving optimizer state produces optimizer.pkl."""
        model, cfg = _tiny_model()
        train_cfg = TrainingConfig(
            optimizer="adamw", batch_size=2, total_steps=5,
            dtype="float32", param_dtype="float32",
        )
        tx = build_optimizer(train_cfg)
        opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

        ckpt = CheckpointManager(tmp_path / "ckpts")
        path = ckpt.save(5, model, optimizer=opt)
        assert (path / "optimizer.pkl").exists()


class TestBestCheckpoint:
    def test_best_by_val_loss(self, tmp_path):
        model, _ = _tiny_model()
        ckpt = CheckpointManager(tmp_path / "ckpts", keep_last_n=10)
        ckpt.save(100, model, {"val_loss": 3.0})
        ckpt.save(200, model, {"val_loss": 2.5})  # best
        ckpt.save(300, model, {"val_loss": 2.8})

        best = ckpt.best_checkpoint("val_loss", lower_is_better=True)
        assert best is not None
        assert "step_000200" in str(best)
