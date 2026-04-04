"""Tests for TokenDataset: windowing, train/val split, HDF5 loading."""
from __future__ import annotations
import pytest
import h5py
import numpy as np
from pathlib import Path

from nanochat.data.dataset import TokenDataset, IGNORE_INDEX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_h5(path: Path, n_tokens: int = 2000) -> Path:
    """Write a simple sequential token array to an HDF5 file."""
    tokens = np.arange(n_tokens, dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("tokens", data=tokens)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTokenDatasetInit:
    def test_loads_without_error(self, tmp_path):
        h5 = _make_h5(tmp_path / "tok.h5")
        ds = TokenDataset(h5, seq_len=32)
        assert len(ds) > 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TokenDataset(tmp_path / "nonexistent.h5", seq_len=32)

    def test_missing_tokens_key_raises(self, tmp_path):
        h5 = tmp_path / "bad.h5"
        with h5py.File(h5, "w") as f:
            f.create_dataset("data", data=np.zeros(100, dtype=np.int32))
        with pytest.raises(KeyError):
            TokenDataset(h5, seq_len=32)

    def test_n_windows_correct(self, tmp_path):
        h5 = _make_h5(tmp_path / "tok.h5", n_tokens=1000)
        ds = TokenDataset(h5, seq_len=32, split="train", val_fraction=0.0)
        # (n_tokens - seq_len) // seq_len  (stride defaults to seq_len)
        expected_windows = (1000 - 32) // 32
        assert abs(len(ds) - expected_windows) <= 2  # allow small rounding diff


class TestTokenDatasetSplit:
    def test_train_val_no_overlap(self, tmp_path):
        h5 = _make_h5(tmp_path / "tok.h5", n_tokens=2000)
        train_ds = TokenDataset(h5, seq_len=32, split="train", val_fraction=0.1)
        val_ds = TokenDataset(h5, seq_len=32, split="val", val_fraction=0.1)
        assert train_ds._end == val_ds._start

    def test_invalid_split_raises(self, tmp_path):
        h5 = _make_h5(tmp_path / "tok.h5")
        with pytest.raises(ValueError):
            TokenDataset(h5, seq_len=32, split="test")


class TestTokenDatasetGetItem:
    def test_returns_dict(self, tmp_path):
        h5 = _make_h5(tmp_path / "tok.h5")
        ds = TokenDataset(h5, seq_len=32)
        item = ds[0]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item

    def test_shapes_correct(self, tmp_path):
        h5 = _make_h5(tmp_path / "tok.h5")
        ds = TokenDataset(h5, seq_len=32)
        item = ds[0]
        assert item["input_ids"].shape == (32,)
        assert item["labels"].shape == (32,)
        assert item["attention_mask"].shape == (32,)

    def test_labels_shifted_by_one(self, tmp_path):
        """labels[t] should equal input_ids[t+1] (pre-shifted convention)."""
        h5 = _make_h5(tmp_path / "tok.h5")
        ds = TokenDataset(h5, seq_len=32)
        item = ds[0]
        # labels = input_ids shifted right by 1
        np.testing.assert_array_equal(
            item["labels"][:-1], item["input_ids"][1:]
        )

    def test_attention_mask_all_ones(self, tmp_path):
        h5 = _make_h5(tmp_path / "tok.h5")
        ds = TokenDataset(h5, seq_len=32)
        item = ds[0]
        assert np.all(item["attention_mask"] == 1)

    def test_index_out_of_bounds_raises(self, tmp_path):
        h5 = _make_h5(tmp_path / "tok.h5")
        ds = TokenDataset(h5, seq_len=32)
        with pytest.raises(IndexError):
            _ = ds[len(ds)]

    def test_different_indices_give_different_windows(self, tmp_path):
        h5 = _make_h5(tmp_path / "tok.h5")
        ds = TokenDataset(h5, seq_len=32)
        assert len(ds) >= 2
        i0 = ds[0]["input_ids"]
        i1 = ds[1]["input_ids"]
        assert not np.array_equal(i0, i1)


class TestRealShakespeare:
    def test_shakespeare_h5_loads(self):
        h5_path = Path("data/shakespeare_char.h5")
        if not h5_path.exists():
            pytest.skip("shakespeare_char.h5 not available")

        ds = TokenDataset(h5_path, seq_len=128, split="train")
        assert len(ds) > 1000
        item = ds[0]
        assert item["input_ids"].shape == (128,)
        assert item["labels"].shape == (128,)
        # labels should be input_ids shifted by 1
        np.testing.assert_array_equal(item["labels"][:-1], item["input_ids"][1:])

    def test_shakespeare_train_val_sizes(self):
        h5_path = Path("data/shakespeare_char.h5")
        if not h5_path.exists():
            pytest.skip("shakespeare_char.h5 not available")

        train_ds = TokenDataset(h5_path, seq_len=128, split="train", val_fraction=0.1)
        val_ds = TokenDataset(h5_path, seq_len=128, split="val", val_fraction=0.1)
        # Train should be roughly 9x larger than val
        assert len(train_ds) > len(val_ds) * 5
