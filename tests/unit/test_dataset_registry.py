"""Tests for the dataset registry."""
from __future__ import annotations

import pytest

from nanochat.data.registry import DatasetSpec, DATASET_REGISTRY, get_dataset


class TestDatasetRegistry:
    """Test the dataset registry and its entries."""

    def test_known_datasets_registered(self):
        expected_names = {"shakespeare_char", "tinystories", "openwebtext", "smoltalk", "gsm8k"}
        registered = set(DATASET_REGISTRY.keys())
        assert expected_names.issubset(registered), (
            f"Missing datasets: {expected_names - registered}"
        )

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError, match="Unknown dataset"):
            get_dataset("nonexistent_dataset_xyz")

    def test_dataset_spec_fields(self):
        for name, spec in DATASET_REGISTRY.items():
            assert isinstance(spec, DatasetSpec)
            assert spec.name == name
            assert isinstance(spec.hf_path, str)
            assert len(spec.hf_path) > 0
            assert isinstance(spec.text_column, str)
            assert len(spec.text_column) > 0
            assert isinstance(spec.split_map, dict)
            assert "train" in spec.split_map
            assert spec.tokenizer in ("char", "bpe")
            assert isinstance(spec.description, str)

    def test_get_dataset_returns_correct_spec(self):
        spec = get_dataset("shakespeare_char")
        assert spec.name == "shakespeare_char"
        assert "shakespeare" in spec.hf_path.lower() or "shakespeare" in spec.description.lower()
        assert spec.tokenizer == "char"

    def test_get_dataset_case_insensitive(self):
        spec1 = get_dataset("Shakespeare_Char")
        spec2 = get_dataset("SHAKESPEARE_CHAR")
        assert spec1.name == spec2.name

    def test_tinystories_spec(self):
        spec = get_dataset("tinystories")
        assert spec.hf_path == "roneneldan/TinyStories"
        assert spec.text_column == "text"
        assert spec.tokenizer == "bpe"
        assert "train" in spec.split_map

    def test_openwebtext_spec(self):
        spec = get_dataset("openwebtext")
        assert spec.hf_path == "Skylion007/openwebtext"
        assert spec.tokenizer == "bpe"

    def test_gsm8k_spec(self):
        spec = get_dataset("gsm8k")
        assert spec.hf_path == "openai/gsm8k"
        assert spec.hf_name == "main"

    def test_smoltalk_spec(self):
        spec = get_dataset("smoltalk")
        assert spec.hf_path == "HuggingFaceTB/smoltalk"

    def test_dataset_spec_frozen(self):
        spec = get_dataset("tinystories")
        with pytest.raises(AttributeError):
            spec.name = "modified"  # type: ignore
