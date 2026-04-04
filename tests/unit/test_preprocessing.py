"""Tests for data preprocessing: tokenization, HDF5 output, idempotency."""
from __future__ import annotations
import pytest
import h5py
import numpy as np
from pathlib import Path

from nanochat.tokenizer.char import CharTokenizer
from nanochat.data.preprocessing import preprocess_and_tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """\
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
"""


@pytest.fixture
def char_tok():
    return CharTokenizer.from_text(SAMPLE_TEXT)


# ---------------------------------------------------------------------------
# CharTokenizer tests
# ---------------------------------------------------------------------------

class TestCharTokenizer:
    def test_vocab_contains_specials(self, char_tok):
        assert char_tok.bos_id == 1
        assert char_tok.eos_id == 2
        assert char_tok.pad_id == 0

    def test_encode_decode_round_trip(self, char_tok):
        original = "Hello, World!"
        # Only chars in vocab round-trip; others map to <pad>
        text = "Speak, speak."  # chars in SAMPLE_TEXT
        ids = char_tok.encode(text)
        decoded = char_tok.decode(ids)
        assert decoded == text

    def test_encode_with_bos_eos(self, char_tok):
        ids = char_tok.encode("abc", add_bos=True, add_eos=True)
        assert ids[0] == char_tok.bos_id
        assert ids[-1] == char_tok.eos_id

    def test_decode_skips_specials(self, char_tok):
        ids = [char_tok.bos_id, char_tok.encode("Hi")[0], char_tok.eos_id]
        decoded = char_tok.decode(ids, skip_special_tokens=True)
        assert "<bos>" not in decoded
        assert "<eos>" not in decoded

    def test_from_text_no_duplicates(self):
        tok = CharTokenizer.from_text("aabbcc")
        # Each char should appear once
        assert list(tok._ch2id.keys()).count("a") == 1

    def test_save_load_round_trip(self, char_tok, tmp_path):
        path = tmp_path / "vocab.json"
        char_tok.save(path)
        loaded = CharTokenizer.load(path)
        assert loaded.vocab_size == char_tok.vocab_size
        assert loaded._ch2id == char_tok._ch2id

    def test_batch_encode_shape(self, char_tok):
        texts = ["Speak.", "All:"]
        ids, mask = char_tok.encode_batch(texts, max_length=16)
        assert ids.shape == (2, 16)
        assert mask.shape == (2, 16)


# ---------------------------------------------------------------------------
# Preprocessing pipeline tests
# ---------------------------------------------------------------------------

class TestPreprocessAndTokenize:
    def test_creates_hdf5(self, tmp_path, char_tok):
        text_path = tmp_path / "sample.txt"
        text_path.write_text(SAMPLE_TEXT)
        out_path = tmp_path / "tokens.h5"

        result = preprocess_and_tokenize(
            source=str(text_path),
            tokenizer=char_tok,
            output_path=out_path,
        )

        assert out_path.exists()
        assert result["n_tokens"] > 0

    def test_hdf5_has_tokens_dataset(self, tmp_path, char_tok):
        text_path = tmp_path / "sample.txt"
        text_path.write_text(SAMPLE_TEXT)
        out_path = tmp_path / "tokens.h5"
        preprocess_and_tokenize(source=str(text_path), tokenizer=char_tok,
                                 output_path=out_path)

        with h5py.File(out_path, "r") as f:
            assert "tokens" in f
            tokens = f["tokens"][:]
            assert tokens.dtype == np.int32
            assert len(tokens) > 0

    def test_tokens_include_eos_separators(self, tmp_path, char_tok):
        text_path = tmp_path / "sample.txt"
        text_path.write_text(SAMPLE_TEXT)
        out_path = tmp_path / "tokens.h5"
        preprocess_and_tokenize(source=str(text_path), tokenizer=char_tok,
                                 output_path=out_path)

        with h5py.File(out_path, "r") as f:
            tokens = f["tokens"][:]
        assert char_tok.eos_id in tokens

    def test_idempotent_skip_if_exists(self, tmp_path, char_tok):
        text_path = tmp_path / "sample.txt"
        text_path.write_text(SAMPLE_TEXT)
        out_path = tmp_path / "tokens.h5"

        r1 = preprocess_and_tokenize(source=str(text_path), tokenizer=char_tok,
                                      output_path=out_path)
        r2 = preprocess_and_tokenize(source=str(text_path), tokenizer=char_tok,
                                      output_path=out_path)
        assert r1["n_tokens"] == r2["n_tokens"]

    def test_max_samples_limits_lines(self, tmp_path, char_tok):
        # Write many lines
        lines = "\n".join(f"Line {i}." for i in range(100))
        text_path = tmp_path / "many.txt"
        text_path.write_text(lines)
        out_path = tmp_path / "limited.h5"

        r_full = preprocess_and_tokenize(source=str(text_path), tokenizer=char_tok,
                                          output_path=out_path)

        out_path2 = tmp_path / "limited2.h5"
        r_limited = preprocess_and_tokenize(source=str(text_path), tokenizer=char_tok,
                                             output_path=out_path2, max_samples=10)

        assert r_limited["n_tokens"] < r_full["n_tokens"]

    def test_real_shakespeare_hdf5(self):
        """Verify the actual shakespeare_char.h5 we created."""
        h5_path = Path("data/shakespeare_char.h5")
        if not h5_path.exists():
            pytest.skip("shakespeare_char.h5 not created yet")

        with h5py.File(h5_path, "r") as f:
            assert "tokens" in f
            tokens = f["tokens"][:]
            assert len(tokens) > 1_000_000  # should be ~1.1M
            assert tokens.dtype == np.int32
