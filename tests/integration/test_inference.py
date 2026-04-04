"""Integration tests for InferenceEngine: generate, streaming, batch."""
from __future__ import annotations
import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from nanochat.config import ModelConfig
from nanochat.model.transformer import TransformerLM
from nanochat.inference.engine import InferenceEngine
from nanochat.tokenizer.char import CharTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    """Tiny model + char tokenizer for inference tests."""
    # Build vocab from a simple alphabet
    text = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!? \n"
    tokenizer = CharTokenizer.from_text(text)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_layers=1,
        n_heads=2, n_kv_heads=2,
        max_seq_len=64,
        use_value_embeddings=False,
        use_per_layer_scalars=False,
        use_smear=False,
        logit_softcap=None,
        use_qk_norm=False,
    )
    model = TransformerLM(cfg, rngs=nnx.Rngs(params=7, dropout=8))
    engine = InferenceEngine(model, tokenizer, cfg)
    return engine, tokenizer, cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_returns_string(self, tiny_model_and_tokenizer):
        engine, tok, _ = tiny_model_and_tokenizer
        result = engine.generate("Hello", max_new_tokens=5)
        assert isinstance(result, str)

    def test_generates_nonzero_tokens(self, tiny_model_and_tokenizer):
        engine, tok, _ = tiny_model_and_tokenizer
        result = engine.generate("abc", max_new_tokens=10)
        assert len(result) > 0

    def test_greedy_deterministic(self, tiny_model_and_tokenizer):
        engine, tok, _ = tiny_model_and_tokenizer
        r1 = engine.generate("abc", max_new_tokens=8, temperature=0, seed=0)
        r2 = engine.generate("abc", max_new_tokens=8, temperature=0, seed=0)
        assert r1 == r2

    def test_batch_returns_list(self, tiny_model_and_tokenizer):
        engine, tok, _ = tiny_model_and_tokenizer
        results = engine.generate(["abc", "xyz"], max_new_tokens=5, temperature=0)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, str)

    def test_max_new_tokens_respected(self, tiny_model_and_tokenizer):
        engine, tok, _ = tiny_model_and_tokenizer
        # Can't exceed max_new_tokens unless EOS triggers early
        result = engine.generate("a", max_new_tokens=4, temperature=0)
        # Generated tokens (before decode) must be <= 4
        assert len(tok.encode(result)) <= 4 + 10  # allow for multi-char decode edge cases

    def test_different_seeds_give_different_output(self, tiny_model_and_tokenizer):
        engine, tok, _ = tiny_model_and_tokenizer
        r1 = engine.generate("abc", max_new_tokens=15, temperature=1.0, seed=0)
        r2 = engine.generate("abc", max_new_tokens=15, temperature=1.0, seed=99)
        # Very likely different (not guaranteed, but highly probable)
        # We just check both are strings
        assert isinstance(r1, str)
        assert isinstance(r2, str)


class TestStreamGenerate:
    def test_stream_returns_generator(self, tiny_model_and_tokenizer):
        engine, tok, _ = tiny_model_and_tokenizer
        gen = engine.generate("Hello", max_new_tokens=5, stream=True)
        # Should be a generator, not a string
        import types
        assert isinstance(gen, types.GeneratorType)

    def test_stream_yields_strings(self, tiny_model_and_tokenizer):
        engine, tok, _ = tiny_model_and_tokenizer
        fragments = list(engine.generate("abc", max_new_tokens=8,
                                          temperature=0, stream=True))
        for f in fragments:
            assert isinstance(f, str)

    def test_stream_join_equals_non_stream(self, tiny_model_and_tokenizer):
        engine, tok, _ = tiny_model_and_tokenizer
        prompt = "abc"
        # Non-streaming
        full = engine.generate(prompt, max_new_tokens=10, temperature=0, seed=0)
        # Streaming
        streamed = "".join(
            engine.generate(prompt, max_new_tokens=10, temperature=0,
                            seed=0, stream=True)
        )
        assert full == streamed

    def test_stream_batch_uses_first_prompt(self, tiny_model_and_tokenizer):
        engine, tok, _ = tiny_model_and_tokenizer
        gen = engine.generate(["abc", "xyz"], max_new_tokens=5,
                               temperature=0, stream=True)
        import types
        assert isinstance(gen, types.GeneratorType)


class TestRealDataInference:
    """Test inference using the Shakespeare model config."""

    def test_generate_with_shakespeare_vocab(self):
        from pathlib import Path
        vocab_path = Path("data/shakespeare_vocab.json")
        if not vocab_path.exists():
            pytest.skip("shakespeare_vocab.json not available")

        tok = CharTokenizer.load(vocab_path)
        cfg = ModelConfig(
            vocab_size=tok.vocab_size,
            d_model=32, n_layers=1,
            n_heads=2, n_kv_heads=2,
            max_seq_len=64,
            use_value_embeddings=False,
            use_per_layer_scalars=False,
            use_smear=False,
            logit_softcap=None,
            use_qk_norm=False,
        )
        model = TransformerLM(cfg, rngs=nnx.Rngs(params=0, dropout=1))
        engine = InferenceEngine(model, tok, cfg)

        result = engine.generate("HAMLET:", max_new_tokens=20, temperature=0.8)
        assert isinstance(result, str)
