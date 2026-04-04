"""Tests for the lm-eval harness adapter."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nanochat.config import ModelConfig
from nanochat.model.transformer import TransformerLM
from nanochat.tokenizer.char import CharTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def nano_model_and_tokenizer():
    """Build a tiny model and character tokenizer for testing."""
    text = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"
    tokenizer = CharTokenizer.from_text(text)
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=64,
    )
    model = TransformerLM(cfg, rngs=nnx.Rngs(params=0, dropout=1))
    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_adapter_import_without_lm_eval():
    """Importing the adapter module should not crash when lm-eval is absent.

    We temporarily hide the lm_eval package and reimport the adapter to
    verify the graceful ImportError path.
    """
    # Save and remove lm_eval from sys.modules if present
    saved_modules = {}
    for key in list(sys.modules):
        if key == "lm_eval" or key.startswith("lm_eval."):
            saved_modules[key] = sys.modules.pop(key)

    # Also remove our adapter so it gets reimported
    adapter_key = "nanochat.evaluation.lm_eval_adapter"
    if adapter_key in sys.modules:
        saved_modules[adapter_key] = sys.modules.pop(adapter_key)

    # Block lm_eval from importing
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def mock_import(name, *args, **kwargs):
        if name == "lm_eval" or name.startswith("lm_eval."):
            raise ImportError("mocked: lm_eval not installed")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        # Force reimport
        import importlib
        # We need a fresh import - manually remove from sys.modules
        for key in list(sys.modules):
            if "lm_eval_adapter" in key:
                del sys.modules[key]

        # Import should succeed (the module defines stubs)
        spec = importlib.util.find_spec("nanochat.evaluation.lm_eval_adapter")
        # The module file exists; we just verify no crash on import attempt
        # by checking the _LM_EVAL_AVAILABLE flag
        from nanochat.evaluation import lm_eval_adapter as adapter_mod
        # After the patched import, it may or may not have reset; the key test
        # is that the module loaded without raising.

    # Restore
    sys.modules.update(saved_modules)

    # Re-import cleanly so other tests work
    import importlib
    if adapter_key in sys.modules:
        importlib.reload(sys.modules[adapter_key])


def test_tok_encode_decode_roundtrip(nano_model_and_tokenizer):
    """tok_encode -> tok_decode should roundtrip cleanly for simple text."""
    model, tokenizer, cfg = nano_model_and_tokenizer

    from nanochat.evaluation.lm_eval_adapter import NanoChatJAXModel, _LM_EVAL_AVAILABLE

    if not _LM_EVAL_AVAILABLE:
        pytest.skip("lm-eval not installed")

    adapter = NanoChatJAXModel(model, tokenizer, cfg, batch_size=2)

    original = "hello world"
    encoded = adapter.tok_encode(original)
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    assert len(encoded) > 0

    decoded = adapter.tok_decode(encoded)
    assert decoded == original


def test_loglikelihood_returns_correct_format(nano_model_and_tokenizer):
    """loglikelihood should return (float, bool) tuples with finite values."""
    model, tokenizer, cfg = nano_model_and_tokenizer

    from nanochat.evaluation.lm_eval_adapter import NanoChatJAXModel, _LM_EVAL_AVAILABLE

    if not _LM_EVAL_AVAILABLE:
        pytest.skip("lm-eval not installed")

    from lm_eval.api.instance import Instance

    adapter = NanoChatJAXModel(model, tokenizer, cfg, batch_size=2)

    # Create mock requests
    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("hello ", "world"),
            idx=0,
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("the cat sat on ", "the mat"),
            idx=1,
        ),
    ]

    results = adapter.loglikelihood(requests)

    assert len(results) == 2
    for ll, is_greedy in results:
        assert isinstance(ll, float), f"Expected float log-likelihood, got {type(ll)}"
        assert isinstance(is_greedy, bool), f"Expected bool is_greedy, got {type(is_greedy)}"
        assert np.isfinite(ll), f"Log-likelihood should be finite, got {ll}"
        assert ll <= 0.0, f"Log-likelihood should be non-positive, got {ll}"


def test_generate_until_stops_at_eos(nano_model_and_tokenizer):
    """generate_until should produce a string and respect stop sequences."""
    model, tokenizer, cfg = nano_model_and_tokenizer

    from nanochat.evaluation.lm_eval_adapter import NanoChatJAXModel, _LM_EVAL_AVAILABLE

    if not _LM_EVAL_AVAILABLE:
        pytest.skip("lm-eval not installed")

    from lm_eval.api.instance import Instance

    adapter = NanoChatJAXModel(model, tokenizer, cfg, batch_size=1)

    requests = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(
                "once upon a ",
                {"until": ["\n", "."], "max_gen_toks": 32, "temperature": 0.0},
            ),
            idx=0,
        ),
    ]

    results = adapter.generate_until(requests)

    assert len(results) == 1
    generated = results[0]
    assert isinstance(generated, str)
    # The result should not contain any of the stop strings
    assert "\n" not in generated
    assert "." not in generated
