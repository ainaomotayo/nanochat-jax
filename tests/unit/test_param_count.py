"""Tests for parameter counting."""
from flax import nnx
from nanochat.model.transformer import TransformerLM
from nanochat.model.param_count import count_params, estimate_flops_per_token


def test_count_params(nano_config):
    model = TransformerLM(nano_config, rngs=nnx.Rngs(0))
    counts = count_params(model)
    assert "total" in counts
    assert counts["total"] > 0


def test_flops_positive(nano_config):
    flops = estimate_flops_per_token(nano_config)
    assert flops > 0
