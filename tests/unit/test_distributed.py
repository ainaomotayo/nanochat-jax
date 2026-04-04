"""Tests for distributed training utilities."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanochat.core.device import setup_device, reset_for_testing
from nanochat.training.distributed import (
    DistributedConfig,
    setup_distributed,
    shard_batch,
)


@pytest.fixture(autouse=True)
def _ensure_device():
    """Ensure device is set up for distributed tests."""
    reset_for_testing()
    setup_device("cpu")
    yield
    reset_for_testing()


def test_setup_distributed_single_device():
    """setup_distributed on 1 device should return a (1,1) mesh."""
    config = setup_distributed()
    assert isinstance(config, DistributedConfig)
    # On CPU we expect exactly 1 device.
    n = len(jax.devices())
    # mesh.shape is a dict {axis_name: size} in JAX
    shape_vals = list(config.mesh.shape.values())
    assert shape_vals[0] * shape_vals[1] == n
    assert config.data_axis == "data"
    assert config.model_axis == "model"


def test_distributed_config_has_mesh():
    """DistributedConfig should have a valid mesh attribute."""
    config = setup_distributed()
    assert config.mesh is not None
    assert hasattr(config.mesh, "shape")
    assert len(config.mesh.shape) == 2


def test_shard_batch_preserves_shapes():
    """shard_batch should preserve array shapes and dtypes."""
    config = setup_distributed()
    B, S = 4, 16
    key = jax.random.PRNGKey(0)

    batch = {
        "input_ids": jax.random.randint(key, (B, S), 0, 256),
        "labels": jax.random.randint(key, (B, S), 0, 256),
        "attention_mask": jnp.ones((B, S), dtype=jnp.int32),
    }

    sharded = shard_batch(batch, config.mesh, config.data_axis)

    for name in batch:
        assert sharded[name].shape == batch[name].shape, (
            f"Shape mismatch for '{name}': "
            f"{sharded[name].shape} vs {batch[name].shape}"
        )
        assert sharded[name].dtype == batch[name].dtype, (
            f"Dtype mismatch for '{name}': "
            f"{sharded[name].dtype} vs {batch[name].dtype}"
        )


def test_shard_batch_passes_through_non_arrays():
    """Non-array values in the batch should be passed through."""
    config = setup_distributed()
    batch = {
        "input_ids": jnp.ones((2, 8), dtype=jnp.int32),
        "metadata": "some_string",
        "step": 42,
    }

    sharded = shard_batch(batch, config.mesh, config.data_axis)
    assert sharded["metadata"] == "some_string"
    assert sharded["step"] == 42
