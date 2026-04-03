"""Distributed training utilities for JAX."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import structlog
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from nanochat.config import ModelConfig

logger = structlog.get_logger()


def create_device_mesh(
    mesh_shape: tuple[int, int] | None = None,
    axis_names: tuple[str, str] = ("data", "model"),
) -> Mesh:
    """Create a 2D JAX device mesh for data x model parallelism.

    If mesh_shape is None, auto-factors n_devices preferring
    more data parallelism over model parallelism.

    Args:
        mesh_shape: (data_parallel, model_parallel) or None for auto
        axis_names: Names for mesh dimensions

    Returns:
        JAX Mesh object

    Raises:
        RuntimeError: If no devices available
        ValueError: If mesh_shape doesn't match device count
    """
    devices = jax.devices()
    n_devices = len(devices)

    if n_devices < 1:
        raise RuntimeError("No JAX devices available")

    if mesh_shape is None:
        # Auto-factor: prefer more data parallelism
        dp = n_devices
        mp = 1
        for d in range(1, n_devices + 1):
            if n_devices % d == 0:
                m = n_devices // d
                if d >= m:
                    dp, mp = d, m
        mesh_shape = (dp, mp)

    dp, mp = mesh_shape
    if dp * mp != n_devices:
        raise ValueError(
            f"mesh_shape {mesh_shape} requires {dp * mp} devices, "
            f"but {n_devices} available"
        )

    device_array = np.array(devices[:n_devices]).reshape(dp, mp)
    mesh = Mesh(device_array, axis_names=axis_names)

    logger.info("mesh_created", shape=mesh_shape, n_devices=n_devices)
    return mesh


def get_partition_specs(cfg: ModelConfig) -> dict[str, P]:
    """Return PartitionSpec for model parameters.

    Megatron-LM style tensor parallelism:
    - Embeddings: shard along vocab dimension
    - Q/K/V: shard along head dimension (output)
    - Out proj: shard along head dimension (input)
    - FFN up/gate: shard along d_ff dimension (output)
    - FFN down: shard along d_ff dimension (input)
    - Norms: replicated
    """
    return {
        "embed": P("model", None),
        "q_proj": P(None, "model"),
        "k_proj": P(None, "model"),
        "v_proj": P(None, "model"),
        "out_proj": P("model", None),
        "gate_proj": P(None, "model"),
        "up_proj": P(None, "model"),
        "down_proj": P("model", None),
        "norm": P(None),
        "lm_head": P(None, "model"),
    }
