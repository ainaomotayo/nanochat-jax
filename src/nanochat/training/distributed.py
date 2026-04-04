"""Distributed training utilities for JAX.

Provides mesh creation, batch sharding, and partition specs for data-
and model-parallel training.  All functions are designed as no-ops on a
single device so that the same training loop works unchanged on a
laptop or a multi-device pod.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import structlog
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from nanochat.config import ModelConfig
from nanochat.core.device import make_mesh

logger = structlog.get_logger(__name__)


@dataclass
class DistributedConfig:
    """Configuration produced by :func:`setup_distributed`.

    Attributes:
        mesh: The JAX device mesh.
        data_axis: Name of the data-parallel mesh axis.
        model_axis: Name of the model-parallel mesh axis.
    """

    mesh: Mesh = field(repr=False)
    data_axis: str = "data"
    model_axis: str = "model"


def setup_distributed(
    mesh_shape: tuple[int, int] | None = None,
) -> DistributedConfig:
    """Set up distributed training and return a config.

    Auto-selects a mesh shape when *mesh_shape* is ``None``:

    ========== ============
    n_devices  mesh (dp,mp)
    ========== ============
    1          (1, 1)
    2-4        (n, 1)
    8          (4, 2)
    ========== ============

    For single-device setups the mesh is trivial ``(1, 1)`` and all
    sharding operations become no-ops.

    Args:
        mesh_shape: Explicit ``(data_parallel, model_parallel)`` shape,
            or ``None`` for automatic selection.

    Returns:
        A :class:`DistributedConfig` with the mesh ready to use.
    """
    n_devices = len(jax.devices())

    if mesh_shape is None:
        if n_devices == 1:
            mesh_shape = (1, 1)
        elif n_devices <= 4:
            mesh_shape = (n_devices, 1)
        elif n_devices == 8:
            mesh_shape = (4, 2)
        else:
            # General case: prefer data parallelism.
            mp = 2 if n_devices >= 8 else 1
            dp = n_devices // mp
            mesh_shape = (dp, mp)

    mesh = make_mesh(mesh_shape, axis_names=("data", "model"))

    logger.info(
        "distributed.setup",
        mesh_shape=mesh_shape,
        n_devices=n_devices,
    )

    return DistributedConfig(mesh=mesh, data_axis="data", model_axis="model")


def shard_batch(
    batch: dict[str, Any],
    mesh: Mesh,
    data_axis: str = "data",
) -> dict[str, Any]:
    """Shard batch arrays along the data axis.

    Each array in *batch* is placed on the mesh with its leading
    (batch) dimension partitioned across the *data_axis*.  Non-array
    values are passed through unchanged.

    For a single-device mesh this is effectively a no-op: the
    :class:`NamedSharding` wraps the array but introduces no
    communication or copy.

    Args:
        batch: Mapping of name to array (or scalar).
        mesh: The device mesh from :class:`DistributedConfig`.
        data_axis: Name of the data-parallel mesh axis.

    Returns:
        A new dict with the same keys, arrays sharded along *data_axis*.
    """
    sharding = NamedSharding(mesh, P(data_axis))

    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, (jax.Array, jnp.ndarray, np.ndarray)):
            arr = jnp.asarray(value)
            result[key] = jax.device_put(arr, sharding)
        else:
            result[key] = value

    return result


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
