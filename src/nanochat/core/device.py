"""Device abstraction layer for NanoChat-JAX.

Provides a unified interface for CPU, GPU (CUDA), and TPU backends.
Call ``setup_device(device_type)`` exactly once at program startup
before any JAX operations.

All subsequent code uses the constants and helpers exported from this
module.  Never import ``jax.devices()`` or set ``jax.config`` anywhere
else in the codebase.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import structlog

log = structlog.get_logger()


class DeviceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


# ── Module-level state (set once by setup_device) ─────────────
_DEVICE_TYPE: DeviceType | None = None
COMPUTE_DTYPE: jnp.dtype = jnp.float32
PARAM_DTYPE: jnp.dtype = jnp.float32
INDEX_DTYPE: jnp.dtype = jnp.int32


def setup_device(device: str | DeviceType) -> DeviceType:
    """Initialise JAX for the target device.

    Must be called exactly once before any JAX computation.
    Idempotent if called again with the same argument.

    Args:
        device: One of ``"cpu"``, ``"gpu"``, ``"tpu"`` (case-insensitive).

    Returns:
        The :class:`DeviceType` that was configured.

    Raises:
        RuntimeError: If the device was already set to a *different* type,
            or if the requested backend is unavailable.
        ValueError: If *device* is not a recognised string.
    """
    global _DEVICE_TYPE, COMPUTE_DTYPE

    dtype = DeviceType(device.lower() if isinstance(device, str) else device.value)

    if _DEVICE_TYPE is not None and _DEVICE_TYPE != dtype:
        raise RuntimeError(
            f"Device already set to {_DEVICE_TYPE}. Cannot change to {dtype}. "
            "Call setup_device only once per process."
        )
    if _DEVICE_TYPE == dtype:
        return dtype

    if dtype == DeviceType.CPU:
        os.environ["JAX_PLATFORMS"] = "cpu"
        COMPUTE_DTYPE = jnp.float32
        log.info("device.setup", device="cpu", compute_dtype="float32")

    elif dtype == DeviceType.GPU:
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
        try:
            devices = jax.devices("gpu")
        except RuntimeError as exc:
            raise RuntimeError(
                f"GPU requested but JAX cannot find CUDA devices: {exc}\n"
                "Install jax[cuda12] or check CUDA installation."
            ) from exc
        COMPUTE_DTYPE = jnp.bfloat16
        log.info(
            "device.setup",
            device="gpu",
            n_devices=len(devices),
            compute_dtype="bfloat16",
        )

    elif dtype == DeviceType.TPU:
        os.environ.setdefault("JAX_PLATFORMS", "tpu")
        if "TPU_WORKER_ID" in os.environ or "CLOUD_TPU_TASK_ID" in os.environ:
            jax.distributed.initialize()
            log.info(
                "device.tpu_distributed_init",
                n_hosts=jax.process_count(),
                host_id=jax.process_index(),
            )
        try:
            devices = jax.devices("tpu")
        except RuntimeError as exc:
            raise RuntimeError(
                f"TPU requested but JAX cannot find TPU devices: {exc}\n"
                "Set PJRT_DEVICE=TPU and ensure libtpu is installed."
            ) from exc
        COMPUTE_DTYPE = jnp.bfloat16
        log.info(
            "device.setup",
            device="tpu",
            n_devices=len(devices),
            compute_dtype="bfloat16",
        )

    _DEVICE_TYPE = dtype
    return dtype


def get_device_type() -> DeviceType:
    """Return the currently configured device type.

    Raises:
        RuntimeError: If :func:`setup_device` has not been called yet.
    """
    if _DEVICE_TYPE is None:
        raise RuntimeError("setup_device() has not been called yet.")
    return _DEVICE_TYPE


def get_devices() -> list[jax.Device]:
    """Return all available devices of the configured type."""
    dt = get_device_type()
    if dt == DeviceType.CPU:
        return jax.devices("cpu")
    return jax.devices(dt.value)


def n_devices() -> int:
    """Return the number of available accelerators."""
    return len(get_devices())


def is_main_process() -> bool:
    """``True`` on the process that should log, save checkpoints, etc."""
    return jax.process_index() == 0


def cast_to_compute(x: jax.Array) -> jax.Array:
    """Cast *x* to the compute dtype for the current device."""
    return x.astype(COMPUTE_DTYPE)


def cast_batch(
    batch: dict[str, Any],
    keys_to_cast: set[str] | None = None,
) -> dict[str, Any]:
    """Cast float arrays in *batch* to :data:`COMPUTE_DTYPE`.

    Integer arrays (input_ids, labels, masks) are left as int32.

    Args:
        batch: Mapping of name to :class:`jax.Array`.
        keys_to_cast: If *None*, cast all float arrays.
    """
    result: dict[str, Any] = {}
    for k, v in batch.items():
        if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating):
            if keys_to_cast is None or k in keys_to_cast:
                result[k] = v.astype(COMPUTE_DTYPE)
            else:
                result[k] = v
        else:
            result[k] = v
    return result


def make_mesh(
    mesh_shape: tuple[int, int] | None = None,
    axis_names: tuple[str, str] = ("data", "model"),
) -> jax.sharding.Mesh:
    """Create a 2-D device mesh for tensor x data parallelism.

    If *mesh_shape* is ``None``, auto-selects a shape that prefers data
    parallelism for <=4 devices and splits for >=8.

    Args:
        mesh_shape: ``(dp, mp)`` or ``None`` for auto.
        axis_names: Names for the two mesh axes.

    Returns:
        A :class:`jax.sharding.Mesh`.

    Raises:
        ValueError: If the product of *mesh_shape* does not match
            the number of available devices.
    """
    from jax.sharding import Mesh

    devices = np.array(get_devices())
    n = len(devices)

    if mesh_shape is None:
        mp = 1 if n <= 4 else 2
        dp = n // mp
        mesh_shape = (dp, mp)

    if mesh_shape[0] * mesh_shape[1] != n:
        raise ValueError(
            f"mesh_shape {mesh_shape} product {mesh_shape[0] * mesh_shape[1]} "
            f"!= n_devices {n}"
        )

    return Mesh(devices.reshape(mesh_shape), axis_names=axis_names)


def device_info() -> dict[str, Any]:
    """Return a dict of device info suitable for logging."""
    dt = get_device_type()
    devs = get_devices()
    return {
        "device_type": dt.value,
        "n_devices": len(devs),
        "compute_dtype": str(COMPUTE_DTYPE),
        "param_dtype": str(PARAM_DTYPE),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in devs[:4]],  # cap for readability
    }


def reset_for_testing() -> None:
    """Reset module state — **only for unit tests**."""
    global _DEVICE_TYPE, COMPUTE_DTYPE
    _DEVICE_TYPE = None
    COMPUTE_DTYPE = jnp.float32
