"""Core infrastructure: device abstraction, utilities."""

from nanochat.core.device import (
    DeviceType,
    setup_device,
    get_device_type,
    get_devices,
    n_devices,
    is_main_process,
    cast_to_compute,
    cast_batch,
    make_mesh,
    device_info,
    COMPUTE_DTYPE,
    PARAM_DTYPE,
    INDEX_DTYPE,
)

__all__ = [
    "DeviceType",
    "setup_device",
    "get_device_type",
    "get_devices",
    "n_devices",
    "is_main_process",
    "cast_to_compute",
    "cast_batch",
    "make_mesh",
    "device_info",
    "COMPUTE_DTYPE",
    "PARAM_DTYPE",
    "INDEX_DTYPE",
]
