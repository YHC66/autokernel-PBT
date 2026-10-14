"""Device and toolchain facts every NVIDIA backend records, read through torch.

Shared by the Triton and CUDA backends so that the two setups of a translation pair, run
on one device in one session, describe that device identically -- a difference in how
two backends spelled the same driver would read as a difference between setups.

torch is imported inside the functions: these modules must import cleanly on a machine
with no torch, so that every structural test runs on CPU.
"""

from __future__ import annotations

from typing import Any

from autokernel_pbt.props.backends.telemetry import MISSING


def cuda_device_facts() -> dict[str, Any]:
    """The device group's backend-independent fields, for torch's current device."""
    import torch

    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multi_processor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "driver_version": driver_version(torch),
        # The CUDA toolkit torch was BUILT against, which is not the driver.
        "runtime_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def driver_version(torch: Any) -> Any:
    """The CUDA version the installed DRIVER supports, as "major.minor", or MISSING.

    `torch.version.cuda` was recorded here before, which is the toolkit torch was built
    with -- a property of the wheel, not of the machine -- so driver and runtime always
    read identically and a driver mismatch was invisible in every artifact.

    `cuDriverGetVersion`, which `_cuda_getDriverVersion` wraps, encodes 1000*major +
    10*minor: 12090 is a 12.9 driver. It is formatted like `runtime_version` ("12.8")
    so the two compare directly. A private torch API, so probed: absent, raising, zero
    (no driver) or not an int all record MISSING rather than a guess.
    """
    try:
        encoded = torch._C._cuda_getDriverVersion()
    except Exception:  # noqa: BLE001 - probed, not asserted
        return MISSING
    if isinstance(encoded, bool) or not isinstance(encoded, int) or encoded <= 0:
        return MISSING
    return f"{encoded // 1000}.{encoded % 1000 // 10}"
