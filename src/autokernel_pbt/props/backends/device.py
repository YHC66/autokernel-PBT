"""Device and toolchain facts every NVIDIA backend records, read through torch and the driver.

Shared by the Triton and CUDA backends so that the two setups of a translation pair, run
on one device in one session, describe that device identically -- a difference in how
two backends spelled the same driver would read as a difference between setups.

torch and ctypes' driver library are loaded inside the functions: these modules must
import cleanly on a machine with no torch and no driver, so that every structural test
runs on CPU.
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
        "driver_version": driver_version(),
        # The CUDA toolkit torch was BUILT against, which is not the driver.
        "runtime_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def _load_libcuda() -> Any:
    """The driver library itself, by its Linux name. Raises where there is no driver."""
    import ctypes

    return ctypes.CDLL("libcuda.so.1")


def driver_version() -> Any:
    """The CUDA version the installed DRIVER supports, as "major.minor", or MISSING.

    `torch.version.cuda` was recorded here before, which is the toolkit torch was built
    with -- a property of the wheel, not of the machine -- so driver and runtime always
    read identically and a driver mismatch was invisible in every artifact.

    Asked of the driver library through `cuDriverGetVersion`, which encodes 1000*major +
    10*minor: 12090 is a 12.9 driver. It is formatted like `runtime_version` ("12.8") so
    the two compare directly. This went through torch's private
    `torch._C._cuda_getDriverVersion` until the first device session (2026-10-20, RTX
    A6000) showed torch 2.7.0 has no such attribute, so every row on Lambda Stack would
    have recorded MISSING. The driver library is the authority, and it is loaded wherever
    either backend can launch a kernel. Probed: no library, a failed call or zero (no
    driver) record MISSING rather than a guess.
    """
    import ctypes

    encoded = ctypes.c_int(0)
    try:
        status = _load_libcuda().cuDriverGetVersion(ctypes.byref(encoded))
    except Exception:  # noqa: BLE001 - probed, not asserted
        return MISSING
    if status != 0 or encoded.value <= 0:
        return MISSING
    return f"{encoded.value // 1000}.{encoded.value % 1000 // 10}"
