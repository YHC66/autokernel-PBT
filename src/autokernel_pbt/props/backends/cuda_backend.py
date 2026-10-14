"""Device backend for CUDA C++ kernels compiled with NVRTC.

The translation segment's first target setup (feature 0012). It mirrors `TritonBackend`
deliberately -- same protocol, same failure discipline, the same telemetry schema -- so
that the two setups of a translation pair are recorded by code that differs only where
the toolchains do. A kernel that fails is *data*, never an exception that escapes: an
escaping exception aborts a recording whose earlier executions have already cost rented
hardware time.

The status mapping:

* `CudaCompilationError` -> COMPILE_ERROR. NVRTC rejected the source, or the driver could
  not load it or find its entry point; either way the kernel never ran and says nothing
  about numerics. Classified by type, never by message.
* `InputMutatedError` -> LAUNCH_ERROR with `input_mutated` set. Raised by the launcher, the
  only layer holding the device buffers; see `triton_kernel.InputMutatedError`.
* `OutputContractError` -> OUTPUT_ERROR.
* `CudaToolchainError` is NOT mapped: it is raised. The toolchain failed, not the kernel --
  booked as the kernel's LAUNCH_ERROR it would count against a kernel that never compiled,
  and every later kernel would fail the same way. A bad call; the recorder keeps the
  completed groups in `<out>.partial` before the error propagates.
* anything else -> LAUNCH_ERROR. That includes a launcher refusing an input it would
  misread, a failed `cuLaunchKernel`, and an asynchronous fault surfaced by the
  synchronize after it.

Execution is not bitwise reproducible across launches in general, and nothing here relies
on it: the recorded execution is the one every arm scores.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from autokernel_pbt.props.backends.base import (
    OUTPUT_NAME,
    TELEMETRY_BACKEND,
    TELEMETRY_WALL_MS,
    ExecutionResult,
    OutputContractError,
    Status,
    kernel_inputs,
)
from autokernel_pbt.props.backends.cuda_kernel import CudaKernel
from autokernel_pbt.props.backends.cuda_toolchain import CudaCompilationError, CudaToolchainError
from autokernel_pbt.props.backends.device import cuda_device_facts
from autokernel_pbt.props.backends.telemetry import MISSING, extract
from autokernel_pbt.props.backends.triton_kernel import InputMutatedError
from autokernel_pbt.props.case import Case


class CudaBackend:
    """Executes `CudaKernel` adapters and records device telemetry."""

    name = "cuda"

    def __init__(self, device_probe: Any = None) -> None:
        # Injected so the backend is exercisable with no CUDA present. The default reads
        # torch lazily, inside itself, so this module imports cleanly on a machine with
        # no torch.
        self.device_probe = device_probe or cuda_device_facts

    def run(self, kernel: Any, case: Case) -> ExecutionResult:
        if not isinstance(kernel, CudaKernel):
            # A bad *call*, not bad data: it can only come from a coding error and costs
            # nothing to re-run.
            msg = (
                f"{type(self).__name__} requires a CudaKernel adapter, got "
                f"{type(kernel).__name__}; wrap the CUDA source first"
            )
            raise TypeError(msg)

        inputs = kernel_inputs(case)
        start = time.perf_counter()
        try:
            output = kernel(**inputs)
        except CudaToolchainError:
            raise
        except CudaCompilationError as exc:
            return self._failed(kernel, case, start, Status.COMPILE_ERROR, exc)
        except InputMutatedError as exc:
            # Not redundant with the generic handler below: it sets the flag that
            # separates "wrote to its own input" from "crashed" in the artifacts.
            return self._failed(
                kernel, case, start, Status.LAUNCH_ERROR, exc, flags={"input_mutated": True}
            )
        except OutputContractError as exc:
            return self._failed(kernel, case, start, Status.OUTPUT_ERROR, exc)
        except Exception as exc:  # noqa: BLE001 - a failing kernel is data, not an error
            return self._failed(kernel, case, start, Status.LAUNCH_ERROR, exc)

        return ExecutionResult(
            case=case,
            outputs={OUTPUT_NAME: output},
            telemetry=self._telemetry(kernel, start),
            status=Status.OK,
        )

    def _telemetry(
        self, kernel: CudaKernel, start: float, flags: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        device = self._probe(self.device_probe)
        device.update(self._probe(kernel.toolchain_versions))
        payload = extract(
            kernel.compiled,
            device=device,
            launch=kernel.launch_telemetry(),
            flags=flags,
        )
        payload["shared_bytes"] = _shared_per_block(payload)
        payload[TELEMETRY_BACKEND] = self.name
        payload[TELEMETRY_WALL_MS] = (time.perf_counter() - start) * 1000.0
        payload["kernel_source_hash"] = kernel.source_hash
        return payload

    @staticmethod
    def _probe(read: Any) -> dict[str, Any]:
        """A device-group reading, or an empty one -- every key MISSING -- if it fails.

        Runs on the failure path too. After a sticky CUDA error every later CUDA call
        raises, the probe's included, and an unguarded probe would re-raise from inside
        the error handler and escape `run` on the row that most needed recording.
        """
        try:
            return dict(read())
        except Exception:  # noqa: BLE001 - probed, not asserted
            return {}

    def _failed(
        self,
        kernel: CudaKernel,
        case: Case,
        start: float,
        status: Status,
        exc: BaseException,
        flags: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        return ExecutionResult(
            case=case,
            telemetry=self._telemetry(kernel, start, flags),
            status=status,
            error="".join(traceback.format_exception(exc)),
        )


def _shared_per_block(payload: dict[str, Any]) -> Any:
    """The shared memory one block is given: static plus dynamic, or MISSING.

    The quantity Triton's `metadata.shared` reports, which is what `shared_bytes` means on
    a Triton row: Triton's kernels use dynamic shared memory only, and the launch passes
    exactly that much. The driver's `SHARED_SIZE_BYTES` is the static part alone, so for a
    port using `extern __shared__` it is 0 while each block really gets `width * 4`
    bytes; recorded as `shared_bytes` it would compare the two setups on different
    quantities.
    """
    static = payload.get("static_shared_bytes")
    dynamic = payload.get("dynamic_shared_bytes")
    if isinstance(static, bool) or isinstance(dynamic, bool):
        return MISSING
    if isinstance(static, int) and isinstance(dynamic, int):
        return static + dynamic
    return MISSING
