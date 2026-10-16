"""CudaBackend tests, CPU-only.

Status mapping, telemetry assembly and error classification are structural, and are
tested here with a fake toolchain and an injected device probe. The device path itself is
covered by gpu-marked tests in tests/gpu/, run by hand on the instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OUTPUT_NAME, Status
from autokernel_pbt.props.backends.cuda_backend import CudaBackend
from autokernel_pbt.props.backends.cuda_kernel import CudaKernel
from autokernel_pbt.props.backends.cuda_toolchain import (
    CudaCompilationError,
    CudaToolchainError,
)
from autokernel_pbt.props.backends.telemetry import (
    MISSING,
    SCHEMA_VERSION,
    TELEMETRY_SCHEMA_VERSION,
    declared_keys,
)
from autokernel_pbt.props.backends.triton_kernel import InputMutatedError
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.table import ExecutionTable

NVRTC_LOG = 'kernel.cu(4): error: identifier "undefined_thing" is undefined'


def _case() -> Case:
    return Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="relu",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.ones((2, 3), dtype=np.float32)},
    )


def _kernel(toolchain, launcher=None, **overrides) -> CudaKernel:
    defaults = {
        "kernel_id": "k",
        "source": 'extern "C" __global__ void k(const float* x, float* y, int n) {}',
        "entry": "k",
        "grid": lambda shape, params: (shape[0],),
        "block": lambda shape, params: (32,),
        "launcher": launcher or (lambda **kw: np.zeros((2, 3), dtype=np.float32)),
        "toolchain": toolchain,
    }
    defaults.update(overrides)
    return CudaKernel(**defaults)


def _backend(**overrides) -> CudaBackend:
    defaults = {"device_probe": lambda: {"device_name": "fake", "compute_capability": "8.6"}}
    defaults.update(overrides)
    return CudaBackend(**defaults)


def _raising(exc: Exception):
    def launcher(**kw):
        raise exc

    return launcher


def test_a_successful_run_reports_ok_and_the_output(fake_toolchain):
    result = _backend().run(_kernel(fake_toolchain()), _case())
    assert result.status is Status.OK
    assert result.outputs[OUTPUT_NAME].shape == (2, 3)


def test_a_compile_failure_maps_to_compile_error(fake_toolchain):
    """The criterion CUDA_COMPILE_ERROR_IS_DISTINGUISHED_FROM_LAUNCH_ERROR.

    A port that never compiled says nothing about numerics; one that launched and crashed
    may. Both are INCONCLUSIVE in every arm, but a translation corpus needs to tell them
    apart -- and the compiler's log is the repair hint a translating agent gets.
    """
    failure = CudaCompilationError("NVRTC rejected the kernel source", log=NVRTC_LOG)
    result = _backend().run(_kernel(fake_toolchain(fail_compile=failure)), _case())
    assert result.status is Status.COMPILE_ERROR
    assert NVRTC_LOG in result.error


def test_a_same_named_exception_from_elsewhere_is_not_a_compile_error(fake_toolchain):
    # Classified by the toolchain's type, never by name or message.
    class CudaCompilationError(Exception):
        pass

    launcher = _raising(CudaCompilationError("CudaCompilationError: not the toolchain's"))
    result = _backend().run(_kernel(fake_toolchain(), launcher=launcher), _case())
    assert result.status is Status.LAUNCH_ERROR


class _Weird(Exception):
    """An exception type nothing in the backend knows about."""


#: Every way a CUDA kernel call can fail, and the one Status each must reach.
_FAILURES = [
    ("compile", CudaCompilationError("NVRTC rejected the kernel source"), Status.COMPILE_ERROR),
    ("launcher", InputMutatedError("kernel modified its input"), Status.LAUNCH_ERROR),
    ("launcher", RuntimeError("cuLaunchKernel failed: CUDA_ERROR_LAUNCH_FAILED"), Status.LAUNCH_ERROR),
    ("launcher", TypeError("the CUDA ports read float32; got float64"), Status.LAUNCH_ERROR),
    ("launcher", _Weird(), Status.LAUNCH_ERROR),
    ("output", None, Status.OUTPUT_ERROR),
]


@pytest.mark.parametrize(("where", "exc", "status"), _FAILURES)
def test_status_mapping_is_total(fake_toolchain, where, exc, status):
    """The criterion CUDA_STATUS_MAPPING_IS_TOTAL.

    Every exception a kernel can raise reaches exactly one Status, and none escapes: an
    escaping exception aborts a recording whose earlier rows were already paid for.
    """
    if where == "compile":
        kernel = _kernel(fake_toolchain(fail_compile=exc))
    elif where == "launcher":
        kernel = _kernel(fake_toolchain(), launcher=_raising(exc))
    else:
        kernel = _kernel(fake_toolchain(), launcher=lambda **kw: None)
    assert _backend().run(kernel, _case()).status is status


def test_an_input_mutation_reported_by_the_launcher_is_flagged(fake_toolchain):
    # The check lives in the launcher, which holds the device buffers. The flag is what
    # separates "wrote to its own input" from "crashed": both are LAUNCH_ERROR.
    launcher = _raising(InputMutatedError("kernel modified its input tensor(s) ['x'] on device"))
    result = _backend().run(_kernel(fake_toolchain(), launcher=launcher), _case())
    assert result.status is Status.LAUNCH_ERROR
    assert result.telemetry["input_mutated"] is True


def test_an_ordinary_crash_does_not_set_the_mutation_flag(fake_toolchain):
    result = _backend().run(
        _kernel(fake_toolchain(), launcher=_raising(RuntimeError())), _case()
    )
    assert result.telemetry["input_mutated"] is False


def test_a_cuda_artifact_fills_its_fields_and_leaves_triton_s_missing(fake_toolchain):
    """The criterion A_CUDA_ARTIFACT_FILLS_ITS_FIELDS_AND_NO_OTHERS.

    The driver reports registers, local and static shared memory, and the PTX is hashed.
    `n_spills`, `num_warps` and `num_stages` are Triton's quantities: local memory
    includes spills but is not a spill count, so filling `n_spills` from it would
    mislabel it. They stay MISSING, as Triton's launch keys and version do.
    """
    kernel = _kernel(
        fake_toolchain(), params={"eps": 1e-5}, shared_bytes=lambda block, params: 128
    )
    telemetry = _backend().run(kernel, _case()).telemetry

    assert telemetry[TELEMETRY_SCHEMA_VERSION] == SCHEMA_VERSION == 2
    assert telemetry["backend"] == "cuda"
    assert set(declared_keys()) <= set(telemetry)
    # The fake artifact's values are pairwise distinct, so each assertion below fails if
    # its field is read from any other attribute.
    assert telemetry["n_regs"] == 24
    assert telemetry["local_bytes"] == 16
    assert telemetry["static_shared_bytes"] == 48
    assert telemetry["max_threads_per_block"] == 1024
    assert telemetry["ptx_version"] == 80
    assert telemetry["binary_version"] == 86
    assert isinstance(telemetry["ptx_hash"], str)
    assert len(telemetry["ptx_hash"]) == 16
    assert telemetry["block"] == [32]
    assert telemetry["grid"] == [2]
    assert telemetry["dynamic_shared_bytes"] == 128
    assert telemetry["compile_options"] == ["--gpu-architecture=compute_86"]
    assert telemetry["kernel_params"] == {"eps": 1e-5}
    assert telemetry["nvrtc_version"] == "12.8"
    assert telemetry["cuda_bindings_version"] == "12.8.0"
    for triton_only in ("n_spills", "num_warps", "num_stages", "constexprs", "triton_version"):
        assert telemetry[triton_only] is MISSING, triton_only


def test_shared_bytes_is_what_one_block_is_given_static_plus_dynamic(fake_toolchain):
    """The quantity Triton's `metadata.shared` reports, so the two setups compare.

    The driver's SHARED_SIZE_BYTES is static shared memory only. A port that uses
    `extern __shared__` has 0 there while each block gets `width * 4` bytes, so recorded
    as `shared_bytes` it would compare a Triton row's allocation with a CUDA row's zero.
    """
    kernel = _kernel(fake_toolchain(), shared_bytes=lambda block, params: block[0] * 4)
    telemetry = _backend().run(kernel, _case()).telemetry
    assert telemetry["shared_bytes"] == 48 + 32 * 4


def test_shared_bytes_is_missing_when_either_part_is(fake_toolchain):
    failure = CudaCompilationError("rejected")
    telemetry = _backend().run(_kernel(fake_toolchain(fail_compile=failure)), _case()).telemetry
    assert telemetry["static_shared_bytes"] is MISSING
    assert telemetry["shared_bytes"] is MISSING


def test_a_toolchain_failure_is_raised_not_booked_against_the_kernel(fake_toolchain):
    """The criterion A_TOOLCHAIN_FAILURE_IS_A_BAD_CALL.

    NVRTC failing for a reason that is not this kernel's source or options would fail
    every later kernel the same way; booked as LAUNCH_ERROR, it would count against
    kernels that were never compiled.
    """
    failure = CudaToolchainError("nvrtcCompileProgram failed: NVRTC_ERROR_INTERNAL_ERROR")
    with pytest.raises(CudaToolchainError, match="NVRTC_ERROR_INTERNAL_ERROR"):
        _backend().run(_kernel(fake_toolchain(fail_compile=failure)), _case())


def test_a_compile_failure_records_its_options_but_no_artifact(fake_toolchain):
    failure = CudaCompilationError("rejected")
    telemetry = _backend().run(_kernel(fake_toolchain(fail_compile=failure)), _case()).telemetry
    assert telemetry["compile_options"] == ["--gpu-architecture=compute_86"]
    assert telemetry["n_regs"] is MISSING
    assert telemetry["ptx_hash"] is MISSING


def test_telemetry_carries_the_kernel_source_hash(fake_toolchain):
    telemetry = _backend().run(_kernel(fake_toolchain()), _case()).telemetry
    assert len(telemetry["kernel_source_hash"]) == 16


def test_a_failing_device_probe_degrades_the_device_group_to_missing(fake_toolchain):
    # After a sticky CUDA error every later CUDA call raises, the probe's included; the
    # probe runs on the failure path, where an escape would abort the paid run.
    def sticky():
        msg = "CUDA error: an illegal memory access was encountered"
        raise RuntimeError(msg)

    launcher = _raising(RuntimeError("CUDA error: an illegal memory access was encountered"))
    result = _backend(device_probe=sticky).run(
        _kernel(fake_toolchain(), launcher=launcher), _case()
    )
    assert result.status is Status.LAUNCH_ERROR
    assert result.telemetry["device_name"] is MISSING


def test_failing_toolchain_versions_degrade_to_missing(fake_toolchain):
    def broken():
        msg = "nvrtc unloaded"
        raise RuntimeError(msg)

    result = _backend().run(_kernel(fake_toolchain(versions=broken)), _case())
    assert result.status is Status.OK
    assert result.telemetry["nvrtc_version"] is MISSING


def test_a_plain_callable_is_rejected_as_a_bad_call():
    with pytest.raises(TypeError, match="requires a CudaKernel adapter"):
        _backend().run(lambda **kw: np.zeros((2, 3), dtype=np.float32), _case())


def test_the_backend_is_named_for_the_report_layer():
    assert CudaBackend.name == "cuda"


def test_a_cuda_row_survives_the_table_write(fake_toolchain, tmp_path):
    # The write happens once, after every execution has been paid for, so the whole CUDA
    # telemetry payload -- lists, dicts, MISSING -- must encode and round-trip.
    kernel = _kernel(fake_toolchain(), params={"eps": 1e-5})
    result = _backend().run(kernel, _case())
    ExecutionTable(tmp_path / "run").write([result])
    (row,) = ExecutionTable(tmp_path / "run").read()
    assert row.telemetry["block"] == [32]
    assert row.telemetry["kernel_params"] == {"eps": 1e-5}
    assert row.telemetry["n_spills"] is None
