"""NvrtcToolchain's own logic, on CPU, against fake `cuda-python` bindings.

The fakes follow the bindings' conventions: every call returns a tuple whose first element
is a result enum, and output buffers are `bytes` objects the library writes into. What
they cannot show is that the real bindings behave this way on a given release -- that is
what the device tests in tests/gpu/ are for.
"""

from __future__ import annotations

import ctypes
import enum
import sys
import types

import numpy as np
import pytest

from autokernel_pbt.props.backends.cuda_toolchain import (
    CudaCompilationError,
    CudaToolchainError,
    NvrtcToolchain,
    load_bindings,
    marshal,
)
from autokernel_pbt.props.backends.telemetry import MISSING

#: Sources the fake NVRTC treats specially.
REJECTED = "this source does not compile"
BAD_OPTION = "--not-an-nvrtc-option"
MANGLED = "mangled_entry"
UNLOADABLE = "this ptx does not load"
INTERNAL = "this source trips an internal NVRTC error"
COMPILER_LOG = 'kernel.cu(1): error: expected a ";"'


class CUresult(enum.Enum):
    CUDA_SUCCESS = 0
    CUDA_ERROR_INVALID_PTX = 218
    CUDA_ERROR_NOT_FOUND = 500
    CUDA_ERROR_LAUNCH_FAILED = 719
    CUDA_ERROR_ILLEGAL_ADDRESS = 700


class nvrtcResult(enum.Enum):
    NVRTC_SUCCESS = 0
    NVRTC_ERROR_INVALID_OPTION = 5
    NVRTC_ERROR_COMPILATION = 6
    NVRTC_ERROR_INTERNAL_ERROR = 11


class CUfunction_attribute(enum.Enum):
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6


class CUdevice_attribute(enum.Enum):
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


def _write(buffer: bytes, data: bytes) -> None:
    """Write into a bytes object through its pointer, as the C library does."""
    assert len(buffer) >= 2, "a one-byte buffer is CPython's shared cached object"
    ctypes.memmove(ctypes.c_char_p(buffer), data, min(len(data), len(buffer)))


class FakeNvrtc:
    nvrtcResult = nvrtcResult

    def __init__(self) -> None:
        self.buffers: list[bytes] = []
        self.destroyed = 0
        self.fail_everything = False

    def nvrtcCreateProgram(self, src, name, n_headers, headers, include_names):
        return nvrtcResult.NVRTC_SUCCESS, {"src": src.decode(), "log": b""}

    def nvrtcCompileProgram(self, program, n_options, options):
        decoded = [option.decode() for option in options]
        assert n_options == len(options)
        if self.fail_everything or BAD_OPTION in decoded:
            program["log"] = b"nvrtc: error: unrecognized option\x00"
            return (nvrtcResult.NVRTC_ERROR_INVALID_OPTION,)
        if INTERNAL in program["src"]:
            return (nvrtcResult.NVRTC_ERROR_INTERNAL_ERROR,)
        if REJECTED in program["src"]:
            program["log"] = COMPILER_LOG.encode() + b"\x00"
            return (nvrtcResult.NVRTC_ERROR_COMPILATION,)
        program["ptx"] = f"// ptx for {decoded}\n{program['src']}".encode() + b"\x00"
        return (nvrtcResult.NVRTC_SUCCESS,)

    def nvrtcGetProgramLogSize(self, program):
        return nvrtcResult.NVRTC_SUCCESS, max(len(program["log"]), 1)

    def nvrtcGetProgramLog(self, program, buffer):
        self.buffers.append(buffer)
        _write(buffer, program["log"] or b"\x00")
        return (nvrtcResult.NVRTC_SUCCESS,)

    def nvrtcGetPTXSize(self, program):
        return nvrtcResult.NVRTC_SUCCESS, len(program["ptx"])

    def nvrtcGetPTX(self, program, buffer):
        self.buffers.append(buffer)
        _write(buffer, program["ptx"])
        return (nvrtcResult.NVRTC_SUCCESS,)

    def nvrtcDestroyProgram(self, program):
        self.destroyed += 1
        return (nvrtcResult.NVRTC_SUCCESS,)

    def nvrtcVersion(self):
        return nvrtcResult.NVRTC_SUCCESS, 12, 8


class FakeDriver:
    CUresult = CUresult
    CUfunction_attribute = CUfunction_attribute
    CUdevice_attribute = CUdevice_attribute

    def __init__(self) -> None:
        self.launches: list[tuple] = []
        self.launch_result = CUresult.CUDA_SUCCESS
        self.sync_result = CUresult.CUDA_SUCCESS
        self.loads = 0
        self.unavailable: set[str] = set()

    def CUstream(self, handle):
        return ("stream", handle)

    def cuInit(self, flags):
        return (CUresult.CUDA_SUCCESS,)

    def cuDeviceGet(self, index):
        return CUresult.CUDA_SUCCESS, ("device", index)

    def cuDevicePrimaryCtxRetain(self, device):
        return CUresult.CUDA_SUCCESS, ("context", device)

    def cuCtxSetCurrent(self, context):
        return (CUresult.CUDA_SUCCESS,)

    def cuDeviceGetAttribute(self, attribute, device):
        major = attribute is CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        return CUresult.CUDA_SUCCESS, 8 if major else 6

    def cuModuleLoadData(self, ptx):
        self.loads += 1
        if UNLOADABLE.encode() in ptx:
            return CUresult.CUDA_ERROR_INVALID_PTX, None
        return CUresult.CUDA_SUCCESS, {"ptx": ptx}

    def cuModuleGetFunction(self, module, name):
        if name.decode() == MANGLED:
            return CUresult.CUDA_ERROR_NOT_FOUND, None
        return CUresult.CUDA_SUCCESS, ("function", name.decode())

    def cuFuncGetAttribute(self, attribute, function):
        if attribute.name in self.unavailable:
            return CUresult.CUDA_ERROR_NOT_FOUND, 0
        # Pairwise distinct, so an attribute read into the wrong field cannot pass.
        values = {"NUM_REGS": 24, "SHARED_SIZE_BYTES": 48, "LOCAL_SIZE_BYTES": 16,
                  "MAX_THREADS_PER_BLOCK": 1024, "PTX_VERSION": 80, "BINARY_VERSION": 86}
        return CUresult.CUDA_SUCCESS, values[attribute.name.removeprefix("CU_FUNC_ATTRIBUTE_")]

    def cuLaunchKernel(self, function, gx, gy, gz, bx, by, bz, shared, stream, params, extra):
        self.launches.append((function, (gx, gy, gz), (bx, by, bz), shared, stream, params))
        return (self.launch_result,)

    def cuStreamSynchronize(self, stream):
        return (self.sync_result,)


def _bindings():
    return types.SimpleNamespace(driver=FakeDriver(), nvrtc=FakeNvrtc())


def _toolchain(bindings=None) -> NvrtcToolchain:
    return NvrtcToolchain(bindings=bindings or _bindings(), device_index=0)


SOURCE = 'extern "C" __global__ void k(float* y) { y[0] = 2.0f; }'


def test_a_toolchain_that_cannot_compile_its_self_test_raises_at_construction():
    """The criterion THE_TOOLCHAIN_REFUSES_TO_START_BROKEN.

    If a trivial kernel does not compile, load and resolve, nothing will, and the fault is
    the toolchain's. Raised at construction it stops the run before any row exists;
    discovered per kernel, it would be booked as one LAUNCH_ERROR per row.
    """
    bindings = _bindings()
    bindings.nvrtc.fail_everything = True
    with pytest.raises(RuntimeError, match="cannot compile, load and resolve a trivial kernel"):
        NvrtcToolchain(bindings=bindings, device_index=0)


def test_only_a_compilation_failure_is_a_compile_error():
    """The criterion NVRTC_COMPILE_FAILURE_IS_TYPED.

    A source NVRTC rejects, or compile options the kernel itself asked for that NVRTC
    rejects, is that kernel never compiling: COMPILE_ERROR, with the log. Any other NVRTC
    failure is the toolchain's, `CudaToolchainError`, never booked against a kernel.
    """
    toolchain = _toolchain()
    with pytest.raises(CudaCompilationError) as rejected:
        toolchain.compile(f"// {REJECTED}\n{SOURCE}", "k", ())
    assert rejected.value.log == COMPILER_LOG

    with pytest.raises(CudaCompilationError, match="rejected the kernel's compile options"):
        toolchain.compile(SOURCE, "k", (BAD_OPTION,))

    with pytest.raises(CudaToolchainError, match="NVRTC_ERROR_INTERNAL_ERROR") as internal:
        toolchain.compile(f"// {INTERNAL}\n{SOURCE}", "k", ())
    assert not isinstance(internal.value, CudaCompilationError)


def test_an_option_rejection_with_no_kernel_options_is_the_toolchain_s():
    # Only the harness's architecture option was passed, so the rejection is of that --
    # which the self-test proved once, and which is no kernel's fault if it now fails.
    bindings = _bindings()
    toolchain = _toolchain(bindings)
    bindings.nvrtc.fail_everything = True
    with pytest.raises(CudaToolchainError, match="NVRTC_ERROR_INVALID_OPTION"):
        toolchain.compile(SOURCE, "fresh", ())


def test_an_argument_of_an_unmarshallable_type_is_refused():
    """The criterion KERNEL_ARGUMENTS_ARE_MARSHALLED_BY_EXPLICIT_TYPE.

    A CUDA `int` handed an 8-byte value, or a `float` handed a double, reads the wrong
    bytes and computes a plausible wrong answer with no error. Only explicitly typed
    values pass.
    """

    class DeviceBuffer:
        def data_ptr(self) -> int:
            return 0xBEEF

    values, kinds = marshal((DeviceBuffer(), np.int32(7), np.float32(1e-5)))
    assert values == (0xBEEF, 7, pytest.approx(1e-5))
    assert kinds == (ctypes.c_void_p, ctypes.c_int, ctypes.c_float)

    for bad in (7, 1e-5, np.int64(7), np.float64(1e-5), True, np.ones(3, np.float32)):
        with pytest.raises(TypeError, match="has no C width"):
            marshal((DeviceBuffer(), bad))


def test_an_entry_point_that_is_not_extern_c_is_a_compile_error():
    # A C++ kernel's name is mangled; the module compiles and has no function by that
    # name. It never became runnable, which is the compile-error class.
    with pytest.raises(CudaCompilationError, match="extern \"C\""):
        _toolchain().compile(SOURCE, MANGLED, ())


def test_ptx_the_driver_cannot_load_is_a_compile_error():
    with pytest.raises(CudaCompilationError, match="CUDA_ERROR_INVALID_PTX"):
        _toolchain().compile(f"// {UNLOADABLE}\n{SOURCE}", "k", ())


def test_the_artifact_carries_the_attributes_the_ptx_and_the_resolved_options():
    artifact = _toolchain().compile(SOURCE, "k", ("-lineinfo",))
    assert artifact.n_regs == 24
    assert artifact.local_bytes == 16
    assert artifact.static_shared_bytes == 48
    assert artifact.max_threads_per_block == 1024
    assert artifact.ptx_version == 80
    assert artifact.binary_version == 86
    assert artifact.options == ("-lineinfo", "--gpu-architecture=compute_86")
    assert SOURCE in artifact.asm["ptx"]
    assert "\x00" not in artifact.asm["ptx"]


def test_an_attribute_the_driver_will_not_report_is_missing():
    bindings = _bindings()
    bindings.driver.unavailable = {"CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"}
    artifact = _toolchain(bindings).compile(SOURCE, "k", ())
    assert artifact.local_bytes is MISSING
    assert artifact.n_regs == 24


def test_the_arch_is_the_device_s_own_virtual_architecture():
    # Virtual (compute_86), so NVRTC emits PTX -- what is hashed -- and the driver
    # compiles it for the exact device at load time. `--gpu-architecture=` is the spelling
    # NVRTC documents; nvcc's `--gpu-arch` is not an NVRTC option.
    assert _toolchain().arch == "compute_86"
    resolved = _toolchain().resolved_options(("-lineinfo",))
    assert resolved == ("-lineinfo", "--gpu-architecture=compute_86")


def test_the_self_test_passes_only_the_architecture_option():
    bindings = _bindings()
    seen = []
    compile_program = bindings.nvrtc.nvrtcCompileProgram

    def recording(program, n_options, options):
        seen.append([option.decode() for option in options])
        return compile_program(program, n_options, options)

    bindings.nvrtc.nvrtcCompileProgram = recording
    _toolchain(bindings)
    assert seen == [["--gpu-architecture=compute_86"]]


def test_artifacts_are_cached_per_source_entry_and_options():
    bindings = _bindings()
    toolchain = _toolchain(bindings)
    loads = bindings.driver.loads
    first = toolchain.compile(SOURCE, "k", ())
    assert toolchain.compile(SOURCE, "k", ()) is first
    assert bindings.driver.loads == loads + 1
    assert toolchain.compile(SOURCE, "k", ("-lineinfo",)) is not first


def test_every_program_is_destroyed_including_a_rejected_one():
    bindings = _bindings()
    toolchain = _toolchain(bindings)
    before = bindings.nvrtc.destroyed
    with pytest.raises(CudaCompilationError):
        toolchain.compile(f"// {REJECTED}\n{SOURCE}", "k", ())
    toolchain.compile(SOURCE, "k", ())
    assert bindings.nvrtc.destroyed == before + 2


def test_no_output_buffer_is_cpython_s_shared_single_byte_object():
    # An empty log reports size 1. `b" " * 1` is CPython's cached object, and the library
    # writes through its pointer -- a NUL into every `b" "` in the process.
    bindings = _bindings()
    toolchain = _toolchain(bindings)
    with pytest.raises(CudaCompilationError):
        toolchain.compile(f"// {REJECTED}\n{SOURCE}", "k", ())
    assert bindings.nvrtc.buffers
    assert all(len(buffer) >= 2 for buffer in bindings.nvrtc.buffers)
    assert b" " == bytes([32])


def test_a_launch_passes_the_geometry_and_marshalled_arguments():
    bindings = _bindings()
    toolchain = _toolchain(bindings)
    artifact = toolchain.compile(SOURCE, "k", ())
    toolchain.launch(artifact, grid=(4,), block=(256,), shared_bytes=1024, args=(np.int32(3),))
    (launch,) = bindings.driver.launches
    function, grid, block, shared, _stream, params = launch
    assert function == ("function", "k")
    assert grid == (4, 1, 1)
    assert block == (256, 1, 1)
    assert shared == 1024
    assert params == ((3,), (ctypes.c_int,))


@pytest.mark.parametrize("where", ["launch", "synchronize"])
def test_a_failed_launch_or_an_asynchronous_fault_is_a_runtime_error(where):
    # Both are launch errors: the kernel compiled, and running it failed. The synchronize
    # is what attributes an asynchronous fault to the launch that caused it.
    bindings = _bindings()
    toolchain = _toolchain(bindings)
    artifact = toolchain.compile(SOURCE, "k", ())
    if where == "launch":
        bindings.driver.launch_result = CUresult.CUDA_ERROR_LAUNCH_FAILED
    else:
        bindings.driver.sync_result = CUresult.CUDA_ERROR_ILLEGAL_ADDRESS
    with pytest.raises(RuntimeError, match="CUDA_ERROR_") as raised:
        toolchain.launch(artifact, grid=(1,), block=(32,), shared_bytes=0, args=())
    assert not isinstance(raised.value, CudaCompilationError)


def test_versions_are_reported():
    assert _toolchain().versions()["nvrtc_version"] == "12.8"


def test_the_newer_binding_layout_is_preferred(monkeypatch):
    newer = types.SimpleNamespace(name="cuda.bindings")
    older = types.SimpleNamespace(name="cuda.cuda")
    for name, module in {
        "cuda.bindings.driver": newer,
        "cuda.bindings.nvrtc": newer,
        "cuda.cuda": older,
        "cuda.nvrtc": older,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    assert load_bindings().driver is newer


def test_the_older_binding_layout_is_found_when_the_newer_is_absent(monkeypatch):
    older = types.SimpleNamespace(name="cuda.cuda")
    monkeypatch.setitem(sys.modules, "cuda.bindings.driver", None)
    monkeypatch.setitem(sys.modules, "cuda.bindings.nvrtc", None)
    monkeypatch.setitem(sys.modules, "cuda.cuda", older)
    monkeypatch.setitem(sys.modules, "cuda.nvrtc", older)
    assert load_bindings().driver is older


def test_missing_bindings_name_the_package_to_install(monkeypatch):
    for name in ("cuda.bindings.driver", "cuda.bindings.nvrtc", "cuda.cuda", "cuda.nvrtc"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match="pip install cuda-python"):
        load_bindings()
