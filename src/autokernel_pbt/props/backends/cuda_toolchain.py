"""Compiling and launching CUDA C++ kernels: NVRTC and the driver API, behind a protocol.

A CUDA kernel reaches the device in three steps, and each has its own failure class:

1. **NVRTC compiles the source to PTX** for the device's virtual architecture. A kernel
   whose source does not compile (`NVRTC_ERROR_COMPILATION`), or whose own requested
   compile options NVRTC rejects (`NVRTC_ERROR_INVALID_OPTION` with options present), never
   compiled: that is `CudaCompilationError`. The harness's own option, the architecture, is
   proven by the self-test compile at construction, which passes no kernel options. Every
   other NVRTC failure is the toolchain's -- `CudaToolchainError` -- and the backend raises
   it rather than booking it against a kernel.
2. **The driver loads the PTX** and looks up the entry point. A port whose entry point is
   not `extern "C"` compiles cleanly and then has no function of that name: its name was
   C++-mangled. That is a real translation defect, and it never became runnable, so it is
   a compile error too, as is PTX the driver rejects. Any other failure here -- a context
   left dead by an earlier kernel's fault, out of memory -- is not this kernel's, and is
   a plain `RuntimeError`, booked as a launch error would be in that context.
3. **The driver launches it** on torch's current stream, against torch's device buffers.
   A failed launch, or an asynchronous fault surfaced by the synchronize after it, is a
   launch error: a plain `RuntimeError`, which the backend books as LAUNCH_ERROR.

The NVRTC option is `--gpu-architecture=`, the spelling NVRTC's documentation lists (with
`-arch`); nvcc's `--gpu-arch` is not an NVRTC option.

`NvrtcToolchain` is the only code in this project that imports NVIDIA's `cuda-python`.
Everything above it -- the adapter, the backend, the ports' launch contract -- is tested
on CPU against a fake toolchain, and this module's own logic against fake bindings. The
bindings are *probed*, not asserted: they moved from `cuda.{cuda,nvrtc}` to
`cuda.bindings.{driver,nvrtc}` in the 12.x series, and a moved module must not abort a
paid run. Function attributes are probed one by one for the same reason, and an
unavailable one is MISSING.
"""

from __future__ import annotations

import ctypes
import importlib
import types
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from autokernel_pbt.props.backends.telemetry import MISSING

#: Where `cuda-python` keeps the driver and NVRTC modules, newest layout first.
_BINDING_LOCATIONS = (
    ("cuda.bindings.driver", "cuda.bindings.nvrtc"),
    ("cuda.cuda", "cuda.nvrtc"),
)

#: The program name NVRTC reports in its log. Fixed rather than derived from the entry
#: point, so that two compiles of one source produce byte-identical logs.
_PROGRAM_NAME = b"kernel.cu"

#: Compiled once at construction. If it does not compile, load and resolve, nothing will,
#: and the reason is the toolchain's -- so it is raised as a bad call, not recorded.
_SELF_TEST_SOURCE = 'extern "C" __global__ void autokernel_self_test(float* y) { y[0] = 1.0f; }'
_SELF_TEST_ENTRY = "autokernel_self_test"

#: `cuModuleLoadData` results that reject the image itself: the PTX does not assemble for
#: this device (the driver runs ptxas on it), or the driver cannot use what it produced.
#: Probed by name, like the attributes; a member a release lacks never matches.
_IMAGE_REJECTED = (
    "CUDA_ERROR_INVALID_PTX",
    "CUDA_ERROR_INVALID_IMAGE",
    "CUDA_ERROR_NO_BINARY_FOR_GPU",
    "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
)

#: telemetry field -> CUfunction_attribute member name. `SHARED_SIZE_BYTES` is *static*
#: shared memory only; the dynamic shared memory a launch is given is not an attribute of
#: the function, so the backend adds it to make `shared_bytes` (see `cuda_backend`).
_ATTRIBUTES = {
    "n_regs": "CU_FUNC_ATTRIBUTE_NUM_REGS",
    "static_shared_bytes": "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",
    "local_bytes": "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",
    "max_threads_per_block": "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
    "ptx_version": "CU_FUNC_ATTRIBUTE_PTX_VERSION",
    "binary_version": "CU_FUNC_ATTRIBUTE_BINARY_VERSION",
}


class CudaToolchainError(RuntimeError):
    """The toolchain failed, not the kernel: a bad *call*, which the backend raises.

    An NVRTC failure other than a rejected source or rejected kernel options (an internal
    error, out of memory, a program that cannot be created), or a driver call outside a
    launch. Booked as the kernel's LAUNCH_ERROR it would count against a kernel that was
    never compiled -- and every later kernel would fail the same way.
    """


class CudaCompilationError(Exception):
    """The kernel never became runnable: NVRTC rejected it, or the driver could not load
    it or find its entry point.

    Its own type, so the backend classifies by type rather than by message -- the same
    discipline as Triton's compile errors in feature 0007. `log` is the compiler's own
    output, kept whole: it is the most useful repair hint a translating agent can get.
    """

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(f"{message}\n{log}" if log else message)
        self.log = log


@dataclass(frozen=True)
class CudaArtifact:
    """What compiling one entry point produced.

    The attribute names are the ones `telemetry.extract` probes first, so the artifact is
    read by the same code that reads a Triton CompiledKernel. `asm` is shaped like
    Triton's (`{"ptx": text}`) for the same reason: the PTX is hashed, never stored.
    """

    entry: str
    options: tuple[str, ...]
    asm: dict[str, str]
    n_regs: Any = MISSING
    static_shared_bytes: Any = MISSING
    local_bytes: Any = MISSING
    max_threads_per_block: Any = MISSING
    ptx_version: Any = MISSING
    binary_version: Any = MISSING
    #: The driver's function handle. Opaque to everything but the toolchain that made it.
    function: Any = field(default=None, repr=False, compare=False)


class CudaToolchain(Protocol):
    """Compiles CUDA C++ and launches the result. `NvrtcToolchain` is the real one."""

    def resolved_options(self, options: tuple[str, ...]) -> tuple[str, ...]:
        """The options `compile` will actually pass for these requested ones."""
        ...

    def compile(self, source: str, entry: str, options: tuple[str, ...]) -> CudaArtifact: ...

    def launch(
        self,
        artifact: CudaArtifact,
        *,
        grid: tuple[int, ...],
        block: tuple[int, ...],
        shared_bytes: int,
        args: tuple[Any, ...],
    ) -> None: ...

    def versions(self) -> dict[str, Any]:
        """`nvrtc_version` and `cuda_bindings_version`, each a value or MISSING."""
        ...


def marshal(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Kernel arguments as the driver takes them: values, and the C type of each.

    Only three kinds are accepted, each by *explicit* type: a device buffer (anything with
    `data_ptr()`, i.e. a torch tensor) as a pointer, `np.int32` as `int`, `np.float32` as
    `float`. A bare Python `int` or `float` is refused rather than guessed: a CUDA `int`
    parameter handed an 8-byte value, or a `float` handed a double, reads the wrong bytes,
    computes a plausible wrong answer, and raises nothing. Refusing it is a bad call,
    raised before the launch.
    """
    values: list[Any] = []
    kinds: list[Any] = []
    for position, arg in enumerate(args):
        if isinstance(arg, np.int32):
            values.append(int(arg))
            kinds.append(ctypes.c_int)
        elif isinstance(arg, np.float32):
            values.append(float(arg))
            kinds.append(ctypes.c_float)
        elif callable(getattr(arg, "data_ptr", None)):
            values.append(int(arg.data_ptr()))
            kinds.append(ctypes.c_void_p)
        else:
            msg = (
                f"kernel argument {position} is {type(arg).__name__}; pass a device buffer, "
                f"np.int32 or np.float32 -- a bare Python number has no C width, and the "
                f"wrong width reads the wrong bytes without raising"
            )
            raise TypeError(msg)
    return tuple(values), tuple(kinds)


def load_bindings() -> types.SimpleNamespace:
    """`cuda-python`'s driver and NVRTC modules, from wherever this release keeps them."""
    failures: list[str] = []
    for driver_name, nvrtc_name in _BINDING_LOCATIONS:
        try:
            driver = importlib.import_module(driver_name)
            nvrtc = importlib.import_module(nvrtc_name)
        except Exception as exc:  # noqa: BLE001 - probed: absent or broken means "not here"
            failures.append(f"{driver_name}: {exc!r}")
            continue
        return types.SimpleNamespace(driver=driver, nvrtc=nvrtc)
    msg = "cuda-python is not importable (pip install cuda-python); tried " + "; ".join(failures)
    raise ImportError(msg)


def _name(code: Any) -> str:
    return str(getattr(code, "name", code))


class NvrtcToolchain:
    """NVRTC to PTX, the driver to load and launch. Compiled artifacts are cached.

    `bindings` is injected so this class is exercisable on CPU; by default it is
    `load_bindings()`. `device_index` defaults to torch's current device when torch is
    importable, else 0. `arch` defaults to the device's own compute capability, as a
    *virtual* architecture (`compute_86`): NVRTC emits PTX, which is what is hashed into
    the telemetry, and the driver compiles it for the exact device at load time.
    """

    def __init__(
        self,
        bindings: Any = None,
        device_index: int | None = None,
        arch: str | None = None,
    ) -> None:
        self._b = bindings or load_bindings()
        self._driver = self._b.driver
        self._nvrtc = self._b.nvrtc
        self._device = self._ensure_context(self._device_index(device_index))
        self.arch = arch or self._virtual_arch(self._device)
        self._cache: dict[tuple[str, str, tuple[str, ...]], CudaArtifact] = {}
        try:
            self.compile(_SELF_TEST_SOURCE, _SELF_TEST_ENTRY, ())
        except Exception as exc:
            msg = (
                f"the NVRTC toolchain cannot compile, load and resolve a trivial kernel for "
                f"{self.arch}; nothing would run, and the fault is the toolchain's, not any "
                f"kernel's: {exc}"
            )
            raise CudaToolchainError(msg) from exc

    # -- the protocol ------------------------------------------------------------------

    def resolved_options(self, options: tuple[str, ...]) -> tuple[str, ...]:
        return (*options, f"--gpu-architecture={self.arch}")

    def compile(self, source: str, entry: str, options: tuple[str, ...]) -> CudaArtifact:
        key = (source, entry, tuple(options))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        resolved = self.resolved_options(tuple(options))
        ptx = self._ptx(source, resolved, requested=tuple(options))
        function = self._load(ptx, entry)
        artifact = CudaArtifact(
            entry=entry,
            options=resolved,
            asm={"ptx": ptx.rstrip(b"\x00").decode("utf-8", errors="replace")},
            function=function,
            **{name: self._attribute(function, member) for name, member in _ATTRIBUTES.items()},
        )
        self._cache[key] = artifact
        return artifact

    def launch(
        self,
        artifact: CudaArtifact,
        *,
        grid: tuple[int, ...],
        block: tuple[int, ...],
        shared_bytes: int,
        args: tuple[Any, ...],
    ) -> None:
        values, kinds = marshal(args)
        gx, gy, gz = _dims(grid, "grid")
        bx, by, bz = _dims(block, "block")
        stream = self._current_stream()
        (err,) = self._driver.cuLaunchKernel(
            artifact.function,
            gx,
            gy,
            gz,
            bx,
            by,
            bz,
            int(shared_bytes),
            stream,
            (values, kinds),
            0,
        )
        self._check_launch(err, f"cuLaunchKernel({artifact.entry})")
        # Synchronized here, not left to the device-to-host copy: an asynchronous fault
        # (an illegal address) must be attributed to the launch that caused it, and
        # raised while the backend is still classifying this row.
        (err,) = self._driver.cuStreamSynchronize(stream)
        self._check_launch(err, f"cuStreamSynchronize after {artifact.entry}")

    def versions(self) -> dict[str, Any]:
        return {
            "nvrtc_version": self._nvrtc_version(),
            "cuda_bindings_version": _distribution_version(),
        }

    # -- compile, load, read ----------------------------------------------------------

    def _ptx(self, source: str, options: tuple[str, ...], requested: tuple[str, ...]) -> bytes:
        nvrtc = self._nvrtc
        err, program = nvrtc.nvrtcCreateProgram(source.encode("utf-8"), _PROGRAM_NAME, 0, [], [])
        self._check_nvrtc(err, "nvrtcCreateProgram")
        try:
            encoded = [option.encode("utf-8") for option in options]
            (err,) = nvrtc.nvrtcCompileProgram(program, len(encoded), encoded)
            if err == nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION:
                msg = "NVRTC rejected the kernel source"
                raise CudaCompilationError(msg, self._log(program))
            if err == nvrtc.nvrtcResult.NVRTC_ERROR_INVALID_OPTION and requested:
                # The architecture option was proven by the self-test, which requests none,
                # so a rejection here is of the options this kernel asked for: part of the
                # kernel, which never compiled.
                msg = f"NVRTC rejected the kernel's compile options {list(requested)}"
                raise CudaCompilationError(msg, self._log(program))
            # Anything else is not the kernel's fault, and must not be booked as if it were.
            self._check_nvrtc(err, "nvrtcCompileProgram", log=self._log(program))
            err, size = nvrtc.nvrtcGetPTXSize(program)
            self._check_nvrtc(err, "nvrtcGetPTXSize")
            buffer = _out_buffer(size)
            (err,) = nvrtc.nvrtcGetPTX(program, buffer)
            self._check_nvrtc(err, "nvrtcGetPTX")
            return buffer
        finally:
            nvrtc.nvrtcDestroyProgram(program)

    def _log(self, program: Any) -> str:
        nvrtc = self._nvrtc
        try:
            err, size = nvrtc.nvrtcGetProgramLogSize(program)
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                return ""
            buffer = _out_buffer(size)
            nvrtc.nvrtcGetProgramLog(program, buffer)
        except Exception:  # noqa: BLE001 - a log is a courtesy; its absence is not a fault
            return ""
        return buffer.rstrip(b"\x00").decode("utf-8", errors="replace").strip()

    def _load(self, ptx: bytes, entry: str) -> Any:
        """Load the PTX and resolve the entry point; a compile error only if the KERNEL is why.

        The driver reports more than the image through these calls. After a kernel faults
        (an illegal address), the context is dead and every later call returns that
        fault, this one included; a load can also run out of memory. Booked as
        COMPILE_ERROR, either would charge a fault class to a kernel that never had it,
        and differ from the Triton row recorded beside it. So only a rejected image, or a
        missing entry point, is the kernel's; any other code is a plain RuntimeError,
        which the backend books as LAUNCH_ERROR -- as a launch in that context would be.
        """
        driver = self._driver
        err, module = driver.cuModuleLoadData(ptx)
        if err != driver.CUresult.CUDA_SUCCESS:
            if self._is(err, _IMAGE_REJECTED):
                msg = f"the driver could not load the PTX: {_name(err)}"
                raise CudaCompilationError(msg)
            msg = f"cuModuleLoadData failed: {_name(err)}, which does not reject the image"
            raise RuntimeError(msg)
        err, function = driver.cuModuleGetFunction(module, entry.encode("utf-8"))
        if err != driver.CUresult.CUDA_SUCCESS:
            self._unload(module)
            if self._is(err, ("CUDA_ERROR_NOT_FOUND",)):
                msg = (
                    f"no entry point {entry!r} in the compiled module ({_name(err)}); a "
                    f'kernel declared without extern "C" compiles, and its name is '
                    f"C++-mangled"
                )
                raise CudaCompilationError(msg)
            msg = f"cuModuleGetFunction({entry!r}) failed: {_name(err)}"
            raise RuntimeError(msg)
        return function

    def _is(self, err: Any, members: tuple[str, ...]) -> bool:
        """Whether `err` is one of the named CUresult members this binding release has."""
        result = self._driver.CUresult
        return any(hasattr(result, name) and err == getattr(result, name) for name in members)

    def _unload(self, module: Any) -> None:
        """Release a module that will never be used. In a dead context this fails too."""
        try:
            self._driver.cuModuleUnload(module)
        except Exception:  # noqa: BLE001, S110 - cleanup must not mask the classification
            pass

    def _attribute(self, function: Any, member: str) -> Any:
        driver = self._driver
        try:
            attribute = getattr(driver.CUfunction_attribute, member)
            err, value = driver.cuFuncGetAttribute(attribute, function)
        except Exception:  # noqa: BLE001 - probed, not asserted
            return MISSING
        if err != driver.CUresult.CUDA_SUCCESS or isinstance(value, bool):
            return MISSING
        try:
            return int(value)
        except (TypeError, ValueError):
            return MISSING

    # -- device and context ------------------------------------------------------------

    @staticmethod
    def _device_index(device_index: int | None) -> int:
        if device_index is not None:
            return device_index
        try:
            import torch

            return int(torch.cuda.current_device())
        except Exception:  # noqa: BLE001 - torch absent or not initialized: device 0
            return 0

    def _ensure_context(self, index: int) -> Any:
        """Make the device's *primary* context current: the one torch's runtime uses.

        Kernels launched here read and write torch's buffers, so they must run in the
        context those buffers were allocated in. The runtime API torch uses is defined to
        use the primary context, so retaining it here shares it, whichever side of the
        process initializes CUDA first.
        """
        driver = self._driver
        (err,) = driver.cuInit(0)
        self._check(err, "cuInit")
        err, device = driver.cuDeviceGet(index)
        self._check(err, f"cuDeviceGet({index})")
        err, context = driver.cuDevicePrimaryCtxRetain(device)
        self._check(err, "cuDevicePrimaryCtxRetain")
        (err,) = driver.cuCtxSetCurrent(context)
        self._check(err, "cuCtxSetCurrent")
        return device

    def _virtual_arch(self, device: Any) -> str:
        driver = self._driver
        attribute = driver.CUdevice_attribute
        err, major = driver.cuDeviceGetAttribute(
            attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
        )
        self._check(err, "cuDeviceGetAttribute(major)")
        err, minor = driver.cuDeviceGetAttribute(
            attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
        )
        self._check(err, "cuDeviceGetAttribute(minor)")
        return f"compute_{int(major)}{int(minor)}"

    def _current_stream(self) -> Any:
        """torch's current stream, so the launch is ordered with torch's own copies."""
        handle = 0
        try:
            import torch

            handle = int(torch.cuda.current_stream().cuda_stream)
        except Exception:  # noqa: BLE001 - no torch: the legacy default stream, 0
            handle = 0
        return self._driver.CUstream(handle)

    def _nvrtc_version(self) -> Any:
        try:
            err, major, minor = self._nvrtc.nvrtcVersion()
        except Exception:  # noqa: BLE001 - probed, not asserted
            return MISSING
        if err != self._nvrtc.nvrtcResult.NVRTC_SUCCESS:
            return MISSING
        return f"{int(major)}.{int(minor)}"

    # -- result codes ------------------------------------------------------------------

    def _check(self, err: Any, what: str) -> None:
        """A driver call outside a launch: its failure is the toolchain's."""
        if err != self._driver.CUresult.CUDA_SUCCESS:
            msg = f"{what} failed: {_name(err)}"
            raise CudaToolchainError(msg)

    def _check_launch(self, err: Any, what: str) -> None:
        """A launch or the synchronize after it: the kernel ran and failed, which is data."""
        if err != self._driver.CUresult.CUDA_SUCCESS:
            msg = f"{what} failed: {_name(err)}"
            raise RuntimeError(msg)

    def _check_nvrtc(self, err: Any, what: str, log: str = "") -> None:
        if err != self._nvrtc.nvrtcResult.NVRTC_SUCCESS:
            msg = f"{what} failed: {_name(err)}"
            raise CudaToolchainError(f"{msg}\n{log}" if log else msg)


def _out_buffer(size: int) -> bytes:
    """A fresh, zero-filled buffer for NVRTC to write `size` bytes into.

    The bindings take a `bytes` object and write through its internal pointer -- the
    documented pattern is `b" " * size`. For `size == 1`, which is what an empty log
    reports (just the terminator), CPython hands back its cached single-byte object, so
    NVRTC would write a NUL into the interpreter's shared `b" "`, everywhere at once. At
    least two bytes are always allocated, which CPython never caches, and zero-filled, so
    the unused tail strips cleanly.
    """
    return bytes(max(int(size), 2))


def _dims(dims: tuple[int, ...], what: str) -> tuple[int, int, int]:
    """Pad a 1-, 2- or 3-D launch geometry to the driver's three dimensions."""
    if not 1 <= len(dims) <= 3:
        msg = f"{what} must have 1 to 3 dimensions, got {dims!r}"
        raise ValueError(msg)
    padded = [int(d) for d in dims] + [1] * (3 - len(dims))
    return padded[0], padded[1], padded[2]


def _distribution_version() -> Any:
    """The installed `cuda-python` distribution's version, probed by both its names."""
    from importlib import metadata

    for name in ("cuda-bindings", "cuda-python"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return MISSING
