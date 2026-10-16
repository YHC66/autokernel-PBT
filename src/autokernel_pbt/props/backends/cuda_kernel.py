"""The callable adapter that lets a CUDA C++ kernel satisfy the `Backend` protocol.

The same move as `TritonKernel`, for the same reason: `Backend.run` calls its kernel as
`kernel(**numpy_inputs)`, and widening that protocol would make the backends stop being
substitutable -- which is what makes a cross-setup comparison possible at all. A CUDA
kernel is a source string, an entry point, a launch geometry and a launcher; the adapter
holds all four and is callable exactly as the protocol expects.

What differs from Triton is when compilation happens. Triton compiles lazily inside the
launch and hands the artifact back through a callback. Here the adapter compiles
*before* it launches, through the injected toolchain, so the artifact is the adapter's own
record from the start and a compile failure surfaces as `CudaCompilationError` before any
device buffer exists.

The toolchain and the launcher are injected. On CPU there is no NVRTC to compile with and
no device to launch on; injecting both is what makes every structural behaviour in this
module testable without a GPU.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

from autokernel_pbt.props.backends.base import single_output
from autokernel_pbt.props.backends.cuda_toolchain import CudaArtifact, CudaToolchain
from autokernel_pbt.props.backends.telemetry import MISSING, _Missing

HASH_CHARS = 16

#: (shape of the primary input, params) -> launch dimensions.
Geometry = Callable[[tuple[int, ...], dict[str, Any]], tuple[int, ...]]


def _no_shared_memory(block: tuple[int, ...], params: dict[str, Any]) -> int:
    return 0


def _plain(value: Any) -> bool:
    """Whether `repr(value)` is the same in every process: scalars, and tuples of them."""
    if isinstance(value, tuple):
        return all(_plain(item) for item in value)
    return value is None or isinstance(value, (bool, int, float, str, bytes))


class CudaKernel:
    """One CUDA kernel, callable as `kernel(**numpy_inputs) -> np.ndarray`.

    `source` is the CUDA C++ text NVRTC compiles and `entry` the `extern "C"` function to
    launch. `grid` and `block` derive the launch geometry from the primary input's shape
    and `params`; `shared_bytes` derives the dynamic shared memory from the block. The
    launcher receives all of them and may not compute its own: see `__call__`.
    `params` holds the kernel's scalar parameters other than shapes (layernorm's `eps`),
    passed to the launcher and recorded as `kernel_params`.
    """

    def __init__(
        self,
        kernel_id: str,
        source: str,
        entry: str,
        grid: Geometry,
        block: Geometry,
        launcher: Callable[..., Any],
        toolchain: CudaToolchain,
        *,
        params: dict[str, Any] | None = None,
        shared_bytes: Callable[[tuple[int, ...], dict[str, Any]], int] = _no_shared_memory,
        options: tuple[str, ...] = (),
    ) -> None:
        self.kernel_id = kernel_id
        self.source = source
        self.entry = entry
        self.grid = grid
        self.block = block
        self.launcher = launcher
        self.toolchain = toolchain
        self.params = dict(params or {})
        self.shared_bytes = shared_bytes
        self.options = tuple(options)
        self.compiled: CudaArtifact | None = None
        self._launch: dict[str, Any] | None = None

    @property
    def source_hash(self) -> str | _Missing:
        """Content identity of what will actually run, or MISSING if unreadable.

        The kernel text and the launcher's source are both hashed: the launcher carries
        the argument order and the device-buffer handling, and a change there changes the
        kernel as surely as an edit to its body. The compile options are part of what
        runs too, and they are recorded separately, as `compile_options`. Unreadable
        launcher source is MISSING, never a hash of its name -- the defect `TritonKernel`
        once shipped, where editing a kernel body left its hash unchanged.

        A launcher's source does not include the values it closed over, and those can
        decide what runs: `kernels/cuda/ladder.py`'s `_row_launcher(scalars)` returns one
        body for every port, and `scalars` alone sets which parameters follow `n_cols` and
        in what order. Captured plain data is hashed by value. Anything else captured is
        MISSING: its `repr` may carry an address, which would give one launcher a
        different hash in every process.
        """
        if not isinstance(self.source, str):
            return MISSING
        try:
            launcher_source = inspect.getsource(self.launcher)
            captured = inspect.getclosurevars(self.launcher).nonlocals
        except (OSError, TypeError):
            return MISSING
        if not all(_plain(value) for value in captured.values()):
            return MISSING
        material = f"{self.source}\n{self.entry}\n{launcher_source}\n{sorted(captured.items())!r}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:HASH_CHARS]

    def launch_telemetry(self) -> dict[str, Any]:
        """The launch group of the telemetry schema, as actually used.

        `grid`, `block` and `dynamic_shared_bytes` are the geometry derived for the most
        recent call and handed to its launcher -- recorded as soon as it is derived, so a
        call that then failed to compile or launch still says what it was about to run,
        exactly as Triton's `grid` does. They are None before the first call, and after a
        call whose geometry could not be derived at all. `compile_options` is what was
        passed to the compiler, recorded even when the compile then failed -- a rejected
        option set is part of the finding.
        """
        launch = self._launch or {}
        return {
            "grid": launch.get("grid"),
            "block": launch.get("block"),
            "dynamic_shared_bytes": launch.get("shared_bytes"),
            "compile_options": launch.get("compile_options"),
            "kernel_params": dict(self.params),
        }

    def toolchain_versions(self) -> dict[str, Any]:
        return dict(self.toolchain.versions())

    def __call__(self, **inputs: np.ndarray) -> np.ndarray:
        # Per-call state is cleared FIRST. One adapter serves every case of its shape, so
        # a call that fails before recording -- a compile failure, a geometry function
        # that raises -- would otherwise report the previous call's artifact and geometry
        # as its own.
        self.compiled = None
        self._launch = None
        # Copied, not aliased: `readonly_inputs` may have flipped the host arrays
        # non-writeable, and `torch.from_numpy` warns on a non-writeable array, which this
        # project turns into an error. The copy is negligible next to the host-to-device
        # transfer that follows.
        writable = {name: np.array(value, copy=True) for name, value in inputs.items()}
        primary = next(iter(writable.values()))
        grid = tuple(self.grid(primary.shape, self.params))
        block = tuple(self.block(primary.shape, self.params))
        shared_bytes = int(self.shared_bytes(block, self.params))
        compile_options = self.toolchain.resolved_options(self.options)
        self._launch = {
            "grid": list(grid),
            "block": list(block),
            "shared_bytes": shared_bytes,
            "compile_options": list(compile_options),
        }
        self.compiled = self.toolchain.compile(self.source, self.entry, self.options)
        # The launcher is HANDED the geometry rather than recomputing it. If it computed
        # its own, the recorded launch would be a label next to the behaviour rather than
        # a description of it, and the artifacts could report a geometry that never ran.
        return single_output(
            self.launcher(
                artifact=self.compiled,
                toolchain=self.toolchain,
                grid=grid,
                block=block,
                shared_bytes=shared_bytes,
                params=dict(self.params),
                **writable,
            )
        )
