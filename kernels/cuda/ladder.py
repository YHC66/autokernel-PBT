"""CUDA ports of the known Triton ladder kernels: the translation segment's first target.

Each port lives in its own `.cu` file beside this module, and that file's text is the
kernel's identity (`CudaKernel.source_hash`). This module supplies what the `.cu` files
cannot: the launch geometry, the launch contract, and the factories `scripts/gpu_record.py`
calls, keyed exactly as `kernels/triton/ladder.py`'s `KERNELS` is, so one recording command
selects the source or the port by `--backend` alone.

THE BLOCK WIDTH IS DERIVED FROM THE ROW, like the Triton tile, and for the reasons
`kernels/triton/ladder.py` records: a fixed width would make every shape compile to one
artifact and every compiled field constant. It differs from the tile in one way that
matters. Threads stride across the row, so a row wider than the block is not dropped --
it is folded, sequentially per thread, before the block's tree. The CUDA port has no
tile-tail defect to inherit, and its reduction tree differs from Triton's exactly when
`n_cols > width`. `proofs/cross-setup-reductions/` proves the two compute the same
function in exact arithmetic; the cross-setup reference ratio measures how far apart
their roundings land.

Unlike the Triton ladder, this module imports cleanly with no torch and no CUDA: torch is
imported inside the launcher, and the toolchain is built on first use. Only launching a
kernel needs a device.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from autokernel_pbt.props.backends.cuda_kernel import CudaKernel
from autokernel_pbt.props.backends.cuda_toolchain import CudaToolchain, NvrtcToolchain
from autokernel_pbt.props.backends.triton_kernel import InputMutatedError, device_digest

SOURCE_DIR = Path(__file__).resolve().parent

#: The CUDA limit on threads per block, on every architecture this project targets.
MAX_THREADS = 1024
#: One warp. A narrower block leaves most of the warp idle and buys nothing.
MIN_THREADS = 32


def threads_for(n_cols: int) -> int:
    """The block width for a row of `n_cols`: the next power of two, within [32, 1024].

    A power of two because the shared-memory tree halves it; at most 1024 because that is
    the device's limit, and the strided loop folds any wider row. A row of 1 still gets a
    full warp, whose idle threads hold the reduction's identity.
    """
    width = 1 << max(n_cols - 1, 0).bit_length()
    return min(max(width, MIN_THREADS), MAX_THREADS)


def _rows_grid(shape: tuple[int, ...], params: dict[str, Any]) -> tuple[int, ...]:
    """One block per row. `shape` is the primary input's shape."""
    return (shape[0],)


def _rows_block(shape: tuple[int, ...], params: dict[str, Any]) -> tuple[int, ...]:
    return (threads_for(shape[-1]),)


def _one_float_per_thread(block: tuple[int, ...], params: dict[str, Any]) -> int:
    """The `partial[]` array the softmax and layernorm trees reduce in."""
    return block[0] * np.dtype(np.float32).itemsize


@cache
def default_toolchain() -> NvrtcToolchain:
    """The process's NVRTC toolchain, built on first use and shared by every port.

    Built when a *factory* is called, which `gpu_record.py` does outside `Backend.run`: a
    toolchain that cannot start is the harness's fault and must raise as a bad call, not
    be booked as one LAUNCH_ERROR per row. Cached so the self-test compile and the context
    setup happen once.
    """
    return NvrtcToolchain()


def _row_launcher(scalars: tuple[str, ...]):
    """A launcher for `kernel(const float* x, float* y, int n_cols, <scalars as float>)`.

    `scalars` names the params passed after `n_cols`, in the kernel's parameter order,
    each as a float32. The geometry is the one the adapter hands in, never recomputed.
    """

    def launch(*, artifact, toolchain, grid, block, shared_bytes, params, **inputs):
        x = inputs["x"]
        # Refused before any device call. The kernels read `const float*`: a float64 or
        # int array would be reinterpreted byte for byte and produce a plausible wrong
        # answer with no error. A row-major 2-D matrix is the only layout they index.
        if x.dtype != np.float32:
            msg = f"the CUDA ports read float32; got {x.dtype}, which they would misread"
            raise TypeError(msg)
        if x.ndim != 2:
            msg = f"the CUDA ports index a 2-D (rows, cols) matrix; got shape {x.shape}"
            raise ValueError(msg)
        import torch

        # `.contiguous()` on device: the kernels index row-major, and a column-major host
        # array would arrive with strides the pointer arithmetic ignores.
        xd = torch.as_tensor(x, device="cuda").contiguous()
        yd = torch.empty_like(xd)
        args = (xd, yd, np.int32(x.shape[1]), *(np.float32(params[name]) for name in scalars))

        # The input-mutation check, on the device buffer -- the only place it can fire.
        before = device_digest(xd)
        toolchain.launch(artifact, grid=grid, block=block, shared_bytes=shared_bytes, args=args)
        if device_digest(xd) != before:
            msg = "kernel modified its input tensor(s) ['x'] on device"
            raise InputMutatedError(msg)
        # No astype: a dtype defect stays visible instead of being cast away here.
        return yd.cpu().numpy()

    return launch


def _source(name: str) -> str:
    return (SOURCE_DIR / f"{name}.cu").read_text()


def relu_kernel(n_cols: int, toolchain: CudaToolchain | None = None) -> CudaKernel:
    return CudaKernel(
        kernel_id="relu_cuda",
        source=_source("relu"),
        entry="relu_rows",
        grid=_rows_grid,
        block=_rows_block,
        launcher=_row_launcher(()),
        toolchain=toolchain or default_toolchain(),
    )


def softmax_kernel(n_cols: int, toolchain: CudaToolchain | None = None) -> CudaKernel:
    return CudaKernel(
        kernel_id="softmax_cuda",
        source=_source("softmax"),
        entry="softmax_rows",
        grid=_rows_grid,
        block=_rows_block,
        launcher=_row_launcher(()),
        toolchain=toolchain or default_toolchain(),
        shared_bytes=_one_float_per_thread,
    )


def layernorm_kernel(n_cols: int, toolchain: CudaToolchain | None = None) -> CudaKernel:
    # eps matches tasks.LAYERNORM_EPS and the Triton port's EPS constexpr: a port
    # targeting a different eps would differ from its source for a reason that is not a
    # translation defect. Recorded as the float32 value the kernel actually receives
    # (9.99999974e-06), not the Python literal, so `kernel_params` describes the launch.
    from autokernel_pbt.props.tasks import LAYERNORM_EPS

    return CudaKernel(
        kernel_id="layernorm_cuda",
        source=_source("layernorm"),
        entry="layernorm_rows",
        grid=_rows_grid,
        block=_rows_block,
        launcher=_row_launcher(("eps",)),
        toolchain=toolchain or default_toolchain(),
        params={"eps": float(np.float32(LAYERNORM_EPS))},
        shared_bytes=_one_float_per_thread,
    )


#: Keyed as `kernels/triton/ladder.py`'s KERNELS is. The sweep and at-scale tasks are
#: softmax again, so they reuse the softmax port.
KERNELS = {
    "relu": relu_kernel,
    "softmax": softmax_kernel,
    "layernorm": layernorm_kernel,
    "tolerance_sweep": softmax_kernel,
    "softmax_at_scale": softmax_kernel,
}
