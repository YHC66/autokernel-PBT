"""Device tests for the CUDA ports and the NVRTC toolchain. Run by hand on the instance.

Deliberately NOT acceptance criteria, as with the Triton device tests: a criterion whose
evidence exists only on rented hardware would leave tests/spec red everywhere else. These
are what confirm, on one real release of cuda-python and one real driver, the conventions
the CPU tests' fakes assume -- result tuples, buffers written in place, attribute names.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OUTPUT_NAME, Status
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.tasks import REFERENCES, TASKS

pytestmark = pytest.mark.gpu

LADDER = ("relu", "softmax", "layernorm")


@pytest.fixture(scope="module")
def cuda_python():
    return pytest.importorskip("cuda.bindings", reason="cuda-python is not installed")


@pytest.fixture(scope="module")
def toolchain(torch_cuda, cuda_python):
    from kernels.cuda.ladder import default_toolchain

    return default_toolchain()


def _case(x: np.ndarray) -> Case:
    return Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="relu",
        dtype=str(x.dtype),
        shape=x.shape,
        tensors={"x": x},
    )


def _probe_kernel(toolchain, source: str, entry: str):
    from autokernel_pbt.props.backends.cuda_kernel import CudaKernel
    from kernels.cuda.ladder import _row_launcher, _rows_block, _rows_grid

    return CudaKernel(
        kernel_id=entry,
        source=source,
        entry=entry,
        grid=_rows_grid,
        block=_rows_block,
        launcher=_row_launcher(()),
        toolchain=toolchain,
    )


@pytest.mark.parametrize("task_id", LADDER)
def test_the_cuda_port_agrees_with_its_numpy_reference(task_id, toolchain):
    from autokernel_pbt.props.backends.cuda_backend import CudaBackend
    from kernels.cuda.ladder import KERNELS

    backend = CudaBackend()
    task = TASKS[task_id]
    for group in Generator(task.domain, seed=0).generate(len(task.domain.shapes)):
        for case in group.cases:
            result = backend.run(KERNELS[task_id](case.shape[-1]), case)
            assert result.status is Status.OK, result.error
            expected = REFERENCES[task_id](x=case.tensors["x"])
            # Loose on purpose: this asserts the port is not grossly wrong. How close it
            # should be is what TE1 measures, not a threshold to assume in advance.
            assert np.allclose(result.outputs[OUTPUT_NAME], expected, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("task_id", ["softmax", "layernorm"])
def test_a_row_wider_than_the_block_is_folded_not_dropped(task_id, toolchain):
    # The Triton tile drops a row's tail when the tile is too narrow; the CUDA port's
    # strided loop must not. 4096 columns against a 1024-thread block: four per thread.
    from autokernel_pbt.props.backends.cuda_backend import CudaBackend
    from kernels.cuda.ladder import KERNELS

    x = np.random.default_rng(0).standard_normal((3, 4096)).astype(np.float32)
    result = CudaBackend().run(KERNELS[task_id](4096), _case(x))
    assert result.status is Status.OK, result.error
    assert result.telemetry["block"] == [1024]
    expected = REFERENCES[task_id](x=x)
    assert np.allclose(result.outputs[OUTPUT_NAME], expected, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("task_id", LADDER)
def test_compiled_telemetry_is_populated_on_device(task_id, toolchain):
    """The reason for the schema: these must not come back MISSING.

    If one does, the bindings changed an attribute name or a return convention on this
    release -- the kind of thing the smoke session exists to find while the instance is up.
    """
    from autokernel_pbt.props.backends.cuda_backend import CudaBackend
    from autokernel_pbt.props.backends.telemetry import MISSING
    from kernels.cuda.ladder import KERNELS

    x = np.ones((4, 129), dtype=np.float32)
    result = CudaBackend().run(KERNELS[task_id](129), _case(x))
    assert result.status is Status.OK, result.error
    for key in (
        "n_regs",
        "local_bytes",
        "max_threads_per_block",
        "ptx_version",
        "binary_version",
        "ptx_hash",
        "nvrtc_version",
        "driver_version",
    ):
        assert result.telemetry[key] is not MISSING, f"{key} came back MISSING on device"


def test_a_kernel_that_fails_to_compile_is_a_compile_error_on_device(toolchain):
    """A REAL NVRTC compile failure, with its log, classified by the real backend."""
    from autokernel_pbt.props.backends.cuda_backend import CudaBackend

    source = 'extern "C" __global__ void broken(const float* x, float* y, int n) { y[0] = z; }'
    result = CudaBackend().run(
        _probe_kernel(toolchain, source, "broken"), _case(np.ones((1, 4), np.float32))
    )
    assert result.status is Status.COMPILE_ERROR, result.error
    assert "undefined" in result.error


def test_a_kernel_without_extern_c_is_a_compile_error_on_device(toolchain):
    # Compiles, and its name is mangled, so the driver finds no `plain` to launch.
    from autokernel_pbt.props.backends.cuda_backend import CudaBackend

    source = "__global__ void plain(const float* x, float* y, int n) { y[0] = x[0]; }"
    result = CudaBackend().run(
        _probe_kernel(toolchain, source, "plain"), _case(np.ones((1, 4), np.float32))
    )
    assert result.status is Status.COMPILE_ERROR, result.error


def test_a_kernel_that_writes_to_its_input_is_caught_on_device(toolchain):
    """The integrity check firing for real; it cannot be exercised off-device."""
    from autokernel_pbt.props.backends.triton_kernel import InputMutatedError

    source = (
        'extern "C" __global__ void vandal(float* x, float* y, int n) {\n'
        "  const size_t base = (size_t)blockIdx.x * n;\n"
        "  for (int c = threadIdx.x; c < n; c += blockDim.x) { y[base + c] = x[base + c]; "
        "x[base + c] += 1.0f; }\n"
        "}\n"
    )
    with pytest.raises(InputMutatedError):
        _probe_kernel(toolchain, source, "vandal")(x=np.ones((4, 8), dtype=np.float32))


def test_an_empty_compiler_log_leaves_the_interpreter_s_bytes_intact(toolchain):
    # NVRTC writes through the buffer it is handed; a one-byte buffer would have been
    # CPython's shared cached object. Checked on the real library, after real compiles.
    toolchain.compile('extern "C" __global__ void ok(float* y) { y[0] = 3.0f; }', "ok", ())
    assert b" " == bytes([32])
    assert b"\x00" == bytes([0])


@pytest.mark.parametrize("task_id", LADDER)
def test_the_cuda_port_and_its_triton_source_agree_on_byte_identical_inputs(task_id, toolchain):
    """The smoke version of TE1, on one device: the known kernel and its port.

    Loose, like the reference check above: the measurement of how far apart they land is
    the experiment's, under its pre-registration, not this test's.
    """
    pytest.importorskip("triton", reason="triton is not installed")
    from autokernel_pbt.props.backends.cuda_backend import CudaBackend
    from autokernel_pbt.props.backends.triton_backend import TritonBackend
    from kernels.cuda.ladder import KERNELS as CUDA
    from kernels.triton.ladder import KERNELS as TRITON

    task = TASKS[task_id]
    for group in Generator(task.domain, seed=42).generate(len(task.domain.shapes)):
        for case in group.cases:
            source = TritonBackend().run(TRITON[task_id](case.shape[-1]), case)
            port = CudaBackend().run(CUDA[task_id](case.shape[-1]), case)
            assert source.status is port.status is Status.OK, (source.error, port.error)
            assert np.allclose(
                port.outputs[OUTPUT_NAME], source.outputs[OUTPUT_NAME], rtol=1e-3, atol=1e-5
            )
