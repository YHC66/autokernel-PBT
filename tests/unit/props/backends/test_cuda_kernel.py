"""CudaKernel adapter tests, all CPU.

The adapter's structure is where wiring bugs live -- protocol conformance, the recorded
launch geometry, source identity, stale per-call state -- and all of it is checkable
against a fake toolchain. The device path is covered by tests/gpu/, run on the instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OutputContractError
from autokernel_pbt.props.backends.cuda_kernel import CudaKernel
from autokernel_pbt.props.backends.cuda_toolchain import CudaCompilationError
from autokernel_pbt.props.backends.telemetry import MISSING

SOURCE = 'extern "C" __global__ void relu_rows(const float* x, float* y, int n) {}'


def _zeros(**kw):
    return np.zeros_like(kw["x"])


def _adapter(toolchain, **overrides) -> CudaKernel:
    defaults = {
        "kernel_id": "relu_cuda",
        "source": SOURCE,
        "entry": "relu_rows",
        "grid": lambda shape, params: (shape[0],),
        "block": lambda shape, params: (32,),
        "launcher": _zeros,
        "toolchain": toolchain,
    }
    defaults.update(overrides)
    return CudaKernel(**defaults)


def test_adapter_is_callable_with_numpy_kwargs(fake_toolchain):
    """The criterion CUDA_ADAPTER_SATISFIES_THE_BACKEND_PROTOCOL.

    `Backend.run` calls its kernel as `kernel(**kernel_inputs(case))`. The adapter must
    satisfy that unchanged, or the two setups of a translation pair would need different
    protocols and stop being substitutable.
    """
    out = _adapter(fake_toolchain())(x=np.ones((2, 3), dtype=np.float32))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)


def test_the_launcher_receives_exactly_the_geometry_that_is_recorded(fake_toolchain):
    """The criterion RECORDED_GEOMETRY_IS_THE_GEOMETRY_LAUNCHED.

    If the launcher computed its own grid, block or shared memory, the recorded launch
    would be a label beside the behaviour rather than a description of it, and the
    artifacts could report a geometry that never ran.
    """
    seen = {}

    def launcher(*, artifact, toolchain, grid, block, shared_bytes, params, **inputs):
        seen.update(artifact=artifact, grid=grid, block=block, shared_bytes=shared_bytes)
        return np.zeros_like(inputs["x"])

    kernel = _adapter(
        fake_toolchain(),
        launcher=launcher,
        grid=lambda shape, params: (shape[0], 2),
        block=lambda shape, params: (64,),
        shared_bytes=lambda block, params: block[0] * 4,
    )
    kernel(x=np.ones((5, 7), dtype=np.float32))

    launch = kernel.launch_telemetry()
    assert seen["grid"] == (5, 2)
    assert launch["grid"] == [5, 2]
    assert seen["block"] == (64,)
    assert launch["block"] == [64]
    assert seen["shared_bytes"] == 256
    assert launch["dynamic_shared_bytes"] == 256
    # The artifact launched is the one the adapter holds -- which is what telemetry reads.
    assert seen["artifact"] is kernel.compiled


def test_editing_the_kernel_source_changes_the_hash(fake_toolchain):
    """The criterion THE_SOURCE_HASH_IDENTIFIES_THE_KERNEL_TEXT.

    `kernel_id` is a label; two runs must not both call something `relu_cuda` and mean
    different code. And a launcher whose source cannot be read hashes to MISSING, never to
    a hash of its name, which looks like an identity while two bodies share it.
    """
    toolchain = fake_toolchain()
    original = _adapter(toolchain).source_hash
    assert isinstance(original, str)
    assert len(original) == 16
    assert _adapter(toolchain).source_hash == original
    edited = _adapter(toolchain, source=SOURCE.replace("{}", "{ y[0] = 0.0f; }"))
    assert edited.source_hash != original

    unreadable = eval("lambda **kw: None")
    assert _adapter(toolchain, launcher=unreadable).source_hash is MISSING


def test_the_launcher_is_part_of_the_kernel_s_identity(fake_toolchain):
    # The launcher carries the argument order and the buffer handling; changing it changes
    # what runs as surely as editing the body.
    def other(**kw):
        return np.ones_like(kw["x"])

    toolchain = fake_toolchain()
    assert _adapter(toolchain).source_hash != _adapter(toolchain, launcher=other).source_hash


def test_a_failed_compile_does_not_report_the_previous_artifact(fake_toolchain):
    """The criterion A_FAILED_CALL_REPORTS_NO_STALE_ARTIFACT.

    One adapter serves every case of its shape. A call that fails before it compiles must
    not inherit the previous call's artifact -- the row would carry another execution's
    registers and PTX hash as its own.
    """
    toolchain = fake_toolchain()
    kernel = _adapter(toolchain)
    kernel(x=np.ones((2, 3), dtype=np.float32))
    assert kernel.compiled is not None

    toolchain.fail_compile = CudaCompilationError("NVRTC rejected the kernel source")
    with pytest.raises(CudaCompilationError):
        kernel(x=np.ones((5, 7), dtype=np.float32))
    assert kernel.compiled is None
    # The geometry this call derived is its own, not the previous call's.
    assert kernel.launch_telemetry()["grid"] == [5]


def test_a_geometry_function_that_raises_leaves_no_stale_geometry(fake_toolchain):
    calls = []

    def grid(shape, params):
        calls.append(shape)
        if len(calls) == 2:
            msg = "no grid for this shape"
            raise ValueError(msg)
        return (shape[0],)

    kernel = _adapter(fake_toolchain(), grid=grid)
    kernel(x=np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="no grid for this shape"):
        kernel(x=np.ones((5, 7), dtype=np.float32))
    assert kernel.launch_telemetry()["grid"] is None
    assert kernel.compiled is None


def test_the_compile_precedes_the_launch(fake_toolchain):
    # A source that never compiled must never reach a device buffer.
    launched = []

    def launcher(**kw):
        launched.append(kw)
        return np.zeros((2, 3), dtype=np.float32)

    kernel = _adapter(
        fake_toolchain(fail_compile=CudaCompilationError("rejected")), launcher=launcher
    )
    with pytest.raises(CudaCompilationError):
        kernel(x=np.ones((2, 3), dtype=np.float32))
    assert launched == []


def test_compile_options_are_recorded_even_when_the_compile_fails(fake_toolchain):
    # A rejected option set is part of the finding, so it is recorded before compiling.
    kernel = _adapter(
        fake_toolchain(fail_compile=CudaCompilationError("rejected")),
        options=("--use_fast_math",),
    )
    with pytest.raises(CudaCompilationError):
        kernel(x=np.ones((2, 3), dtype=np.float32))
    assert kernel.launch_telemetry()["compile_options"] == [
        "--use_fast_math",
        "--gpu-architecture=compute_86",
    ]


def test_the_toolchain_compiles_the_adapter_s_own_source_entry_and_options(fake_toolchain):
    toolchain = fake_toolchain()
    _adapter(toolchain, options=("-lineinfo",))(x=np.ones((2, 3), dtype=np.float32))
    assert toolchain.compiles == [(SOURCE, "relu_rows", ("-lineinfo",))]


def test_kernel_params_are_handed_to_the_launcher_and_recorded(fake_toolchain):
    seen = {}

    def launcher(*, params, **kw):
        seen.update(params)
        return np.zeros_like(kw["x"])

    kernel = _adapter(fake_toolchain(), launcher=launcher, params={"eps": 1e-5})
    kernel(x=np.ones((2, 3), dtype=np.float32))
    assert seen == {"eps": 1e-5}
    assert kernel.launch_telemetry()["kernel_params"] == {"eps": 1e-5}


def test_launch_telemetry_is_empty_before_the_first_call(fake_toolchain):
    # Nothing has launched, so there is no geometry to report.
    launch = _adapter(fake_toolchain()).launch_telemetry()
    assert launch["grid"] is None
    assert launch["block"] is None
    assert launch["dynamic_shared_bytes"] is None


def test_the_launcher_receives_writable_copies(fake_toolchain):
    # `readonly_inputs` may have flipped the host arrays read-only, and torch warns on a
    # read-only array -- which this project turns into an error.
    seen = {}

    def launcher(**kw):
        seen["writeable"] = kw["x"].flags.writeable
        return np.zeros_like(kw["x"])

    x = np.ones((2, 3), dtype=np.float32)
    x.flags.writeable = False
    _adapter(fake_toolchain(), launcher=launcher)(x=x)
    assert seen["writeable"] is True


def test_a_launcher_returning_nothing_is_an_output_contract_error(fake_toolchain):
    with pytest.raises(OutputContractError):
        _adapter(fake_toolchain(), launcher=lambda **kw: None)(x=np.ones((2, 3), np.float32))
