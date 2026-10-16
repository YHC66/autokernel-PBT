"""Structure of the CUDA ports, checked on CPU against a fake toolchain.

Nothing here launches a kernel. The block-width derivation, the launch contract and the
launcher's refusals all happen before the first call into torch or the driver.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autokernel_pbt.props.tasks import LAYERNORM_EPS, TASKS
from kernels.cuda import ladder

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_threads_for_is_the_next_power_of_two_clamped_to_a_warp_and_the_device_limit():
    widths = [ladder.threads_for(n) for n in (1, 7, 32, 33, 129, 1024, 1025, 16384)]
    assert widths == [32, 32, 32, 64, 256, 1024, 1024, 1024]


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_every_shape_in_use_gets_a_power_of_two_width_within_the_limit(task_id):
    """The criterion THE_BLOCK_WIDTH_HONOURS_THE_LAUNCH_CONTRACT.

    The shared-memory tree halves the width, so a width that is not a power of two
    skips a partial; above 1024 the launch is refused by the device. Unlike the Triton
    tile there is no lower bound tied to the row: the strided loop folds any width.
    """
    for shape in TASKS[task_id].domain.shapes:
        width = ladder.threads_for(shape[-1])
        assert width & (width - 1) == 0, (shape, width)
        assert ladder.MIN_THREADS <= width <= ladder.MAX_THREADS, (shape, width)


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_every_port_launches_a_power_of_two_width_within_the_limit(fake_toolchain, task_id):
    # The ports' own geometry, not the helper: a port whose block bypassed `threads_for`
    # would pass the helper's test and drop partial sums on device. Checked through the
    # adapter's recorded launch, which is the geometry the launcher is handed.
    for shape in TASKS[task_id].domain.shapes:
        kernel = ladder.KERNELS[task_id](shape[-1], toolchain=fake_toolchain())
        kernel.launcher = lambda **kw: np.zeros_like(kw["x"])
        kernel(x=np.ones(shape, dtype=np.float32))
        (width,) = kernel.launch_telemetry()["block"]
        assert width & (width - 1) == 0, (task_id, shape, width)
        assert ladder.MIN_THREADS <= width <= ladder.MAX_THREADS, (task_id, shape, width)
        assert width >= min(shape[-1], ladder.MAX_THREADS), (task_id, shape, width)


@pytest.mark.parametrize("factory", [ladder.softmax_kernel, ladder.layernorm_kernel])
def test_a_non_float32_input_is_refused_before_any_device_call(fake_toolchain, factory):
    """The criterion THE_PORTS_REFUSE_INPUT_THEY_WOULD_MISREAD.

    The kernels read `const float*`. A float64 matrix would be reinterpreted byte for
    byte -- a plausible wrong answer and no error. Refused before torch is touched or the
    driver is called: the fake toolchain records no launch.
    """
    toolchain = fake_toolchain()
    kernel = factory(8, toolchain=toolchain)
    with pytest.raises(TypeError, match="read float32; got float64"):
        kernel(x=np.ones((2, 8), dtype=np.float64))
    with pytest.raises(ValueError, match="index a 2-D"):
        kernel(x=np.ones((2, 2, 8), dtype=np.float32))
    assert toolchain.launches == []


@pytest.mark.parametrize(
    ("factory", "name", "entry"),
    [
        (ladder.relu_kernel, "relu", "relu_rows"),
        (ladder.softmax_kernel, "softmax", "softmax_rows"),
        (ladder.layernorm_kernel, "layernorm", "layernorm_rows"),
    ],
)
def test_each_port_is_its_own_cu_file_and_entry_point(fake_toolchain, factory, name, entry):
    kernel = factory(8, toolchain=fake_toolchain())
    assert kernel.source == (ladder.SOURCE_DIR / f"{name}.cu").read_text()
    # extern "C", or the entry point is C++-mangled and the driver cannot find it.
    assert re.search(rf'extern "C" __global__ void {entry}\(', kernel.source)
    assert kernel.entry == entry
    assert kernel.kernel_id == f"{name}_cuda"


def test_the_tree_kernels_get_one_float_of_shared_memory_per_thread(fake_toolchain):
    toolchain = fake_toolchain()
    for factory in (ladder.softmax_kernel, ladder.layernorm_kernel):
        kernel = factory(129, toolchain=toolchain)
        assert kernel.shared_bytes((256,), kernel.params) == 256 * 4
    relu = ladder.relu_kernel(129, toolchain=toolchain)
    assert relu.shared_bytes((256,), relu.params) == 0


def test_the_layernorm_port_records_the_task_s_eps_as_launched(fake_toolchain):
    # A port targeting a different eps would differ from its source for a reason that is
    # not a translation defect. Recorded as the float32 the kernel receives, so
    # `kernel_params` describes the launch rather than the Python literal.
    kernel = ladder.layernorm_kernel(8, toolchain=fake_toolchain())
    (eps,) = kernel.params.values()
    assert eps == float(np.float32(LAYERNORM_EPS))
    assert float(np.float32(eps)) == eps
    assert eps != LAYERNORM_EPS


def test_the_ports_cover_every_task(fake_toolchain):
    assert set(ladder.KERNELS) == set(TASKS)


def test_the_geometry_is_one_block_per_row_of_the_derived_width(fake_toolchain):
    kernel = ladder.softmax_kernel(129, toolchain=fake_toolchain())
    assert kernel.grid((7, 129), kernel.params) == (7,)
    assert kernel.block((7, 129), kernel.params) == (256,)


def test_the_module_imports_without_torch_or_a_toolchain():
    # Unlike the Triton ladder, which imports torch and triton at module scope.
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    code = (
        "import sys\n"
        f"sys.path[:0] = [{str(REPO_ROOT / 'src')!r}, {str(REPO_ROOT)!r}]\n"
        "import kernels.cuda.ladder\n"
        "assert 'torch' not in sys.modules\n"
        "assert not any(name.startswith('cuda.') for name in sys.modules)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
