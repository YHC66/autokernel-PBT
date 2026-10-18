"""The log-ratio headroom record's numbers, pinned.

Run from the repository root (outside ``testpaths``, so run explicitly):

    .venv/bin/python -m pytest experiments/generation/log-ratio-headroom -q

``docs/measurements/2026-10-14-log-ratio-headroom-prereg.md`` quotes these numbers, and the
paper may. A change to the gate's cases, the generator or the recorded-run reader that
moves them must fail here, not leave the record quoting stale values. One assertion per
test, each about a different population.
"""

from __future__ import annotations

import headroom
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def correct_cpu() -> list[headroom.Spread]:
    return [s for k in headroom.correct_cpu_kernels().values() for s in headroom.on_gate_cases(k)]


@pytest.fixture(scope="module")
def correct_device() -> list[headroom.Spread]:
    return [s for name in headroom.DEVICE_RUNS for s in headroom.on_device_run(name)]


def test_correct_cpu_cases_reach_the_recorded_maximum(correct_cpu):
    """522 cases, at most 7.29 eps, no zero or subnormal output."""
    observed = (
        len(correct_cpu),
        round(max(s.eps for s in correct_cpu), 2),
        sum(s.zeros + s.subnormals for s in correct_cpu),
    )
    assert observed == (522, 7.29, 0)


def test_recorded_a10_executions_reach_the_recorded_maximum(correct_device):
    """30 executions, at most 12.3 eps, under the registered threshold of 30."""
    observed = (
        len(correct_device),
        round(max(s.eps for s in correct_device), 1),
        sum(s.zeros + s.subnormals for s in correct_device),
    )
    assert observed == (30, 12.3, 0)


def test_every_differing_wrong_case_sits_seven_orders_above(correct_device):
    """The post-hoc criterion: the weakest wrong output that differs from the reference."""
    weakest = min(
        s.eps
        for kernel in headroom.WRONG.values()
        for s in headroom.on_gate_cases(kernel)
        if s.differs
    )
    assert f"{weakest:.3g}" == "2.67e+07"
