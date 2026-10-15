"""The symmetry probe: six wrong kernels the gate admits, one output symmetry each.

Every law in the softmax and layernorm bundles reads an output alone (finite, in
[0, 1], rows sum to one; zero mean, unit variance) or compares an output with another
output (shift invariance). None ties an output value to the input value it came from.
So any post-processing of a correct output that keeps each row's constraints, and
commutes with a per-row input shift, passes the whole bundle: rolling the rows,
reversing the columns, negating a layernorm, sorting each of its rows. The paper's
Proposition 1 (a) and (c) states this; this script measures it on the gate, unmodified.

Every probe kernel is the task's own reference with one symmetry applied to its output,
so whatever a row gets wrong it gets wrong through that symmetry alone. Each task also
has two controls: the reference itself, which the gate must admit, and a positive
control, which the gate must reject -- without it, a gate that admitted everything would
reproduce every probe row.

These kernels reconstruct the probe the paper's Table 3 quoted as ``[P]`` (re-run
2026-10-01 by a script that was not kept). They are built from the table's row labels,
and they stand in for it only because they reproduce its numbers. See
``docs/measurements/2026-10-14-symmetry-probe.md``. Each row carries what every arm and
the gate must say about it, and ``test_symmetry_probe.py`` pins that.

    python symmetry_probe.py     prints the matrix as measured
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pbt_gate
from saboteurs import Saboteur, first_failing_property, log_ratio_spread_eps, unnormalized

from autokernel_pbt.props.oracle import (
    ALLCLOSE_PROPERTY,
    REFERENCE_PROPERTY,
    AllcloseOracle,
    DeclarativeOracle,
    HybridOracle,
    ReferenceOracle,
)
from autokernel_pbt.props.properties import RowsHaveUnitVariance, RowsSumToOne
from autokernel_pbt.props.tasks import layernorm_reference, softmax_reference
from autokernel_pbt.props.verdict import Verdict

ALLCLOSE = AllcloseOracle.name
REFERENCE = ReferenceOracle.name
DECLARATIVE = DeclarativeOracle.name
HYBRID = HybridOracle.name
ARMS = (ALLCLOSE, REFERENCE, DECLARATIVE, HYBRID)

Kernel = Callable[[np.ndarray], np.ndarray]


def after(reference: Kernel, symmetry: Callable[[np.ndarray], np.ndarray]) -> Kernel:
    """The reference's output with ``symmetry`` applied to it.

    The parameter is named ``x`` because the backend calls ``kernel(**inputs)``.
    Contiguous, so a reversed view cannot reach the backend as a negative stride.
    """

    def kernel(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(symmetry(reference(x)))

    return kernel


def layernorm_undivided(x: np.ndarray) -> np.ndarray:
    """Positive control: the row centered and never divided by its standard deviation.

    The defect ``rows_have_unit_variance`` exists to catch. The layernorm domain draws
    from U(-10, 10) precisely so that an undivided row's variance (~33) sits far from 1
    (``LAYERNORM`` in ``props/tasks.py``).
    """
    wide = np.asarray(x, dtype=np.float64)
    return (wide - np.mean(wide, axis=-1, keepdims=True)).astype(x.dtype)


# Seven of nine at most. The single-column rungs (1, 1) and (17, 1) make every kernel
# below exactly right: a one-element softmax is 1.0 and a one-element layernorm is 0.0,
# whatever symmetry is applied. Rolling rows is also the identity on the single-row
# rung (1, 64), so a rolled kernel can be caught on six.
_CAUGHT_BY_NO_ARM = {ALLCLOSE: (0, ""), REFERENCE: (0, ""), DECLARATIVE: (0, ""), HYBRID: (0, "")}


def _caught_only_against_a_reference(groups: int) -> dict[str, tuple[int, str]]:
    """Rejected by every arm that compares with the reference, passed by the declarative
    arm. The hybrid arm catches through its reference half: its declarative half passed,
    so nothing short-circuited."""
    return {
        ALLCLOSE: (groups, ALLCLOSE_PROPERTY),
        REFERENCE: (groups, REFERENCE_PROPERTY),
        DECLARATIVE: (0, ""),
        HYBRID: (groups, REFERENCE_PROPERTY),
    }


def _caught_by_every_arm_through(law: str) -> dict[str, tuple[int, str]]:
    """A positive control: the declarative arm catches it through ``law``, and the hybrid
    arm through the same law, because its declarative half fails first."""
    return {
        ALLCLOSE: (7, ALLCLOSE_PROPERTY),
        REFERENCE: (7, REFERENCE_PROPERTY),
        DECLARATIVE: (7, law),
        HYBRID: (7, law),
    }


PROBES: dict[str, Saboteur] = {
    "softmax_control": Saboteur(
        "softmax(x), the reference", "softmax", softmax_reference, _CAUGHT_BY_NO_ARM, True
    ),
    "softmax_rows_rolled": Saboteur(
        "softmax, rows rolled by one",
        "softmax",
        after(softmax_reference, lambda y: np.roll(y, 1, axis=0)),
        _caught_only_against_a_reference(6),
        True,
    ),
    "softmax_columns_reversed": Saboteur(
        "softmax, columns reversed",
        "softmax",
        after(softmax_reference, lambda y: y[:, ::-1]),
        _caught_only_against_a_reference(7),
        True,
    ),
    "softmax_positive_control": Saboteur(
        "exp(x - max), never divided",
        "softmax",
        unnormalized,
        _caught_by_every_arm_through(RowsSumToOne.name),
        False,
    ),
    "layernorm_control": Saboteur(
        "LN(x), the reference", "layernorm", layernorm_reference, _CAUGHT_BY_NO_ARM, True
    ),
    "layernorm_negated": Saboteur(
        "-LN(x)",
        "layernorm",
        after(layernorm_reference, np.negative),
        _caught_only_against_a_reference(7),
        True,
    ),
    "layernorm_columns_reversed": Saboteur(
        "LN, columns reversed",
        "layernorm",
        after(layernorm_reference, lambda y: y[:, ::-1]),
        _caught_only_against_a_reference(7),
        True,
    ),
    "layernorm_rows_sorted": Saboteur(
        "LN, each row sorted",
        "layernorm",
        after(layernorm_reference, lambda y: np.sort(y, axis=-1)),
        _caught_only_against_a_reference(7),
        True,
    ),
    "layernorm_rows_rolled": Saboteur(
        "LN, rows rolled by one",
        "layernorm",
        after(layernorm_reference, lambda y: np.roll(y, 1, axis=0)),
        _caught_only_against_a_reference(6),
        True,
    ),
    "layernorm_positive_control": Saboteur(
        "x - mean, never divided",
        "layernorm",
        layernorm_undivided,
        _caught_by_every_arm_through(RowsHaveUnitVariance.name),
        False,
    ),
}

#: The declarative results that are not a PASS, on a row whose declarative arm passes:
#: (law, case id), all INCONCLUSIVE. Layernorm's unit-variance law has nothing to judge
#: where every row is a constant input, which layernorm maps to zeros -- the
#: single-column rungs, base case and shifted partner -- so it abstains there, on the
#: correct reference too. The other laws judge those groups. Softmax abstains nowhere.
DECLARATIVE_ABSTENTIONS: dict[str, frozenset[tuple[str, str]]] = {
    "softmax": frozenset(),
    "layernorm": frozenset(
        (RowsHaveUnitVariance.name, f"layernorm-{group}-base{partner}")
        for group in ("g00006", "g00008")
        for partner in ("", "::shift_rows")
    ),
}


@contextmanager
def declarative_results() -> Iterator[list]:
    """Record every result the gate's own declarative arm produces while the block runs.

    ``check_task`` counts only FAIL groups, so a declarative "0/9" reads the same whether
    the arm judged every case and approved it or abstained on every case. The record
    needs to tell those apart. Observed on the arm the gate builds, not on a copy; the
    hybrid arm holds the same object, so its declarative half is recorded too.
    """
    seen: list = []
    original = DeclarativeOracle.evaluate

    def evaluate(self, rows):
        results = original(self, rows)
        seen.extend(results)
        return results

    DeclarativeOracle.evaluate = evaluate
    try:
        yield seen
    finally:
        DeclarativeOracle.evaluate = original


def main() -> None:
    print(
        f"numpy {np.__version__}, "
        f"GATE_SEED={pbt_gate.GATE_SEED}, GATE_GROUPS={pbt_gate.GATE_GROUPS}"
    )
    header = ["row", "kernel", *ARMS, "gate", "declarative inconclusive", "log-ratio spread (eps)"]
    print("| " + " | ".join(header) + " |")
    print("|---" * len(header) + "|")
    for name, probe in PROBES.items():
        with declarative_results() as seen:
            verdict = pbt_gate.check_task(probe.kernel, probe.task)
        cells = {a.arm: a for a in verdict.arms}
        shown = [
            f"{cells[arm].verdict} {cells[arm].groups_failed}/{cells[arm].groups_total}"
            + (f" ({first_failing_property(cells[arm])})" if cells[arm].groups_failed else "")
            for arm in ARMS
        ]
        gate = "admits" if verdict.passed else "rejects"
        abstained = sum(r.verdict is Verdict.INCONCLUSIVE for r in seen)
        # The log-ratio law completes the softmax bundle only; layernorm's completing law
        # is the affine tie, which this probe does not measure.
        spread = (
            f"{log_ratio_spread_eps(probe.kernel, probe.task):.3g}"
            if probe.task == "softmax"
            else "-"
        )
        row = [name, probe.label, *shown, gate, f"{abstained} of {len(seen)}", spread]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
