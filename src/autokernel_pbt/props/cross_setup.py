"""A port judged against the known kernel it was mapped from (feature 0013).

The translation segment maps a known kernel from one setup to another. Its recorded
executions are the most direct evidence of what the port must compute: the same inputs,
byte for byte, and with GPU first, the same device. This module turns a source run into a
reference the existing arms can use unchanged, and guards the one dangerous step --
deciding that two separately recorded runs are about the same inputs.

*Keyed by content, never by id.* Case ids are a pure function of `(seed, index)`, so two
runs of one task under one seed share every id even if their inputs differ: a new
generator, a changed relation, a different draw order. A recording keyed by id would
score the port against outputs computed from other inputs, and nothing would look wrong.
`SourceRecording` answers by the bytes of the inputs it is handed, so an input the source
never saw has no answer at all.

*The arms are the existing ones.* `ReferenceOracle` and `AllcloseOracle` call their
reference as `reference_fn(**kernel_inputs(case))`; a recording is callable exactly so.
The source arms differ from their parents only in `name`, so every difference between
`reference` and `reference_source` on a run is the reference and nothing else.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np

from autokernel_pbt.props.backends.base import OUTPUT_NAME, ExecutionResult, Status, kernel_inputs
from autokernel_pbt.props.oracle import (
    AllcloseOracle,
    HybridOracle,
    ReferenceOracle,
    ReferenceUnavailable,
)


class UnpairedInputError(ValueError):
    """A reference was asked about inputs the source run never executed.

    A bad *call*: `pair_runs` refuses any pair where this could happen, so reaching it
    means a caller scored a run it never paired. Approximating -- the nearest input, the
    same case id -- would compare the port against outputs of other inputs.
    """


def content_key(inputs: Mapping[str, np.ndarray]) -> str:
    """A digest of a kernel's inputs: every name, dtype, shape and byte.

    Dtype and shape are hashed alongside the bytes because the same bytes read as float32
    (8, 8) and as float64 (4, 8) are different inputs.
    """
    digest = hashlib.sha256()
    for name in sorted(inputs):
        # Shape from the array as given: `np.ascontiguousarray` is `ndmin=1`, so a 0-d
        # input would otherwise hash as (1,) -- the same key as a different input.
        array = np.asarray(inputs[name])
        digest.update(f"{name}\0{array.dtype.str}\0{array.shape}\0".encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class SourceRecording:
    """A source run's recorded outputs, callable as `recording(**inputs) -> ndarray`.

    `fingerprint` is the source table's corpus fingerprint, from the same read as `rows`;
    it is what `pairing.json` records, so a re-recorded source is detectable later.
    """

    def __init__(self, rows: list[ExecutionResult], fingerprint: str = "") -> None:
        self.fingerprint = fingerprint
        self._by_content: dict[str, ExecutionResult] = {}
        for row in rows:
            key = content_key(kernel_inputs(row.case))
            seen = self._by_content.get(key)
            if seen is not None and not _same_answer(seen, row):
                msg = (
                    f"the source run gives two different answers for one input: cases "
                    f"{seen.case.case_id!r} and {row.case.case_id!r} have byte-identical "
                    f"inputs and differ in status or output, so it is not a reference"
                )
                raise ValueError(msg)
            self._by_content.setdefault(key, row)

    def __call__(self, **inputs: np.ndarray) -> np.ndarray:
        row = self._by_content.get(content_key(inputs))
        if row is None:
            msg = (
                "no source execution has these input bytes; the runs were not paired "
                "(see cross_setup.pair_runs), and the nearest answer is not an answer"
            )
            raise UnpairedInputError(msg)
        if row.status != Status.OK or OUTPUT_NAME not in row.outputs:
            msg = (
                f"the source's own execution of {row.case.case_id!r} was "
                f"{row.status}; the source, not the port, has no answer here"
            )
            raise ReferenceUnavailable(msg)
        # A copy: the arms are handed this array, and one writing into it must not change
        # what every later arm is compared against.
        return np.array(row.outputs[OUTPUT_NAME], copy=True)


def _same_answer(a: ExecutionResult, b: ExecutionResult) -> bool:
    if a.status != b.status:
        return False
    out_a, out_b = a.outputs.get(OUTPUT_NAME), b.outputs.get(OUTPUT_NAME)
    if out_a is None or out_b is None:
        return out_a is out_b
    return (
        out_a.dtype == out_b.dtype
        and out_a.shape == out_b.shape
        and out_a.tobytes() == out_b.tobytes()
    )


def pair_runs(
    source: list[ExecutionResult],
    target: list[ExecutionResult],
    *,
    source_fingerprint: str,
    target_fingerprint: str,
    require_known_source: bool = True,
) -> None:
    """Refuse a source/target pair that is not one corpus judged by a known kernel.

    Each check raises its own message, so one refusal can never be mistaken for another.
    """
    if source_fingerprint and source_fingerprint == target_fingerprint:
        msg = (
            f"source and target are the same recording ({source_fingerprint}); a port "
            f"compared with itself agrees everywhere and measures nothing"
        )
        raise ValueError(msg)

    tasks = {row.case.task_id for row in source} | {row.case.task_id for row in target}
    if len(tasks) != 1:
        msg = f"the runs span task(s) {sorted(tasks)}; a pair is one task recorded twice"
        raise ValueError(msg)

    source_by_id = {row.case.case_id: row for row in source}
    target_by_id = {row.case.case_id: row for row in target}
    if source_by_id.keys() != target_by_id.keys():
        only_source = sorted(source_by_id.keys() - target_by_id.keys())
        only_target = sorted(target_by_id.keys() - source_by_id.keys())
        msg = (
            f"the runs are different corpora: cases only in the source {only_source}, "
            f"only in the target {only_target}"
        )
        raise ValueError(msg)

    regrouped = sorted(
        case_id
        for case_id, row in target_by_id.items()
        if row.case.group_id != source_by_id[case_id].case.group_id
    )
    if regrouped:
        msg = f"case(s) {regrouped} sit in different groups in the two runs"
        raise ValueError(msg)

    different = sorted(
        case_id
        for case_id, row in target_by_id.items()
        if content_key(kernel_inputs(row.case))
        != content_key(kernel_inputs(source_by_id[case_id].case))
    )
    if different:
        msg = (
            f"case(s) {different} carry the same id and different input bytes in the two "
            f"runs; ids are a function of (seed, index), not of the inputs"
        )
        raise ValueError(msg)

    if require_known_source:
        unknown = sorted(
            row.case.case_id for row in source if row.kernel_is_broken is not False
        )
        if unknown:
            msg = (
                f"the source is not stated correct on case(s) {unknown} "
                f"(kernel_is_broken must be False); a port is judged against a KNOWN "
                f"kernel -- pass require_known_source=False to judge against an unknown one"
            )
            raise ValueError(msg)


class SourceAllcloseOracle(AllcloseOracle):
    """T0: the field's default, `allclose(port, source)`, unmodified."""

    name = "allclose_source"


class SourceReferenceOracle(ReferenceOracle):
    """T1: the normalized test ratio of the port against the source's recorded outputs."""

    name = "reference_source"


class SourceHybridOracle(HybridOracle):
    """The law bundle first, then the ratio against the source where the laws found nothing."""

    name = "hybrid_source"
