"""SourceRecording, pair_runs and the source arms (feature 0013), on hand-built rows.

Each refusal is paired with its own message, so a check that fires for the wrong reason
fails its test rather than certifying another check's case.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OUTPUT_NAME, ExecutionResult, Status
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.cross_setup import (
    SourceAllcloseOracle,
    SourceHybridOracle,
    SourceRecording,
    SourceReferenceOracle,
    UnpairedInputError,
    content_key,
    pair_runs,
)
from autokernel_pbt.props.driver import ARM_NAMES, TRANSLATION_ARM_NAMES, arm_order
from autokernel_pbt.props.oracle import AllcloseOracle, HybridOracle, ReferenceOracle
from autokernel_pbt.props.verdict import Verdict

RNG = np.random.default_rng(7)
X = RNG.standard_normal((3, 8)).astype(np.float32)
Y = RNG.random((3, 8)).astype(np.float32)


def _row(
    x: np.ndarray = X,
    y: np.ndarray | None = Y,
    *,
    case_id: str = "softmax-g00000-base",
    group_id: str = "softmax-g00000",
    task_id: str = "softmax",
    status: Status = Status.OK,
    broken: bool | None = False,
) -> ExecutionResult:
    case = Case(
        case_id=case_id,
        group_id=group_id,
        relation="base",
        task_id=task_id,
        dtype=str(x.dtype),
        shape=x.shape,
        tensors={"x": x},
    )
    outputs = {} if y is None else {OUTPUT_NAME: y}
    return ExecutionResult(
        case=case, outputs=outputs, status=status, kernel_id="k", kernel_is_broken=broken
    )


def _pair(source, target, **kw):
    kw.setdefault("source_fingerprint", "source-run")
    kw.setdefault("target_fingerprint", "target-run")
    pair_runs(source, target, **kw)


def test_the_recording_answers_with_the_source_output_for_the_same_input_bytes():
    """The criterion THE_SOURCE_ANSWERS_FOR_THE_SAME_INPUT_BYTES.

    A different array object holding the same bytes is the same input: that is what the
    port's run hands the arm, after a round trip through its own execution table.
    """
    recording = SourceRecording([_row()])
    answer = recording(x=X.copy())
    assert answer.dtype == Y.dtype
    assert np.array_equal(answer, Y)


def test_an_input_the_source_never_saw_is_refused():
    """The criterion AN_INPUT_THE_SOURCE_NEVER_SAW_IS_A_BAD_CALL.

    One flipped bit, a different dtype over the same bytes, or the same bytes in another
    shape is a different input; the nearest answer is not an answer.
    """
    recording = SourceRecording([_row()])
    flipped = X.copy()
    flipped.view(np.uint32)[0, 0] ^= 1
    for different in (flipped, X.view(np.int32), X.reshape(6, 4)):
        with pytest.raises(UnpairedInputError, match="no source execution has these input bytes"):
            recording(x=different)
    with pytest.raises(UnpairedInputError):
        recording(y=X)


@pytest.mark.parametrize("arm", [SourceReferenceOracle, SourceAllcloseOracle])
def test_a_case_the_source_failed_is_inconclusive_for_the_port(arm):
    """The criterion A_FAILED_SOURCE_ROW_IS_INCONCLUSIVE_NAMING_THE_SOURCE."""
    recording = SourceRecording([_row(y=None, status=Status.LAUNCH_ERROR)])
    (result,) = arm(recording).evaluate([_row()])
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "the source, not the port" in result.detail
    assert "launch_error" in result.detail


def test_an_arm_cannot_corrupt_the_source_recording():
    """The criterion AN_ARM_CANNOT_CORRUPT_THE_SOURCE.

    Compared against a copy taken before: comparing against the row's own output array
    would compare an aliased array with itself, and pass whatever was written into it.
    """
    output = Y.copy()
    before = output.copy()
    recording = SourceRecording([_row(y=output)])
    answer = recording(x=X)
    answer[...] = 0.0
    assert np.array_equal(recording(x=X), before)
    assert np.array_equal(output, before)


def test_two_answers_for_one_input_are_refused():
    """The criterion AN_AMBIGUOUS_SOURCE_IS_REFUSED.

    One input with two answers is not a reference. The same answer twice is harmless.
    """
    other = _row(y=Y + 1.0, case_id="softmax-g00001-base", group_id="softmax-g00001")
    with pytest.raises(ValueError, match="two different answers for one input"):
        SourceRecording([_row(), other])
    failed = _row(y=None, status=Status.LAUNCH_ERROR, case_id="c1", group_id="g1")
    with pytest.raises(ValueError, match="two different answers for one input"):
        SourceRecording([_row(), failed])
    same = _row(case_id="softmax-g00001-base", group_id="softmax-g00001")
    assert np.array_equal(SourceRecording([_row(), same])(x=X), Y)


_OTHER = X + 1.0


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (_row(task_id="layernorm"), r"span task\(s\) \['layernorm', 'softmax'\]"),
        (_row(case_id="softmax-g00009-base"), "different corpora: cases only in the source"),
        (_row(group_id="softmax-g00009"), r"sit in different groups"),
        (_row(x=_OTHER), "same id and different input bytes"),
    ],
    ids=["task", "case-ids", "groups", "input-bytes"],
)
def test_pairing_refuses_runs_over_different_corpora(target, message):
    """The criterion PAIRING_REQUIRES_ONE_CORPUS.

    Ids are a function of (seed, index): two runs under different seeds share every id
    and differ in every input, which is exactly the "input-bytes" case.
    """
    with pytest.raises(ValueError, match=message):
        _pair([_row()], [target])


def test_pairing_requires_a_known_source_other_than_the_target():
    """The criterion PAIRING_REQUIRES_A_KNOWN_SOURCE."""
    for unknown in (None, True):
        with pytest.raises(ValueError, match="source is not stated correct"):
            _pair([_row(broken=unknown)], [_row()])
    _pair([_row(broken=None)], [_row()], require_known_source=False)

    with pytest.raises(ValueError, match="same recording"):
        _pair([_row()], [_row()], source_fingerprint="one-run", target_fingerprint="one-run")


def test_a_matching_pair_is_accepted():
    _pair([_row(broken=False)], [_row(broken=True)])


def test_translation_scoring_has_exactly_seven_named_arms():
    # Literal, not the module's own constant: a constant cut to two arms would otherwise
    # certify itself.
    assert TRANSLATION_ARM_NAMES == (
        "allclose",
        "reference",
        "declarative",
        "hybrid",
        "allclose_source",
        "reference_source",
        "hybrid_source",
    )


def test_the_four_arm_order_is_unchanged_by_the_generalization():
    """The criterion THE_FOUR_ARM_ORDER_IS_UNCHANGED.

    `arm_order` gained a `names` parameter. Every run recorded so far was ordered by the
    four-arm draw, so for every seed the default must give exactly what it gave before.
    The literal for seed 42 was computed by the code before the change.
    """
    assert arm_order(42) == ["reference", "hybrid", "allclose", "declarative"]
    for seed in range(200):
        rng = np.random.default_rng([seed, 0xA6])
        before = [ARM_NAMES[index] for index in rng.permutation(len(ARM_NAMES))]
        assert arm_order(seed) == before
    assert sorted(arm_order(42, TRANSLATION_ARM_NAMES)) == sorted(TRANSLATION_ARM_NAMES)


@pytest.mark.parametrize(
    ("child", "parent", "name"),
    [
        (SourceReferenceOracle, ReferenceOracle, "reference_source"),
        (SourceAllcloseOracle, AllcloseOracle, "allclose_source"),
    ],
)
def test_a_source_arm_differs_from_its_parent_only_in_name(child, parent, name):
    # So every difference between `reference` and `reference_source` on a run is the
    # reference, and nothing about how it is compared.
    recording = SourceRecording([_row()])
    port = _row(y=Y * np.float32(1.0 + 3e-7))
    assert child.name == name
    assert [replace(r, property_name="") for r in child(recording).evaluate([port])] == [
        replace(r, property_name="") for r in parent(recording).evaluate([port])
    ]


def test_the_source_hybrid_is_named_for_its_reference():
    assert SourceHybridOracle.name == "hybrid_source"
    assert issubclass(SourceHybridOracle, HybridOracle)


def test_content_key_covers_name_dtype_shape_and_bytes():
    base = content_key({"x": X})
    assert content_key({"x": X.copy()}) == base
    assert content_key({"y": X}) != base
    assert content_key({"x": X.view(np.int32)}) != base
    assert content_key({"x": X.reshape(6, 4)}) != base
    assert content_key({"x": np.asfortranarray(X)}) == base
    # `np.ascontiguousarray` is ndmin=1: a 0-d input and a (1,) input are different.
    assert content_key({"x": np.float32(2.0)}) != content_key({"x": np.array([2.0], np.float32)})
