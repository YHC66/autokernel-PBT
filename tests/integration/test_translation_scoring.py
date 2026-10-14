"""Cross-setup scoring end to end on CPU: a port judged against the kernel it was mapped from.

Two NumPy "setups" stand in for Triton and CUDA. The source reduces with NumPy's pairwise
sum; the correct port folds sequentially (`cumsum`), so the two are correct and differ in
the last bits -- the regime a real CUDA port of a Triton kernel lives in. The wrong port
reproduces the tile-tail defect measured on hardware: a tile narrower than the row, so the
tail is never reduced and never written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autokernel_pbt.metrics.rates import rates_from_run, rates_from_translation
from autokernel_pbt.props import driver
from autokernel_pbt.props.driver import (
    PAIRING_FILE,
    SOURCE_ARM_NAMES,
    TRANSLATION_ARM_NAMES,
    read_run,
    read_translation,
    run_task,
    score_translation,
)
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import LAYERNORM, REFERENCES, SOFTMAX
from autokernel_pbt.props.verdict import Verdict, summarize

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
N_GROUPS = len(SOFTMAX.domain.shapes)
#: The wrong port's tile. Ladder rows wider than this have a tail it never touches.
TILE = 16


def _source(x: np.ndarray) -> np.ndarray:
    """The known kernel: float32 throughout, NumPy's pairwise reductions."""
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def _port(x: np.ndarray) -> np.ndarray:
    """A correct port: the same function, the denominator folded sequentially."""
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / np.cumsum(e, axis=-1)[..., -1:]


def _tail_dropping_port(x: np.ndarray) -> np.ndarray:
    """A wrong port: one tile of TILE lanes per row, as a too-narrow Triton BLOCK does."""
    head = x[..., :TILE]
    shifted = head - head.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    y = np.zeros_like(x)
    y[..., :TILE] = e / e.sum(axis=-1, keepdims=True)
    return y


#: The temperature of a known kernel that is NOT the reference. softmax(beta * x) satisfies
#: every law of the softmax bundle (the recorded temperature blind spot) and differs from
#: softmax(x) on every row wider than one column, so it separates "judged against the
#: source" from "judged against the reference" without the laws taking sides.
BETA = np.float32(1.1)


def _warm_source(x: np.ndarray) -> np.ndarray:
    return _source(BETA * x)


def _warm_port(x: np.ndarray) -> np.ndarray:
    return _port(BETA * x)


def _record(
    run_dir: Path, kernel, kernel_id: str, *, seed: int = SEED, broken: bool | None = False
):
    run_task(
        task=SOFTMAX,
        kernel=kernel,
        reference_fn=REFERENCES["softmax"],
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        n_groups=N_GROUPS,
        seed=seed,
        kernel_id=kernel_id,
        kernel_is_broken=broken,
    )


def _verdicts(arms, name: str) -> dict[str, Verdict]:
    """One arm's verdict per group, the unit at which arms are comparable."""
    (arm,) = [arm for arm in arms if arm.arm == name]
    by_group: dict[str, list] = {}
    for result in arm.results:
        by_group.setdefault(result.group_id, []).append(result)
    return {group: summarize(results) for group, results in by_group.items()}


@pytest.fixture
def source_run(tmp_path) -> Path:
    run = tmp_path / "source"
    _record(run, _source, "softmax_source")
    return run


def test_a_correct_port_is_scored_by_seven_arms_and_the_pair_is_recorded(source_run, tmp_path):
    """The criterion TRANSLATION_SCORING_WRITES_SEVEN_ARMS_AND_THE_PAIRING."""
    target = tmp_path / "port"
    _record(target, _port, "softmax_port")
    score_translation(
        task=SOFTMAX,
        source_dir=source_run,
        target_dir=target,
        repo_root=REPO_ROOT,
        reference_fn=REFERENCES["softmax"],
    )
    rows, arms, pairing = read_translation(target, source_run)

    assert [arm.arm for arm in arms] == list(TRANSLATION_ARM_NAMES)
    groups = {row.case.group_id for row in rows}
    for name in TRANSLATION_ARM_NAMES:
        verdicts = _verdicts(arms, name)
        assert set(verdicts) == groups, name
        assert Verdict.FAIL not in verdicts.values(), (name, verdicts)

    # The two setups really did round differently: otherwise this test proves nothing
    # about the "correct but different" regime it stands in for.
    source_rows = {row.case.case_id: row for row in ExecutionTable(source_run).read()}
    assert any(
        not np.array_equal(row.outputs["y"], source_rows[row.case.case_id].outputs["y"])
        for row in rows
    )

    _, source_fingerprint = ExecutionTable(source_run).read_with_fingerprint()
    _, target_fingerprint = ExecutionTable(target).read_with_fingerprint()
    assert pairing["source_fingerprint"] == source_fingerprint
    assert pairing["target_fingerprint"] == target_fingerprint
    assert pairing["source_kernel_ids"] == ["softmax_source"]
    assert pairing["target_kernel_ids"] == ["softmax_port"]
    assert pairing["task_id"] == "softmax"


def test_a_tail_dropping_port_is_caught_against_the_source(source_run, tmp_path):
    """The criterion A_WRONG_PORT_IS_CAUGHT_AGAINST_THE_SOURCE.

    Caught on exactly the groups whose rows are wider than the tile, by both source
    arms; on the narrow rows the port is correct and must not be booked.
    """
    target = tmp_path / "port"
    _record(target, _tail_dropping_port, "softmax_tail_dropping", broken=True)
    score_translation(
        task=SOFTMAX,
        source_dir=source_run,
        target_dir=target,
        repo_root=REPO_ROOT,
        reference_fn=REFERENCES["softmax"],
    )
    rows, arms, _ = read_translation(target, source_run)
    wide = {row.case.group_id for row in rows if row.case.shape[-1] > TILE}
    assert wide, "the corpus must reach rows wider than the tile"
    for name in ("allclose_source", "reference_source"):
        failed = {group for group, v in _verdicts(arms, name).items() if v is Verdict.FAIL}
        assert failed == wide, name


def test_read_translation_refuses_a_re_recorded_source(source_run, tmp_path):
    """The criterion A_RE_RECORDED_SOURCE_INVALIDATES_THE_PAIR.

    Re-recorded with the same kernel and seed, the source's ids and inputs are identical;
    only its fingerprint says it is a different recording -- and the source arms'
    verdicts are about the old one.
    """
    target = tmp_path / "port"
    _record(target, _port, "softmax_port")
    score_translation(
        task=SOFTMAX,
        source_dir=source_run,
        target_dir=target,
        repo_root=REPO_ROOT,
        reference_fn=REFERENCES["softmax"],
    )
    _record(source_run, _source, "softmax_source")
    with pytest.raises(ValueError, match="re-recorded or replaced after scoring"):
        read_translation(target, source_run)


def test_runs_under_different_seeds_are_refused_before_scoring(source_run, tmp_path):
    # Same task, same ids -- ids are (seed, index) -- and different inputs everywhere.
    target = tmp_path / "port"
    _record(target, _port, "softmax_port", seed=SEED + 1)
    with pytest.raises(ValueError, match="same id and different input bytes"):
        score_translation(
            task=SOFTMAX,
            source_dir=source_run,
            target_dir=target,
            repo_root=REPO_ROOT,
            reference_fn=REFERENCES["softmax"],
        )
    # Nothing was written over the target's own four-arm scores.
    assert not (target / PAIRING_FILE).exists()


def test_a_run_scored_without_a_source_is_not_read_as_a_translation(source_run):
    with pytest.raises(ValueError, match="was not scored as a translation"):
        read_translation(source_run, source_run)


def test_the_source_arms_are_the_last_three():
    assert TRANSLATION_ARM_NAMES[-3:] == SOURCE_ARM_NAMES


def _score(source: Path, target: Path, **kw) -> None:
    score_translation(
        task=SOFTMAX,
        source_dir=source,
        target_dir=target,
        repo_root=REPO_ROOT,
        reference_fn=REFERENCES["softmax"],
        **kw,
    )


def _failed(arms, name: str) -> set[str]:
    return {group for group, v in _verdicts(arms, name).items() if v is Verdict.FAIL}


def test_the_source_arms_judge_against_the_source_and_not_the_reference(tmp_path):
    """The criterion SOURCE_ARMS_JUDGE_AGAINST_THE_SOURCE.

    When the source and the reference agree, an arm wired to the wrong one passes every
    other test. Here they disagree on every multi-column row, in both directions.
    """
    source = tmp_path / "warm-source"
    _record(source, _warm_source, "softmax_beta_source")

    faithful = tmp_path / "faithful-port"
    _record(faithful, _warm_port, "softmax_beta_port")
    _score(source, faithful)
    rows, arms, _ = read_translation(faithful, source)
    differing = {row.case.group_id for row in rows if row.case.shape[-1] > 1}
    assert len(differing) == 7
    for name in ("allclose", "reference", "hybrid"):
        assert _failed(arms, name) == differing, name
    for name in ("allclose_source", "reference_source", "hybrid_source", "declarative"):
        assert _failed(arms, name) == set(), name

    correct = tmp_path / "correct-port"
    _record(correct, _port, "softmax_port")
    _score(source, correct)
    _, arms, _ = read_translation(correct, source)
    for name in ("allclose_source", "reference_source", "hybrid_source"):
        assert _failed(arms, name) == differing, name
    for name in ("allclose", "reference", "hybrid", "declarative"):
        assert _failed(arms, name) == set(), name


@pytest.fixture
def scored(source_run, tmp_path) -> Path:
    target = tmp_path / "port"
    _record(target, _port, "softmax_port")
    _score(source_run, target)
    return target


def test_a_crash_between_the_scores_and_the_pairing_leaves_an_unreadable_run(
    source_run, scored, tmp_path, monkeypatch
):
    """The criterion A_HALF_WRITTEN_TRANSLATION_IS_REFUSED.

    Re-scored against another source, with the pairing write failing: the old pairing
    must already be gone, so the new scores cannot be read as judged against the old
    source.
    """
    other = tmp_path / "other-source"
    _record(other, _warm_source, "softmax_beta_source")

    def crash(*args, **kw):
        msg = "power cut"
        raise OSError(msg)

    monkeypatch.setattr(driver, "_write_pairing", crash)
    with pytest.raises(OSError, match="power cut"):
        _score(other, scored)
    for source in (source_run, other):
        with pytest.raises(ValueError, match="source-arm scores and no pairing.json"):
            read_translation(scored, source)


def test_read_run_refuses_a_run_scored_against_a_source(scored):
    """The criterion READ_RUN_REFUSES_SOURCE_ARMS: the metrics path cannot skip the pairing."""
    with pytest.raises(ValueError, match="read it with read_translation"):
        read_run(scored)
    with pytest.raises(ValueError, match="read it with read_translation"):
        rates_from_run(scored)


def test_rates_from_translation_reports_all_seven_arms(source_run, scored):
    assert list(rates_from_translation(scored, source_run)) == list(TRANSLATION_ARM_NAMES)


def _rewrite_pairing(run: Path, **changes) -> None:
    path = run / PAIRING_FILE
    pairing = json.loads(path.read_text())
    pairing.update(changes)
    path.write_text(json.dumps(pairing))


def test_a_pairing_written_for_another_target_recording_is_refused(source_run, scored):
    _rewrite_pairing(scored, target_fingerprint="0" * 32 + "-" + "0" * 64)
    with pytest.raises(ValueError, match="pairs target corpus"):
        read_translation(scored, source_run)


def test_a_pairing_written_for_other_arms_is_refused(source_run, scored):
    _rewrite_pairing(scored, scored_arms=list(TRANSLATION_ARM_NAMES[:4]))
    with pytest.raises(ValueError, match="re-scored after pairing"):
        read_translation(scored, source_run)


def test_scoring_refuses_a_task_other_than_the_one_recorded(source_run, tmp_path):
    target = tmp_path / "port"
    _record(target, _port, "softmax_port")
    with pytest.raises(ValueError, match="record task 'softmax', not 'layernorm'"):
        score_translation(
            task=LAYERNORM,
            source_dir=source_run,
            target_dir=target,
            repo_root=REPO_ROOT,
            reference_fn=REFERENCES["layernorm"],
        )


def test_an_unknown_source_is_refused_unless_waived_and_the_waiver_is_recorded(tmp_path):
    source = tmp_path / "unknown-source"
    _record(source, _source, "softmax_unknown", broken=None)
    target = tmp_path / "port"
    _record(target, _port, "softmax_port")
    with pytest.raises(ValueError, match="source is not stated correct"):
        _score(source, target)
    _score(source, target, require_known_source=False)
    _, _, pairing = read_translation(target, source)
    assert pairing["require_known_source"] is False
