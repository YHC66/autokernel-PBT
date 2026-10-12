"""The rigidity audit, pinned one symmetry at a time.

Run from the repository root (outside ``testpaths``, so the main suite does not collect it):

    .venv/bin/python -m pytest experiments/verification/formal-rigidity -q

Each test is the sole catcher of one way the audit could rot: a symmetry the Lean file proves
going unrecognized (sampling too tight, or a transformation implemented wrongly), a rejection
attributed to the wrong axiom, or the instrument finding about the harness's raw comparison
disappearing silently.
"""

from __future__ import annotations

import pytest
import rigidity as rg


def _table(axioms, transformations):
    return {f.transformation: (f.admitted, f.rejecting_axiom) for f in rg.audit(axioms, transformations)}


@pytest.fixture(scope="module")
def explog():
    return _table(rg.WHY3_EXPLOG, rg.EXP_TRANSFORMATIONS)


@pytest.fixture(scope="module")
def trig():
    return _table(rg.WHY3_TRIG, rg.TRIG_TRANSFORMATIONS)


def test_why3_explog_admits_the_positive_temperature_family(explog):
    assert {t for t, (ok, _) in explog.items() if ok} == {"identity", "exp(2x)", "exp(x/2)"}


def test_why3_explog_rejects_negative_temperature_through_monotonicity(explog):
    assert explog["exp(-x)"] == (False, "exp_increasing")


def test_why3_explog_rejects_zero_temperature_through_log_exp(explog):
    assert explog["exp(0x)"] == (False, "Log_exp")


def test_the_papers_three_axioms_admit_negative_temperature_too():
    table = _table(rg.EXP_LAWS, rg.EXP_TRANSFORMATIONS)
    assert {t for t, (ok, _) in table.items() if ok} == {"identity", "exp(2x)", "exp(x/2)", "exp(-x)"}


def test_the_tangent_axiom_alone_removes_the_family():
    table = _table(rg.WHY3_EXPLOG_TANGENT, rg.EXP_TRANSFORMATIONS)
    assert {t for t, (ok, _) in table.items() if ok} == {"identity"}
    assert table["exp(2x)"] == (False, "Tangent")


def test_double_precision_bounds_on_e_remove_the_family():
    table = _table(rg.WHY3_EXPLOG_EBOUNDS, rg.EXP_TRANSFORMATIONS)
    assert {t for t, (ok, _) in table.items() if ok} == {"identity"}
    assert table["exp(2x)"] == (False, "E_bounds")


def test_why3_square_is_rigid_and_sign_is_pinned_by_sqrt_positive():
    table = _table(rg.WHY3_SQUARE, rg.SQRT_TRANSFORMATIONS)
    assert {t for t, (ok, _) in table.items() if ok} == {"identity"}
    assert table["-sqrt(x)"] == (False, "Sqrt_positive")


def test_why3_trigonometry_admits_exactly_the_sampled_frequencies_congruent_to_one_mod_four(trig):
    assert {t for t, (ok, _) in trig.items() if ok} == {"identity", "cos,sin(5x)", "cos,sin(-3x)"}


def test_why3_trigonometry_rejects_doubling_through_cos_pi(trig):
    assert trig["cos,sin(2x)"] == (False, "Cos_pi")


def test_the_harness_raw_comparison_rejects_true_trigonometric_axioms():
    """The instrument finding: relative-only closeness fails identities whose sides pass near 0.

    Which axiom trips depends on the sample range; that at least one true axiom is rejected on
    a non-zero fraction of samples does not.
    """
    rates = rg.spurious_rejections(rg.WHY3_TRIG)
    worst = max(rates, key=rates.get)
    assert rates[worst] > 0.0, rates


def test_the_floored_comparison_accepts_every_true_axiom():
    for axioms, transformations in rg.SUITES:
        table = _table(axioms, transformations)
        assert table["identity"] == (True, ""), axioms.name
