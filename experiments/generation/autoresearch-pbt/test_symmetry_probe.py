"""The symmetry probe, pinned one row at a time.

Run from the repository root. The experiment is outside ``testpaths``, so the main
suite does not collect it:

    .venv/bin/python -m pytest experiments/generation/autoresearch-pbt -q

What is pinned is a *baseline*, not a target. Six of these kernels are wrong and the
gate admits them, because every law in both bundles reads outputs alone
(``docs/measurements/2026-10-14-symmetry-probe.md``). Completing the bundles is a
separate experiment; when it lands, these tests are meant to fail, and the rows in
``symmetry_probe.py`` are re-recorded deliberately rather than loosened to pass.

One assertion per test, and each is the sole catcher of a break in the gate the other
two cannot see, as in ``test_gate_saboteurs.py``:

* the pairing: an arm going blind, or starting to catch, or catching through a
  different property;
* the decision: the gate deciding on an arm other than the declarative one, or
  admitting a kernel its decisive arm rejected;
* the judgement: a declarative PASS that is really an abstention, beyond the four the
  record names on layernorm's single-column rungs.
"""

from __future__ import annotations

import pbt_gate
import pytest
from saboteurs import first_failing_property
from symmetry_probe import DECLARATIVE, DECLARATIVE_ABSTENTIONS, PROBES

from autokernel_pbt.props.oracle import DeclarativeOracle
from autokernel_pbt.props.verdict import Verdict

pytestmark = pytest.mark.integration

#: Results the declarative arm produces per kernel on the gate's 9 groups: the arm's own
#: and the hybrid arm's declarative half, which evaluates every group. Pinned beside the
#: abstentions so that a recording that saw nothing cannot pass.
DECLARATIVE_RESULTS = 126


@pytest.mark.parametrize("name", sorted(PROBES))
def test_each_probe_row_is_caught_by_exactly_its_paired_arms(name: str):
    """Equality over all four arms, never a subset check: the arm expected to start
    catching is the declarative one, the day the bundles are completed."""
    probe = PROBES[name]
    verdict = pbt_gate.check_task(probe.kernel, probe.task)
    observed = {arm.arm: (arm.groups_failed, first_failing_property(arm)) for arm in verdict.arms}
    assert observed == probe.expected


@pytest.mark.parametrize("name", sorted(PROBES))
def test_the_gate_admits_exactly_the_rows_it_is_recorded_admitting(name: str):
    """The decision, which no arm's row carries. Every probe row is caught by three arms,
    so the pairing stays green if the gate switches to deciding on any of them; the
    positive controls are the rows a gate that admits everything would get wrong."""
    probe = PROBES[name]
    assert pbt_gate.check_task(probe.kernel, probe.task).passed is probe.admitted


@pytest.mark.parametrize(
    "name", sorted(n for n, p in PROBES.items() if p.expected[DECLARATIVE] == (0, ""))
)
def test_a_declarative_pass_is_a_judgement_not_an_abstention(monkeypatch, name: str):
    """Every declarative result is a PASS except the abstentions the record names.

    Observed through the arm ``check_task`` actually builds, not a copy. The count is
    pinned with the abstentions in one assertion, so an empty recording fails rather than
    matching softmax's empty set of abstentions vacuously.
    """
    seen = []
    original = DeclarativeOracle.evaluate

    def evaluate(self, rows):
        results = original(self, rows)
        seen.extend(results)
        return results

    monkeypatch.setattr(DeclarativeOracle, "evaluate", evaluate)
    probe = PROBES[name]
    pbt_gate.check_task(probe.kernel, probe.task)

    not_passed = {
        (r.property_name, r.case_id, r.verdict) for r in seen if r.verdict is not Verdict.PASS
    }
    expected = {
        (law, case_id, Verdict.INCONCLUSIVE) for law, case_id in DECLARATIVE_ABSTENTIONS[probe.task]
    }
    assert (len(seen), not_passed) == (DECLARATIVE_RESULTS, expected)
