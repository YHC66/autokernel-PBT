"""The Why3 real-theory audit compiles, closes every goal, and rests on Lean's standard axioms.

Same contract as ``test_softmax_bundle_lean.py``: skipped without the pinned toolchain; otherwise
the unique catcher of an open goal (``sorryAx``), a smuggled ``axiom``, or a silently deleted
theorem.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = ROOT / "proofs" / "why3-real-theories"
PROOF = PROOF_DIR / "Why3RealTheories.lean"

AUDITED = (
    "Why3RealTheories.why3ExpLog_scaled",
    "Why3RealTheories.why3ExpLog_rejects_nonpos",
    "Why3RealTheories.softmaxRowWith_scaleExp",
    "Why3RealTheories.certificate_is_a_family",
    "Why3RealTheories.value_pins",
    "Why3RealTheories.tangent_rigid",
    "Why3RealTheories.value_freedom_pos",
    "Why3RealTheories.value_freedom_neg",
    "Why3RealTheories.no_approximation_bound_pos",
    "Why3RealTheories.no_approximation_bound_neg",
    "Why3RealTheories.tangent_two_sided",
    "Why3RealTheories.pow_invariant",
    "Why3RealTheories.exp2_kernel_not_certifiable",
    "Why3RealTheories.why3Square_categorical",
    "Why3RealTheories.why3Trig_scaled_five",
    "Why3RealTheories.why3Trig_rejects_two",
)
STANDARD = {"propext", "Classical.choice", "Quot.sound"}

_AXIOMS = re.compile(r"'(?P<name>\S+)' depends on axioms: \[(?P<axioms>[^\]]*)\]")


def _pinned_lean() -> Path | None:
    pin = (PROOF_DIR / "lean-toolchain").read_text().strip()
    name = pin.replace("/", "--").replace(":", "---")
    home = Path(os.environ.get("ELAN_HOME", Path.home() / ".elan"))
    lean = home / "toolchains" / name / "bin" / "lean"
    return lean if lean.exists() else None


@pytest.mark.formal
def test_the_why3_audit_compiles_on_standard_axioms_only():
    lean = _pinned_lean()
    if lean is None:
        pytest.skip("the pinned Lean toolchain is not installed")
    done = subprocess.run(
        [str(lean), PROOF.name], cwd=PROOF_DIR, capture_output=True, text=True, timeout=600, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    text = PROOF.read_text()
    assert "sorry" not in text, "a goal is left open in the source"
    assert not re.search(r"^\s*axiom\s", text, re.MULTILINE), "a declared axiom is smuggled in"
    audited = {
        m["name"]: set(filter(None, map(str.strip, m["axioms"].split(","))))
        for m in _AXIOMS.finditer(done.stdout)
    }
    missing = [name for name in AUDITED if name not in audited]
    assert not missing, f"not audited by #print axioms: {missing}"
    beyond = {name: axioms - STANDARD for name, axioms in audited.items() if axioms - STANDARD}
    assert not beyond, f"theorems resting on non-standard axioms: {beyond}"
