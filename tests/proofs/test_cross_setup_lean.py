"""The cross-setup consistency proof compiles in Lean, on Lean's standard axioms only.

`proofs/cross-setup-reductions/lean/` proves that the known Triton ladder kernels and their
CUDA ports compute the same function in exact arithmetic (see the directory's README).
Skipped when the pinned toolchain is absent, as GPU tests are when CUDA is. When it runs it
is the unique catcher of a goal left open, an axiom smuggled in, or a theorem deleted.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = ROOT / "proofs" / "cross-setup-reductions" / "lean"
PROOF = PROOF_DIR / "CrossSetup.lean"

#: The six theorems the shared statement names; a renamed or deleted one fails here.
AUDITED = tuple(
    f"CrossSetup.{name}"
    for name in (
        "strided_cover",
        "tile_reduce_eq_spec",
        "tile_reduce_drops_tail",
        "block_reduce_eq_spec",
        "softmax_consistent",
        "layernorm_consistent",
    )
)
STANDARD = {"propext", "Classical.choice", "Quot.sound"}
_AXIOMS = re.compile(r"'(?P<name>\S+)' depends on axioms: \[(?P<axioms>[^\]]*)\]")
_ESCAPES = re.compile(r"\b(sorry|admit|axiom|native_decide)\b")


def _pinned_lean() -> Path | None:
    pin = (PROOF_DIR / "lean-toolchain").read_text().strip()
    name = pin.replace("/", "--").replace(":", "---")
    home = Path(os.environ.get("ELAN_HOME", Path.home() / ".elan"))
    lean = home / "toolchains" / name / "bin" / "lean"
    return lean if lean.exists() else None


@pytest.mark.formal
def test_the_lean_consistency_proof_compiles_on_standard_axioms_only():
    lean = _pinned_lean()
    if lean is None:
        pytest.skip("the pinned Lean toolchain is not installed")
    done = subprocess.run(
        [str(lean), PROOF.name],
        cwd=PROOF_DIR,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert not _ESCAPES.search(PROOF.read_text()), "an escape hatch is in the source"
    audited = {
        m["name"]: set(filter(None, map(str.strip, m["axioms"].split(","))))
        for m in _AXIOMS.finditer(done.stdout)
    }
    missing = [name for name in AUDITED if name not in audited]
    assert not missing, f"not audited by #print axioms: {missing}"
    beyond = {name: axioms - STANDARD for name, axioms in audited.items() if axioms - STANDARD}
    assert not beyond, f"theorems resting on non-standard axioms: {beyond}"
