"""The cross-setup consistency proof compiles in Rocq, closed under the global context.

The same six theorems as the Lean proof, in Rocq 9 with Stdlib only. Compiled in a
temporary copy, so the `.vo`/`.glob` artefacts never land in the checkout. Skipped when no
`rocq` is installed. `Print Assumptions` answers "Closed under the global context" only
for a theorem resting on no axiom at all; anything else lists the axioms instead.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT / "proofs" / "cross-setup-reductions" / "rocq" / "CrossSetup.v"

AUDITED = (
    "strided_cover",
    "tile_reduce_eq_spec",
    "tile_reduce_drops_tail",
    "block_reduce_eq_spec",
    "softmax_consistent",
    "layernorm_consistent",
)
CLOSED = "Closed under the global context"
_ESCAPES = re.compile(r"\b(Admitted|admit|Axiom|Axioms|Parameter|Parameters|Conjecture)\b")


@pytest.mark.formal
def test_the_rocq_consistency_proof_compiles_closed_under_the_global_context(tmp_path):
    rocq = shutil.which("rocq")
    if rocq is None:
        pytest.skip("rocq is not installed")
    source = PROOF.read_text()
    assert not _ESCAPES.search(source), "an escape hatch is in the source"
    # One `Print Assumptions` per audited theorem, in order: each answer below is then
    # attributable to exactly one name.
    printed = re.findall(r"^Print Assumptions (\w+)\.", source, flags=re.MULTILINE)
    assert printed == list(AUDITED), printed

    shutil.copy(PROOF, tmp_path / PROOF.name)
    done = subprocess.run(
        [rocq, "compile", PROOF.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    answers = [line for line in done.stdout.splitlines() if line.strip()]
    assert answers == [CLOSED] * len(AUDITED), done.stdout
