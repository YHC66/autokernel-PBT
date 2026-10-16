"""The cross-setup consistency proof builds in Isabelle/HOL, with no cheating allowed.

The same six theorems as the Lean and Rocq proofs. Built from a temporary copy so nothing
lands in the checkout (Isabelle writes its heaps and logs under `~/.isabelle` anyway).
Skipped when no `isabelle` is installed.

`quick_and_dirty=false` is forced on the command line rather than trusted from the user's
settings: in that mode `sorry` is an error ("Cheating requires quick_and_dirty mode"), so
a build that succeeds has proved every goal. The source scan catches what the build
cannot: an `axiomatization` is not cheating as far as Isabelle is concerned.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = ROOT / "proofs" / "cross-setup-reductions" / "isabelle"
THEORY = PROOF_DIR / "CrossSetup.thy"

AUDITED = (
    "strided_cover",
    "tile_reduce_eq_spec",
    "tile_reduce_drops_tail",
    "block_reduce_eq_spec",
    "softmax_consistent",
    "layernorm_consistent",
)
_ESCAPES = re.compile(r"\b(sorry|oops|axiomatization|axioms|quick_and_dirty)\b")


@pytest.mark.formal
def test_the_isabelle_consistency_proof_builds_without_cheating(tmp_path):
    isabelle = shutil.which("isabelle")
    if isabelle is None:
        pytest.skip("isabelle is not installed")
    source = THEORY.read_text()
    assert not _ESCAPES.search(source), "an escape hatch is in the source"
    for name in AUDITED:
        assert re.search(rf"^\s*(theorem|lemma)\s+{name}\b", source, flags=re.MULTILINE), name

    copy = tmp_path / "isabelle"
    shutil.copytree(PROOF_DIR, copy)
    done = subprocess.run(
        [isabelle, "build", "-o", "quick_and_dirty=false", "-d", ".", "-v", "CrossSetup"],
        cwd=copy,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Finished CrossSetup" in done.stdout, done.stdout
