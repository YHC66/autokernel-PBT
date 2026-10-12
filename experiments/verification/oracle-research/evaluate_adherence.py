"""Mechanical checks on one adherence sandbox: compile every Lean file, scan for escape hatches,
read `#print axioms`, pull out the main theorem text, and list the kernel's verdict from RESULT.md.

    .venv/bin/python experiments/verification/oracle-research/evaluate_adherence.py <sandbox-dir> [...]

What this does NOT do (no adapter exists): execute the agent's spec/model readings (C2–C4) or
sample its axioms (C6). Those are judged by reading, and the record says so.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

LEAN = Path.home() / ".elan/toolchains/leanprover--lean4---v4.33.0/bin/lean"
_AXIOMS = re.compile(r"'(?P<name>\S+)' depends on axioms: \[(?P<axioms>[^\]]*)\]")
STANDARD = {"propext", "Classical.choice", "Quot.sound"}


def evaluate(sandbox: Path) -> dict:
    out = {"sandbox": str(sandbox), "kernel": (sandbox / "kernel.py").read_text().strip().splitlines()[-1][:80]}
    files = []
    for f in sorted((sandbox / "proof").glob("*.lean")):
        if f.name == "Skeleton.lean":
            continue
        done = subprocess.run([str(LEAN), f.name], cwd=f.parent, capture_output=True, text=True, timeout=600, check=False)
        text = f.read_text()
        audited = {m["name"]: set(filter(None, map(str.strip, m["axioms"].split(",")))) for m in _AXIOMS.finditer(done.stdout)}
        theorems = re.findall(r"^(?:theorem|lemma)\s+(\S+)", text, re.MULTILINE)
        classes = re.findall(r"^class\s+(\S+)[^\n]*\n((?:\s+\S[^\n]*\n)+)", text, re.MULTILINE)
        files.append({
            "file": f.name, "returncode": done.returncode, "lines": text.count("\n"),
            "has_sorry": "declaration uses 'sorry'" in (done.stdout + done.stderr),  # the compiler's word, not a grep of comments "declares_axiom": bool(re.search(r"^\s*axiom\s", text, re.MULTILINE)),
            "print_axioms": {k: sorted(v) for k, v in audited.items()},
            "nonstandard": {k: sorted(v - STANDARD) for k, v in audited.items() if v - STANDARD},
            "theorems": theorems,
            "classes": [(name, body.strip()[:600]) for name, body in classes],
            "errors": [ln for ln in (done.stdout + done.stderr).splitlines() if "error" in ln][:8],
        })
    out["lean"] = files
    result = (sandbox / "RESULT.md").read_text() if (sandbox / "RESULT.md").exists() else ""
    m = re.search(r"verdict[^\n]*?(adheres|refuted|undecided)", result, re.IGNORECASE)
    out["result_verdict"] = m.group(1).lower() if m else None
    out["result_md_lines"] = result.count("\n")
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(json.dumps(evaluate(Path(arg).resolve()), indent=1))
