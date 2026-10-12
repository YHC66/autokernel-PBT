"""Score one agent sandbox, blind to its arm: soundness, held-out completeness, Lean status.

    .venv/bin/python experiments/verification/oracle-research/evaluate.py <sandbox-dir> [...]

Loads the sandbox's own ``gate.py`` (hence its ``laws.py`` and vendored property layer) by path,
so the completed bundle under test is exactly the one the agent left behind, and runs every
held-out kernel through it. A held-out kernel counts as *rejected* when the declarative arm
fails every group the reference arm fails; *admitted* when the declarative arm fails no group;
*partial* otherwise. Soundness is the declarative arm failing zero groups on the reference and
on each correct rewrite. Prints one JSON object per sandbox.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import heldout

LEAN = Path.home() / ".elan/toolchains/leanprover--lean4---v4.33.0/bin/lean"
_AXIOMS = re.compile(r"'(?P<name>\S+)' depends on axioms: \[(?P<axioms>[^\]]*)\]")
STANDARD = {"propext", "Classical.choice", "Quot.sound"}


def _load_gate(sandbox: Path):
    # Each sandbox has its own vendored package; isolate the import by resetting sys.modules.
    for name in [m for m in sys.modules if m.startswith("autokernel_pbt") or m in {"laws", "gate"}]:
        del sys.modules[name]
    sys.path.insert(0, str(sandbox))
    spec = importlib.util.spec_from_file_location("gate", sandbox / "gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    sys.path.pop(0)
    return gate


def _verdict(res: dict) -> tuple[str, dict]:
    d_failed, total, d_first = res["declarative"]
    r_failed, _, _ = res["reference"]
    if d_failed == 0:
        v = "admitted"
    elif d_failed >= r_failed:
        v = "rejected"
    else:
        v = "partial"
    return v, {"declarative": f"{d_failed}/{total}", "reference": f"{r_failed}/{total}", "first": d_first}


def _lean(sandbox: Path) -> dict:
    # Anywhere in the sandbox, not only proof/: tests-only agents were not given proof/ and
    # several wrote Lean files of their own elsewhere. Scratch drafts are excluded.
    files = sorted(
        p for p in sandbox.rglob("*.lean")
        if p.name != "Skeleton.lean" and "scratch" not in p.relative_to(sandbox).parts
        and "autokernel_pbt" not in p.relative_to(sandbox).parts
    )
    out = []
    for f in files:
        done = subprocess.run([str(LEAN), f.name], cwd=f.parent, capture_output=True, text=True, timeout=600, check=False)
        text = f.read_text()
        audited = {m["name"]: set(filter(None, map(str.strip, m["axioms"].split(",")))) for m in _AXIOMS.finditer(done.stdout)}
        out.append({
            "file": str(f.relative_to(sandbox)),
            "returncode": done.returncode,
            "has_sorry": "declaration uses 'sorry'" in (done.stdout + done.stderr),  # the compiler's word, not a grep of comments
            "declares_axiom": bool(re.search(r"^\s*axiom\s", text, re.MULTILINE)),
            "print_axioms": {k: sorted(v) for k, v in audited.items()},
            "nonstandard": {k: sorted(v - STANDARD) for k, v in audited.items() if v - STANDARD},
            "errors": [ln for ln in (done.stdout + done.stderr).splitlines() if "error" in ln][:10],
            "lines": text.count("\n"),
        })
    return {"files": out}


def evaluate(sandbox: Path) -> dict:
    task = (sandbox / "TASK").read_text().strip()
    gate = _load_gate(sandbox)
    report: dict = {"sandbox": str(sandbox), "task": task}
    # the agent's completed bundle: what laws.py adds
    import laws

    report["added_laws"] = [p.name for p in (*laws.EXTRA_CASE_PROPERTIES, *laws.EXTRA_GROUP_PROPERTIES)]
    # soundness
    sound = {}
    ref_res = gate.check(gate.reference(), True)
    sound["reference"] = _verdict(ref_res)[1]
    for name, k in heldout.CORRECT[task].items():
        sound[name] = _verdict(gate.check(k, True))[1]
    report["soundness"] = sound
    report["sound"] = all(v["declarative"].startswith("0/") for v in sound.values())
    # held-out completeness, old bundle and completed bundle
    held = {}
    for name, k in heldout.WRONG[task].items():
        old = _verdict(gate.check(k, False))
        new = _verdict(gate.check(k, True))
        held[name] = {"old_bundle": old[0], "completed": new[0], "detail": new[1]}
    report["heldout"] = held
    report["heldout_complete"] = all(v["completed"] == "rejected" for v in held.values())
    report["admitted_after"] = sorted(n for n, v in held.items() if v["completed"] != "rejected")
    # the agent's own admitted kernels, under old and completed bundle
    own = {}
    for p in sorted((sandbox / "admitted").glob("*.py")) if (sandbox / "admitted").exists() else []:
        if p.name.startswith("_"):
            continue  # helper modules (_family.py, __init__.py), not kernels
        try:
            k = gate.load_kernel(p)
            own[p.name] = {"old_bundle": _verdict(gate.check(k, False))[0], "completed": _verdict(gate.check(k, True))[0]}
        except Exception as exc:  # noqa: BLE001
            own[p.name] = {"error": f"{type(exc).__name__}: {exc}"}
    report["own_admitted"] = own
    report["lean"] = _lean(sandbox)
    report["result_md_present"] = (sandbox / "RESULT.md").exists()
    return report


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    for arg in sys.argv[1:]:
        print(json.dumps(evaluate(Path(arg).resolve()), indent=1))
