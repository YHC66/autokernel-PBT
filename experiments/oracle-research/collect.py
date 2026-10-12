"""Copy each sandbox's durable artifacts into the repository, so the record outlives /private/tmp.

    .venv/bin/python experiments/oracle-research/collect.py <dest-dir> <sandbox-dir> [...]

Kept per sandbox: README.md (the protocol text the agent saw), TASK/SANDBOX, RESULT.md, laws.py,
admitted/, correct_variants/, extras/, every agent-written *.py and *.txt/*.md at the top level,
and proof/*.lean plus proof/README.md. Dropped: the vendored property layer, the venv symlink,
__pycache__, proof/scratch, and the template's Skeleton.lean (identical everywhere; one copy is
kept at <dest-dir>/_template/).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

DROP_DIRS = {"autokernel_pbt", "__pycache__", "venv", "scratch", ".pytest_cache"}


def collect(dest: Path, sandbox: Path) -> int:
    out = dest / sandbox.name
    n = 0
    for p in sandbox.rglob("*"):
        rel = p.relative_to(sandbox)
        if any(part in DROP_DIRS for part in rel.parts) or p.is_symlink():
            continue
        if p.is_dir():
            continue
        if rel.parts[0] == "proof" and p.name == "Skeleton.lean":
            continue
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        n += 1
    return n


if __name__ == "__main__":
    dest = Path(sys.argv[1]).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for arg in sys.argv[2:]:
        sb = Path(arg).resolve()
        n = collect(dest, sb)
        total += n
        print(f"{sb.name}: {n} files")
        skel = sb / "proof" / "Skeleton.lean"
        if skel.exists() and not (dest / "_template" / "Skeleton.lean").exists():
            (dest / "_template").mkdir(exist_ok=True)
            shutil.copy2(skel, dest / "_template" / "Skeleton.lean")
    print(f"total {total} files into {dest}")
