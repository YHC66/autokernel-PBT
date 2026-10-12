"""Post-hoc, NOT pre-registered: every agent's completed bundle against every agent's admitted
kernels for the same task.

    .venv/bin/python experiments/verification/oracle-research/cross_evaluate.py <task> <sandbox-dir> [...]

The pre-registered completeness measure is the fixed held-out set in ``heldout.py``. This script
asks a second question the agents' own output makes possible: pooling the wrong kernels all
agents exhibited (loaded from each sandbox's ``admitted/``), which completed bundles reject all
of them? A kernel is included only if the pooled reference arm rejects it on at least one group
(so correct rewrites misfiled as "admitted" are dropped) and is evaluated under each sandbox's
own ``gate.py`` + ``laws.py``. Prints a matrix: rows = bundles, columns = kernel sources.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import _load_gate, _verdict


def _kernels_from(sandbox: Path, gate) -> dict[str, object]:
    out = {}
    for p in sorted((sandbox / "admitted").glob("*.py")) if (sandbox / "admitted").exists() else []:
        if p.name.startswith("_"):
            continue
        try:
            out[f"{sandbox.name}/{p.stem}"] = gate.load_kernel(p)
        except BaseException as exc:  # noqa: BLE001 - agent code; a file that cannot load is skipped and named
            out[f"{sandbox.name}/{p.stem}"] = exc
    return out


def main(task: str, dirs: list[str]) -> None:
    sandboxes = [Path(d).resolve() for d in dirs]
    sandboxes = [s for s in sandboxes if (s / "TASK").read_text().strip() == task]
    # Pool kernels, each loaded through its own sandbox's loader (same TASK name, so any gate works).
    pooled: dict[str, object] = {}
    for s in sandboxes:
        gate = _load_gate(s)
        pooled.update(_kernels_from(s, gate))
    loadable = {k: v for k, v in pooled.items() if not isinstance(v, BaseException)}
    skipped = sorted(k for k, v in pooled.items() if isinstance(v, BaseException))
    # Reference-arm filter, under the first sandbox's gate (the reference arm is identical everywhere).
    gate0 = _load_gate(sandboxes[0])
    wrong = {}
    for name, k in loadable.items():
        try:
            res = gate0.check(k, False)
        except BaseException as exc:  # noqa: BLE001
            skipped.append(f"{name} ({type(exc).__name__})")
            continue
        if res["reference"][0] > 0 and res["declarative"][0] == 0:
            wrong[name] = k
    print(f"task={task}: {len(pooled)} admitted files pooled from {len(sandboxes)} sandboxes; "
          f"{len(wrong)} are wrong-and-admitted under the old bundle; skipped {len(skipped)}: {skipped}")
    matrix = {}
    for s in sandboxes:
        gate = _load_gate(s)
        row = {}
        for name, k in wrong.items():
            try:
                row[name] = _verdict(gate.check(k, True))[0]
            except BaseException as exc:  # noqa: BLE001
                row[name] = f"error:{type(exc).__name__}"
        matrix[s.name] = row
    print("\n| bundle from | rejects all pooled | admitted (pooled) |")
    print("|---|---|---|")
    for b, row in matrix.items():
        adm = sorted(n for n, v in row.items() if v != "rejected")
        print(f"| {b} | {not adm} | {len(adm)}: {', '.join(adm)[:300]} |")
    Path(HERE, f"cross_{task}.json").write_text(json.dumps({"wrong": sorted(wrong), "matrix": matrix, "skipped": skipped}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
