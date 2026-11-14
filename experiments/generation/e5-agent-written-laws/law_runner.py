"""Runs one agent-written ``laws.py`` on one kernel's recorded cases. Sandboxed by ``e5.py``.

    python -I -B law_runner.py WORKDIR

WORKDIR holds ``laws.py`` and ``cases.npz``: ``x_NNN`` and ``y_NNN`` per case in the gate's order, and ``meta``,
a JSON list of [group, is_base] per case. In each group the base case comes first, then its ``shift_rows``
partner, whose rows are the base's rows plus one constant per row. Prints one JSON line: per group, the laws that
failed, each with its detail. A law that raises or returns something unreadable has failed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

ANSWER = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)

import numpy as np


def _verdict(fn, *args) -> tuple[bool, str]:
    try:
        out = fn(*args)
        ok, detail = out
        return bool(ok), str(detail)[:200]
    except BaseException as exc:  # noqa: BLE001 -- the law's failure is the record
        return False, f"raised {type(exc).__name__}: {str(exc)[:150]}"


def main() -> None:
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    workdir = Path(sys.argv[1])
    out: dict = {"groups": {}}
    try:
        spec = importlib.util.spec_from_file_location("laws", workdir / "laws.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        case_laws = dict(getattr(module, "CASE_LAWS", {}) or {})
        shift_laws = dict(getattr(module, "SHIFT_LAWS", {}) or {})
    except BaseException as exc:  # noqa: BLE001
        out["load_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        ANSWER.write(json.dumps(out) + "\n")
        return
    out["laws"] = sorted(case_laws) + sorted(shift_laws)
    data = np.load(workdir / "cases.npz", allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    base: dict = {}
    for i, (group, is_base) in enumerate(meta):
        x, y = data[f"x_{i:03d}"], data[f"y_{i:03d}"]
        x.setflags(write=False)
        failed = out["groups"].setdefault(str(group), {})
        for name, law in case_laws.items():
            ok, detail = _verdict(law, x.copy(), y.copy())
            if not ok and name not in failed:
                failed[name] = detail
        if is_base:
            base[group] = (x, y)
        elif group in base:
            x0, y0 = base[group]
            for name, law in shift_laws.items():
                ok, detail = _verdict(law, x0.copy(), y0.copy(), x.copy(), y.copy())
                if not ok and name not in failed:
                    failed[name] = detail
    ANSWER.write(json.dumps(out) + "\n")
    ANSWER.flush()


if __name__ == "__main__":
    main()
