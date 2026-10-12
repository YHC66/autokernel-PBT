"""Aggregate the oracle-research experiment: agent reports × blind evaluation, by task and arm.

    .venv/bin/python experiments/oracle-research/summarize.py <reports.json> <eval-dir>

``reports.json`` is the list of ``{"cell": {task, arm, r, dir}, "report": {...}}`` objects the
workflow returned; ``eval-dir`` holds one ``<sandbox-name>.json`` per sandbox from
``evaluate.py``. Prints a Markdown table per task and the counts the pre-registration asks for:
claims, overclaims (claim of completeness the held-out set refutes), held-out completeness,
soundness, compiled Lean uniqueness proofs, and elapsed seconds.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main(reports_path: str, eval_dir: str) -> None:
    reports = json.loads(Path(reports_path).read_text())
    evals = {p.stem: json.loads(p.read_text()) for p in Path(eval_dir).glob("*.json")}
    rows = []
    for item in reports:
        cell, rep = item["cell"], item["report"] or {}
        name = Path(cell["dir"]).name
        ev = evals.get(name, {})
        claims = bool(rep.get("claims_complete"))
        complete = bool(ev.get("heldout_complete"))
        lean = ev.get("lean") or {}
        lean_ok = any(f["returncode"] == 0 and not f["has_sorry"] and not f["declares_axiom"] and not f["nonstandard"] for f in lean.get("files", []))
        rows.append({
            "name": name, "task": cell["task"], "arm": cell["arm"], "r": cell["r"],
            "claims": claims, "heldout_complete": complete, "overclaim": claims and not complete,
            "underclaim": (not claims) and complete,
            "sound": bool(ev.get("sound")), "admitted_after": ev.get("admitted_after", []),
            "n_admitted_families": len(rep.get("admitted_families") or []),
            "n_laws": len(rep.get("added_laws") or []), "laws": [l.get("name") for l in rep.get("added_laws") or []],
            "lean_ok": lean_ok, "lean_files": [f["file"] for f in lean.get("files", [])],
            "elapsed": rep.get("elapsed_seconds"), "outside": rep.get("outside_paths_read") or [],
            "remark": (rep.get("remarks_on_formal_ambiguity") or "")[:160],
            "assumptions": rep.get("primitive_assumptions") or [],
        })
    for task in ("softmax", "layernorm"):
        print(f"\n## {task}\n")
        print("| cell | claims complete | held-out complete | overclaim | sound | admitted after | #families | laws | Lean ok | elapsed s | outside reads |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted((r for r in rows if r["task"] == task), key=lambda r: (r["arm"], r["r"])):
            print(f"| {r['name']} | {r['claims']} | {r['heldout_complete']} | {'**yes**' if r['overclaim'] else 'no'} | {r['sound']} | {', '.join(r['admitted_after']) or '—'} | {r['n_admitted_families']} | {', '.join(r['laws'])} | {r['lean_ok'] if r['arm']=='F' else 'n/a'} | {r['elapsed']} | {len(r['outside'])} |")
    print("\n## Counts by arm\n")
    print("| task | arm | n | claims | held-out complete | overclaims | sound | Lean ok | median elapsed s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for task in ("softmax", "layernorm"):
        for arm in ("T", "F"):
            sub = [r for r in rows if r["task"] == task and r["arm"] == arm]
            if not sub:
                continue
            el = [r["elapsed"] for r in sub if isinstance(r["elapsed"], (int, float))]
            print(f"| {task} | {arm} | {len(sub)} | {sum(r['claims'] for r in sub)} | {sum(r['heldout_complete'] for r in sub)} | {sum(r['overclaim'] for r in sub)} | {sum(r['sound'] for r in sub)} | {sum(r['lean_ok'] for r in sub) if arm=='F' else 'n/a'} | {statistics.median(el) if el else '—'} |")
    print("\n## Remarks on formal ambiguity (F arm)\n")
    for r in rows:
        if r["arm"] == "F":
            print(f"- {r['name']}: {r['remark'] or '(none)'}")
            for a in r["assumptions"]:
                print(f"    - {a}")
    Path(eval_dir, "summary_rows.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
