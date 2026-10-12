"""Grep the workflow's agent transcripts for reads outside the sandbox: the protocol's self-report
check. Prints, per agent, the distinct outside paths that appear in tool inputs.

    .venv/bin/python experiments/oracle-research/audit_transcripts.py <workflow-transcript-dir>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_MARKERS = ("autokernel research/repo", "autokernel-research/repo", "/repo/", "docs/measurements", "saboteurs.py", "proofs/softmax-bundle", "paper/")


def main(tdir: str) -> None:
    for f in sorted(Path(tdir).glob("agent-*.jsonl")):
        label = ""
        meta = f.with_suffix(".meta.json")
        if meta.exists():
            try:
                label = json.loads(meta.read_text()).get("label", "")
            except Exception:  # noqa: BLE001
                label = ""
        hits = set()
        for line in f.read_text(errors="ignore").splitlines():
            if not any(m in line for m in REPO_MARKERS):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                d = {"unparsed": line}  # a torn line still gets scanned as text below
            # only tool INPUTS count as reads; the system prompt and our own prompt may mention paths
            blob = json.dumps(d.get("message", d)) if isinstance(d, dict) else line
            for m in REPO_MARKERS:
                for found in re.findall(r"[^\s\"']*" + re.escape(m) + r"[^\s\"']*", blob):
                    if "tool_use" in blob or "command" in blob:
                        hits.add(found[:160])
        print(f"{f.name} [{label}]: {len(hits)} suspicious tool-input path(s)")
        for h in sorted(hits)[:10]:
            print(f"    {h}")


if __name__ == "__main__":
    main(sys.argv[1])
