"""TE1 scoring (runbook §5) and the pre-registered summary (prereg §1 'reported').

Run from the checkout root. Scores every task with score_translation, then reads the
scores back through read_translation and reports, per task: group verdicts per arm,
min/median/max ratio for reference and reference_source, and the fraction of output
elements bitwise equal between the Triton and CUDA recordings.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from autokernel_pbt.props.backends.base import OUTPUT_NAME
from autokernel_pbt.props.driver import read_translation, score_translation
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import REFERENCES, TASKS
from autokernel_pbt.props.verdict import summarize

TASK_IDS = ("relu", "softmax", "layernorm", "tolerance_sweep", "softmax_at_scale")
RATIO = re.compile(r"ratio[= ]([0-9.eE+-]+|inf|nan)")

summary: dict = {}
for task_id in TASK_IDS:
    src_dir, tgt_dir = f"runs/te1-triton-{task_id}", f"runs/te1-cuda-{task_id}"
    t0 = time.time()
    score_translation(
        task=TASKS[task_id],
        source_dir=src_dir,
        target_dir=tgt_dir,
        repo_root=Path("."),
        reference_fn=REFERENCES[task_id],
        arm_order_seed=0,
    )
    rows, arms, pairing = read_translation(tgt_dir, src_dir)
    group_of_case = {row.case.case_id: row.case.group_id for row in rows}

    per_arm: dict = {}
    for arm in arms:
        by_group: dict[str, list] = {}
        for result in arm.results:
            by_group.setdefault(result.group_id or group_of_case[result.case_id], []).append(
                result
            )
        verdicts = Counter(summarize(rs).name for rs in by_group.values())
        entry: dict = {"groups": dict(sorted(verdicts.items()))}
        if arm.arm in ("reference", "reference_source"):
            ratios = []
            for result in arm.results:
                m = RATIO.search(result.detail)
                if m:
                    ratios.append(float(m.group(1)))
            if ratios:
                entry["ratio_min_median_max"] = [
                    min(ratios),
                    statistics.median(ratios),
                    max(ratios),
                ]
                entry["n_ratios"] = len(ratios)
        failing = sorted(
            {
                (r.group_id or group_of_case[r.case_id], r.property_name, r.detail[:120])
                for r in arm.results
                if r.verdict.name == "FAIL"
            }
        )
        if failing:
            entry["failing"] = failing[:12]
        per_arm[arm.arm] = entry

    source = {r.case.case_id: r for r in ExecutionTable(src_dir).read()}
    equal = total = 0
    multi_col_unequal = []
    for row in rows:
        a = np.asarray(source[row.case.case_id].outputs[OUTPUT_NAME])
        b = np.asarray(row.outputs[OUTPUT_NAME])
        same = a.view(np.uint8).reshape(a.size, -1) == b.view(np.uint8).reshape(b.size, -1)
        same = same.all(axis=1)
        equal += int(same.sum())
        total += int(same.size)
        if row.case.shape[-1] > 1 and not same.all():
            multi_col_unequal.append(row.case.case_id)

    summary[task_id] = {
        "rows": len(rows),
        "groups": len(set(group_of_case.values())),
        "arms": per_arm,
        "bitwise_equal_fraction": equal / total,
        "elements": total,
        "multi_column_cases_not_bitwise_equal": len(multi_col_unequal),
    }
    print(f"{task_id}: scored in {time.time() - t0:.1f} s", file=sys.stderr)

Path(__file__).with_name("summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
