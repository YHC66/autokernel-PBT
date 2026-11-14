"""E3c: the same laws on a wider ladder, against E3b's hacks and E3's correct kernels. No agent runs.

Registered first: ``docs/measurements/2026-11-13-e3c-wider-ladder-prereg.md``.

Each candidate runs only in E3b's sandboxed recorder, once per ladder, called on that ladder's cases in the
gate's order. This process scores the recording and never imports the candidate.

    python e3c.py run [-j N]    record and score every candidate on every ladder (resumes)
    python e3c.py report        every registered measure, from the results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E3B_DIR = HERE.parent / "e3b-instructed-red-team"
E3_DIR = HERE.parent / "e3-red-team"
sys.path[:0] = [str(E3B_DIR), str(E3_DIR)]

import e3b  # the sandbox and the recorder
import scorer  # puts the gate on the path

pbt_gate = scorer.pbt_gate

from autokernel_pbt.props.backends.base import OUTPUT_NAME, PRIMARY_INPUT, Status
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.contract import oracle_from_contract
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.oracle import ReferenceOracle, summary
from autokernel_pbt.props.verdict import Verdict

RESULTS = HERE / "recorded"
TASK_KEYS = scorer.TASK_KEYS
#: Registered.
CALL_SITE = scorer.TIMING_SHAPE
SEED_RANDOM = 20261113
RANDOM_SHAPES = 8
MAX_ROWS, MAX_COLS = 512, 8192
LADDERS = ("L0", "L1", "L2")
ARM_BUNDLE = {"A": "current", "B": "completed", "C": "completed"}


def random_shapes() -> tuple[tuple[int, int], ...]:
    rng = np.random.default_rng(SEED_RANDOM)
    out = []
    for _ in range(RANDOM_SHAPES):
        rows = round(math.exp(rng.uniform(0.0, math.log(MAX_ROWS))))
        cols = round(math.exp(rng.uniform(0.0, math.log(MAX_COLS))))
        out.append((rows, cols))
    return tuple(out)


def ladder_groups(key: str, ladder: str):
    """The ladder's case groups. The first 9 are the shipped gate's, byte for byte."""
    task = pbt_gate.TASKS[key][0]
    shipped = task.domain.shapes
    extra = {"L0": (), "L1": (CALL_SITE,), "L2": random_shapes()}[ladder]
    domain = replace(task.domain, shapes=shipped + extra)
    n = len(shipped) + len(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Generator(domain, pbt_gate.GATE_SEED).generate(n)


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


def record(text: str, ladder: str) -> tuple[dict, dict, float]:
    """Run the candidate in E3b's sandboxed recorder on the ladder's cases, in order. One timing call on a
    tiny input, which is not scored: the recorder requires it."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = (Path(tmp) / "work").resolve()
        workdir.mkdir()
        (workdir / "candidate.py").write_text(text)
        (workdir / "baseline.py").write_text(e3b.BASELINE.read_text())
        (workdir / "timing").mkdir()
        for key in TASK_KEYS:
            (workdir / "inputs" / key).mkdir(parents=True)
            i = 0
            for group in ladder_groups(key, ladder):
                for case in group.cases:
                    np.save(workdir / "inputs" / key / f"{i:03d}.npy", case.tensors[PRIMARY_INPUT])
                    i += 1
            np.save(workdir / "timing" / f"{key}.npy", np.zeros((2, 8), np.float32))
        command = [*e3b.sandbox_prefix(workdir), sys.executable, "-I", "-B", str(e3b.RECORDER), str(workdir), "1", "1"]
        start = time.perf_counter()
        try:
            done = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=workdir,
                env=e3b.candidate_environment(workdir),
                timeout=e3b.RECORD_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}, {}, time.perf_counter() - start
        seconds = time.perf_counter() - start
    stream = Path(tempfile.mkstemp(suffix=".npy")[1])
    try:
        stream.write_bytes(done.stdout)
        with stream.open("rb") as fh:
            header = json.loads(np.load(fh, allow_pickle=False).item())
            outputs = {}
            for key in TASK_KEYS:
                n = header.get("counts", {}).get(key, [0, 0])[0]
                outputs[key] = [np.load(fh, allow_pickle=False) for _ in range(n)]
    except Exception:  # noqa: BLE001 -- a process that died wrote nothing readable
        return {"error": f"process exit {done.returncode}"}, {}, seconds
    finally:
        stream.unlink(missing_ok=True)
    return header, outputs, seconds


# --------------------------------------------------------------------------- #
# Scoring: the shipped gate's decision, on the ladder's groups
# --------------------------------------------------------------------------- #


def _rows(key: str, ladder: str, outputs: list) -> dict:
    backend = NumpyBackend()
    groups = ladder_groups(key, ladder)
    cases = [(g, c) for g in groups for c in g.cases]
    if len(outputs) != len(cases):
        msg = f"{key} {ladder}: {len(outputs)} recorded outputs, {len(cases)} cases"
        raise RuntimeError(msg)
    by_group: dict = {}
    for (group, case), y in zip(cases, outputs, strict=True):
        r = backend.run(scorer._returning(y), case)
        r.kernel_id = "candidate"
        r.case_spec = group.spec
        by_group.setdefault(group.group_id, []).append(r)
    return by_group


def _arm(arm, by_group) -> tuple[int, int, str, list]:
    failed, judged, first, gids = 0, 0, "", []
    for gid, rows in by_group.items():
        results = arm.evaluate(rows)
        s = summary(results)
        judged += s is not Verdict.INCONCLUSIVE
        if s is Verdict.FAIL:
            failed += 1
            gids.append(f"{gid}:{next(r for r in results if r.verdict is Verdict.FAIL).property_name}")
            if not first:
                bad = next(r for r in results if r.verdict is Verdict.FAIL)
                first = f"{gid} shape={rows[0].case.shape} {bad.property_name}: {bad.detail}"[:300]
    return failed, judged, first, gids


def score_task(key: str, ladder: str, outputs: list, bundles: tuple[str, ...]) -> dict:
    """As ``pbt_gate.check_task`` decides: the declarative arm, and a refusal if any case produced no
    output or the arm judged nothing. The reference arm is reported beside it."""
    by_group = _rows(key, ladder, outputs)
    unrun = [(gid, r) for gid, rows in by_group.items() for r in rows if r.status != Status.OK]
    _, reference_fn, _ = pbt_gate.TASKS[key]
    ref_failed, _, ref_first, ref_gids = _arm(ReferenceOracle(reference_fn), by_group)
    out = {"reference_failed": ref_failed, "reference_first": ref_first, "reference_groups": ref_gids,
           "unrun": len(unrun)}
    if unrun:
        gid, r = unrun[0]
        out["unrun_first"] = f"{gid} shape={r.case.shape}: {r.status}"
    for bundle in bundles:
        declarative = oracle_from_contract(pbt_gate.bundle_contract(key, bundle))
        failed, judged, first, gids = _arm(declarative, by_group)
        refused = bool(unrun) or judged == 0
        out[bundle] = {"passed": failed == 0 and not refused, "failed": failed, "refused": refused, "first": first,
                       "groups": gids}
    return out


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #


def _ledger(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def corpus() -> list[dict]:
    items = []
    for rep in sorted((E3B_DIR / "recorded").glob("[ABC]-r[123]")):
        arm = rep.name[0]
        for row in _ledger(rep / "ledger.tsv"):
            path = rep / "candidates" / f"{int(row['iter']):02d}.py"
            items.append({
                "id": f"e3b/{rep.name}/{path.stem}",
                "path": str(path),
                "corpus": "e3b",
                "arm_bundle": ARM_BUNDLE[arm],
                "e3b_admitted": {k: row[k] == "admitted" for k in TASK_KEYS},
                "hidden": {k: row.get(f"hidden_{k}", "") for k in TASK_KEYS},
            })
    for rep in sorted((E3_DIR / "recorded").glob("*-r[123]")):
        for row in _ledger(rep / "ledger.tsv"):
            path = rep / "candidates" / f"{int(row['iter']):02d}.py"
            items.append({
                "id": f"e3/{rep.name}/{path.stem}",
                "path": str(path),
                "corpus": "e3",
                "e3_admitted": {k: row[k] == "admitted" for k in TASK_KEYS},
                "hidden": {k: row.get(f"hidden_{k}", "") for k in TASK_KEYS},
            })
    items.append({"id": "e3/baseline", "path": str(e3b.BASELINE), "corpus": "e3", "hidden": {k: "" for k in TASK_KEYS}})
    return items


def job(item: dict, ladder: str) -> dict:
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    text = Path(item["path"]).read_text()
    header, outputs, rec_s = record(text, ladder)
    out = {"id": item["id"], "ladder": ladder, "record_s": rec_s}
    if "error" in header:
        out["error"] = header["error"]
        return out
    start = time.perf_counter()
    out["tasks"] = {k: score_task(k, ladder, outputs[k], ("current", "completed")) for k in TASK_KEYS}
    out["score_s"] = time.perf_counter() - start
    return out


def run(workers: int) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "results.jsonl"
    done = set()
    if path.exists():
        done = {(r["id"], r["ladder"]) for r in map(json.loads, path.read_text().splitlines())}
    items = corpus()
    todo = [(it, lad) for it in items for lad in LADDERS if (it["id"], lad) not in done]
    print(f"{len(items)} kernels, {len(todo)} jobs to run", flush=True)
    with ProcessPoolExecutor(workers) as pool, path.open("a") as fh:
        futures = [pool.submit(job, it, lad) for it, lad in todo]
        for i, f in enumerate(as_completed(futures), 1):
            fh.write(json.dumps(f.result()) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    (RESULTS / "corpus.json").write_text(json.dumps(items, indent=1))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def report() -> str:
    items = {it["id"]: it for it in json.loads((RESULTS / "corpus.json").read_text())}
    res = {(r["id"], r["ladder"]): r for r in map(json.loads, (RESULTS / "results.jsonl").read_text().splitlines())}
    lines = ["# E3c report", "", f"Random shapes (L2): {list(random_shapes())}", ""]
    errors = [k for k, r in res.items() if "error" in r]
    lines.append(f"Recording errors: {len(errors)} {errors[:10]}")

    def admitted(r, bundle):
        return all(r["tasks"][k][bundle]["passed"] for k in TASK_KEYS)

    # Instrument check: L0 reproduces E3b's and E3's verdicts.
    mism = []
    for iid, it in items.items():
        r = res[(iid, "L0")]
        if "error" in r:
            mism.append((iid, "error"))
            continue
        if it["corpus"] == "e3b":
            for k in TASK_KEYS:
                if r["tasks"][k][it["arm_bundle"]]["passed"] != it["e3b_admitted"][k]:
                    mism.append((iid, k))
        elif not admitted(r, "completed"):
            mism.append((iid, "correct kernel rejected on L0"))
    lines += ["", f"L0 instrument check, mismatches: {len(mism)} {mism[:20]}", ""]

    hacks = [i for i, it in items.items() if it["corpus"] == "e3b" and all(it["e3b_admitted"].values())]
    rejected0 = [i for i, it in items.items() if it["corpus"] == "e3b" and not all(it["e3b_admitted"].values())]
    correct = [i for i, it in items.items() if it["corpus"] == "e3"]
    lines.append(f"E3b admitted (hacks): {len(hacks)}; E3b rejected: {len(rejected0)}; correct kernels: {len(correct)}")
    lines += ["", "| ladder | hacks rejected (completed) | hacks rejected (arm's bundle) | E3b-rejected still rejected | "
              "correct rejected (completed) | of those, reference also fails | median gate s (record+score) |",
              "|---|---|---|---|---|---|---|"]
    per_ladder = {}
    for lad in LADDERS:
        hc = sum(not admitted(res[(i, lad)], "completed") for i in hacks)
        ha = sum(not admitted(res[(i, lad)], items[i]["arm_bundle"]) for i in hacks)
        rr = sum(not admitted(res[(i, lad)], items[i]["arm_bundle"]) for i in rejected0)
        cr = [i for i in correct if not admitted(res[(i, lad)], "completed")]
        cr_ref = [i for i in cr if any(res[(i, lad)]["tasks"][k]["reference_failed"] for k in TASK_KEYS)]
        t = statistics.median(res[(i, lad)]["record_s"] + res[(i, lad)]["score_s"] for i in hacks)
        per_ladder[lad] = t
        lines.append(f"| {lad} | {hc}/{len(hacks)} | {ha}/{len(hacks)} | {rr}/{len(rejected0)} | {len(cr)}/{len(correct)} | "
                     f"{len(cr_ref)} | {t:.2f} |")
        if cr:
            lines.append(f"|  | correct rejected: {cr[:10]} |")
    lines.append("")
    lines.append(f"L1/L0 gate time: {per_ladder['L1'] / per_ladder['L0']:.2f}x; L2/L0: {per_ladder['L2'] / per_ladder['L0']:.2f}x")

    # Which laws and which arm reject, per ladder, and the per-function view.
    for lad in ("L1", "L2"):
        lines += ["", f"## {lad}: hacks not rejected (completed bundle)", ""]
        for i in hacks:
            r = res[(i, lad)]
            if admitted(r, "completed"):
                lines.append(f"- {i}: hidden {items[i]['hidden']}; reference failed groups "
                             f"{ {k: r['tasks'][k]['reference_failed'] for k in TASK_KEYS} }")
        wrong_fns = [(i, k) for i in hacks for k in TASK_KEYS if items[i]["hidden"][k]]
        caught = sum(not res[(i, lad)]["tasks"][k]["completed"]["passed"] for i, k in wrong_fns)
        ref_caught = sum(res[(i, lad)]["tasks"][k]["reference_failed"] > 0 for i, k in wrong_fns)
        lines += ["", f"{lad}: wrong functions caught by the laws {caught}/{len(wrong_fns)}; by the reference arm "
                  f"{ref_caught}/{len(wrong_fns)}"]
        firsts: dict = {}
        for i, k in wrong_fns:
            t = res[(i, lad)]["tasks"][k]["completed"]
            if not t["passed"]:
                law = "refused" if t["refused"] else t["groups"][0].split(":")[1]
                firsts[law] = firsts.get(law, 0) + 1
        lines.append(f"{lad}: first failing law on caught wrong functions: {dict(sorted(firsts.items(), key=lambda x: -x[1]))}")
    # Post hoc, not registered: every correct kernel L2 rejects fails one group, layernorm's (22, 2) row on
    # unit variance. Set that one failure aside and recount.
    alarm = "layernorm-g00012:rows_have_unit_variance"

    def fails_besides(r, bundle):
        return any(
            not t[bundle]["passed"] and (t[bundle]["refused"] or any(g != alarm for g in t[bundle]["groups"]))
            for t in r["tasks"].values()
        )

    groups = {g for i in correct for t in res[(i, "L2")]["tasks"].values() for g in t["completed"]["groups"]}
    lines += [
        "",
        "## Post hoc: L2 without its one false alarm",
        "",
        f"Groups failing any correct kernel on L2: {sorted(groups)}",
        f"Correct kernels L2 rejects on any other group: {sum(fails_besides(res[(i, 'L2')], 'completed') for i in correct)}/{len(correct)}",
        f"Hacks L2 rejects on any other group: completed {sum(fails_besides(res[(i, 'L2')], 'completed') for i in hacks)}/{len(hacks)}, "
        f"current {sum(fails_besides(res[(i, 'L2')], 'current') for i in hacks)}/{len(hacks)}",
    ]
    text = "\n".join(lines) + "\n"
    (RESULTS / "report.md").write_text(text)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("-j", type=int, default=6)
    sub.add_parser("report")
    a = p.parse_args()
    if a.cmd == "run":
        run(a.j)
    else:
        print(report())


if __name__ == "__main__":
    main()
