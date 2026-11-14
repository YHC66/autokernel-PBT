"""E5: an agent writes a task's laws itself, with (TF) or without (T) an obligation to prove in Lean that they
determine the function. Registered first: ``docs/measurements/2026-11-13-e5-agent-written-laws-prereg.md``.

    python e5.py control               the positive and negative controls: no model
    python e5.py run ARM TASK R        one replicate (ARM T or TF, TASK softmax or layernorm, R r1..r5 or pilot)
    python e5.py all                   every counted replicate, rotating; re-running resumes
    python e5.py score                 score every final bundle, and the shipped ones, on the corpus
    python e5.py report                every registered measure, from the scores

The agent session is E3b's (``claude -p``, Read/Edit/Write, no shell). Lean runs sandboxed, and so does every
agent-written law: this process never imports one.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "e3c-wider-ladder")]

import e3c  # the ladders, the recorder and the corpus; puts E3b, E3 and the gate on the path

e3b, scorer, pbt_gate = e3c.e3b, e3c.scorer, e3c.pbt_gate

from autokernel_pbt.props.backends.base import PRIMARY_INPUT
from autokernel_pbt.props.case import BASE_RELATION

RESULTS = HERE / "recorded"
LEAN_DIR = HERE / "lean"
LAW_RUNNER = HERE / "law_runner.py"
TOOLCHAIN = Path.home() / ".elan" / "toolchains" / "leanprover--lean4---v4.33.0"
LEAN = TOOLCHAIN / "bin" / "lean"
#: Registered.
ARMS = ("T", "TF")
TASKS = ("softmax", "layernorm")
REPLICATES = 5
ROUNDS = {"T": 1, "TF": 4}
LADDER = "L1"
SHIPPED_GROUPS = 9  # L0 is L1's first 9 groups
STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
LEAN_TIMEOUT_S = 180
LAW_TIMEOUT_S = 300
LEDGER_COLUMNS = (
    "round", "laws_py", "laws_lean", "closed", "axioms", "lean_exit", "seconds", "cost_usd", "usage",
    "outside_attempts", "outside_successes", "note",
)

TASK_TEXT = {
    "softmax": {
        "NAME": "softmax",
        "FUNCTION": (
            "`softmax(x)`: row-wise softmax over the last axis of a 2-D float32 array `x`,\n"
            "`y[i, j] = exp(x[i, j]) / sum_k exp(x[i, k])`. The output has `x`'s shape and dtype."
        ),
        "PRIMITIVE": "`exp`",
        "TOLERANCE_NOTE": "",
        "HYPOTHESIS": "(hne : xs ≠ [])",
        "HYPOTHESIS_FORALL": "xs ≠ []",
        "TARGET": "softmaxRow",
        "CLASSES": "[HasExp F] [ExpLaws F]",
    },
    "layernorm": {
        "NAME": "layernorm",
        "FUNCTION": (
            "`layernorm(x)`: row-wise layer normalization over the last axis of a 2-D float32 array `x`, with no\n"
            "affine parameters: `y[i, j] = (x[i, j] - m_i) / sqrt(v_i + 1e-5)`, where `m_i` is row i's mean and\n"
            "`v_i` its population variance (divide by the row length). The output has `x`'s shape and dtype."
        ),
        "PRIMITIVE": "the mean, the variance and `sqrt`",
        "TOLERANCE_NOTE": (
            "The `1e-5` inside the square root is part of the function, and its effect is largest on rows whose "
            "variance is small."
        ),
        "HYPOTHESIS": "(hv : 0 < var xs)",
        "HYPOTHESIS_FORALL": "0 < E5.var xs",
        "TARGET": "lnRow",
        "CLASSES": "[HasSqrt F] [SqrtLaws F]",
    },
}


# --------------------------------------------------------------------------- #
# The program the agent reads
# --------------------------------------------------------------------------- #


def _fill(template: str, values: dict) -> str:
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def program(arm: str, task: str) -> str:
    values = dict(TASK_TEXT[task])
    if arm == "TF":
        values["FORMAL"] = _fill((HERE / "formal.md").read_text(), values)
        values["FORMAL_FILE"] = " and `Laws.lean`"
    else:
        values["FORMAL"] = ""
        values["FORMAL_FILE"] = ""
    text = _fill((HERE / "program.md").read_text(), values)
    if re.search(r"\{[A-Z_]+\}", text):
        msg = f"unfilled placeholder in the program: {re.findall(r'{[A-Z_]+}', text)}"
        raise RuntimeError(msg)
    return text


# --------------------------------------------------------------------------- #
# Lean
# --------------------------------------------------------------------------- #


def lean_sandbox(workdir: Path) -> list[str]:
    """No network, no writes outside ``workdir``, no reads in the home directory except the toolchain: a Lean
    file can run code while it is elaborated."""
    tool = shutil.which("sandbox-exec")
    if tool is None:
        msg = "refusing to compile a model's Lean without sandbox-exec"
        raise RuntimeError(msg)
    profile = (
        "(version 1)(allow default)(deny network*)(deny file-write*)"
        f'(allow file-write* (subpath "{workdir.resolve()}"))'
        '(allow file-write* (literal "/dev/null"))'
        f'(deny file-read* (subpath "{Path.home().resolve()}"))'
        f'(allow file-read* (subpath "{TOOLCHAIN.resolve()}"))'
        # Lean resolves its own path, which stats the directories above the toolchain: metadata only.
        "(allow file-read-metadata)"
    )
    return [tool, "-p", profile]


def lean_check(body: str, task: str) -> dict:
    """Compile preamble + body + obligation. Closed only with exit 0, no error, and standard axioms only."""
    source = (
        (LEAN_DIR / f"{task}_preamble.lean").read_text()
        + "\n-- ===== the agent's Laws.lean =====\n"
        + body
        + "\n-- ===== end of the agent's Laws.lean =====\n"
        + (LEAN_DIR / f"{task}_suffix.lean").read_text()
    )
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp).resolve()
        (work / "Check.lean").write_text(source)
        try:
            done = subprocess.run(
                [*lean_sandbox(work), str(LEAN), "Check.lean"],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=LEAN_TIMEOUT_S,
                env={"PATH": "/usr/bin:/bin", "HOME": str(work), "TMPDIR": str(work)},
                check=False,
            )
            log, code = done.stdout + done.stderr, done.returncode
        except subprocess.TimeoutExpired:
            log, code = f"timeout after {LEAN_TIMEOUT_S} s", -1
    axioms: list[str] | None = None
    m = re.search(r"'e5_obligation' depends on axioms: \[([^\]]*)\]", log)
    if m:
        axioms = [a.strip() for a in m.group(1).split(",") if a.strip()]
    elif "'e5_obligation' does not depend on any axioms" in log:
        axioms = []
    closed = code == 0 and axioms is not None and set(axioms) <= STANDARD_AXIOMS and "error" not in log
    return {"closed": closed, "axioms": axioms, "exit": code, "log": log}


# --------------------------------------------------------------------------- #
# A replicate
# --------------------------------------------------------------------------- #


def replicate_dir(arm: str, task: str, rep: str) -> Path:
    return RESULTS / f"{arm}-{task}-{rep}"


def read_ledger(directory: Path) -> list[dict]:
    path = directory / "ledger.tsv"
    if not path.is_file():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _append_ledger(directory: Path, row: dict) -> None:
    path = directory / "ledger.tsv"
    new = not path.is_file()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLUMNS, delimiter="\t", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def history(directory: Path, task: str = "") -> str:
    lines = ["# History", ""]
    if task:
        start = len((LEAN_DIR / f"{task}_preamble.lean").read_text().splitlines()) + 2
        lines += [f"In the compiler output below, your `Laws.lean` starts at line {start} of the combined file.", ""]
    rows = read_ledger(directory)
    if not rows:
        return "\n".join([*lines, "This is the first round.", ""])
    for row in rows:
        r = int(row["round"])
        log = (directory / f"round{r}" / "lean.log").read_text() if (directory / f"round{r}" / "lean.log").is_file() else ""
        lines += [
            f"## Round {r}",
            "",
            f"- `laws.py` written: {row['laws_py']}; `Laws.lean` written: {row['laws_lean']}",
            f"- Proof closed: {row['closed']}. Axioms: {row['axioms'] or 'none reported'}. Lean exit code: {row['lean_exit']}",
            "",
            "Compiler output (the file is the preamble, then your `Laws.lean`, then the obligation; line numbers "
            "count from the top of that combined file):",
            "",
            "```",
            log[-6000:].strip() or "(none)",
            "```",
            "",
        ]
    return "\n".join(lines)


def prepare(workspace: Path, arm: str, task: str, directory: Path) -> None:
    (workspace / "PROGRAM.md").write_text(program(arm, task))
    (workspace / "HISTORY.md").write_text(history(directory, task if arm == "TF" else ""))
    if arm == "TF":
        (workspace / "PREAMBLE.lean").write_text((LEAN_DIR / f"{task}_preamble.lean").read_text())


def run_replicate(arm: str, task: str, rep: str, agent=None) -> None:
    agent = agent or e3b.claude_agent
    directory = replicate_dir(arm, task, rep)
    directory.mkdir(parents=True, exist_ok=True)
    rows = read_ledger(directory)
    if any(r["closed"] == "True" for r in rows) or len(rows) >= ROUNDS[arm]:
        print(f"{directory.name}: done")
        return
    with tempfile.TemporaryDirectory() as tmp:
        workspace = (Path(tmp) / "work").resolve()
        workspace.mkdir()
        last = directory / f"round{len(rows) - 1}" if rows else None
        for name in ("laws.py", "Laws.lean", "NOTE.txt"):
            if last and (last / name).is_file():
                shutil.copy(last / name, workspace / name)
        for r in range(len(rows), ROUNDS[arm]):
            prepare(workspace, arm, task, directory)
            out = directory / f"round{r}"
            out.mkdir(exist_ok=True)
            session = agent(workspace, out / "transcript.jsonl")
            if session.successes:
                (directory / e3b.STOPPED).write_text("\n".join(session.successes))
                msg = f"{directory.name} round {r}: a tool call outside the workspace succeeded: {session.successes}"
                raise e3b.RunStopped(msg)
            for name in ("laws.py", "Laws.lean", "NOTE.txt"):
                if (workspace / name).is_file():
                    shutil.copy(workspace / name, out / name)
            check = {"closed": "", "axioms": None, "exit": ""}
            if arm == "TF":
                body = (workspace / "Laws.lean").read_text() if (workspace / "Laws.lean").is_file() else ""
                check = lean_check(body, task)
                (out / "lean.log").write_text(check["log"])
            row = {
                "round": r,
                "laws_py": (workspace / "laws.py").is_file(),
                "laws_lean": (workspace / "Laws.lean").is_file(),
                "closed": check["closed"],
                "axioms": ",".join(check["axioms"]) if check["axioms"] is not None else "",
                "lean_exit": check["exit"],
                "seconds": round(session.seconds, 1),
                "cost_usd": session.cost_usd if session.cost_usd is not None else "",
                "usage": session.usage,
                "outside_attempts": len(session.attempts),
                "outside_successes": len(session.successes),
                "note": ((workspace / "NOTE.txt").read_text().strip()[:300] if (workspace / "NOTE.txt").is_file() else ""),
            }
            _append_ledger(directory, row)
            print(f"{directory.name} round {r}: closed={row['closed']} axioms={row['axioms']} ${row['cost_usd']}", flush=True)
            if check["closed"] is True:
                break


def final_laws(directory: Path) -> Path | None:
    rows = read_ledger(directory)
    if not rows:
        return None
    path = directory / f"round{rows[-1]['round']}" / "laws.py"
    return path if path.is_file() else None


# --------------------------------------------------------------------------- #
# Controls: no model
# --------------------------------------------------------------------------- #


def control() -> dict:
    """The Lean check on known files. Expected: the reference solution closes; a `sorry`, an extra axiom, a
    file that fails to compile and one that stops early do not. A `Laws` that restates the function closes too:
    that is reference-in-disguise, which the hand audit catches, not Lean."""
    out = {}
    proof = re.compile(r"(theorem uniqueness[\s\S]*?:= by)[\s\S]*$")
    for task in TASKS:
        good = (LEAN_DIR / f"{task}_control.lean").read_text()
        target, hyp = TASK_TEXT[task]["TARGET"], TASK_TEXT[task]["HYPOTHESIS_FORALL"].replace("E5.", "")
        cases = {
            "reference solution (expect True)": good,
            "sorry (expect False)": proof.sub(r"\1\n  sorry\n", good),
            "extra axiom (expect False)": proof.sub(r"\1\n  exact cheat.elim\n", good).replace(
                "theorem uniqueness", "axiom cheat : False\n\ntheorem uniqueness", 1),
            "does not compile (expect False)": good + "\ntheorem broken : (1 : Nat) = 2 := rfl\n",
            "stops early (expect False)": good + "\n#exit\n",
            "restates the function (expect True; the audit's job)": good.split("def Laws")[0]
            + f"def Laws (f : List F → List F) : Prop := ∀ xs : List F, {hyp} → f xs = {target} xs\n\n"
            + f"theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) {TASK_TEXT[task]['HYPOTHESIS']} :\n"
            + f"    f xs = {target} xs := h xs {TASK_TEXT[task]['HYPOTHESIS'].split()[0][1:]}\n",
        }
        for name, body in cases.items():
            out[f"{task}: {name}"] = lean_check(body, task)["closed"]
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

FAMILY_STOCK_SOFTMAX = (
    "def softmax(x):\n"
    "    e = np.exp(x - np.max(x, axis=-1, keepdims=True))\n"
    "    return e / np.sum(e, axis=-1, keepdims=True)\n"
)
FAMILY_STOCK_LAYERNORM = (
    "def layernorm(x):\n"
    "    centered = x - np.mean(x, axis=-1, keepdims=True)\n"
    "    variance = np.mean(centered * centered, axis=-1, keepdims=True)\n"
    "    return centered / np.sqrt(variance + np.float32(1e-5))\n"
)
#: The 10 families the current bundle admits (Table gap of the paper), as kernel files.
FAMILIES = {
    "softmax": {
        "beta_2": "z = 2 * x\n    e = np.exp(z - np.max(z, axis=-1, keepdims=True))\n    return e / np.sum(e, axis=-1, keepdims=True)",
        "beta_0": "z = 0 * x\n    e = np.exp(z - np.max(z, axis=-1, keepdims=True))\n    return e / np.sum(e, axis=-1, keepdims=True)",
        "beta_minus_1": "z = -x\n    e = np.exp(z - np.max(z, axis=-1, keepdims=True))\n    return e / np.sum(e, axis=-1, keepdims=True)",
        "uniform_fill": "return np.full(x.shape, 1.0 / x.shape[-1], dtype=x.dtype)",
        "rows_rolled": "return np.ascontiguousarray(np.roll(_sm(x), 1, axis=0))",
        "columns_reversed": "return np.ascontiguousarray(_sm(x)[:, ::-1])",
    },
    "layernorm": {
        "negated": "return -_ln(x)",
        "columns_reversed": "return np.ascontiguousarray(_ln(x)[:, ::-1])",
        "rows_sorted": "return np.sort(_ln(x), axis=-1)",
        "rows_rolled": "return np.ascontiguousarray(np.roll(_ln(x), 1, axis=0))",
    },
}


def family_source(task: str, name: str) -> str:
    helpers = FAMILY_STOCK_SOFTMAX.replace("def softmax", "def _sm") + FAMILY_STOCK_LAYERNORM.replace("def layernorm", "def _ln")
    body = FAMILIES[task][name]
    wrong = f"def {task}(x):\n    {body}\n"
    other = FAMILY_STOCK_LAYERNORM if task == "softmax" else FAMILY_STOCK_SOFTMAX
    return f'"""Family {task}/{name}: wrong on {task} only."""\n\nimport numpy as np\n\n{helpers}\n{wrong}\n{other}'


def scoring_corpus() -> list[dict]:
    """Every kernel, and per task whether its function is wrong (True), correct (False), or unknown (None:
    an E3b candidate the gate rejected, which no hidden checker scored)."""
    items = []
    for task, names in FAMILIES.items():
        for name in names:
            items.append({
                "id": f"family/{task}/{name}", "source": family_source(task, name), "kind": "family",
                "wrong": {t: t == task for t in TASKS},
            })
    for it in e3c.corpus():
        src = Path(it["path"]).read_text()
        if it["corpus"] == "e3b":
            admitted = all(it["e3b_admitted"].values())
            wrong = {t: bool(it["hidden"][t]) if admitted else None for t in TASKS}
            items.append({"id": it["id"], "source": src, "kind": "e3b", "wrong": wrong})
        else:
            items.append({"id": it["id"], "source": src, "kind": "correct", "wrong": {t: False for t in TASKS}})
    fast = e3b.e3._fast_exp_kernel()
    items.append({"id": "fast_exp", "source": fast, "kind": "fastexp", "wrong": {t: None for t in TASKS}})
    return items


def bundles() -> dict[str, tuple[str, Path]]:
    """Every final agent bundle: name -> (task, laws.py)."""
    out = {}
    for d in sorted(RESULTS.glob("T*-*-r*")):
        arm, task, rep = d.name.split("-")
        path = final_laws(d)
        if path is not None and rep != "pilot":
            out[d.name] = (task, path)
    return out


def _cases(task: str, outputs: list) -> tuple[dict, list]:
    groups = e3c.ladder_groups(task, LADDER)
    arrays, meta, i = {}, [], 0
    for g_index, group in enumerate(groups):
        for case in group.cases:
            arrays[f"x_{i:03d}"] = case.tensors[PRIMARY_INPUT]
            arrays[f"y_{i:03d}"] = outputs[i]
            meta.append([g_index, case.relation == BASE_RELATION])
            i += 1
    return arrays, meta


def run_laws(laws: Path, arrays: dict, meta: list) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        work = (Path(tmp) / "work").resolve()
        work.mkdir()
        shutil.copy(laws, work / "laws.py")
        # The sandbox hides the home directory, so the runner goes in with the data.
        shutil.copy(LAW_RUNNER, work / "law_runner.py")
        np.savez(work / "cases.npz", meta=np.array(json.dumps(meta)), **arrays)
        try:
            done = subprocess.run(
                [*e3b.sandbox_prefix(work), sys.executable, "-I", "-B", str(work / "law_runner.py"), str(work)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=work,
                env=e3b.candidate_environment(work), timeout=LAW_TIMEOUT_S, check=False,
            )
            return json.loads(done.stdout.splitlines()[-1])
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        except Exception as exc:  # noqa: BLE001 -- the runner died: the bundle's failure, recorded
            return {"error": f"runner: {type(exc).__name__}"}


def score_kernel(item: dict, agent_bundles: dict) -> dict:
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    header, outputs, _ = e3c.record(item["source"], LADDER)
    out = {"id": item["id"], "tasks": {}}
    if "error" in header:
        out["error"] = header["error"]
        return out
    for task in TASKS:
        ys = outputs[task]
        refused = any(scorer._is_reason(y) for y in ys)
        shipped = e3c.score_task(task, LADDER, ys, ("current", "completed"))
        t = {"refused": refused, "shipped": {}, "agent": {}}
        for b in ("current", "completed"):
            gids = [int(g.split(":")[0].rsplit("g", 1)[1]) for g in shipped[b]["groups"]]
            t["shipped"][b] = {"L1": sorted(set(gids)), "refused": shipped[b]["refused"]}
        if not refused:
            arrays, meta = _cases(task, ys)
            for name, (btask, path) in agent_bundles.items():
                if btask != task:
                    continue
                r = run_laws(path, arrays, meta)
                if "groups" in r:
                    failing = {int(g): v for g, v in r["groups"].items() if v}
                    r = {"L1": sorted(failing), "laws": sorted({k for v in failing.values() for k in v}),
                         "first": next((f"g{g}: {next(iter(v.items()))}" for g, v in sorted(failing.items())), ""),
                         "load_error": r.get("load_error", "")}
                t["agent"][name] = r
        out["tasks"][task] = t
    return out


def score(workers: int = 8) -> None:
    agent_bundles = bundles()
    items = scoring_corpus()
    print(f"{len(items)} kernels, {len(agent_bundles)} agent bundles", flush=True)
    path = RESULTS / "scores.jsonl"
    with ProcessPoolExecutor(workers) as pool, path.open("w") as fh:
        futures = [pool.submit(score_kernel, it, agent_bundles) for it in items]
        for i, f in enumerate(as_completed(futures), 1):
            fh.write(json.dumps(f.result()) + "\n")
            if i % 50 == 0:
                print(f"{i}/{len(items)}", flush=True)
    meta = {it["id"]: {"kind": it["kind"], "wrong": it["wrong"]} for it in items}
    (RESULTS / "scored_corpus.json").write_text(json.dumps({"kernels": meta, "bundles": {k: [t, str(p.relative_to(HERE))] for k, (t, p) in agent_bundles.items()}}, indent=1))


def _rejected(entry: dict, ladder: str) -> bool:
    if "error" in entry or entry.get("load_error"):
        return True
    if entry.get("refused"):
        return True
    groups = entry.get("L1", [])
    return any(g < SHIPPED_GROUPS for g in groups) if ladder == "L0" else bool(groups)


def report() -> str:
    corpus = json.loads((RESULTS / "scored_corpus.json").read_text())
    kernels, agent_bundles = corpus["kernels"], corpus["bundles"]
    scores = {r["id"]: r for r in map(json.loads, (RESULTS / "scores.jsonl").read_text().splitlines())}
    lines = ["# E5 report", ""]
    names = [("shipped:current", None), ("shipped:completed", None)] + [(b, agent_bundles[b][0]) for b in sorted(agent_bundles)]
    for ladder in ("L1", "L0"):
        lines += [f"## Ladder {ladder}", "",
                  "| bundle | task | families rejected | wrong functions rejected (E3b) | correct rejected | fast-exp | "
                  "proof closed | laws |", "|---|---|---|---|---|---|---|---|"]
        for name, btask in names:
            for task in (TASKS if btask is None else (btask,)):
                fam = fam_n = wrong = wrong_n = corr = corr_n = 0
                fast = ""
                for kid, k in kernels.items():
                    s = scores.get(kid, {})
                    if "error" in s:
                        continue
                    t = s["tasks"][task]
                    if name.startswith("shipped:"):
                        entry = dict(t["shipped"][name.split(":")[1]], refused=t["refused"] or t["shipped"][name.split(":")[1]]["refused"])
                    else:
                        entry = dict(t["agent"].get(name, {}), refused=t["refused"])
                    rej = _rejected(entry, ladder)
                    if k["kind"] == "family" and k["wrong"][task]:
                        fam_n += 1
                        fam += rej
                    elif k["kind"] == "e3b" and k["wrong"][task]:
                        wrong_n += 1
                        wrong += rej
                    elif k["kind"] == "correct":
                        corr_n += 1
                        corr += rej
                    elif k["kind"] == "fastexp" and task == "softmax":
                        fast = "rejected" if rej else "admitted"
                closed, laws = "", ""
                if not name.startswith("shipped:"):
                    rows = read_ledger(RESULTS / name)
                    closed = rows[-1]["closed"] if rows else ""
                    laws = ", ".join(next((t["agent"][name].get("laws", []) for s in scores.values() if "tasks" in s
                                           for t in [s["tasks"][task]] if name in t["agent"]), []))[:80]
                lines.append(f"| {name} | {task} | {fam}/{fam_n} | {wrong}/{wrong_n} | {corr}/{corr_n} | {fast} | {closed} | {laws} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    (RESULTS / "report.md").write_text(text)
    return text


def run_all() -> None:
    rotation = [(arm, task, f"r{k}") for k in range(1, REPLICATES + 1) for task in TASKS for arm in ARMS]
    for arm, task, rep in rotation:
        if (replicate_dir(arm, task, rep) / e3b.STOPPED).exists():
            print(f"{arm}-{task}-{rep}: STOPPED, skipping")
            continue
        run_replicate(arm, task, rep)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("control")
    r = sub.add_parser("run")
    r.add_argument("arm", choices=ARMS)
    r.add_argument("task", choices=TASKS)
    r.add_argument("rep")
    sub.add_parser("all")
    s = sub.add_parser("score")
    s.add_argument("-j", type=int, default=8)
    sub.add_parser("report")
    sub.add_parser("program").add_argument("arm")
    a = p.parse_args()
    if a.cmd == "control":
        print(json.dumps(control(), indent=1))
    elif a.cmd == "run":
        run_replicate(a.arm, a.task, a.rep)
    elif a.cmd == "all":
        run_all()
    elif a.cmd == "score":
        score(a.j)
    elif a.cmd == "report":
        print(report())


if __name__ == "__main__":
    main()
