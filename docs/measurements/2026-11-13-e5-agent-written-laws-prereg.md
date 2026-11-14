# Pre-registration: E5. Can an agent write its own complete laws, and does a proof obligation help?

**Date:** 2026-11-13 (registered before this experiment's code exists and before any agent has run; results are
appended below the line, never edited above it)
**Segment:** generation
**Instance:** Apple M4 Pro, CPU only; the project's pinned environment (NumPy 1.26.4); Lean 4 at the toolchain
pinned in `proofs/softmax-bundle/lean-toolchain`
**Companions:** `2026-10-01-softmax-bundle-mechanized.md` (Proposition 1 in Lean), the two E2 pre-registrations
(the completing laws), `2026-11-13-e3c-wider-ladder-prereg.md` (the ladder and corpora reused here)

**Status:** registered, not run. The formal arm needs the Lean toolchain installed on the instance, which it is not
yet.

## 0. The question

Every law in the gate so far was written by people, and the gap Proposition 1 names was found by people. The
paper's claim is that an agent equipped with property-based testing and formal verification produces more accurate
software, so the agent has to write the checks itself. **When an agent writes a task's laws from its definition,
are they complete, and does an obligation to prove they pin the function make them complete more often?**

## 1. Construction

**Tasks:** softmax and layernorm (no affine parameters), row-wise over the last axis, float32.

**Arms.**

| Arm | The agent writes | Rounds |
|---|---|---|
| **T** tests | `laws.py`: reference-free laws, each `law(x, y) -> (ok, detail)`, using NumPy only | 1 |
| **TF** tests + proof | `laws.py` as in T, and `Laws.lean`: the same laws over the reals with `exp` as an axiomatised function, and a theorem that any `y` satisfying them equals the task's function of `x` | up to 4 |

In TF the loop compiles `Laws.lean` with the pinned toolchain (core Lean 4, no Mathlib, as `SoftmaxBundle.lean`).
It returns the compiler's output and any `sorry` or non-standard axiom in `HISTORY.md` for the next round. The
final `laws.py` is the one scored, whether or not the proof closed. Whether it closed is recorded.

**The agent,** as in E3 and E3b: `claude -p` with `claude-sonnet-5-5`, a fresh session each round, tools Read, Edit
and Write, no shell. It is given the task's mathematical definition, the law interface, float32's machine epsilon,
the input shapes it may assume (any rows, columns 1 to 8192), and nothing about the gate's existing laws, their
known gaps, Proposition 1, or any kernel. *Reference-free* means a law must not compute the task's function of `x`
and compare against it. Each final bundle is read by hand. A bundle with a law that does is reported as
reference-in-disguise and scored separately.

**Scale:** 5 replicates per arm per task, so 20 replicates and at most 50 sessions.

**Scoring,** offline, with the laws run in E3b's sandbox, never in the scorer's process. A function is admitted if no
law fails on any case of ladder L1 (the shipped 9 groups plus the call-site shape, E3c). Also reported on L0.

- **Completeness:** the 10 wrong families the current bundle admits (softmax: β = 2, 0, −1, the 1/n fill, rolled,
  reversed; layernorm: −LN, reversed, sorted, rolled), and E3b's 282 wrong functions.
- **Calibration:** the 121 correct kernels of E3c's corpus. Fast-exp is reported on its own.
- **Reference points:** the shipped current and completed bundles, scored the same way.

## 2. Already known, so not predictions

The current bundle admits all 10 families and the completed bundle rejects all 10 on L0 (E2). On L1 the completed
bundle rejects 246 of the 282 wrong functions and no correct kernel (E3c).

## 3. Predictions

- **Q1.** In T, at most 2 of the 5 softmax bundles reject all 6 softmax families. Agents writing tests alone
  reproduce the human gap: row sums, range and shift invariance, but no law that fixes the temperature.
- **Q2.** In TF, more bundles reject all of their task's families than in T, on each task. **Falsifier for the
  formal half of the thesis:** TF does no better than T on either task.
- **Q3.** At least one of the 10 T bundles rejects a correct kernel: an agent-written tolerance that is wrong.
  This is descriptive, and it measures the calibration side of an agent writing its own tests.
- **Q4.** In TF, at least 6 of the 10 proofs close with no `sorry` within 4 rounds.

## 4. What is NOT measured

- Kernels the agent writes against its own laws (E4 closes that loop).
- GPU kernels and other tasks.
- Whether a hand-written prompt that names the gap helps: no arm does.

## 5. Deviation rules

As in E3b. A prompt or tool change after the first counted session is a deviation, reported below the line, and the
affected replicates are re-run in full.

---

## Results
