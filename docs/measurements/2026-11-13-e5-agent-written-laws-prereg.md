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

## The instrument as built (2026-11-13), before any session

`experiments/generation/e5-agent-written-laws/`: `e5.py` (the loop, the Lean check, the scorer), `law_runner.py` (runs
one `laws.py`, sandboxed), `program.md` and `formal.md` (what the agent reads), and `lean/` (the fixed preambles and
obligations, and a reference solution per task).

- **The Lean model.**
  - **Softmax:** `softmaxRow` over an abstract ordered field with an opaque `exp` and the `ExpLaws` of
    `SoftmaxBundle.lean`.
  - **Layernorm:** `lnRow xs = (x − mean)/sqrt(var)` with an opaque `sqrt`, where `sqrt` is positive on positives
    and squares back. The real-number function has no ε, and the obligation is over rows with `0 < var xs`. The
    row length enters as a sum of ones, since core's `Field` gives no `Nat` coercion.
  - **The obligation:** `∀ xs, (xs ≠ [] | 0 < var xs) → f xs = target xs`, for any `f` with `Laws f`. Here `f` is
    the kernel on one row, so shift laws can be stated.
- **The Lean check.** It compiles the preamble, the agent's file and the obligation with the pinned toolchain,
  under `sandbox-exec`: no network, no writes outside a temporary directory, and no reads in the home
  directory except the toolchain (metadata reads allowed, which Lean needs to find itself). A probe confirmed
  that a file reading or writing in the home directory fails. *Closed* means exit 0, no error, and `#print
  axioms e5_obligation` listing only `propext`, `Classical.choice` and `Quot.sound`.
- **Controls (`python e5.py control`), all as expected:**
  - Closes: the reference solution, for each task. So does a `Laws` that restates the function; catching that
    is the hand audit's job, not Lean's.
  - Does not close: a `sorry`, an extra axiom, a file that fails to compile, or a file that stops early with
    `#exit`.
- **The scorer.**
  - Each kernel is recorded once on L1, in E3b's sandboxed recorder.
  - The shipped bundles are scored as in E3c. Each agent `laws.py` runs in its own sandboxed process, which gets
    copies of the cases and never sees the repository.
  - L0 is L1's first 9 groups, which are the same cases called in the same order.
  - A kernel that produced no output on some case is refused under every bundle, as the shipped gate refuses it.
- **Scorer control.** Two scripted softmax bundles were scored: one of the current kind (range, row sums, shift
  invariance) and one with a log-ratio law added. The first admits all 6 softmax families and the second rejects
  all 6. Both admit the starting kernel, an E3 correct kernel and fast-exp. Both reject E3b's 12,442× hack on
  the call-site group.
- **The corpus** adds fast-exp (E3's copy), reported on its own. E3b's 20 gate-rejected candidates have no hidden
  label, so they are not counted.
- **Pilot:** one replicate per arm and task (`r = pilot`), not counted, to check the loop end to end.

## The pilot (2026-11-13), before any counted session

One replicate per arm and task, in `recorded/<arm>-<task>-pilot/`. Not counted, and no prompt or code changed after
it. 4 sessions, $1.14 in all.

- **The loop works end to end.** Every session wrote `laws.py`. Both TF sessions wrote `Laws.lean`, and both
  proofs closed in round 0 with the standard axioms only. That is fewer rounds than Q4 assumed would be needed.
- **Hand audit of the two Lean files: neither restates the function.**
  - Softmax: length, non-negativity, row sum 1, and a pairwise ratio law, `y_p · exp x_q = y_q · exp x_p`.
  - Layernorm: zero mean, mean square 1, `f xs` a non-negative multiple of `xs − mean`, and shift invariance.
- **Already visible, and not a result.** The T softmax pilot also wrote a log-ratio law, unprompted. If the
  counted replicates do the same, Q1 is falsified. Agents writing laws from the definition would then not
  reproduce the human gap. The prediction is left as registered.
- **Scorer check on the pilot bundles.** Each bundle admits the starting kernel and an E3 correct kernel. Each
  rejects its task's sample family (β = 2 and −LN) and E3b's 12,442× hack on the call-site group. One T
  layernorm law raises a `TypeError` on −LN's output, and that counts as a rejection, as registered.

The 20 counted replicates started right after, unchanged.

## Results (2026-11-16)

All 20 counted replicates ran: 23 sessions in all (3 TF replicates took two rounds), costing $6.98.

**Infrastructure.** On 2026-11-13 the account's session limit stopped the run during `T-layernorm-r4`'s first
session. As registered for E3b, a failed session costs no iteration. The run was resumed on 2026-11-16 with nothing
changed, and that session was re-run from the start. The interrupted log is in `recorded/infrastructure-2026-11-13.log`.

**Hand audit.** No final `laws.py` computes the function and compares with it, and no `Laws.lean` restates it. Every
softmax bundle, in both arms, has a law that fixes the temperature: a log-ratio law, or a pairwise or pivot ratio
`y_i·e^{x_j} = y_j·e^{x_i}`. Every layernorm bundle has a law that ties `y` to `x` with a sign:
- an affine law with a positive slope,
- `y` a non-negative multiple of `x − mean`,
- a monotonicity law, or
- the covariance of `y` and `x` equal to the row's standard deviation.

L1, from `recorded/report.md`. Every agent bundle of a task scores the same on families and on correct kernels.

| Bundles | Families rejected | Correct kernels rejected | E3b wrong functions rejected | Proofs closed |
|---|---|---|---|---|
| T softmax (5) | 6/6, each | 0/121, each | 98–99/134 | n/a |
| TF softmax (5) | 6/6, each | 0/121, each | 98/134 | 5/5 |
| T layernorm (5) | 4/4, each | 0/121, each | 133/148 | n/a |
| TF layernorm (5) | 4/4, each | 0/121, each | 133/148 | 5/5 |
| shipped current | 0/6, 0/4 | 0/121 | 62/134, 145/148 | |
| shipped completed | 6/6, 4/4 | 0/121 | 99/134, 147/148 | |

On L0 every agent bundle also rejects all of its task's families and no correct kernel. Fast-exp is admitted by
every bundle, agent and shipped.

- **Q1 is falsified.** All 5 T softmax bundles reject all 6 families; the prediction was at most 2. Agents writing
  laws from the definition did not reproduce the human gap of Proposition 1, not once in 10 softmax bundles.
- **Q2 is falsified.** TF does no better than T on either task: both arms are at the ceiling. By the falsifier as
  registered, these runs give no evidence that the proof obligation makes agent-written laws more complete.
- **Q3 is falsified.** No T bundle rejects a correct kernel.
- **Q4 holds.** All 10 TF proofs closed, 7 in round 0 and 3 in round 1, with standard axioms only.
- **Where agent laws are weaker: float tolerance.** All 10 agent layernorm bundles admit the same 14 E3b
  layernorm functions that the shipped completed bundle rejects. All 14 are numerically inaccurate rather than
  structurally wrong, for example a one-pass variance or a floored variance. The shipped bundle rejects each one
  only through shift invariance on the call-site group, at a ratio of 33–82 against its threshold of 30; the
  reference arm rejects them too. The agents' tolerances are looser. The Lean proofs are about exact arithmetic,
  so by construction they say nothing about this.

**What this says.** On these two tasks, the agent writes laws that are complete in exact arithmetic from the
definition alone, and proves them complete when asked. The gap Proposition 1 describes was in the laws people
wrote, and the agent did not repeat it. Two things are not shown:
- **That the proof obligation helps.** The arms tie at the ceiling.
- **That the result carries to tasks less familiar than softmax and layernorm.** The model may simply know their
  characterizations.

The measured weakness of agent-written laws is their float tolerance. That half of the check is not covered by
a proof.
