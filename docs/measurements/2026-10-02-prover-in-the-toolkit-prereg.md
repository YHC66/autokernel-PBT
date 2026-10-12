# Pre-registration: does a proof assistant in the toolkit change what a research agent concludes about an oracle?

**Date:** 2026-10-02 (registered before any agent ran; results are appended below the line, never edited above it)
**Instance:** Apple M4 Pro, macOS 26.6.1, CPU only. Agents: Claude sub-agents (Fable 5.1), one fresh
session per cell, no budget cap, wall-clock self-reported.
**Toolchains:** Python 3.10.20 / NumPy 1.26.4 (sandbox venv, offline); Lean 4 v4.33.0 core, no
Mathlib (`~/.elan/toolchains/leanprover--lean4---v4.33.0/bin/lean`).
**Companions:** `2026-10-01-softmax-temperature-blind-spot.md` (the gap), `2026-10-01-softmax-bundle-mechanized.md`
(the proof about it), `docs/notes/2026-10-01-formal-positioning-review.md` §4 (E4 first; rigidity audit).

## 0. The question

The project's direction is that formal verification should cooperate with property-based testing
inside the agent's own loop. The cheapest place to test that claim is not kernel code but the
*oracle*: deciding whether a property bundle characterizes its function is a research task an
agent can do with tests alone ("I found no admitted wrong kernel") or with a prover ("no admitted
wrong kernel exists"). Does the prover change (a) what the agent finds, (b) whether its
completeness claim is true, (c) what it costs?

A second, smaller experiment runs the first cell of the pre-registered pilot E1 in the one language
installed here: agent-written adherence proofs in core Lean 4, where `exp` must be axiomatized —
the situation of every SMT-backed language — so the prediction that such proofs are temperature-
blind can be checked on agent-written axioms.

## 1. Construction

### Experiment A — oracle research, two arms

| | |
|---|---|
| tasks | softmax (bundle: finite, unit interval, row sums, shift invariance) and layernorm (finite, zero mean, unit variance, shift invariance) — `kernels/tasks/*/acceptance.yaml`, unchanged |
| arms | **T** (tests only): Python, NumPy, the gate, the reference. **F** (tests + prover): the same plus the pinned Lean 4 binary and a compiling skeleton over an abstract ordered field, and the added requirement to prove completeness in Lean |
| cells | 2 tasks × 2 arms × 4 replicates = 16 fresh sub-agents, run concurrently |
| sandbox | a sealed directory per agent: the reference (verbatim from `props/tasks.py`), the bundle, a vendored copy of `src/autokernel_pbt/props/` (17 files, no docs, no saboteurs, no proofs, no paper), `gate.py` (the repo's `pbt_gate.check_task` reduced to the declarative and reference arms, same seed 42, same nine groups), `laws.py` (extension point). Agents are told to read nothing outside the directory and to self-report any violation; the project memory index was moved aside for the duration so sub-agents do not inherit its one-line summary of the known blind spot |
| protocol text | `README_oracle_common.md` + one arm-specific section; identical across arms except for the prover paragraph. Kept with the run artifacts |
| deliverables | `admitted/*.py` (wrong kernels the bundle admits), `laws.py` (completing laws with float tolerance), `correct_variants/*.py`, `RESULT.md` (claim + grounds + elapsed seconds); arm F adds `proof/*.lean` |
| evaluation (blind to arm, by script) | **soundness**: the completed bundle passes the reference and four correct rewrites (exp2, reciprocal-multiply, float32 accumulation, rsqrt / one-pass variance). **completeness**: the completed bundle against a **held-out** set of wrong kernels the agents never saw (below). **verified claim** (F): the Lean file compiles with the pinned binary, no `sorry`, `#print axioms` standard, and the main theorem states uniqueness of the row output. **rigidity**: the assumptions each proof makes about `exp` are tested for invariance under `x ↦ exp(βx)`. **cost**: self-reported elapsed seconds |
| held-out wrong kernels, softmax | softmax(βx) for β ∈ {2, 1/2, −1, 0}; the constant 1/n fill; columns reversed; rows rolled by one; output row i computed from input row i+1; y ∝ 1/(1 + max x − x) ("Cauchy"); y ∝ exp(−(x − max x)²) ("Gaussian"); softmax of the sorted row |
| held-out wrong kernels, layernorm | −LN; columns reversed; rows rolled by one; each row sorted; output row i from input row i+1; a fixed Householder reflection H = I − 2vvᵀ with v ⊥ 1 applied to each output row (preserves zero mean and unit variance; not in the paper) |
| scoring | an agent's bundle is *complete on the held-out set* if the declarative arm rejects every held-out kernel on every group where the reference arm rejects it. A **claim** is the agent's own statement in `RESULT.md`; an **overclaim** is a claim of completeness that the held-out set refutes |

### Experiment B — adherence proofs in core Lean 4 (E1, one language)

| | |
|---|---|
| kernels | k1: stable softmax (correct); k2: softmax(2x), the scale written in plain sight; k3: exp2-based softmax (correct only because 2^(t·log₂e) = e^t); k4: softmax with output rows rolled by one (wrong). All four are admitted by the current bundle; k1 and k3 pass the reference arm, k2 and k4 fail it |
| cells | 4 kernels × 2 replicates = 8 fresh sub-agents |
| sandbox | `kernel.py`, the informal spec (one sentence plus the formula), the Lean skeleton, the venv |
| the ask | decide adherence; if yes, five artifacts (spec S, model M, hypotheses H, theorem ∀xs, H xs → S xs (M xs), proof); if no, a refutation (Lean proof of the negation or a concrete input run against the formula). List every assumption about primitives |
| evaluation | compile + `sorry`/`axiom` scan + `#print axioms` (C1); the theorem has the fixed shape (C5, structural); model fidelity by reading M against the kernel (C4, by inspection — no adapter exists yet, so this is not the executable check the design calls for); rigidity audit of the stated `exp`/`exp2` assumptions (C6′); false certificates (an "adheres" verdict on k2 or k4) |

## 2. Predictions, registered before the first run

Experiment A:

- **P1.** Both arms find the temperature family on softmax in at least 3 of 4 replicates.
- **P2.** At least one T-arm agent overclaims: it states completeness and the held-out set refutes
  it. Among F-arm agents whose Lean file compiles with a uniqueness theorem, there are zero
  overclaims — a compiled uniqueness proof over the field is the thing the held-out set cannot
  refute, up to the float form.
- **P3.** On layernorm the majority completion in both arms is the affine-tie law or an equivalent
  "y_i = (x_i − μ_i)/s_i" form; at least one agent in each arm proposes a law set the held-out set
  shows incomplete.
- **P4.** F-arm completeness theorems on softmax are relative to an opaque `exp`; no agent adds a
  rigidity axiom (tangent line or numeric bounds); at most one of four remarks that the formal
  statement leaves temperature undetermined.
- **P5.** Arm F costs more wall-clock than arm T: median ratio ≥ 1.5.
- **P6.** At least one agent, in any cell, exhibits an admitted family not among the ten the
  authors recorded (temperature ×4, fill, rolled rows, reversed columns, −LN, LN reversed, LN
  sorted, LN rolled) — the wrong-row, Cauchy, Gaussian or Householder families are the candidates.

Experiment B:

- **P7.** k1 is proven by at least one of two agents; k3 by at least one, and every k3 proof rests
  on a bridging assumption relating exp2 to exp that only sampling can check; k2 and k4 are
  refuted by all four agents; zero false certificates; no agent weakens the spec to admit k2 or k4.
- **P8.** Every proof of k1 or k3 rests only on `exp` assumptions that are preserved by
  `x ↦ exp(βx)`, β > 0 — so each certificate is, as a formal statement, equally a certificate for
  softmax(βx) under reinterpretation of `exp`. Zero agents include an assumption that breaks the
  symmetry.

## 3. What is NOT being measured

Not the kernel-writing loop (E3/E4); not whether agents game a certificate they are rewarded for
(B's prompt is neutral: "decide"); not any absolute capability number — four replicates per cell
is a pilot, and every count below is reported as a count. Agents could in principle read outside
their sandbox; the protocol relies on the instruction and on self-report, and the run's agent
transcripts are grepped for outside paths after the fact.

---

## Deviation recorded before any result was read (2026-10-02)

The first launch of all 24 sessions was cut off after about 19 minutes by the account's usage
limit, with 1.88M sub-agent tokens and 296 tool calls spent and **no** session having written
its `RESULT.md` or returned a report. Every sandbox kept the files its session had produced
(admitted kernels, draft laws, draft proofs). Rather than discard that work, each cell was
relaunched as a **continuation session in the same sandbox**: a fresh agent with no memory of
the first, told that a previous session was interrupted, that its files are present and may be
used, verified or discarded, and that the README still governs. The oracle-research cells were
relaunched first and the adherence cells after them. Consequences for reading the results:

- "one fresh session per cell" became "up to two sequential sessions per cell, sharing a
  directory but not a context"; the arms are affected identically;
- elapsed seconds are the continuation session's own, and under-count the cell's real cost;
  the first session's cost is known only in aggregate (1.88M tokens over 24 cells);
- nothing was read, evaluated or summarized from any sandbox before the relaunch, beyond file
  listings used to confirm the sessions had been working inside their sandboxes.
