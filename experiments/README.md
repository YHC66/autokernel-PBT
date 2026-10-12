# Experiments

Every experiment this project has run or pre-registered, grouped by the paper's two task segments
plus the verification layer they share. Each self-contained experiment lives in its own directory
with a README that says how to run it. Experiments that run through the library itself (the
oracle comparison, the hardware runs) have their code under `src/`, `kernels/` and `scripts/`, and
are indexed here by their records.

```
experiments/
├── generation/          the agent writes or optimizes a kernel
│   └── autoresearch-pbt/      a property-gated optimization loop, and the gate's saboteur matrix
├── translation/         the agent ports a kernel between backends (proposed; nothing has run)
└── verification/        the correctness layer itself: PBT oracles and formal proofs
    ├── formal-rigidity/       sampled symmetry audit of a verifier's axioms
    └── oracle-research/       does a prover in the toolkit change what an agent concludes?
```

Formal proofs live in [`proofs/`](../proofs/), and are compiled by `tests/proofs/` when a Lean
toolchain is present. Results are recorded, dated and append-only, in
[`docs/measurements/`](../docs/measurements/). The ranked summary is
[`docs/findings.md`](../docs/findings.md).

**Status:** **[M]** measured and recorded · **[B]** built · **[ran]** ran, not yet scored ·
**[planned]** pre-registered · **[proposed]** designed, not pre-registered.

## Generation

| Experiment | Question | Status | Headline | Code | Record |
|---|---|---|---|---|---|
| **Property-gated autoresearch loop** | Can a PBT gate make an optimization loop safe, and what does the oracle choice cost? | [M] | −21.2% val_bpb, +27.7% throughput over 26 candidates; 0 in-loop rejections; 6 saboteurs caught; an `allclose` gate cannot accept the baseline | [`generation/autoresearch-pbt/`](./generation/autoresearch-pbt/) | [`2026-08-18-autoresearch-loop-pbt-gate`](../docs/measurements/2026-08-18-autoresearch-loop-pbt-gate.md) |
| **Temperature blind spot** | What wrong kernels does the gate's decisive arm admit? | [M] | The declarative arm admits softmax(βx) at every β, the 1/n fill (52–54× faster) and softmax(−x); the other arms reject them | [`generation/autoresearch-pbt/saboteurs.py`](./generation/autoresearch-pbt/saboteurs.py) | [`2026-10-01-softmax-temperature-blind-spot`](../docs/measurements/2026-10-01-softmax-temperature-blind-spot.md) |
| E2: complete the bundle | Do the log-ratio and affine-tie laws close every admitted family with no new false positive? | [planned] | — | — | paper §5 |
| E3: red-team the gate | With a speed objective, no forbidden list and a hidden checker, is the gate hacked? | [planned] | — | — | paper §5 |
| E1: agent-written adherence proofs | Can an agent certify kernel–spec adherence, in which of 7 languages, and how often falsely? | [planned] | 147 sessions designed | [`src/autokernel_pbt/formal/`](../src/autokernel_pbt/formal/) (core, [B]) | [design](../docs/superpowers/specs/2026-10-01-agent-formal-adherence-design.md) |
| E4: formal feedback | Does returning failed checks and unsolved goals improve the agent beyond a better gate? | [planned] | — | — | paper §5 |

## Translation

Nothing has run. The proposed pipelines (T0–T4) and experiments (TE1–TE4) are in
[`docs/pipelines/translation.md`](../docs/pipelines/translation.md). The evidence so far is the
cross-backend half of the hardware runs below, which show Tier-1 laws moving NumPy → Triton
unchanged. See [`translation/README.md`](./translation/README.md).

## Verification layer (shared by both segments)

### Property-based testing: the oracle comparison

Four arms (`allclose`, normalized reference ratio, declarative law bundle, hybrid), all scored
over byte-identical recorded executions. Code: `src/autokernel_pbt/props/`, `kernels/`,
`scripts/gpu_record.py`.

| Experiment | Status | Headline | Record |
|---|---|---|---|
| False-positive rate on correct kernels | [M] | `allclose` 10/54; the other arms 0/54; a "correct" variant found to be a cancellation bug | [`2026-08-19-false-positive-rate`](../docs/measurements/2026-08-19-false-positive-rate.md), [`2026-08-16-allclose-layernorm-false-positives`](../docs/measurements/2026-08-16-allclose-layernorm-false-positives.md) |
| Detection over mutation corpora | [M] | 20 admitted mutants from 3 corpora, **0** arm disagreements | [`2026-08-18-arm-differentiation-null-result`](../docs/measurements/2026-08-18-arm-differentiation-null-result.md), [`…-blinded-corpus-pilot`](../docs/measurements/2026-08-17-blinded-corpus-pilot.md), [`…-provisional-corpus-pilot`](../docs/measurements/2026-08-17-provisional-corpus-pilot.md) |
| Reference-arm normalization | [M] | `log2 n` over linear `n`; tile-aware bound 3.5× vs 5.9× drift | [`2026-08-17-hybrid-reduction-tree`](../docs/measurements/2026-08-17-hybrid-reduction-tree.md), [`…-normalization-discrepancy-resolved`](../docs/measurements/2026-08-17-normalization-discrepancy-resolved.md) |
| Hardware runs: transfer and scale (NumPy → Triton, A10) | [M] | Laws transfer unchanged up to 29.4 M elements; a tile-tail launch defect caught | [`2026-08-17-first-hardware-run`](../docs/measurements/2026-08-17-first-hardware-run.md), [`…-at-scale-hardware-run`](../docs/measurements/2026-08-17-at-scale-hardware-run.md) |
| compute-sanitizer viability | [M] | Works only with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | [`2026-08-17-sanitizer-viability-probe`](../docs/measurements/2026-08-17-sanitizer-viability-probe.md) |
| Authoring cost (layernorm) | [M], `n = 1` | Declarative arm 176 lines and 1 serious defect, against 77 lines for the reference arm | [`2026-08-16-layernorm-authoring-cost`](../docs/measurements/2026-08-16-layernorm-authoring-cost.md) |

### Formal verification

| Experiment | Status | Headline | Code | Record |
|---|---|---|---|---|
| Proposition 1, mechanized | [M] | The bundle admits softmax(βx) for all β; the log-ratio law pins softmax. Lean 4, no Mathlib, 6 theorems | [`proofs/softmax-bundle/`](../proofs/softmax-bundle/) | [`2026-10-01-softmax-bundle-mechanized`](../docs/measurements/2026-10-01-softmax-bundle-mechanized.md) |
| Why3 real theories, audited | [M] | All 10 `ExpLog` axioms hold for exp(βx); `Square` categorical; trig ×5 alias | [`proofs/why3-real-theories/`](../proofs/why3-real-theories/), [`verification/formal-rigidity/`](./verification/formal-rigidity/) | [`2026-10-02-why3-real-theories-symmetry`](../docs/measurements/2026-10-02-why3-real-theories-symmetry.md) |
| Prover in the toolkit | [ran] | 11 of 24 agent self-reports survive, unscored; not quotable | [`verification/oracle-research/`](./verification/oracle-research/) | [`2026-10-02-prover-in-the-toolkit-prereg`](../docs/measurements/2026-10-02-prover-in-the-toolkit-prereg.md) |
| Formal adherence core (feature 0009) | [B] | Six checks, each the sole catcher of one cheat | [`src/autokernel_pbt/formal/`](../src/autokernel_pbt/formal/) | [design](../docs/superpowers/specs/2026-10-01-agent-formal-adherence-design.md), [ADR 0002](../docs/adr/0002-formal-track.md) |

## Running the experiments' own tests

The experiments' tests sit outside the main suite's `testpaths`, and CI runs them as a separate
step:

```bash
pytest experiments -q
```

## Adding an experiment

1. Pick the segment (`generation/`, `translation/`, `verification/`) and create
   `experiments/<segment>/<name>/` with a README covering the question, how to run it, its
   environment, and its status.
2. Pre-register: write `docs/measurements/YYYY-MM-DD-<name>-prereg.md` with the construction and
   predictions *before* the first run, and append results below its line afterwards. Never edit
   above that line.
3. Copy agent artifacts into the repository the moment each agent finishes. Scratch directories
   under `/private/tmp` are wiped within days, and one experiment lost all 24 sandboxes that way.
4. Add a row to this index and, once results are recorded, to
   [`docs/findings.md`](../docs/findings.md).
5. Keep run directories off synced folders such as OneDrive (see `CLAUDE.md`).
