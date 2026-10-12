# autokernel-PBT

**A correctness layer for agent-written compute kernels.** Property-based testing (PBT) decides
what a kernel-writing agent may keep. Formal verification (FV) certifies what PBT accepted, and
proves where PBT's laws fall short. Each checks the other.

> **PBT means *property-based testing*,** not population-based training.

When an agent optimizes a kernel, many of the cheapest speedups are wrong. Dropping a
max-subtraction, a centering or a divide buys 1.11–2.40×, and a constant-fill "softmax" is 52–54×
faster. Each produces plausible output. The correctness gate is therefore half of the agent's
objective, and this repository builds and measures that half.

## Two task segments

The paper, and so this repository, covers two tasks:

| Segment | The agent… | The correctness question | Status | Pipelines |
|---|---|---|---|---|
| **Autokernel generation** | writes or optimizes a kernel for one backend | Is the faster kernel still the same function? | Gate built and measured; a gated optimization loop run on CPU; formal certificate core built | [G0–G5](docs/pipelines/generation.md) |
| **Autokernel translation** | ports a kernel between backends: CUDA/Triton → TPU (Pallas), Triton → Trainium (NKI) | Does the port compute the source's function on hardware that rounds differently, often with no reference on the target? | **Proposed.** Laws already moved NumPy → Triton unchanged | [T0–T4](docs/pipelines/translation.md) |

Both segments share one substrate: seeded case groups with metamorphic partners, executed once per
backend, recorded byte for byte, and scored offline by competing oracles.

## Where PBT and formal verification paid off

The full ranked list, with every number linked to its record, is in
**[`docs/findings.md`](docs/findings.md)**.

| | Gain | Number | Seg. |
|---|---|---|---|
| PBT | **Calibration**: the gate accepts correct kernels | `allclose` rejects 10/54 correct-kernel groups (0.185); law, reference and hybrid arms reject 0/54. An `allclose`-gated loop cannot accept its own baseline | G+T |
| PBT | **A cheap gate made an optimization loop safe** | −21.2% val_bpb, +27.7% throughput over 26 candidates; 6 faster-but-wrong optimizations caught | G |
| PBT | **Laws port across backends** with no second implementation | NumPy → Triton unchanged, up to 29.4 M elements; a scale-only launch defect caught | T |
| PBT → FV | **A gap found by testing became an exact characterization** | The gate admits 10 wrong-kernel families. Proposition 1, mechanized in Lean, characterizes them, and one added law per task closes them | G+T |
| FV | **Verifier axioms share the blind spot** | All 10 axioms of Why3's `real.ExpLog` hold for exp(βx), so a softmax certificate holds at *some* temperature | G+T |
| FV × PBT | **Certificates whose claims are executed** | 6 checks run every formal artifact on PBT's cases, each the only catcher of one cheat (built; pilot pre-registered) | G |

**Where they did not help:** on detection, the four oracle arms tie, with 0 disagreements over 20
admitted mutants. Where a cheap, trusted reference exists, laws can only match it. Their value is
calibration, and working where no reference exists, which is the translation setting.

## Repository map

```
experiments/           self-contained experiments, by segment (index: experiments/README.md)
├── generation/          autoresearch-pbt: the property-gated optimization loop + saboteur matrix
├── translation/         proposed; nothing has run yet
└── verification/        formal-rigidity, oracle-research: experiments on the correctness layer
proofs/                Lean 4 proofs (Proposition 1; Why3's real theories), compiled by tests/proofs
src/autokernel_pbt/
├── props/               the PBT core: generator, case groups, backends (NumPy, Triton),
│                        execution and score tables, the four oracle arms, driver.run_task
├── formal/              the formal-certificate core: the six checks (feature 0009)
├── corpus/, metrics/    mutation-corpus admission and the detection / false-positive metrics
└── harness/, cli/       the feature 0001/0002 skeleton and the `akpbt` CLI
kernels/               Triton ladder kernels, per-task law bundles (acceptance.yaml), mutants
docs/                  findings, pipelines, designs, ADRs, dated measurements (index: docs/README.md)
specs/                 feature specs with machine-checkable acceptance criteria
paper/                 the AAAI workshop draft; every number comes from paper/numbers.tex
scripts/               GPU bootstrap and recording, the clean-commit helper
reference/             the tiered paper library and the PBT literature notes
tests/                 unit, integration, spec-derived, GPU and formal tests
harness/, contracts/   the original evaluation skeleton and interface notes (see their READMEs)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not gpu" -q        # the whole CPU suite; must be green and warning-free
pytest experiments -q         # the experiments' own tests (saboteur matrix, rigidity audit)
pytest -m gpu                 # needs CUDA and the [gpu] extra (torch + triton)
```

Formal tests (`-m formal`) compile the Lean proofs and are skipped when no Lean toolchain is
installed. The pinned version is in each `proofs/*/lean-toolchain`. The optimization loop needs
its own NumPy 2.x environment: see
[`experiments/generation/autoresearch-pbt/README.md`](experiments/generation/autoresearch-pbt/README.md).

## Working in this repository

- **Spec before code.** A feature gets `specs/features/NNNN-*/spec.md` and `acceptance.yaml`
  first, and its spec test starts red ([`specs/README.md`](specs/README.md),
  [ADR 0001](docs/adr/0001-sdd-tdd.md)).
- **Pre-register measurements.** Write the record's construction and predictions in
  `docs/measurements/` before the run, and never edit a result afterwards.
- **Review adversarially.** Verify claims by re-running them, and make every assertion the unique
  catcher of some saboteur. The standard, and the module contracts that cost review rounds to
  establish, are in [`CLAUDE.md`](CLAUDE.md). Read it before changing `src/`.
- **Commits.** Conventional subjects (`feat:`, `fix:`, `docs:`, `spec:`, `test:`) with a body
  that explains why. Agents commit through `scripts/git_commit_clean.sh`, never plain
  `git commit` (see `CLAUDE.md`).
- **Keep run directories off OneDrive and other sync folders.** Sync renamed recorded payloads and
  made runs unreadable once.

## Reference library

PDFs live under `reference/L0-start-here/` … `L5-advanced-topics/` and are gitignored. Fetch them
with `scripts/download_reference_pdfs.sh`, from `reference/manifest.csv`. The PBT literature
review is [`reference/PBT-property-based-testing/`](reference/PBT-property-based-testing/).

## License

MIT
