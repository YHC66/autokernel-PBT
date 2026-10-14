# Feature 0013: Cross-setup scoring — a port judged against the known kernel it was mapped from

## Problem

Translation maps a **known** kernel from one setup to another. The known kernel's recorded
executions are the most direct evidence of what the port must compute: same inputs, byte for
byte, and — with GPU first — the same device. Today every reference arm takes a reference
*function*, a NumPy recomputation, so a port can only be judged against an idealized reference and
never against the kernel it was translated from. The pipelines this unblocks are T0 (the field's
default: `allclose` against the source) and T1 (the normalized ratio against the source), in
`docs/pipelines/translation.md`.

The two runs being compared were recorded separately — on different backends, possibly on different
days. Pairing them is therefore the dangerous step. Case ids are a pure function of `(seed, index)`,
so two runs of the same task under the same seed join on every key *even if their inputs differ*;
a pairing that trusts ids would score a port against a source that never saw the port's inputs.

## Design

**A source recording is a reference function, keyed by input content.** `SourceRecording` is built
from the source run's rows and is callable as `recording(**kernel_inputs(case))` — exactly the
contract `ReferenceOracle` and `AllcloseOracle` already call their reference with — returning the
source's recorded output *for those input bytes*. Keying by content rather than by case id makes
byte-identical inputs a property of the lookup itself: an input the source never saw has no answer,
and that is a bad call (raise), never an approximation. Reusing the existing arms unchanged means
every difference between `reference` and `reference_source` is the reference and nothing else.

**A failed source row is the source's failure, not the port's.** The recording raises
`ReferenceUnavailable`, a new exception that both reference arms catch *narrowly* and map to
INCONCLUSIVE with a detail naming the source — the same discipline as `ExactDtypeError`. Booking it
as FAIL would count the source's crash against the port; letting it propagate would abort a scoring
pass over executions already paid for.

**Pairing is verified before anything is scored.** `pair_runs` refuses: runs of different tasks;
different case-id sets or group structure; any case whose inputs differ by a single byte; a source
not stated correct (`kernel_is_broken is False` on every row — the "known kernel" premise, waivable
explicitly for the open question of an untrusted source); a source and target that are the same
recording; and a source whose rows give two different answers for one input.

**Scoring writes seven arms over the port's run.** `driver.score_translation` scores the target run
with the four arms of feature 0006 against the NumPy reference (absolute correctness), plus three
against the source recording: `allclose_source` (T0), `reference_source` (T1) and `hybrid_source`
(the law bundle first, then the source ratio). All seven judge the same rows from one read, so the
comparison between T0, T1 and T2 (`declarative`) is over byte-identical executions, as every
comparison in this project is. Beside `scores.parquet` it writes `pairing.json`: the source run's
directory and corpus fingerprint, the target's, and both runs' kernel ids and backends.
`pairing.json` also records the arms scored and whether the known-source premise was waived.
`driver.read_translation` is the read path for the source arms: from one read of the target it
refuses the pair if the scores no longer belong to the target run, if the pairing was written for
another recording of the target or for other arms, or if the source has been re-recorded since.
`read_run` — and so `metrics.rates_from_run` — refuses a run with source arms outright, because it
cannot check them; `metrics.rates_from_translation(target, source)` is their rates path.

**A crash cannot mix two sources.** Scores and pairing are two files. `score_translation` deletes
the old pairing *before* writing the new scores, so a crash anywhere between the two writes leaves
source-arm scores with no pairing, which every read refuses — never new scores beside a pairing
that names the previous source.

## Scope

1. `ReferenceUnavailable` in `oracle.py`, caught narrowly by `ReferenceOracle` and `AllcloseOracle`.
2. `props/cross_setup.py`: `SourceRecording`, `UnpairedInputError`, `pair_runs`, and the three
   source arms (subclasses differing from their parents only in `name`).
3. `driver.score_translation` and `driver.read_translation`; `arm_order` takes the arm names, with
   the four-arm order unchanged for every existing seed.
4. A CPU integration test with two NumPy "setups" that round differently, so the whole T0/T1/T2 path
   runs before any hardware does.

## Non-goals

- A cross-setup *tolerance*: the source arms use the existing `log2 n` ratio and threshold 30
  unchanged. Whether that is right for two backends that both round is TE2's measurement.
- Tier-2 target constraints (T3) and shared-spec certificates (T4).
- A report layer for the source arms beyond `metrics.rates_from_translation`.

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
