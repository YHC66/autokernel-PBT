# The symmetry probe, recorded: the gate admits six output symmetries of the correct kernels

**Date:** 2026-10-14
**Instance:** Apple M4 Pro, macOS 26.5.2, CPU only
**Environment:** the project's pinned environment (`pip install -e ".[dev]"`): Python 3.12.2,
NumPy 1.26.4, OpenBLAS
**Harness:** `experiments/generation/autoresearch-pbt/pbt_gate.py`, unmodified:
`check_task(kernel, task)`, `GATE_SEED = 42`, 9 groups, NumPy backend
**Companion:** `2026-10-01-softmax-temperature-blind-spot.md`, the four temperature rows of the
same gap, measured on the same gate and cases
**Provenance:** the six rows were first measured on 2026-10-01 by a probe script that was not kept,
and the paper's Table 3 has quoted their counts since then, tagged `[P]` in `paper/numbers.tex`.
This record re-measures them from code that stays in the repo. The kernels are reconstructed from
the table's row labels, and they stand in for the lost probe only because they reproduce its
numbers. The expected counts were known before this run, and a throwaway check earlier the same
day matched them, so this is a reproduction, not a blind prediction.

**Headline.** The gate admits six more wrong kernels, each a correct kernel's output with one
symmetry applied: softmax with its rows rolled or its columns reversed, and layernorm negated, with
its columns reversed, with each row sorted, or with its rows rolled. `allclose`, the reference arm
and the hybrid arm reject every one of them on every group where the kernel is wrong (6 or 7 of 9),
and the declarative arm, on which the gate decides, rejects none. Every count in the paper's
Table 3 reproduces, so those rows move from `[P]` to `[M]`.

---

## 1. Construction

Stated before the numbers, per the repo standard.

| | |
|---|---|
| gate | `pbt_gate.check_task`, unmodified. Its decision is the declarative arm's alone; all four arms are reported |
| bundles | `kernels/tasks/softmax/acceptance.yaml` and `kernels/tasks/layernorm/acceptance.yaml`, unmodified |
| cases | `GATE_SEED = 42`, `GATE_GROUPS = 9`: one group per ladder rung, each a base case plus one `shift_rows` partner, 18 executions per kernel |
| unit | groups failed of 9, per arm, exactly as `check_task` reports them |
| reproduce | `cd experiments/generation/autoresearch-pbt && ../../../.venv/bin/python symmetry_probe.py` prints §2's table |
| pinned by | `experiments/generation/autoresearch-pbt/test_symmetry_probe.py` (§4) |

Every probe kernel is the task's own reference (`softmax_reference`, `layernorm_reference` in
`src/autokernel_pbt/props/tasks.py`) with one symmetry applied to its output, so whatever a row gets
wrong it gets wrong through that symmetry alone:

```python
def after(reference, symmetry):
    def kernel(x):
        return np.ascontiguousarray(symmetry(reference(x)))
    return kernel

after(softmax_reference, lambda y: np.roll(y, 1, axis=0))   # rows rolled by one
after(layernorm_reference, np.negative)                     # -LN(x)
after(layernorm_reference, lambda y: np.sort(y, axis=-1))   # each row sorted
```

Each task has two controls beside its probe rows:

- **the reference itself**, which the gate must admit with no arm failing;
- **a positive control**, which the gate must reject. Without it, a gate that admitted everything
  would reproduce every probe row. For softmax it is `saboteurs.unnormalized`, exp(x − max) never
  divided, the companion's own positive control. For layernorm it is the row centered and never
  divided, the defect the layernorm domain's U(−10, 10) inputs were widened to expose.

The script also records every result the gate's own declarative arm produces, through the object
`check_task` builds, so a declarative "0/9" can be told apart from an arm that abstained.

## 2. The result

Groups failed of 9, and the gate's decision. `allclose`, the reference arm and the hybrid arm agree
on every row, as Table 3 says, so they share one column here; the script prints them separately.

| kernel | allclose, reference, hybrid | declarative | gate | Table 3 (others / decl.) |
|---|---|---|---|---|
| softmax(x), the reference (control) | 0 | 0 | admits | — |
| **softmax, rows rolled by one** | 6 | **0** | **admits** | 6/9 / 0/9 |
| **softmax, columns reversed** | 7 | **0** | **admits** | 7/9 / 0/9 |
| exp(x − max), never divided (positive control) | 7 | 7 | rejects | 7/9 / 7/9 |
| LN(x), the reference (control) | 0 | 0 | admits | — |
| **−LN(x)** | 7 | **0** | **admits** | 7/9 / 0/9 |
| **LN, columns reversed** | 7 | **0** | **admits** | 7/9 / 0/9 |
| **LN, each row sorted** | 7 | **0** | **admits** | 7/9 / 0/9 |
| **LN, rows rolled by one** | 6 | **0** | **admits** | 6/9 / 0/9 |
| x − mean, never divided (positive control) | 7 | 7 | rejects | — |

**Attribution.** On all six probe rows the hybrid arm catches through `matches_reference`, its
reference half, because its declarative half passed. On the two positive controls it catches
through its declarative half: `rows_sum_to_one` on softmax, `rows_have_unit_variance` on
layernorm.

**The uncaught groups.** The single-column rungs (1, 1) and (17, 1) make every kernel in the table
correct: a one-element softmax is 1.0 and a one-element layernorm is 0.0, whatever symmetry is
applied. That makes 7 of 9 the ceiling for every arm. Rolling rows is also the identity on the
single-row rung (1, 64), hence 6 of 9.

**The declarative 0/9 is a judgement, not an abstention.** Softmax: 0 of 126 recorded declarative
results are `INCONCLUSIVE`. Layernorm: 8 of 126 on every row, the correct reference included. All 8
are `rows_have_unit_variance` on the two single-column rungs, base and shifted partner, counted once
by the declarative arm and once by the hybrid arm's declarative half. Every row there is a constant
input, which layernorm maps to zeros, so a variance law has nothing to judge. The other three laws
do judge those cases and pass them, but `summarize` folds any abstention into the group's verdict,
so the declarative arm reads INCONCLUSIVE on those two groups: it judges 7 of 9 layernorm groups,
and 9 of 9 softmax groups. That changes no count, since every kernel is correct on those two
rungs. Every other declarative result on the probe rows is a PASS.

**The log-ratio spread** (worst row spread of log y − x, in float32 eps; a measurement no arm reads,
computed by `saboteurs.log_ratio_spread_eps`) is 0.956 on the softmax reference, 6.76e7 with the
rows rolled and 8.72e7 with the columns reversed. The positive control's 4.3 matches the companion
record's value for the same kernel. Layernorm's completing law, the affine tie, is not measured here.

## 3. Why it happens

Every law in both bundles reads an output alone (finite, in [0, 1], rows sum to one; zero mean,
unit variance) or compares an output with another output (shift invariance). None ties an output
value to the input value it came from. So any post-processing of a correct output that keeps each
row's constraints, and commutes with a per-row input shift, passes the whole bundle:

- **permuting rows or columns** keeps every row a probability vector, or zero-mean with unit
  variance, and acts on the output alone, so it commutes with an input shift;
- **negating a layernorm** keeps zero mean and unit variance;
- **sorting each layernorm row** permutes within the row, the same way.

That is the paper's Proposition 1 (a) and (c). The softmax log-ratio spread separates both softmax
probe rows from the reference by seven orders of magnitude, which is part (b): with the log-ratio
law added, softmax is the only kernel the bundle admits.

## 4. How it is pinned

`symmetry_probe.py` gives every row, through `saboteurs.Saboteur`, each arm's expected (groups
failed, first-failing property) and the gate's expected decision, and names the four declarative
abstentions above. `test_symmetry_probe.py` asserts them in three tests of one assertion each, as
`test_gate_saboteurs.py` does for the companion's matrix: the pairing, the decision, and the
judgement (every declarative result a PASS except the named abstentions, with the result count
pinned so an empty recording fails). Verified by breaking the gate six ways in memory, without
editing any source file:

| gate defect injected | test that fails | rows that fail |
|---|---|---|
| decide on the hybrid arm instead of the declarative one | the decision | the 6 probe rows |
| admit every kernel | the decision | the 2 positive controls |
| reference arm blind (threshold ∞) | the pairing | 6 probe rows + 2 positive controls |
| `shift_invariance` abstains on one group | the judgement | all 8 rows the declarative arm passes |
| `rows_have_unit_variance` passes constant rows instead of abstaining | the judgement | the 5 layernorm rows the declarative arm passes |
| `shift_invariance` abstains on every group | the decision and the judgement | the 8 rows the declarative arm passes |

Each test is the only catcher of at least one defect. In the last row the gate's own guard fires
first: with every group INCONCLUSIVE, `check_task` refuses the kernel as never judged, so the
decision test sees it too.

## 5. Threats to validity

- **Author-chosen, not blinded.** The symmetries were chosen by reasoning from the bundles'
  invariances. They show the gate *can* admit these kernels; they are not a detection rate and must
  not be pooled with the mutation corpora.
- **A reconstruction.** The 2026-10-01 kernels were not kept. These match Table 3's counts on every
  cell, but they may not be the same code.
- **One seed, one dtype, one backend, one environment:** `GATE_SEED = 42`, float32, NumPy, and
  NumPy 1.26.4 only. The companion found the gate's verdicts identical under NumPy 2.5.2 /
  Accelerate; this record did not re-check that.
- **7 of 9 is a ceiling, not a rate**, and 6 of 9 for rolled rows.

## 6. What is NOT claimed

- **Not that an agent wrote these kernels.** None of the loop's 26 candidates was one of them.
- **Not that these are fast.** Every probe kernel does the reference's work and then more; none is
  a reward hack the way the companion's constant fill is. They are blind spots, not speedups.
- **Not a fix.** The bundles and the decisive arm are unchanged on purpose. Closing these rows is the
  bundle-completion experiment's job (E2), and this record is part of the baseline it is measured
  against. When the bundles are completed, `test_symmetry_probe.py` is meant to fail on the six
  probe rows; they are then re-recorded, not loosened until they pass.
