# Pre-registration: does one tolerance for the log-ratio law accept every correct softmax, device runs included, and reject every wrong family?

**Date:** 2026-10-14 (registered before any code for this measurement exists; results are
appended below the line, never edited above it)
**Segment:** generation, as design input for E2 (complete the bundle). Translation inherits the
answer, because the same law would gate a port with no reference.
**Instance:** Apple M4 Pro, CPU only; the project's pinned environment (NumPy 1.26.4)
**Companions:** `2026-10-01-softmax-temperature-blind-spot.md` (the log-ratio column, first
printed), `2026-10-14-symmetry-probe.md` (two more softmax families), and
`2026-08-17-first-hardware-run.md` / `2026-08-17-at-scale-hardware-run.md` (the device runs
re-scored here)

## 0. The question

Proposition 1(b) says the softmax bundle plus the **log-ratio law** (log y_ij − x_ij constant
along each row) admits softmax and nothing else, over the reals. E2 must turn that law into a
floating-point check, and its pre-registered success criterion includes *no new false positive*
on correct kernels, device groups among them. The paper leaves the law's float form open. This
measurement asks the narrow question that decides whether a simple form can work: **is there one
threshold on the law's statistic that every correct softmax we hold stays under, on the CPU and
on the recorded A10 runs, while every wrong family we hold exceeds it?** It changes no bundle, no
arm and no gate.

## 1. Construction

**The statistic.** For a case with recorded float32 input x (m × n) and output y, in float64:
d_ij = log y_ij − x_ij over the entries with y_ij > 0; per row, spread_i = max_j d_ij − min_j
d_ij; the case's spread is max_i spread_i divided by float32 eps (2⁻²³). Entries with y_ij = 0
are excluded from the spread and counted separately; so are subnormal y_ij. A one-column row has
spread 0 by definition. This is `saboteurs.log_ratio_spread_eps` with the underflow made
explicit instead of turning the spread into ∞.

**The threshold examined:** 30, the gate's existing ratio threshold (`DEFAULT_THRESH`). Chosen
now, not fitted to the results.

**Correct kernels (the law must accept every case):**

| group | kernels | cases |
|---|---|---|
| NumPy, gate cases | `softmax_reference` (float64, cast); the float32 stock kernel (`saboteurs.softmax_beta(1.0)`); NumPy float32 analogues of the two correct Triton rewrites, `exp2(x·log2 e)` and multiply-by-reciprocal; every loop candidate's `softmax` in `candidates/` except fast-exp, and `best/train.py`'s | `pbt_gate` cases: seed 42, 9 groups, base + `shift_rows`, 18 per kernel |
| A10, recorded | the stock Triton softmax in `runs/gpu-softmax/`, `runs/gpu-tolerance_sweep/` and `runs/gpu-scale/` | 18 + 7 + 5 executions, read offline with `ExecutionTable` |

**Wrong kernels (the law must reject):** softmax(2x), softmax(0x), softmax(−x), the constant
1/n fill, rows rolled by one, columns reversed; gate cases. **Reported, not classified:** fast-exp
(`candidates/r3_softmax_fastexp.py`), whose admission the paper leaves to a stated accuracy
budget, and the positive control exp(x − max) undivided, whose log-ratios are exact by
construction and which the row-sum law already rejects.

**Reported per case:** shape, the spread in eps, the row range max_j x − min_j x of the worst row,
and the counts of zero and subnormal outputs.

## 2. Already known, so not predictions

From the companion records, on the gate cases: the stock float32 kernel 4.8 eps, fast-exp 48,
the positive control 4.3, the temperature family 5.2e7 (β = 2 and β = 0) and 1.04e8 (β = −1),
rows rolled 6.76e7, columns reversed 8.72e7, the float64 reference 0.956. A throwaway check of the
device runs has not been made; their spreads are unknown to me at registration.

## 3. Predictions

- **P1 (CPU, correct).** Every correct NumPy kernel's worst gate-case spread is at most **8 eps**.
- **P2 (device, correct).** Every recorded A10 execution's spread is below **30 eps**.
- **P3 (no underflow).** No correct case, CPU or device, has an output equal to 0 or subnormal.
  These domains are N(0, 1) inputs, and shifts move a row uniformly.
- **P4 (one threshold works).** At 30: every correct case is accepted, every wrong family is
  rejected on every multi-column group, and fast-exp (48) is rejected.
- **P5 (margin).** The smallest wrong-family spread exceeds the largest correct spread by at
  least **10⁶**.

**Falsifier.** Any correct case at or above any wrong family's spread: then no single threshold on
this statistic separates them, and E2 needs a different float form (for example, a bound that
scales with the row range).

## 4. What is NOT measured

- **Masked rows.** The model's real call site sends −∞ scores, where y = 0 and log y − x is
  undefined. No recorded run or gate case contains −∞; the law's handling of masked entries is
  E2's design, not this record's.
- **The correct Triton rewrites on device.** The A10 runs of the `exp2` and reciprocal-multiply
  kernels behind the paper's 54-group false-positive figure are not versioned in `runs/`, so only
  their NumPy analogues are measured. E2's device criterion is therefore only partly answered.
- **Layernorm.** Its completing law, the affine tie, needs its own float form.
- **Any agent or loop.** This is an offline measurement over fixed kernels.

## 5. Deviation rules

- The statistic, the threshold and the kernel list do not change after the first number is seen.
- A correct case that underflows is reported with its count, never dropped.
- If a kernel cannot be loaded or run, it is reported as such, not replaced.

---
