# Pre-registration: TE1 — do the laws and the cross-setup ratio hold when the known Triton ladder is mapped to CUDA?

**Date:** 2026-10-12 (registered before any CUDA kernel has run on a device; results are appended
below the line, never edited above it)
**Segment:** translation. First backend pair, GPU first (decided 2026-10-11): Triton → CUDA C++ on
one NVIDIA GPU.
**Code at registration:** branch `translation-cuda-backend` — features 0012 (CUDA backend, NVRTC)
and 0013 (cross-setup scoring), both CPU-tested; the CUDA path itself has never executed.
**Companions:** `docs/pipelines/translation.md` (T0–T4, TE1–TE4),
`proofs/cross-setup-reductions/` (the exact-arithmetic consistency proof of the same port, in Lean
and Rocq), `2026-08-17-first-hardware-run.md` and `2026-08-19-false-positive-rate.md` (the
single-backend numbers these predictions are anchored to).

## 0. The question

The translation segment maps a *known* kernel from one setup to another. The project's claim for it
is that the Tier-1 laws are the portable equivalence contract and that a calibrated ratio against
the known kernel's own recorded outputs accepts a correct port without the false alarms `allclose`
raises. TE1 is the first test of that claim on a real port: the recorded Triton ladder kernels
(`kernels/triton/ladder.py`, the known kernels) against their CUDA ports (`kernels/cuda/*.cu`),
both recorded in one session on one device, on byte-identical inputs.

The exact-arithmetic half is already settled by proof: in Lean and Rocq, the Triton tile model and
the CUDA block model compute the same function as the spec, for every row width and every
power-of-two block width (`proofs/cross-setup-reductions/`). What remains is the floating-point
half, which only an execution can answer: how far apart do two correct kernels with different
reduction trees and different `exp` implementations land, and do the arms tell that apart from a
defect?

## 1. Construction

| | |
|---|---|
| device | one NVIDIA GPU, the cheapest available (the runbook's rule); device, driver, CUDA, torch, Triton, NVRTC and `cuda-python` versions are recorded on every row (telemetry schema v2) |
| source setup | the Triton ladder, **re-recorded in the same session** — not the August runs, which carry schema v1 and another session's driver |
| target setup | the CUDA ports, NVRTC at the device's virtual architecture, no fast-math, default `--fmad` |
| tasks | `relu`, `softmax`, `layernorm`, `tolerance_sweep`, `softmax_at_scale` — every task both ladders key |
| corpus | `Generator`, seed 42, every shape in each domain (the recorder's default), the domain's relations |
| recording | `python3 scripts/gpu_record.py --task <t> --backend triton --out runs/te1-triton-<t>` then `--backend cuda --out runs/te1-cuda-<t>`; stated correct (`kernel_is_broken = False`) |
| scoring (offline, CPU) | `driver.score_translation(source=runs/te1-triton-<t>, target=runs/te1-cuda-<t>, reference_fn=REFERENCES[<t>], arm_order_seed=0)`: seven arms over the CUDA rows. `allclose`, `reference`, `declarative`, `hybrid` against the float64 NumPy reference; `allclose_source` (T0), `reference_source` (T1), `hybrid_source` against the Triton recording. Thresholds unchanged: ratio < 30, `allclose` at NumPy's defaults |
| reported | per task and arm, groups PASS / FAIL / INCONCLUSIVE; per task, min / median / max `reference_source` ratio and the same for `reference`; per task, the fraction of output elements bitwise equal between the two setups; the CUDA telemetry fields that came back MISSING |

Where the two trees coincide and where they do not: for rows up to 1024 columns the CUDA block is
`max(next_pow2(n), 32)` threads, so each thread holds at most one column and the CUDA reduction is
a balanced tree in shared memory, while Triton reduces a tile of `next_pow2(n)` lanes through its
own lowering (per thread, across lanes, across warps). Above 1024 columns — `tolerance_sweep`'s
4095, 4096 and 16384, and `softmax_at_scale`'s 4096 and 8192 — each CUDA thread first folds
`n / 1024` columns sequentially. Whether Triton's `tl.exp` and CUDA's `expf` round identically is
not known to us at registration, and is not assumed either way.

## 2. Predictions, registered before the first run

- **P1 (execution).** Every row of both setups is `ok`. A CUDA row with any other status is a defect
  in the port or the backend, recorded as a finding, and stops the run before scoring (§4).
- **P2 (telemetry).** On every CUDA row, `n_regs`, `local_bytes`, `max_threads_per_block`,
  `ptx_version`, `binary_version`, `ptx_hash`, `nvrtc_version` and `driver_version` are populated.
  `n_spills`, `num_warps` and `num_stages` are MISSING, by design.
- **P3 (relu is exact).** The CUDA and Triton relu outputs are equal (`==`) on every element of
  every case: `max(x, 0)` involves no rounding.
- **P4 (correct but different is real on device).** For softmax and layernorm, at least one case
  with more than one column has outputs that are not bitwise equal between the setups. If none
  does, the cross-setup tolerance question is moot for the ladder, and the record says so.
- **P5 (T1 accepts the correct port).** `reference_source` rejects **0** groups on every task, and
  its largest ratio on any case is below **3**, ten times under the threshold. Anchor: the largest
  correct-kernel ratio against the float64 reference on the A10 was 0.327; two float32 kernels can
  each contribute their own error.
- **P6 (T0 raises false alarms on the correct port).** `allclose_source` rejects **at least 1** of
  the 9 layernorm groups — the translation analogue of `allclose`'s 5/9 on correct layernorm
  kernels, where `atol` sits below float32's resolution for outputs near zero.
- **P7 (the laws transfer; TE1's falsifier).** The `declarative` arm rejects **0** groups on the
  CUDA ports, on every task. The experiment's registered falsifier is any Tier-1 law failing on a
  group that `reference_source` accepts.
- **P8 (absolute correctness).** `reference` (against the float64 NumPy reference) rejects 0 groups
  on the CUDA ports, as it did on the Triton ladder.

## 3. What is NOT being measured

- Detection of wrong ports. That is TE3, which needs a blinded translation mutation corpus.
- An agent porting anything. The CUDA ports were written for the infrastructure (feature 0012) and
  stated correct; TE1 tests the instrument on a known-correct pair, not an agent (TE4).
- A cross-setup *tolerance*. P5 uses the existing `log2 n` ratio; if it fails, TE2 derives one.
- Speed. Wall-clock is recorded on every row, and neither kernel is tuned; no performance claim is
  made from TE1.

## 4. Deviation rules

- If a CUDA row fails (P1), the session records the failure and the run stops. A port fixed after
  seeing device output is a different kernel: it gets a new section below the line, dated, stating
  what changed, and its runs a new directory.
- If the CUDA telemetry comes back MISSING on device (P2), the probe is fixed while the instance is
  up (the runbook's rule) and the recording repeated; the first recording is kept, not overwritten.
- No threshold, tolerance, arm or corpus parameter may change after the first CUDA row is recorded.

---

## Results — 2026-10-20, one session on an RTX A6000

**Device.** Lambda Cloud `gpu_1x_a6000` (us-south-2): NVIDIA RTX A6000, sm_86, 84 SMs, 48 GB.
Lambda Stack 22.04: driver 580.105.08, whose driver API reports CUDA **13.0**; torch 2.7.0 built
against CUDA **12.8**; Triton 3.3.0; NVRTC 12.8; `cuda-bindings` 12.9.10; NumPy 1.26.4. The A6000
($1.09/hr) was the cheapest GPU with capacity at launch, which is §1's rule. An A10 ($1.29/hr) was
also available. The runbook prefers an A10 only so that Triton rows can be compared with the
August runs, and this construction does not need that, because §1 re-records the source in the
same session. The device's 84 SMs (the A10 has 72) are recorded on every row.

**Code.** `main` at 7d04d5a, plus dabe4e1 on branch `te1-a6000-session` (below). Every row of
both setups was recorded at dabe4e1.

**Before any row: the telemetry probe was fixed (§4, P2).** The bootstrap's CPU suite passed (910
passed, 6 skipped). Its device tests then failed 3 of 24, all on one field: `driver_version` came
back MISSING on every CUDA kernel. `device.driver_version` called
`torch._C._cuda_getDriverVersion`, which torch 2.7.0 does not have. The field is shared by both
backends, so the Triton rows would have recorded MISSING too. The fix, dabe4e1, asks libcuda's
`cuDriverGetVersion` directly. After it, the device tests passed 24 of 24, and then recording
began. **P2 was false on the code as registered.** §4's telemetry rule applied, and no recording
had been taken yet, so there is no first recording to keep. The fixed probe shows that the driver
(13.0) and the toolkit torch was built against (12.8) differ on this image. That difference is
what the field exists to show, and before PR #11 it was recorded as the toolkit version.

**Recording.** Both setups were recorded per the runbook, Triton first, under
`runs/te1-{triton,cuda}-<task>/`. `pair_runs` accepted all five pairs on byte-identical inputs
before the instance was terminated. Scored offline on CPU with `score_translation(...,
arm_order_seed=0)`. The summary is regenerated by
`experiments/translation/te1/summarize.py`.

### Predictions

| | Prediction | Observed | Verdict |
|---|---|---|---|
| P1 | every row of both setups `ok` | 62 of 62 Triton rows and 62 of 62 CUDA rows `ok` | **holds** |
| P2 | the eight compiled fields populated on every CUDA row; `n_spills`, `num_warps`, `num_stages` MISSING | after dabe4e1: all eight populated on 62 of 62 CUDA rows. Empty on every row: `n_spills`, `num_warps`, `num_stages` (by design), and `triton_version`, `constexprs` (Triton-only, as the runbook expects) | **failed as registered; holds after the §4 fix** |
| P3 | relu outputs bitwise equal | 1811 of 1811 output elements equal | **holds** |
| P4 | softmax and layernorm: some multi-column case not bitwise equal | softmax 14 cases, layernorm 14 cases; 37.2% and 44.9% of elements bitwise equal | **holds** |
| P5 | `reference_source` rejects 0 groups; largest ratio < 3 | 0 groups rejected on every task; largest ratio **0.986** (layernorm, g00003) | **holds** |
| P6 | `allclose_source` rejects ≥ 1 of 9 layernorm groups | **2 of 9** (g00004, g00005) | **holds** |
| P7 | `declarative` rejects 0 groups on every task (the falsifier) | 0 rejected on every task. Layernorm's g00006 and g00008 are INCONCLUSIVE: every row in them has a constant input, so the unit-variance law defers. This is the same designed 2/9 abstention as the August record | **holds; falsifier not triggered** |
| P8 | `reference` rejects 0 groups on the CUDA ports | 0 on every task | **holds** |

### Group verdicts on the CUDA rows (PASS / FAIL / INCONCLUSIVE)

| task (groups) | `allclose` | `reference` | `declarative` | `hybrid` | `allclose_source` (T0) | `reference_source` (T1) | `hybrid_source` |
|---|---|---|---|---|---|---|---|
| relu (9) | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 |
| softmax (9) | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 |
| layernorm (9) | 4 / **5** / 0 | 9 / 0 / 0 | 7 / 0 / 2 | 7 / 0 / 2 | 7 / **2** / 0 | 9 / 0 / 0 | 7 / 0 / 2 |
| tolerance_sweep (7) | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 |
| softmax_at_scale (5) | 5 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 |

Every FAIL in the table is a false alarm, since both setups are correct. They all come from the
two `allclose` arms on layernorm. `allclose` against float64 rejects 5/9 groups, the same count as
on the August Triton layernorm. Against the known kernel it rejects 2/9.

### Ratios (min / median / max over case results) and bitwise agreement

| task | `reference`, CUDA | `reference`, Triton (context) | `reference_source` | elements bitwise equal |
|---|---|---|---|---|
| relu | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 100% of 1,811 |
| softmax | 0 / 0.0847 / 0.327 | 0 / 0.0847 / 0.327 | 0 / 0.144 / 0.299 | 37.2% of 3,622 |
| layernorm | 0 / 0.166 / 1.00 | 0 / 0.193 / 1.00 | 0 / 0.230 / 0.986 | 44.9% of 3,622 |
| tolerance_sweep | 0.0379 / 0.0552 / 0.327 | 0.0552 / 0.0913 / 0.327 | 0.0552 / 0.0819 / 0.0997 | 25.3% of 101,152 |
| softmax_at_scale | 0.0479 / 0.0667 / 0.100 | 0.0479 / 0.0580 / 0.100 | 0.0546 / 0.0957 / 0.201 | 28.8% of 58,720,256 |

The Triton column was not pre-registered. It was computed in memory by the same `ReferenceOracle`
over the source rows, and nothing was written into the source runs. It is here so the port can be
read against the known kernel's own error. On every task the port's largest distance from float64
equals the known kernel's. Layernorm's maximum of 1.00 occurs on both setups, so it is not a
porting artifact. The softmax maximum, 0.327, matches the A10 figure P5 was anchored to. The
cross-setup ratio is at most one kernel's own error on four tasks; on layernorm, 0.986 against
1.00. On `softmax_at_scale` it is about two: 0.201, against 0.100 for each kernel. Two float32
errors adding is what P5's tenfold margin was registered for.

### What TE1 does and does not establish

- **Established, on one device:** the CUDA backend works on hardware, after one telemetry fix.
  A correct port of the known kernels differs from them bitwise on most elements of every
  reduction, while computing the same function. Against such a port, T1 (`reference_source`) and
  the Tier-1 laws raise no false alarm, with T1's largest ratio 30× under its threshold. T0
  (`allclose_source`) raises 2 on 9 layernorm groups. This is the translation form of result 1 in
  `findings.md`.
- **Not established:** detection of wrong ports, which is TE3; any agent porting anything, which
  is TE4; a cross-backend tolerance, which P5 shows the ladder does not yet need; speed; any
  device other than this one.
