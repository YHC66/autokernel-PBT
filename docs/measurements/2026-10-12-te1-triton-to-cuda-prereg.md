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
