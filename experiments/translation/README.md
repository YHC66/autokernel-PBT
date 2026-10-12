# Translation experiments

**Nothing has run here yet.** This segment is for experiments in which an agent ports a kernel
between backends. GPU comes first: CUDA ↔ Triton on NVIDIA, then CUDA → HIP on AMD, and later TPU
(Pallas/JAX) and Trainium (NKI).

- The proposed pipelines (T0–T4) and experiments (TE1–TE4), with what exists and what each still
  needs: [`docs/pipelines/translation.md`](../../docs/pipelines/translation.md).
- Evidence so far: Tier-1 laws moved NumPy → Triton unchanged on an A10
  ([`first-hardware-run`](../../docs/measurements/2026-08-17-first-hardware-run.md),
  [`at-scale-hardware-run`](../../docs/measurements/2026-08-17-at-scale-hardware-run.md)).

The first experiment to land here is TE1 on GPU: the Triton ladder kernels ported to CUDA and
recorded on one NVIDIA GPU. It needs a CUDA backend under `src/autokernel_pbt/props/backends/`,
built the way the Triton one was (feature 0007), spec first.
