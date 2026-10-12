# Translation experiments

**Nothing has run here yet.** This segment is for experiments in which an agent ports a kernel
between backends: CUDA or Triton → TPU (Pallas/JAX), Triton → Trainium (NKI), and so on.

- The proposed pipelines (T0–T4) and experiments (TE1–TE4), with what exists and what each still
  needs: [`docs/pipelines/translation.md`](../../docs/pipelines/translation.md).
- Evidence so far: Tier-1 laws moved NumPy → Triton unchanged on an A10
  ([`first-hardware-run`](../../docs/measurements/2026-08-17-first-hardware-run.md),
  [`at-scale-hardware-run`](../../docs/measurements/2026-08-17-at-scale-hardware-run.md)).

The first experiment to land here is likely TE1, the laws on a second target backend. It needs a
backend adapter under `src/autokernel_pbt/props/backends/`, built the way the Triton one was
(feature 0007), spec first.
