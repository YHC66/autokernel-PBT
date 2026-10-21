# Translation experiments

The translation segment maps **known kernels** from one setup to another, using property-based
testing and formal verification together so that the port is correct, consistent with the kernel
it came from, and performant on the target. GPU comes first: Triton → CUDA on one NVIDIA GPU, then
CUDA → HIP on AMD, and later TPU (Pallas/JAX) and Trainium (NKI). The pipelines (T0–T4) and
experiments (TE1–TE4) are in [`docs/pipelines/translation.md`](../../docs/pipelines/translation.md).

**The first pair has run on a device once:** TE1, 2026-10-20, one RTX A6000. What exists:

| piece | where | status |
|---|---|---|
| CUDA backend: NVRTC, the driver API, telemetry schema v2 | `src/autokernel_pbt/props/backends/cuda_*.py`, feature [0012](../../specs/features/0012-cuda-backend/spec.md) | [M]: 62 of 62 rows `ok` on device in TE1, after one telemetry fix (`driver_version`) |
| CUDA ports of the known Triton ladder | [`kernels/cuda/`](../../kernels/cuda/) | [M]: correct on device against float64 and against the known kernel (TE1) |
| Scoring a port against the known kernel's recording (T0 `allclose_source`, T1 `reference_source`) | `src/autokernel_pbt/props/cross_setup.py`, `driver.score_translation`, feature [0013](../../specs/features/0013-cross-setup-scoring/spec.md) | [M]: TE1 |
| Proof that the Triton kernels and the CUDA ports compute the same function, exact arithmetic | [`proofs/cross-setup-reductions/`](../../proofs/cross-setup-reductions/): Lean, Rocq and Isabelle/HOL, six theorems each | [M] (machine-checked) |
| **TE1**: Triton → CUDA on one device | [pre-registration and results](../../docs/measurements/2026-10-12-te1-triton-to-cuda-prereg.md), [runbook](../../docs/runbooks/2026-10-12-te1-translation-session.md), [`te1/`](./te1/) | [M]: all 8 predictions hold |

Earlier evidence for the segment: Tier-1 laws moved NumPy → Triton unchanged on an A10
([`first-hardware-run`](../../docs/measurements/2026-08-17-first-hardware-run.md),
[`at-scale-hardware-run`](../../docs/measurements/2026-08-17-at-scale-hardware-run.md)).

## Running TE1

On a GPU instance, after `scripts/gpu_bootstrap.sh` (which installs `cuda-python` and runs the CUDA
device tests), record both setups in one session — the runbook has the full sequence:

```bash
python3 scripts/gpu_record.py --task softmax --backend triton --out runs/te1-triton-softmax
python3 scripts/gpu_record.py --task softmax --backend cuda   --out runs/te1-cuda-softmax
```

Then, at home on CPU:

```python
from pathlib import Path
from autokernel_pbt.props.driver import read_translation, score_translation
from autokernel_pbt.props.tasks import REFERENCES, TASKS

score_translation(
    task=TASKS["softmax"],
    source_dir="runs/te1-triton-softmax",
    target_dir="runs/te1-cuda-softmax",
    repo_root=Path("."),
    reference_fn=REFERENCES["softmax"],
)
rows, arms, pairing = read_translation("runs/te1-cuda-softmax", "runs/te1-triton-softmax")
```

`tests/integration/test_translation_scoring.py` runs exactly this path on CPU, with two NumPy
"setups" that round differently and a tail-dropping port standing in for the measured tile-tail
defect.
