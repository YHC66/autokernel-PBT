# Recorded GPU runs

Every execution this project has recorded on a device, version controlled. A run is not
regenerable: the inputs are a pure function of the seed, but the outputs are what one GPU, one
driver and one compiler produced on one day. Scoring is always offline, so any oracle can be run
over these runs again on a CPU, at any time, without hardware.

| Run | Task | Rows | Groups | Recorded | Record |
|---|---|---|---|---|---|
| `gpu-relu/` | relu | 9 | 9 | 2026-08-17 | [first hardware run](../docs/measurements/2026-08-17-first-hardware-run.md) |
| `gpu-softmax/` | softmax (with `shift_rows` partners) | 18 | 9 | 2026-08-17 | same |
| `gpu-layernorm/` | layernorm (with `shift_rows` partners) | 18 | 9 | 2026-08-17 | same |
| `gpu-tolerance_sweep/` | tolerance_sweep | 7 | 7 | 2026-08-17 | same |
| `gpu-scale/` | softmax_at_scale, up to 29.4 M elements | 5 | 5 | 2026-08-17 | [at-scale hardware run](../docs/measurements/2026-08-17-at-scale-hardware-run.md) |
| `te1-{triton,cuda}-relu/` | relu | 9 each | 9 | 2026-10-20 | [TE1](../docs/measurements/2026-10-12-te1-triton-to-cuda-prereg.md) |
| `te1-{triton,cuda}-softmax/` | softmax (with `shift_rows` partners) | 18 each | 9 | 2026-10-20 | [TE1](../docs/measurements/2026-10-12-te1-triton-to-cuda-prereg.md) |
| `te1-{triton,cuda}-layernorm/` | layernorm (with `shift_rows` partners) | 18 each | 9 | 2026-10-20 | [TE1](../docs/measurements/2026-10-12-te1-triton-to-cuda-prereg.md) |
| `te1-{triton,cuda}-tolerance_sweep/` | tolerance_sweep | 7 each | 7 | 2026-10-20 | [TE1](../docs/measurements/2026-10-12-te1-triton-to-cuda-prereg.md) |
| `te1-{triton,cuda}-softmax_at_scale/` | softmax_at_scale (with partners) | 10 each | 5 | 2026-10-20 | [TE1](../docs/measurements/2026-10-12-te1-triton-to-cuda-prereg.md) |

All five ran on one Lambda Cloud NVIDIA A10 (sm_86, 72 SMs) with Triton 3.3.0, PyTorch 2.7.0 and
CUDA 12.8, seed 42. Every execution's status is `ok`. `gpu-scale/` is 224 MB, and its largest
payload is 64 MB. Everything else totals about 1 MB.

The `te1-*` runs are one translation pair per task: the known Triton kernel (`te1-triton-*`) and its
CUDA port (`te1-cuda-*`), recorded on byte-identical inputs in one session on a Lambda Cloud NVIDIA
RTX A6000 (sm_86, 84 SMs). The stack was Triton 3.3.0, PyTorch 2.7.0 built against CUDA 12.8,
NVRTC 12.8, a driver reporting CUDA 13.0, and seed 42. Every execution's status is `ok`. They carry
telemetry schema v2 and the current on-disk format. Each `te1-cuda-*` directory also holds its
seven-arm scores and the `pairing.json` that names its source. Each `softmax_at_scale` run is
448 MB, with a largest payload of 64 MB; the other eight total about 2 MB.

## Reading a run

```python
from autokernel_pbt.props.table import ExecutionTable
rows = ExecutionTable("runs/gpu-softmax").read()          # list[ExecutionResult]
groups = ExecutionTable("runs/gpu-softmax").read_groups()  # {group_id: [base, partner, ...]}
```

`rows.parquet` is the index (case metadata, status, telemetry), and `tensors/` holds one
safetensors payload per case: inputs plus outputs.

## Known properties of these runs

- **They predate the current on-disk format.** Their fingerprints use the old bare 64-hex form, and
  their payloads are unstamped, so `ExecutionTable` reads them through its *legacy* path. Runs
  recorded now are not readable by code from before PR #11.
- **Partner payload names were restored.** The ids of metamorphic partners contain `::`
  (`softmax-g00000-base::shift_rows`). OneDrive renamed those 18 files to `…base_shift_rows`, which
  made the softmax and layernorm runs unreadable. They are committed under their portable encoded
  names (`…base%3a%3ashift_rows.safetensors`, see `payload_filename`), which the legacy reader
  tries first. The bytes were not touched. All 62 files match an independently restored copy
  byte for byte, and re-scoring every run with every arm reproduces all 523 recorded results,
  detail strings included.
- **Two telemetry fields are weaker than their names.** `driver_version` holds the CUDA toolkit
  version, and `kernel_source_hash` covers only the launcher and the kernel's name, not its
  body. Both were fixed in PR #11 for later runs, so hashes from these runs are not comparable
  with new ones.

## Adding a run

Record on the device with `scripts/gpu_record.py --task <task> --out runs/gpu-<name>` (see
[`docs/runbooks/`](../docs/runbooks/)). Bring the directory home as a tarball, extract it into
`runs/` in a clone that is **not** in a synced folder, add a row above, and commit the run in the
same PR as the record that cites it. Payloads larger than 100 MB cannot be pushed to GitHub, so
split a task across groups or ask before adding Git LFS.
