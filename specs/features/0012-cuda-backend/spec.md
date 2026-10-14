# Feature 0012: CUDA backend, and the CUDA ports of the known Triton ladder

## Problem

The translation segment maps a **known** kernel from one setup to another, and the first pair the
project chose (2026-10-11) is GPU to GPU: the recorded Triton ladder (`kernels/triton/ladder.py`) to
CUDA C++ on the same NVIDIA device. Both kernels then run in one session on byte-identical inputs,
and the source's recorded outputs are a reference on the same hardware. Nothing can record a CUDA
kernel today: `kernels/cuda/` is empty and there is no CUDA backend.

The backend is also the place the one unrecoverable decision of a hardware run is made — what the
telemetry captures — so it is specified, and tested on CPU, before any rented hour.

Feature numbers 0010 and 0011 are reserved for the formal track's adapters and pilot run (ADR 0002).

## Design

**Toolchain: NVRTC, through NVIDIA's `cuda-python` bindings.** A CUDA kernel is a source string
compiled at run time to PTX for the device's virtual architecture, loaded through the driver, and
launched on torch's current stream against torch's device buffers. This was chosen over
`torch.utils.cpp_extension.load_inline` and over CuPy:

- NVRTC compiles a kernel in well under a second, where a C++ extension build takes tens of
  seconds per kernel — and the translation mutation corpus (TE3) and the agent loop (TE4) compile
  hundreds of candidate ports.
- A compile failure is a typed NVRTC result code with a log, not a substring of a build error, so
  `COMPILE_ERROR` is classified by type as feature 0007 requires. A rejected source, or rejected
  options the kernel itself asked for, is the kernel never compiling (`CudaCompilationError`).
  Any other NVRTC failure, and any driver call outside a launch, is the toolchain's
  (`CudaToolchainError`), and the backend *raises* it rather than booking it against the kernel —
  it is a bad call, and it would fail every later kernel the same way.
- The driver's `cuFuncGetAttribute` reports registers, local memory and static shared memory
  directly, and the PTX is in hand to hash. The architecture is passed as
  `--gpu-architecture=compute_XY`, the spelling NVRTC documents (nvcc's `--gpu-arch` is not one).
- The buffers stay torch's, so `device_digest` (the input-mutation check) and
  `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (the sanitizer contract) carry over unchanged.

The bindings are probed, not asserted: `cuda.bindings.{driver,nvrtc}` first, the older
`cuda.{cuda,nvrtc}` second. The toolchain compiles a self-test kernel when it is constructed, so a
missing library or a rejected option is a bad *call*, raised before any execution is recorded,
rather than one `LAUNCH_ERROR` per row.

**Adapter.** `CudaKernel` wraps the source, the entry point, the launch-geometry functions and a
launcher, and is callable as `kernel(**numpy_inputs) -> ndarray` exactly as `TritonKernel` is, so
the two backends stay substitutable. It derives the grid and block from the input's shape and
hands both to the launcher, which may not compute its own; it compiles before launching, so the
artifact is the adapter's own record and a compile failure surfaces as `CudaCompilationError`.
Per-call state is cleared first, so a failed call cannot report the previous call's artifact. Its
`source_hash` covers the kernel text and the launcher's source; unreadable source is `MISSING`.

**Ports.** `kernels/cuda/{relu,softmax,layernorm}.cu`, one self-contained file per kernel so that
each file's text is that kernel's identity. One block per row; threads stride across the row, so no
row width is too wide for the kernel. The block width is derived from the row,
`min(max(next_pow2(n_cols), 32), 1024)`: a power of two, because the shared-memory tree halves
it, and at most the device's 1024-thread limit. The reduction tree therefore *differs* from
Triton's whenever `n_cols > width` — which is the "correct but different" regime translation lives
in, and why a proof that the two compute the same function in exact arithmetic is worth having
(`proofs/cross-setup-reductions/`).

**Telemetry, schema version 2.** Version 1 was written for Triton and has nowhere to put a CUDA
launch: `extract` copies only `grid` and `constexprs` from the launch group, so a block size handed
to it would be silently dropped. Version 2 adds:

| group | key | CUDA | Triton |
|---|---|---|---|
| launch | `block` | threads per block, as derived and handed to the launcher | MISSING |
| launch | `dynamic_shared_bytes` | as derived and handed to the launcher | MISSING |
| launch | `compile_options` | NVRTC options, verbatim | MISSING |
| launch | `kernel_params` | scalar parameters other than shapes (layernorm's `eps`) | MISSING |
| compiled | `local_bytes` | `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` | MISSING |
| compiled | `static_shared_bytes` | `CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`: fixed-size `__shared__` arrays only | MISSING |
| compiled | `max_threads_per_block` | `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` | MISSING |
| compiled | `ptx_version`, `binary_version` | the corresponding attributes | MISSING |
| device | `nvrtc_version`, `cuda_bindings_version` | probed | MISSING |

`n_regs` and `ptx_hash` are populated for CUDA from `CU_FUNC_ATTRIBUTE_NUM_REGS` and the PTX text.
`shared_bytes` keeps its version-1 meaning on both setups — the shared memory one block is given.
Triton's `metadata.shared` is exactly that, since Triton's kernels use dynamic shared memory only;
on a CUDA row the backend sums static and dynamic. The driver's own attribute is the static part
alone, which for a port using `extern __shared__` is 0 while each block gets `width * 4` bytes, so
it is recorded under its own name. The launch geometry is the one derived for the call and handed
to the launcher, recorded as soon as it is derived — so a call that then failed to compile or
launch still says what it was about to run, exactly as Triton's `grid` does. `n_spills`, `num_warps` and `num_stages` are Triton's own quantities and are `MISSING` on a
CUDA row: the driver reports local memory, which includes spills but is not a spill count, so it is
recorded under its own name rather than under one that would mislabel it. Version 1 rows remain
readable; the version recorded on each row is what tells a reader which keys can exist.

## Scope

1. `cuda_toolchain.py`: the `CudaToolchain` protocol, the NVRTC implementation, `CudaCompilationError`.
2. `cuda_kernel.py`: the `CudaKernel` adapter.
3. `cuda_backend.py`: `CudaBackend`, with the status mapping and failure discipline of `TritonBackend`.
4. Telemetry schema version 2.
5. `kernels/cuda/`: the three ports and `ladder.py` (factories, launch contract, `KERNELS`).
6. `scripts/gpu_record.py --backend {triton,cuda}`, defaulting to `triton` so every recorded command
   still means what it meant.
7. Device tests (`tests/gpu/`), the bootstrap installing `cuda-python`, and a runbook section.

## Non-goals

- Scoring one setup's run against another's recorded outputs (feature 0013).
- HIP, TPU (Pallas), Trainium (NKI).
- Fast-math or other correct-but-different CUDA variants: those are TE2's population, written once
  the calibration experiment is pre-registered.
- Multi-tile or two-stage reductions; GEMM, attention.
- Autotuning, or any search over block widths.

## Acceptance

See [acceptance.yaml](./acceptance.yaml). As with feature 0007, nothing here needs a GPU to be
accepted: the device path is covered by `tests/gpu/`, run by hand on the instance, and is not an
acceptance criterion, because a criterion whose evidence exists only on rented hardware would leave
`tests/spec` red on every developer machine and in CI.
