# Runbook: TE1 — the known Triton ladder and its CUDA ports, one GPU session

**Purpose.** Record both setups of the first translation pair on one device, in one session, on
byte-identical inputs — and, before that, prove the CUDA backend works on real hardware. The CUDA
path (NVRTC through `cuda-python`, the driver API) has never executed; everything about it was
tested on CPU against fakes. This session is where the fakes meet the real library.

**Pre-registration:** `docs/measurements/2026-10-12-te1-triton-to-cuda-prereg.md`. Read §4, the
deviation rules, before starting: no threshold, arm or corpus parameter changes once the first CUDA
row is recorded.

**Expected duration.** About an hour. Recording is seconds per task; the rest is bootstrap.

---

## 0. Before you launch anything

Everything in `2026-08-17-lambda-smoke-session.md` §0 applies: rent the **cheapest** NVIDIA GPU
(nothing here is compute-bound; what matters is that the device is *recorded*), and confirm SSH
access before anything else. Rows are recorded with the device's compute capability, so any NVIDIA
architecture NVRTC supports will do — but if an A10 (sm_86) is available, it makes the Triton rows
comparable with the August runs.

## 1. Bootstrap

```bash
ssh lambda
git clone https://github.com/suv11235/autokernel-PBT.git
cd autokernel-PBT
git checkout main                       # translation-cuda-backend merged as PR #16
./scripts/gpu_bootstrap.sh
```

The bootstrap now also installs `cuda-python` pinned to torch's CUDA major, builds the NVRTC
toolchain once (it prints `nvrtc {...} arch compute_XY: OK`), and runs every device test,
including `tests/gpu/test_cuda_kernels.py`.

**If any CUDA device test fails, stop and capture its output.** The likely failures, and what each
means:

| symptom | meaning | fix, while the instance is up |
|---|---|---|
| `cuda-python is not importable` | the pip install failed | install the matching `cuda-python==<major>.*` by hand |
| `the NVRTC toolchain cannot compile, load and resolve a trivial kernel` + `NVRTC_ERROR_INVALID_OPTION` | NVRTC does not know this GPU's architecture | a newer `cuda-python`/NVRTC, or pass `arch=` an older virtual architecture |
| the same message + a `cuInit` / `cuDeviceGet` failure | no driver context | check `nvidia-smi`; restart the instance |
| a compiled field MISSING in `test_compiled_telemetry_is_populated_on_device` | an attribute name moved in this binding release | add the location in `cuda_toolchain._ATTRIBUTES` and re-run |
| a status other than `compile_error` for the uncompilable kernel | NVRTC's result codes differ from what `_ptx` checks | fix the classification, re-run |
| `TypeError` from `cuLaunchKernel` | the kernel-parameter convention differs in this release | adjust `marshal`'s pointer type, re-run |
| `driver_version came back MISSING on device` | the driver probe could not read the driver (seen 2026-10-20: torch 2.7.0 has no `torch._C._cuda_getDriverVersion`; fixed by asking libcuda) | check `device.driver_version` against `cuDriverGetVersion`, re-run |

Debugging on the clock is the one avoidable cost, but these fixes are small and are only cheap now.
Commit any fix on the branch before recording, so the recorded `kernel_source_hash` and the code
agree.

## 2. Record both setups

Triton first: the known kernel. Then the ports.

```bash
for task in relu softmax layernorm tolerance_sweep softmax_at_scale; do
  python3 scripts/gpu_record.py --task "$task" --backend triton --out "runs/te1-triton-$task"
  python3 scripts/gpu_record.py --task "$task" --backend cuda   --out "runs/te1-cuda-$task"
done
```

Each prints a status histogram with its backend. **Anything other than all-`ok` is a finding**
(prediction P1), recorded, not retried blindly: `compile_error` means the port never compiled,
`launch_error` with `input_mutated` set means it wrote to its input.

## 3. Check the CUDA telemetry populated

```bash
python3 - <<'PY'
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.backends.telemetry import declared_keys
rows = ExecutionTable("runs/te1-cuda-softmax").read()
t = rows[0].telemetry
for k in declared_keys():
    print(f"  {k:28} {t.get(k)}")
print("\nempty:", [k for k in declared_keys() if t.get(k) is None])
PY
```

Expected empty (prediction P2): `n_spills`, `num_warps`, `num_stages`, `constexprs`,
`triton_version`. Anything else empty is a decision to make now; see §1's table.

## 4. Bring the runs home, verify, terminate

`softmax_at_scale` is about 225 MB per setup. On the instance:

```bash
tar czf te1-runs.tar.gz runs/te1-*
```

From your laptop, **into the git checkout, never a OneDrive folder** (OneDrive renames payloads;
see `CLAUDE.md`):

```bash
scp lambda:~/autokernel-PBT/te1-runs.tar.gz .
tar xzf te1-runs.tar.gz
```

Verify both setups pair before terminating — this reads every payload and checks the inputs are
byte-identical:

```bash
python3 - <<'PY'
from autokernel_pbt.props.cross_setup import pair_runs
from autokernel_pbt.props.table import ExecutionTable
for task in ("relu", "softmax", "layernorm", "tolerance_sweep", "softmax_at_scale"):
    src, sfp = ExecutionTable(f"runs/te1-triton-{task}").read_with_fingerprint()
    tgt, tfp = ExecutionTable(f"runs/te1-cuda-{task}").read_with_fingerprint()
    pair_runs(src, tgt, source_fingerprint=sfp, target_fingerprint=tfp)
    print(f"{task:18} {len(src):3} source rows, {len(tgt):3} target rows: paired")
PY
```

Then terminate the instance and check the console shows it terminated.

## 5. Score at home, on CPU

```python
from pathlib import Path
from autokernel_pbt.props.driver import score_translation
from autokernel_pbt.props.tasks import REFERENCES, TASKS

for task in ("relu", "softmax", "layernorm", "tolerance_sweep", "softmax_at_scale"):
    score_translation(
        task=TASKS[task],
        source_dir=f"runs/te1-triton-{task}",
        target_dir=f"runs/te1-cuda-{task}",
        repo_root=Path("."),
        reference_fn=REFERENCES[task],
        arm_order_seed=0,
    )
```

Results are appended below the pre-registration's line, then the runs are committed with that
record (`runs/README.md`).
