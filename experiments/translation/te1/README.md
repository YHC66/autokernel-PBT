# TE1: the known Triton ladder mapped to CUDA

Pre-registration and results:
[`docs/measurements/2026-10-12-te1-triton-to-cuda-prereg.md`](../../../docs/measurements/2026-10-12-te1-triton-to-cuda-prereg.md).
Runbook: [`docs/runbooks/2026-10-12-te1-translation-session.md`](../../../docs/runbooks/2026-10-12-te1-translation-session.md).
Runs: `runs/te1-triton-<task>/` (the known kernel) and `runs/te1-cuda-<task>/` (the port), one
RTX A6000, 2026-10-20.

`summarize.py` re-scores every pair with `score_translation` (seven arms, `arm_order_seed=0`) and
writes `summary.json`: per task and arm, group verdicts; min / median / max ratio for `reference`
and `reference_source`; and the fraction of output elements bitwise equal between the setups.
Run it from the repository root, on CPU:

```bash
python experiments/translation/te1/summarize.py
```

Re-scoring rewrites the scores and `pairing.json` inside `runs/te1-cuda-*`. Every verdict and
detail is deterministic, and `summary.json` reproduces exactly. Each arm's `elapsed_s` in
`scores.parquet` is wall-clock time, so that file changes on every re-run.
