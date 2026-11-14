# E3c: a wider ladder against E3b's hacks

Pre-registration and results: [`docs/measurements/2026-11-13-e3c-wider-ladder-prereg.md`](../../../docs/measurements/2026-11-13-e3c-wider-ladder-prereg.md).

No agent runs. E3b's 180 candidates and E3's 120 correct candidates, plus the starting kernel, are re-run in E3b's
sandboxed recorder on three ladders: the shipped one (L0), the shipped one plus the call-site shape (L1), and the
shipped one plus 8 random shapes (L2). The gate's decision, the declarative arm, is replayed on each recording, and
the reference arm is recorded beside it.

```bash
python experiments/generation/e3c-wider-ladder/e3c.py run      # about 2 minutes on 8 workers
python experiments/generation/e3c-wider-ladder/e3c.py report
```

Results go to `recorded/results.jsonl` (one line per kernel and ladder) and `recorded/report.md`.
