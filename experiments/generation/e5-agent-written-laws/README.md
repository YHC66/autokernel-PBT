# E5: agent-written laws, with and without a proof obligation

Pre-registration, instrument and results:
[`docs/measurements/2026-11-13-e5-agent-written-laws-prereg.md`](../../../docs/measurements/2026-11-13-e5-agent-written-laws-prereg.md).

An agent writes a task's laws (`laws.py`) from the function's definition. In arm TF it also writes `Laws.lean`, and
the loop checks a proof that the laws determine the function. Every final `laws.py` is scored on the shipped gate's
ladder plus the call-site shape. The corpus is the 10 families the current bundle admits, E3b's wrong kernels, and
E3's correct ones.

```bash
python experiments/generation/e5-agent-written-laws/e5.py control          # Lean controls, no model
python experiments/generation/e5-agent-written-laws/e5.py run TF softmax r1
python experiments/generation/e5-agent-written-laws/e5.py all              # every counted replicate
python experiments/generation/e5-agent-written-laws/e5.py score && python experiments/generation/e5-agent-written-laws/e5.py report
```

Needs `sandbox-exec` (macOS), the `claude` CLI, and the Lean toolchain pinned in `proofs/softmax-bundle/lean-toolchain`
under `~/.elan/toolchains/`.
