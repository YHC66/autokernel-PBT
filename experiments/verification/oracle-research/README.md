# oracle-research: does a prover in the toolkit change what a research agent concludes?

The experiment registered in `docs/measurements/2026-10-02-prover-in-the-toolkit-prereg.md`:
fresh agents, each in a sealed sandbox, asked whether a task's property bundle characterizes
its reference and, if not, to complete it — with tests only (arm T) or with tests and a pinned
Lean 4 toolchain plus the obligation to prove completeness (arm F). A second, smaller experiment
asks agents to certify kernel adherence in core Lean 4.

```
heldout.py              the held-out wrong kernels and correct rewrites no agent saw (evaluation data)
evaluate.py             scores one oracle-research sandbox blind to its arm: soundness, held-out
                        completeness under the agent's completed bundle, Lean compile/axiom audit
evaluate_adherence.py   compiles and audits one adherence sandbox's Lean files
summarize.py            aggregates agent reports × evaluations into the tables the record quotes
audit_transcripts.py    greps the workflow's agent transcripts for reads outside the sandbox
```

The sandboxes themselves (protocol text, vendored property layer, gate, agents' artifacts) are
kept with the run, outside the repository; the record names their location and every agent's
`RESULT.md` is quoted from there. `heldout.py` must never be copied into a sandbox.

Run the evaluator from the repository root with the repo's venv, one sandbox per argument; it
loads each sandbox's own `gate.py` and `laws.py` by path so the bundle under test is exactly what
the agent left behind.

## Status (2026-10-11): ran, not yet scored

The experiment ran on 2026-10-02 under the pre-registration, including the relaunch recorded
there as a deviation. **11 of 24 cells returned a structured report**: all 8 softmax oracle cells
(4 arm T, 4 arm F), 2 of 8 layernorm cells (both arm T) and 1 of 8 adherence cells (k2, replicate
2). The sandboxes lived under `/private/tmp` and were wiped before any evaluation was recorded, so
`evaluate.py` can no longer be pointed at them. Two things survive, both in `runs/`:

- `reports.json`: each agent's own structured report. These are its claims, not an evaluation of
  them.
- `transcripts/`: all 54 agent sessions in the run's three waves (`1-initial`, then the two
  continuation waves recorded as a deviation), one JSONL file per cell. Alongside them,
  `workflows/` holds the workflow script, which contains every prompt the agents were given, and
  each wave's journal.

Each transcript keeps only the conversation: the prompt, the agent's messages and tool calls, and
the tool results, in order. Harness context entries and request metadata were dropped. Every file
an agent wrote or edited appears in its tool calls, so the sandboxes can be reconstructed from
them and then scored with `evaluate.py`. In all of these files, the user's email and every
machine-specific path are redacted to `<user-email>`, `<scratchpad>`, `<repo>` and `~`.

**No result has been appended below the pre-registration's line, and none of these reports may be
quoted as a finding until one is.** Next time, run `collect.py` the moment each agent
finishes.
