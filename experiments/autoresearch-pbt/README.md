# autoresearch-pbt

An autoresearch-style optimization loop (karpathy/autoresearch), transplanted to
CPU/NumPy, with this repository's property oracles as the correctness gate.

```
prepare.py   fixed: data, byte tokenizer, fixed eval windows, val_bpb, budget
train.py     the agent edits this: NumPy GPT, AdamW, training loop
pbt_gate.py  the four oracle arms over train.py's softmax / layernorm
runner.py    the outer loop: gate -> measure -> keep/revert, ledger in results.tsv
program.md   agent instructions
best/        the current best train.py
results.tsv  the ledger
```

Loop:

```bash
../../.venv/bin/python runner.py baseline
# an agent edits train.py, then:
../../.venv/bin/python runner.py evaluate "note about the change"
../../.venv/bin/python runner.py rigor 3
```

The gate runs **before** the metric, and its decisive arm is the **declarative**
one, not `allclose` — `allclose` false-positives on the baseline's own correct
layernorm (5 of 9 groups), so a loop gated on it cannot accept its own starting
point.
