# Generation experiments

Experiments in which an agent writes or optimizes a kernel and a correctness gate decides what it
keeps. The pipelines these measure (G0–G5, O1) are in
[`docs/pipelines/generation.md`](../../docs/pipelines/generation.md).

| Directory | What it is |
|---|---|
| [`autoresearch-pbt/`](./autoresearch-pbt/) | karpathy/autoresearch's loop on a NumPy GPT, gated by the four oracle arms (pipeline G1), plus the saboteur matrix that pins what the gate admits |
