# Verification experiments

Experiments on the correctness layer itself, which both task segments consume: how good the
property-based oracles are, and what formal proofs about them, or about the verifiers'
own axioms, can show. The oracle-arm comparison and the hardware runs use the library directly
(`src/autokernel_pbt/props/`, `scripts/gpu_record.py`) and are indexed by their records in
[`../README.md`](../README.md).

| Directory | What it is |
|---|---|
| [`formal-rigidity/`](./formal-rigidity/) | A sampled, language-neutral audit of whether a verifier's axioms for `exp`, `sqrt`, `sin`/`cos` admit rescaled models, which are wrong kernels that certificates cannot exclude. Companion to `proofs/why3-real-theories/` |
| [`oracle-research/`](./oracle-research/) | Pre-registered: agents complete a property bundle with tests only, or with tests and a Lean prover; plus a small adherence-proof experiment. Ran 2026-10-02; unscored |
