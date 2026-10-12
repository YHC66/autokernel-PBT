# formal-rigidity: which reinterpretations of its primitives does an axiom set survive?

A proof that a kernel adheres to its spec holds in *every* model of the axioms its primitives
(`exp`, `log`, `sqrt`, …) are given. If some transformation of the primitives preserves every
axiom (rescaling the argument of `exp`, for example), the certificate names a family of functions
rather than one. Check C6 of the certificate design asks whether each axiom is *true* on samples.
This audit asks whether the axiom set is *rigid*: whether anything other than the intended
interpretation also passes.

It is the sampled, language-neutral counterpart of the theorems in
[`proofs/why3-real-theories/`](../../../proofs/why3-real-theories/), and it finds the same
symmetries in seconds per theory: all of Why3's `real.ExpLog` admits `exp(βx)` for β > 0, `Square`
admits nothing, and `Trigonometry` admits a ×5 frequency alias. It also measured that C6, as
designed, rejects some *true* axioms whose two sides pass near zero, on up to 2.4% of samples. C6
needs a floored comparison.

```bash
# from the repository root, with the project venv
.venv/bin/python experiments/verification/formal-rigidity/rigidity.py      # print the audit table
.venv/bin/python -m pytest experiments/verification/formal-rigidity -q     # 11 tests pin it
```

Record: [`docs/measurements/2026-10-02-why3-real-theories-symmetry.md`](../../../docs/measurements/2026-10-02-why3-real-theories-symmetry.md).
