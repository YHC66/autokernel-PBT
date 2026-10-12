# Proofs about a verifier's primitives: Why3 1.8.2's real theories, audited for symmetry

`Why3RealTheories.lean` transcribes the axioms of Why3's `real.ExpLog`, `real.Square` and
`real.Trigonometry` (as published at https://www.why3.org/stdlib/real.html, why3doc 1.8.2) into
core Lean 4 — **no Mathlib** — and proves what their models look like over an abstract ordered
field. Why3 is the proof platform under Frama-C/WP and SPARK/GNATprove, so these are the axioms a
deductive proof about C or Ada numerical code reasons from.

| theorem | says | consequence for a kernel certificate |
|---|---|---|
| `why3ExpLog_scaled` | all ten `ExpLog` axioms hold for `(x ↦ exp (βx), y ↦ log y / β)` for every β > 0 | a proof that code "computes softmax w.r.t. `exp`" is equally a proof about softmax(βx), β > 0 |
| `why3ExpLog_rejects_nonpos` | β ≤ 0 is excluded (`Log_exp` at β = 0, `exp_increasing` at β < 0) | the theory pins the *direction* of `exp`, not its rate: the ranking-reversed softmax(−x) and the uniform 1/n are excluded, softmax(2x) is not |
| `softmaxRowWith_scaleExp` | softmax w.r.t. the rescaled `exp` is softmax at inverse temperature β | the admitted family is exactly the test bundle's temperature family |
| `certificate_is_a_family` | any theorem derived from the axioms transfers to the rescaled pair | the formal content of such a certificate is membership in a family |
| `value_freedom_pos/neg` | for x₀ > 0 and any y > 1 (x₀ < 0 and any 0 < y < 1) some model has `exp x₀ = y` | the numerical value of `exp` away from 0 is not determined by the theory |
| `no_approximation_bound_pos/neg` | for any value v computed without `exp`, and every ε ≥ 0 (x₀ > 0) or ε < 1/4 (x₀ < 0), some model puts `exp x₀` outside [v − ε, v + ε] | **no accuracy bound for a fast-exp, polynomial or bit-trick approximation of `exp` follows from the theory**, however loose |
| `pow_invariant` | Why3's `pow x y = exp (y log x)` is unchanged by the symmetry | an `exp2`/`pow`-based kernel computes the same thing in every model |
| `exp2_kernel_not_certifiable` | for any literal c and every ε, some model separates `pow 2 (c x₀)` from `exp x₀` by more than ε | an `exp2` softmax cannot be certified against an `exp` spec from `ExpLog` + `PowerReal` alone |
| `tangent_two_sided` | with `1 + x ≤ exp x` added: `1 + x₀ ≤ exp x₀` and `exp x₀ (1 − x₀) ≤ 1` in every model | the value is no longer free; halving via `Exp_sum` tightens the band to any accuracy |
| `value_pins` | one axiom fixing `exp` at one non-zero point forces β = 1 | injectivity (`Log_exp`) does the rest — the pattern Why3 uses for π |
| `tangent_rigid` | `1 + x ≤ exp x` for `exp` and its rescaling forces β = 1 | the exact, first-order fix; uses only `exp_positive` and `Exp_sum` |
| `why3Square_categorical` | any two models of `Square` agree on x ≥ 0 | a Why3 certificate for layernorm cannot be read as one for −LN: `Sqrt_positive` breaks the sign symmetry the test bundle has |
| `why3Trig_scaled_five` | `(cos (5x), sin (5x), atan x / 5)` with the same π satisfies all sixteen `Trigonometry` axioms | a rotary-embedding kernel certified against the theory could rotate at 5× the frequency (in general β ≡ 1 mod 4) |
| `why3Trig_rejects_two` | rescaling by 2 breaks `Cos_pi` | the trigonometric symmetry is discrete, not continuous |

Read together: a verifier's standard axiomatization of a transcendental primitive has a symmetry
group, and that group is exactly the set of wrong kernels its certificates cannot exclude. For
`exp` it is a continuous one-parameter family (temperature); for `cos`/`sin` a discrete one
(frequency aliasing); for `sqrt` it is trivial. `experiments/verification/formal-rigidity/rigidity.py` is the
sampled, language-neutral audit that finds the same symmetries in milliseconds; the record is
`docs/measurements/2026-10-02-why3-real-theories-symmetry.md`.

## Check it

```bash
~/.elan/toolchains/leanprover--lean4---v4.33.0/bin/lean proofs/why3-real-theories/Why3RealTheories.lean
```

Exit code 0, no `sorry`, no declared `axiom`, and every `#print axioms` line lists only
`propext`, `Classical.choice`, `Quot.sound`. `tests/proofs/test_why3_real_theories_lean.py` runs
exactly this (marker `formal`, skipped without the toolchain). Compile time: about one second.
