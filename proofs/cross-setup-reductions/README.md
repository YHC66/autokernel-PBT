# Cross-setup consistency: the ladder's reductions, in Lean, Rocq and Isabelle

This directory holds one theorem family, stated once below and proved independently in each
prover, by a separate agent working from this statement alone:

| prover | directory | status | checked by |
|---|---|---|---|
| Lean 4 v4.33.0, core only, no Mathlib | [`lean/`](./lean/) | **proved**: six theorems, standard axioms only, ~1.6 s | `tests/proofs/test_cross_setup_lean.py` |
| Rocq 9.3, Stdlib only | [`rocq/`](./rocq/) | **proved**: six theorems, closed under the global context, ~1.4 s | `tests/proofs/test_cross_setup_rocq.py` |
| Isabelle2025-2, HOL + `HOL-Library.Multiset` | [`isabelle/`](./isabelle/) | **proved**: six theorems, strict build (`quick_and_dirty=false`), no oracles, ~12 s | `tests/proofs/test_cross_setup_isabelle.py` |

The three proofs found the same things the statement had left implicit, now in the table below:
Triton's re-masked variance lanes hold `0.0 * 0.0`, and the CUDA tree is correct only because the
block width is a power of two (`kernels/cuda/ladder.py` guarantees it; with width 6, `partial[2]`
and `partial[5]` never reach `partial[0]` — Isabelle checks this counterexample as
`width_six_drops_partials`). **None of the three needed `n ≥ 1` for the consistency theorems, nor
idempotence of `max`**: the statement below asks for both, and every proof is of the stronger
theorem without them. Each proof states the kernels' literal IEEE behaviour (`0.0 * 0.0 = 0.0`,
and in Isabelle also `exp(-inf - M) = 0`) only as the premise of a side lemma, never of the six. These proofs are reference certificates written
while building the translation segment, not data from the agent-written-proof experiments (E1,
T4), whose agents must write their own.

## What it certifies, and what it leaves to PBT

The translation segment maps a **known** kernel from one setup to another. The first pair is the
recorded Triton ladder (`kernels/triton/ladder.py`) and its CUDA port (`kernels/cuda/*.cu`). The two
kernels reduce each row through different trees:

- **Triton (source).** One tile of width `B`, a power of two, holds the whole row. Lanes `i < n`
  load `x[i]`; masked lanes load the identity of the reduction (`-inf` for max; for the sum, a
  masked lane holds `exp(-inf - M) = 0`). `tl.max` / `tl.sum` reduce the tile as a balanced tree.
- **CUDA (port).** A block of `W` threads, a power of two. Thread `t` folds the columns
  `t, t + W, t + 2W, … < n` sequentially, starting from the identity, into `partial[t]`. The block
  then runs the in-place tree `for s = W/2, W/4, …, 1: if t < s: partial[t] ⊕= partial[t + s]`
  and reads `partial[0]`.

**In exact arithmetic these are the same function, and it is the spec**: that is the theorem. It
holds for *every* `n` and `W`, and for every `B ≥ n`. The hypothesis `n ≤ B` is necessary — with
`B < n` the tile model computes the reduction of the first `B` columns only, which is the
tile-tail defect measured on hardware (softmax rows summing to 1.51; see `CLAUDE.md`).

What the theorem does **not** cover, and who covers it:

| gap | covered by |
|---|---|
| floating-point rounding: the two trees round differently, `expf` vs Triton's `exp` | PBT: the cross-setup reference ratio (feature 0013) on byte-identical inputs |
| the models are hand-transcriptions of the kernel text, not extracted from it | PBT: every model is checked against the recorded executions (check C4 of the formal design) once executable readings exist; until then, by review against the `.cu` / Triton source |
| `exp(-inf) = 0`, `fmaxf` with `-inf`, and `0.0 * 0.0 = 0.0` in Triton's re-masked variance lanes (`ladder.py`, `tl.where(mask, x - mean, 0.0)` squared) | modelled as monoid identities; PBT's recorded outputs are the evidence. The Rocq proof carries the last as an explicit premise of a side lemma (`triton_var_tile_literal`) |
| the Triton tile model is a halving tree, not Triton's generated reduction order (per-thread, then across lanes, then across warps) | nothing needs to: in a commutative monoid every bracketing gives the same result, so the theorem holds for Triton's order too, but it is *stated* about the halving tree. The CUDA model follows the kernel text loop by loop |
| concurrency: `__syncthreads`, races | out of scope (formal design §13); the model is the sequentialized schedule the barriers enforce |

## The statement

Let `(M, ⊕, e)` be a **commutative monoid**. For a row `xs : list M` of length `n`:

```
spec xs = foldl (⊕) e xs
```

**Tile model** (Triton). For `B = 2^k`:

```
tile B xs   = [ if i < n then xs[i] else e  |  i ← 0 .. B-1 ]
tree [a]    = a
tree (l ++ r) = tree l ⊕ tree r          -- |l| = |r| = 2^(k-1)
tileReduce k xs = tree (tile (2^k) xs)
```

**Block model** (CUDA). For `W = 2^k`, as `softmax.cu` and `layernorm.cu` execute it:

```
strided W n t   = [ t, t + W, t + 2W, … ]  restricted to < n,  ascending
partial0 t      = foldl (⊕) e [ xs[c] | c ← strided W n t ]           -- phase 1
step s p t      = if t < s then p t ⊕ p (t + s) else p t              -- one tree level
blockReduce k xs = (step 1 ∘ step 2 ∘ … ∘ step (2^(k-1))) partial0  evaluated at t = 0
```

(the level with `s = W/2` runs first; `k = 0` means one thread and no tree level).

### The six theorems

Use these names, so the tests can audit each one by name.

| name | statement |
|---|---|
| `strided_cover` | for `W ≥ 1`: the concatenation over `t = 0 .. W-1` of `strided W n t` is a permutation of `[0, n)` — every column is visited by exactly one thread, once |
| `tile_reduce_eq_spec` | `length xs ≤ 2^k → tileReduce k xs = spec xs` |
| `tile_reduce_drops_tail` | `tileReduce k xs = spec (take (2^k) xs)`, for every `xs` — so for `length xs > 2^k` the tail is dropped |
| `block_reduce_eq_spec` | `blockReduce k xs = spec xs`, for every `k` and `xs`: no relation between `W` and `n` |
| `softmax_consistent` | the Triton and CUDA softmax models agree with the softmax spec on every column, for every `k_B` with `n ≤ 2^k_B`, every `k_W`, and `n ≥ 1` |
| `layernorm_consistent` | the same for layer normalization |

For the last two, work over an abstract carrier `F` with abstract operations `+`, `-`, `*`, `/`,
`exp`, `sqrt`, a cast `n ↦ (n : F)`, a constant `eps`, and a `max` that is commutative,
associative and idempotent. `(F, +, 0)` is a commutative monoid; for max, use `F` extended with a
bottom element `⊥` (`-inf`), on which `max` with `⊥` as identity is a commutative monoid. **No
axiom about `exp`, `sqrt` or `/` is needed or allowed**: consistency between two setups is a
statement about the reductions only, and the same `exp` appears on both sides.

```
softmaxSpec xs [i]    = exp (xs[i] - M) / S
    where M = the max over xs (finite, since n ≥ 1)
          S = spec_+ [ exp (x - M) | x ← xs ]
softmaxTriton kB xs   = the same, with M and S computed by tileReduce kB
softmaxCuda   kW xs   = the same, with M and S computed by blockReduce kW

layernormSpec xs [i]  = (xs[i] - mean) / sqrt (var + eps)
    where mean = spec_+ xs / n
          var  = spec_+ [ (x - mean) * (x - mean) | x ← xs ] / n
layernormTriton, layernormCuda: the same, with each sum computed by the corresponding model
```

In the Triton softmax model the masked lanes of the sum tile hold `0` directly: that `exp(-inf)`
is `0` is IEEE behaviour, recorded in the table above as a modelling assumption.

## Rules

- No escape hatches: no `sorry` / `admit` / `axiom` (Lean), no `Admitted` / `admit` / `Axiom` /
  `Parameter` used as an assumption (Rocq), no `sorry` / `oops` / `axiomatization` (Isabelle).
  Abstract operations are **parameters of the statement** (section variables, locale parameters,
  explicit arguments), never axioms.
- The model definitions follow the kernel text, loop by loop. A header comment in each file maps
  every definition to the line of `kernels/cuda/*.cu` or `kernels/triton/ladder.py` it models.

## Check them

See each subdirectory's README for the exact command; `tests/proofs/` runs each one when its
toolchain is installed and skips it otherwise.
