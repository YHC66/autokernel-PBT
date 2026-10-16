# Cross-setup reductions in Lean 4 (core only, no Mathlib)

`CrossSetup.lean` proves the theorem family stated in `../README.md`, in one file and with no
`lake` project. The file does not use `sorry`, `admit` or `native_decide`, and it declares no
axioms. Each theorem's `#print axioms` lists only Lean's standard `propext`,
`Classical.choice` and `Quot.sound`.

## Check it

```bash
cd proofs/cross-setup-reductions/lean && ~/.elan/toolchains/leanprover--lean4---v4.33.0/bin/lean CrossSetup.lean
```

`lean-toolchain` pins `leanprover/lean4:v4.33.0`. Call that binary directly: a bare `lean` may
make elan download another release. The run exits with code 0 and prints only the ten
`#print axioms` lines. Three runs on 2026-10-12 took 1.5 s of wall-clock time each
(`/usr/bin/time -p`: real 1.48, 1.48, 1.55; the first cold run took 1.71).

**The proof is not vacuous.** I changed each model in turn and confirmed that the file then
fails to compile:

- the stride `c + W` to `c + 2W`;
- the mask `i < n` to `i + 1 < n`;
- the tree read `p (t + s)` to `p (t + s + 1)`;
- the level `s = 2^j` to `2^j / 2`;
- the loop fuel `n` to `n / 2`.

## What is proved

`M` is any type with a commutative monoid `m : CMonoid M`, and `spec m xs = xs.foldl m.op m.e`.

| theorem | says |
|---|---|
| `strided_cover` | for `W ≥ 1`, `(List.range W).flatMap (strided W n)` is a `List.Perm` of `List.range n`: thread `t < W` of `for (c = t; c < n; c += W)` visits its columns, and every column `< n` is visited exactly once by exactly one thread |
| `tile_reduce_eq_spec` | `xs.length ≤ 2^k → tileReduce m k xs = spec m xs`: if the tile holds the whole row, the Triton model computes the spec |
| `tile_reduce_drops_tail` | `tileReduce m k xs = spec m (xs.take (2^k))` for every row: a row wider than the tile loses its tail (the measured tile-tail defect) |
| `block_reduce_eq_spec` | `blockReduce m k xs = spec m xs` for every `k` and every row; `W = 2^k` and `n` need no relation |
| `softmax_consistent` | for every `kB` with `n ≤ 2^kB` and every `kW`: `softmaxTriton … kB xs = softmaxSpec … xs ∧ softmaxCuda … kW xs = softmaxSpec … xs`. These are equalities of whole rows, so the models agree with the spec on every column |
| `layernorm_consistent` | the same for `layernormTriton`, `layernormCuda` and `layernormSpec` |

The last two hold for an abstract carrier `F` with these inputs:

- `plus : CMonoid F`, which is `(F, +, 0)`.
- `L : MaxOp F`, a `max` together with its commutativity and associativity.
- `O : Ops F`, which holds `sub`, `mul`, `div`, `exp`, `sqrt`, `cast : Nat → F` and `eps`.

Nothing is assumed about `O`. The consistency between setups is a fact about the reductions
alone.

The file also proves four lemmas, also audited by `#print axioms`, that tie the README's model
to the kernel text:

| lemma | says |
|---|---|
| `strided_unfold` | for `W ≥ 1`, `strided W n t = if t < n then t :: strided W n (t + W) else []`. This is the C loop's own recursion, so the fuel in the definition never runs out |
| `blockReduce_eq_loop` | `blockReduce` (the README's composition `step 1 ∘ … ∘ step (2^(k-1))`) equals `treeLoop`, the literal `for (s = W >> 1; s > 0; s >>= 1)` loop |
| `softmaxSpec_cons` | on a non-empty row, `softmaxSpec` is `exp (x - M) / S` at every column, with `M` the finite row max and `S = spec_+ [exp (y - M) | y ← xs]` |
| `triton_var_tile` | the kernel's literal variance tile (lines 105–106: centre under the mask, then square every lane, masked lanes included) reduces to the model's value, given `0 * 0 = 0` |

## Proof route

1. `spec_perm`: a `List.Perm` leaves `spec` unchanged. It follows from core's
   `List.Perm.foldl_eq'`, with right-commutativity derived from `op_assoc` and `op_comm`.
2. `strided_cover`: both lists have no duplicates and the same members
   (`List.perm_ext_iff_of_nodup`). `mem_strided` characterizes thread `t`'s columns as the
   `c < n` with `t ≤ c` and `c ≡ t (mod W)`. Each `strided` list is strictly increasing, and two
   threads `a < b < W` have disjoint residues.
3. Tile: `tile_eq` shows `tile B xs = xs.take B ++ replicate (B - n) e`. `tree_eq_spec` shows
   that the halving tree on a list of length `2^k` is `spec`. Padding with `e` does not change
   the result, which gives `tile_reduce_drops_tail`, and with `List.take_of_length_le` also
   `tile_reduce_eq_spec`.
4. Block: one level preserves the sum of the live slots (`sum_step`), so
   `levels j p 0 = spec [p t | t < 2^j]` (`levels_spec`). The fold of the `W` partials is the
   fold of the concatenated strided column lists (`spec_flatten`), and by `strided_cover` that
   is the fold of the row.
5. Kernels: `softmaxWith_congr` and `layernormWith_congr` rewrite each reduction to `spec`.
   Nothing is unfolded about `exp`, `sqrt`, `/`, `-`, `*` or the cast.

## How the definitions model the kernels

S = `kernels/cuda/softmax.cu`, L = `kernels/cuda/layernorm.cu`, T = `kernels/triton/ladder.py`.

| definition | kernel text | lines |
|---|---|---|
| `lane` | lane `i` of `tl.load(..., mask=offs < n_cols, other=e)`: `xs[i]` if `i < n`, else the identity | T:86–89, 99–101 |
| `tile B xs` | the `BLOCK`-wide tile, `offs = tl.arange(0, BLOCK)` | T:86, 99 |
| `tree`, `tileReduce` | `tl.max(x, axis=0)` and `tl.sum(…, axis=0)`, as the README's balanced halving tree | T:90, 93, 102, 106 |
| hypothesis `n ≤ 2^kB` | `block_for`: `BLOCK` is `n_cols` rounded up to a power of two | T:47–71 |
| `loopCols`, `strided` | `for (int c = tid; c < n_cols; c += width)` | S:34, 51, 64; L:43, 49, 56 |
| `partial0` | identity initialization, the sequential `acc = acc ⊕ x[c]` over the thread's columns, `partial[tid] = acc` | S:33–37, 50–54; L:41–45, 48–52, 18 |
| `step s` | `if (tid < s) partial[tid] = partial[tid] ⊕ partial[tid + s]` | S:40–42, 57–59; L:21–23 |
| `treeLoop` | `for (int s = width >> 1; s > 0; s >>= 1)`, literally | S:39, 56; L:20 |
| `levels`, `blockReduce` | that loop as the README's composition of levels, read at `partial[0]` | S:45, 62; L:26 |
| `blockReduce`'s `W = 2^k` | launch contract: width a power of two (`threads_for` in `kernels/cuda/ladder.py`) | S:14; L:11 |
| `withBot L` | `max` with `-inf` as identity: `negative_infinity()`, `other=-float("inf")` | S:19–21, 33; T:89 |
| `softmaxWith` | `x - row_max`, `exp`, `/ denom`; the max uses one reduction and the denominator another | S:45–66; T:90–93 |
| `softmaxTriton` | max tile with masked lanes `⊥`; sum tile with masked lanes `0` (the README's reading of `exp(-inf - M) = 0`) | T:89–93 |
| `softmaxCuda` | `softmax_rows`, both reductions by `blockReduce` | S:31–66 |
| `lnMean`, `lnVar` | `mean = sum(x) / n`, `var = sum((x - mean)^2) / n` | T:102, 105–106; L:46, 50–53 |
| `layernormWith` | `(x - mean) / sqrt(var + eps)` | T:107; L:54–58 |
| `layernormTriton`, `layernormCuda` | the same, with every sum by `tileReduce` or `blockReduce` | T:97–107; L:32–59 |

**The in-place tree.** At level `s`, thread `t < s` runs
`partial[t] = partial[t] ⊕ partial[t + s]`, and threads `t ≥ s` do nothing. Every write lands
at an index `< s`, and only thread `t` writes index `t`. Every read is either at `t`, by thread
`t` itself before its own write, or at `t + s ≥ s`, an index nobody writes at this level. So no
read sees a write from the same level, whatever the interleaving, and the level is the pure
function `step s p`. The `__syncthreads()` that closes each level (S:43, S:60, L:24) hands the
next level exactly that array. The model is this barrier-enforced schedule. Concurrency itself
is out of scope, as `../README.md` states.

## Lean-specific modelling choices

- **Commutative monoid.** `structure CMonoid` has the fields `op`, `e`, `op_assoc`, `op_comm`
  and `e_op` (a left identity; `CMonoid.op_e` derives the right identity). It is passed as an
  explicit argument, not as a class instance, so that `(F, +, 0)` and `(F ∪ {⊥}, max, ⊥)` can
  coexist. The laws are hypotheses that the caller must discharge.
- **Bottom for max.** `F ∪ {⊥}` is `Option F`, with `none = ⊥ = -inf`. `botMax` extends `max`,
  and `withBot L : CMonoid (Option F)` proves the monoid laws from `L.max_comm` and
  `L.max_assoc` by case analysis.
- **`xs[c]`.** `lane` returns `xs[c]` for `c < n` and `e` otherwise. In the tile, the `else`
  case is the masked lane. The CUDA loops only produce `c < n`, so there the `else` case is
  never taken (`mem_strided`).
- **The strided loop.** Lean needs a termination measure, and the C loop has none when
  `width = 0`. So `loopCols W n fuel c` is the loop body with fuel, and `strided` gives it `n`
  units, which bounds the trip count for every `W ≥ 1`. `strided_unfold` proves that the fuel
  is never the reason the loop stops.
- **The in-place array.** It is a function `Nat → M`. A tree level is `step s`. The loop is
  `treeLoop`, defined by well-founded recursion on `s`, which reproduces `s >>= 1` with Lean's
  `>>>`.
- **`tree` at `k = 0`** is `headD e`, which is `a` on `[a]`. At `k + 1` it splits with
  `take`/`drop (2^k)`. On lists of length `2^k`, the only ones it is applied to, this is the
  README's `tree (l ++ r) = tree l ⊕ tree r`.
- **Rows** are `List F`, and `n` is `xs.length`.

## Deviations from `../README.md`'s statement

None of these weakens the statement. The first two drop a hypothesis.

1. **`softmax_consistent` has no hypothesis `n ≥ 1`.** The models represent the row max in
   `Option F`. A row whose max is `⊥`, which happens only for the empty row, writes no column
   (`[]`). That is what both kernels do when `n_cols = 0`: their store loops run zero times.
   With this convention the equality holds for every `n`. The hypothesis matters only for the
   spec to be the textbook formula with a finite `M`, and `softmaxSpec_cons` proves exactly
   that for every non-empty row.
2. **`max` need not be idempotent.** The README lists commutative, associative and idempotent.
   Only commutativity and associativity are used, so `MaxOp` carries only those two.
3. **`0 * 0 = 0` in the Triton layernorm model.** Following the README ("each sum computed by
   the corresponding model"), the Triton variance tile is the squared row with masked lanes
   holding `0` directly. The kernel's literal tile squares the masked lanes too, so they hold
   `0.0 * 0.0`. `triton_var_tile` proves the two agree under the one hypothesis
   `O.mul 0 0 = 0`. This is an IEEE fact of the same kind as `exp(-inf) = 0`, but the README's
   table of modelling assumptions does not list it. The consistency theorems themselves assume
   nothing about `*`.

## What the proof does not show

- **Floating point.** Rounding, `expf` against Triton's `exp`, and IEEE special values are
  left to PBT, as `../README.md`'s table says.
- **`W` must be a power of two.** This is a real requirement of the kernel, not a convenience
  of the proof. With `W = 6`, the loop runs the levels `s = 3, 1` and never folds `partial[2]`
  or `partial[5]` into `partial[0]`. Evaluating `treeLoop` on six partials of `1` gives `4`,
  not `6` (checked with `#eval` against this file's definitions). `threads_for` guarantees a
  power of two in `[32, 1024]`.
- **Unbounded indices.** Indices are unbounded `Nat`. The kernels use `int c`, and `c + width`
  would overflow only when `n_cols > INT_MAX - width`, far beyond any recorded shape.
- **Triton's reduction tree.** The README models it as the balanced halving tree. Triton's
  `tl.reduce` actually folds within a thread, then across lanes by shuffles, then across warps.
  In exact arithmetic every such bracketing and order gives `spec` (`spec_perm` together with
  associativity), so this choice does not affect the consistency claim. It does matter for
  rounding, which is PBT's job.
- **The final store loop.** It is modelled as `xs.map`. Every column is written exactly once
  (`strided_cover`), and the written value depends only on `x[c]` and on block-uniform
  scalars.
